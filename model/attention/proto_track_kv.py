from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .dot_production_attention import get_multi_stage_dot_production_attention


def _as_float(t: torch.Tensor) -> torch.Tensor:
    if t.dtype.is_floating_point:
        return t
    return t.float()


@dataclass
class ProtoTrackConfig:
    """Configuration bundle for the ProtoTrack-KV cache."""

    window_size: int
    bank_size: int = 64
    pq_subspaces_k: int = 8
    pq_subspaces_v: int = 8
    pq_codewords: int = 16
    pq_update_interval: int = 300
    merge_interval: int = 90
    idle_ttl: int = 120
    alpha0: float = 0.05
    beta0: float = 0.05
    gamma: float = 0.1
    lambda_sp: float = 0.1
    lambda_idle: float = 0.2
    expansion: int = 1
    normalize_every: int = 64
    residual_buffer_size: int = 4096
    reseed_buffer_size: int = 512
    merge_threshold_k: float = 0.15
    merge_threshold_v: float = 0.10
    safety_threshold: float = 0.15
    replay_cap: int = 4

    @classmethod
    def from_dict(cls, data: Dict) -> "ProtoTrackConfig":
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            raise TypeError("proto_track_config must be a dict or ProtoTrackConfig instance")
        return cls(**data)


class PrototypeBank:
    """Maintains a constant-size object-centric prototype bank for a single head."""

    def __init__(
        self,
        config: ProtoTrackConfig,
        dim_key: int,
        dim_value: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.cfg = config
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.device = device
        self.dtype = dtype

        if dim_key % self.cfg.pq_subspaces_k != 0:
            raise ValueError(
                f"Key dimension {dim_key} must be divisible by pq_subspaces_k={self.cfg.pq_subspaces_k}"
            )
        if dim_value % self.cfg.pq_subspaces_v != 0:
            raise ValueError(
                f"Value dimension {dim_value} must be divisible by pq_subspaces_v={self.cfg.pq_subspaces_v}"
            )

        self.num_proto = self.cfg.bank_size
        self.subdim_k = dim_key // self.cfg.pq_subspaces_k
        self.subdim_v = dim_value // self.cfg.pq_subspaces_v

        self.key_centers = torch.zeros((self.num_proto, dim_key), device=device, dtype=dtype)
        self.value_centers = torch.zeros((self.num_proto, dim_value), device=device, dtype=dtype)
        self.mass = torch.zeros((self.num_proto,), device=device, dtype=torch.float32)
        self.mu = torch.zeros((self.num_proto, 2), device=device, dtype=dtype)
        self.sigma = torch.ones((self.num_proto, 2), device=device, dtype=dtype)
        self.last_used = torch.zeros((self.num_proto,), device=device, dtype=torch.long)

        self.hist_k = torch.zeros(
            (self.num_proto, self.cfg.pq_subspaces_k, self.cfg.pq_codewords),
            device=device,
            dtype=torch.int32,
        )
        self.hist_v = torch.zeros(
            (self.num_proto, self.cfg.pq_subspaces_v, self.cfg.pq_codewords),
            device=device,
            dtype=torch.int32,
        )

        self.codebook_k = torch.randn(
            (self.cfg.pq_subspaces_k, self.cfg.pq_codewords, self.subdim_k),
            device=device,
            dtype=torch.float32,
        ) * 0.01
        self.codebook_v = torch.randn(
            (self.cfg.pq_subspaces_v, self.cfg.pq_codewords, self.subdim_v),
            device=device,
            dtype=torch.float32,
        ) * 0.01

        self.residual_buffer_k: deque[torch.Tensor] = deque(maxlen=self.cfg.residual_buffer_size)
        self.residual_buffer_v: deque[torch.Tensor] = deque(maxlen=self.cfg.residual_buffer_size)
        self.reseed_buffer: deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = deque(
            maxlen=self.cfg.reseed_buffer_size
        )

        self.last_pq_refresh_ts = 0
        self.last_merge_ts = 0
        self.token_counter = 0

    # ---------------------------------------------------------------------
    # Core operations
    # ---------------------------------------------------------------------
    def ingest(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        coords: Optional[torch.Tensor],
        timestamp: int,
    ) -> None:
        if keys.numel() == 0:
            return

        if coords is None:
            coords = torch.zeros((keys.size(0), 2), device=keys.device, dtype=keys.dtype)

        # Simplified but more functional version
        keys_norm = F.normalize(keys.float(), dim=-1)
        
        # Find closest prototype for each token (batch operation)
        if self.mass.sum() > 0:
            centers_norm = F.normalize(self.key_centers.float() + 1e-6, dim=-1)
            active_mask = (self.mass > 0)
            if active_mask.any():
                # Use only active prototypes for assignment
                active_centers = centers_norm[active_mask]
                cosine = torch.matmul(keys_norm, active_centers.t())  # (n_tokens, n_active)
                best_active_idx = cosine.argmax(dim=-1)
                active_indices = active_mask.nonzero(as_tuple=False).view(-1)
                proto_indices = active_indices[best_active_idx]
            else:
                # All prototypes are inactive, use first one
                proto_indices = torch.zeros(keys.size(0), dtype=torch.long, device=keys.device)
        else:
            # No prototypes active, use sequential assignment
            proto_indices = torch.arange(min(keys.size(0), self.num_proto), device=keys.device)
            if keys.size(0) > self.num_proto:
                proto_indices = proto_indices.repeat((keys.size(0) + self.num_proto - 1) // self.num_proto)[:keys.size(0)]

        # Update prototypes (simplified EMA)
        for idx in range(keys.size(0)):
            proto_idx = int(proto_indices[idx].item())
            key = keys[idx]
            value = values[idx]
            
            mass = float(self.mass[proto_idx].item())
            alpha = 0.1 if mass > 0 else 1.0  # Full replacement for new prototypes
            beta = 0.1 if mass > 0 else 1.0
            
            if mass == 0:
                self.key_centers[proto_idx] = F.normalize(key.float(), dim=-1)
                self.value_centers[proto_idx] = value
                self.mass[proto_idx] = 1.0
            else:
                self.key_centers[proto_idx] = F.normalize(
                    (1 - alpha) * self.key_centers[proto_idx] + alpha * key,
                    dim=-1,
                )
                self.value_centers[proto_idx] = (1 - beta) * self.value_centers[proto_idx] + beta * value
                self.mass[proto_idx] += 1.0
            
            self.last_used[proto_idx] = timestamp

    def build_pseudo_tokens(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return expected prototype pseudo-tokens and logit biases.

        The returned tensors have shape ``(num_proto * expansion, dim)`` and include
        placeholder tokens (with ``-inf`` bias) for inactive prototypes to ensure a
        constant output size.
        """

        components_per_proto = max(1, self.cfg.expansion)
        total_components = self.num_proto * components_per_proto

        keys = []
        values = []
        biases = []

        for proto_idx in range(self.num_proto):
            mass = float(self.mass[proto_idx].item())
            if mass <= 0:
                zero_key = torch.zeros(self.dim_key, device=self.device, dtype=self.dtype)
                zero_value = torch.zeros(self.dim_value, device=self.device, dtype=self.dtype)
                for _ in range(components_per_proto):
                    keys.append(zero_key)
                    values.append(zero_value)
                    biases.append(float("-inf"))
                continue

            components = self._components_for_proto(proto_idx, components_per_proto)
            for residual_k, residual_v, log_weight in components:
                key = self.key_centers[proto_idx] + residual_k.to(self.key_centers.dtype)
                value = self.value_centers[proto_idx] + residual_v.to(self.value_centers.dtype)
                bias = math.log(max(mass, 1e-6)) + log_weight
                keys.append(key)
                values.append(value)
                biases.append(bias)

        if not keys:
            return (
                torch.zeros((0, self.dim_key), device=self.device, dtype=self.dtype),
                torch.zeros((0, self.dim_value), device=self.device, dtype=self.dtype),
                torch.zeros((0,), device=self.device, dtype=self.dtype),
            )

        key_tensor = torch.stack(keys, dim=0)
        value_tensor = torch.stack(values, dim=0)
        bias_tensor = torch.tensor(biases, device=self.device, dtype=self.dtype)
        return key_tensor, value_tensor, bias_tensor

    def build_replay_tokens(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct a bounded expansion of prototypes for retrieval without bias."""

        max_tokens = max(1, self.cfg.bank_size * max(1, self.cfg.replay_cap))
        key_cache = torch.zeros((max_tokens, self.dim_key), device=self.device, dtype=self.dtype)
        value_cache = torch.zeros((max_tokens, self.dim_value), device=self.device, dtype=self.dtype)
        cursor = 0

        for proto_idx in range(self.num_proto):
            mass = int(min(max(self.mass[proto_idx].item(), 0.0), self.cfg.replay_cap))
            if mass <= 0:
                continue

            residual_k, residual_v, _ = self._components_for_proto(proto_idx, 1)[0]
            key = self.key_centers[proto_idx] + residual_k.to(self.key_centers.dtype)
            value = self.value_centers[proto_idx] + residual_v.to(self.value_centers.dtype)

            available = max_tokens - cursor
            if available <= 0:
                break
            mass = min(mass, available)
            key_repeat = key.unsqueeze(0).expand(mass, -1)
            value_repeat = value.unsqueeze(0).expand(mass, -1)
            key_cache[cursor:cursor + mass] = key_repeat
            value_cache[cursor:cursor + mass] = value_repeat
            cursor += mass

        key_cache = key_cache[:cursor]
        value_cache = value_cache[:cursor]
        return key_cache, value_cache

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _assign_prototype(
        self,
        key_norm: torch.Tensor,
        centers_norm: torch.Tensor,
        coord: torch.Tensor,
        timestamp: int,
    ) -> int:
        empty = (self.mass == 0)
        if empty.any():
            return int(empty.nonzero(as_tuple=False)[0])

        cosine = torch.matmul(centers_norm, key_norm)
        cost = -cosine

        if self.cfg.lambda_sp > 0:
            diff = coord.float()[None, :] - self.mu.float()
            var = torch.clamp(self.sigma.float(), min=1e-3)
            maha = (diff * diff / var).sum(dim=-1)
            cost = cost + self.cfg.lambda_sp * maha

        if self.cfg.lambda_idle > 0:
            idle_mask = (timestamp - self.last_used) > self.cfg.idle_ttl
            cost = cost + self.cfg.lambda_idle * idle_mask.float()

        return int(cost.argmin().item())

    def _update_prototype(
        self,
        proto_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        coord: torch.Tensor,
        timestamp: int,
    ) -> None:
        mass = float(self.mass[proto_idx].item())
        alpha = self.cfg.alpha0 / math.sqrt(mass + 1.0)
        beta = self.cfg.beta0 / math.sqrt(mass + 1.0)

        self.key_centers[proto_idx] = F.normalize(
            (1 - alpha) * self.key_centers[proto_idx] + alpha * key,
            dim=-1,
        )
        self.value_centers[proto_idx] = (1 - beta) * self.value_centers[proto_idx] + beta * value

        gamma = self.cfg.gamma
        self.mu[proto_idx] = (1 - gamma) * self.mu[proto_idx] + gamma * coord
        diff = coord - self.mu[proto_idx]
        var = torch.clamp(diff * diff, min=1e-4)
        self.sigma[proto_idx] = (1 - gamma) * self.sigma[proto_idx] + gamma * var

        self.mass[proto_idx] = self.mass[proto_idx] + 1.0
        self.last_used[proto_idx] = timestamp
        self.token_counter += 1

        residual_k = (key - self.key_centers[proto_idx]).detach()  # after EMA
        residual_v = (value - self.value_centers[proto_idx]).detach()

        self._update_histograms(proto_idx, residual_k, residual_v)
        self.residual_buffer_k.append(residual_k.float().cpu())
        self.residual_buffer_v.append(residual_v.float().cpu())
        self.reseed_buffer.append((key.detach().cpu(), value.detach().cpu(), coord.detach().cpu()))

    def _update_histograms(
        self,
        proto_idx: int,
        residual_k: torch.Tensor,
        residual_v: torch.Tensor,
    ) -> None:
        code_idx_k = self._assign_code(residual_k, self.codebook_k, self.cfg.pq_subspaces_k)
        code_idx_v = self._assign_code(residual_v, self.codebook_v, self.cfg.pq_subspaces_v)

        for s, idx in enumerate(code_idx_k):
            idx = int(idx.item())
            self.hist_k[proto_idx, s, idx] = torch.clamp(
                self.hist_k[proto_idx, s, idx] + 1,
                max=2 ** 16 - 1,
            )

        for s, idx in enumerate(code_idx_v):
            idx = int(idx.item())
            self.hist_v[proto_idx, s, idx] = torch.clamp(
                self.hist_v[proto_idx, s, idx] + 1,
                max=2 ** 16 - 1,
            )

    def _assign_code(
        self,
        residual: torch.Tensor,
        codebook: torch.Tensor,
        num_subspaces: int,
    ) -> torch.Tensor:
        if residual.numel() == 0:
            return torch.zeros((num_subspaces,), device=self.device, dtype=torch.long)

        residual = residual.view(num_subspaces, -1).float()
        diff = residual[:, None, :] - codebook  # (S, m, d_sub)
        dist = torch.sum(diff * diff, dim=-1)
        return dist.argmin(dim=-1)

    def _expected_residual(
        self,
        histogram: torch.Tensor,
        codebook: torch.Tensor,
        num_subspaces: int,
    ) -> torch.Tensor:
        probs = histogram.float()
        norm = probs.sum(dim=-1, keepdim=True).clamp_min(1.0)
        probs = probs / norm
        expected = torch.sum(probs[..., None] * codebook, dim=-2)
        return expected.reshape(num_subspaces * codebook.size(-1))

    def _components_for_proto(
        self,
        proto_idx: int,
        count: int,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
        base_res_k = self._expected_residual(
            self.hist_k[proto_idx], self.codebook_k, self.cfg.pq_subspaces_k
        )
        base_res_v = self._expected_residual(
            self.hist_v[proto_idx], self.codebook_v, self.cfg.pq_subspaces_v
        )

        components: List[Tuple[torch.Tensor, torch.Tensor, float]] = [
            (base_res_k, base_res_v, 0.0)
        ]

        if count <= 1:
            return components

        combos = self._top_code_components(proto_idx, count - 1)
        components.extend(combos)

        while len(components) < count:
            components.append((base_res_k.clone(), base_res_v.clone(), -math.inf))

        return components[:count]

    def _top_code_components(
        self,
        proto_idx: int,
        num_components: int,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
        hist_k = self.hist_k[proto_idx].float()
        norm_k = hist_k.sum(dim=-1, keepdim=True).clamp_min(1.0)
        probs_k = hist_k / norm_k
        top_vals, top_idx = torch.topk(
            probs_k,
            k=min(self.cfg.pq_codewords, max(1, num_components)),
            dim=-1,
        )

        components: List[Tuple[torch.Tensor, torch.Tensor, float]] = []

        for comp in range(num_components):
            idx_selection = []
            log_weight = 0.0
            for s in range(self.cfg.pq_subspaces_k):
                pos = min(comp, top_idx.size(1) - 1)
                idx_code = int(top_idx[s, pos].item())
                prob = float(top_vals[s, pos].item())
                idx_selection.append(idx_code)
                log_weight += math.log(max(prob, 1e-6))

            residual_k_parts = []
            residual_v_parts = []
            for s, idx_code in enumerate(idx_selection):
                residual_k_parts.append(self.codebook_k[s, idx_code])
                idx_v = idx_code % self.cfg.pq_codewords
                if s < self.cfg.pq_subspaces_v:
                    residual_v_parts.append(self.codebook_v[s, idx_v])
                else:
                    residual_v_parts.append(torch.zeros(self.subdim_v, device=self.device, dtype=torch.float32))

            residual_k = torch.cat(residual_k_parts, dim=0)
            residual_v = torch.cat(residual_v_parts, dim=0)
            components.append((residual_k, residual_v, log_weight))

        return components

    def _maintenance(self, timestamp: int) -> None:
        idle_mask = (timestamp - self.last_used) > self.cfg.idle_ttl
        if idle_mask.any():
            self.mass[idle_mask] = torch.floor(self.mass[idle_mask] * 0.5)
            reset_mask = self.mass <= 0
            if reset_mask.any():
                self._reset_prototypes(reset_mask)
            self.last_used[idle_mask] = timestamp

        if timestamp - self.last_pq_refresh_ts >= self.cfg.pq_update_interval:
            self._refresh_codebooks()
            self.last_pq_refresh_ts = timestamp

        if timestamp - self.last_merge_ts >= self.cfg.merge_interval:
            self._merge_and_reseed(timestamp)
            self.last_merge_ts = timestamp

    def _reset_prototypes(self, mask: torch.Tensor) -> None:
        self.key_centers[mask] = 0
        self.value_centers[mask] = 0
        self.hist_k[mask] = 0
        self.hist_v[mask] = 0
        self.mu[mask] = 0
        self.sigma[mask] = 1
        self.mass[mask] = 0

    def _refresh_codebooks(self) -> None:
        if not self.residual_buffer_k or not self.residual_buffer_v:
            return

        residual_k = torch.stack(list(self.residual_buffer_k), dim=0)
        residual_v = torch.stack(list(self.residual_buffer_v), dim=0)

        self._update_codebook(residual_k, self.codebook_k, self.cfg.pq_subspaces_k)
        self._update_codebook(residual_v, self.codebook_v, self.cfg.pq_subspaces_v)

    def _update_codebook(
        self,
        samples: torch.Tensor,
        codebook: torch.Tensor,
        num_subspaces: int,
    ) -> None:
        if samples.numel() == 0:
            return

        samples = samples.to(self.device).view(samples.size(0), num_subspaces, -1)
        for s in range(num_subspaces):
            subspace_samples = samples[:, s, :]
            if subspace_samples.size(0) < self.cfg.pq_codewords:
                continue
            # Simple k-means approximation: randomly sample initial centroids and do a few updates
            try:
                n_samples = subspace_samples.size(0)
                indices = torch.randperm(n_samples)[:self.cfg.pq_codewords]
                centroids = subspace_samples[indices].clone()
                
                # Do a few k-means iterations
                for _ in range(5):
                    distances = torch.cdist(subspace_samples, centroids)
                    assignments = distances.argmin(dim=1)
                    for c in range(self.cfg.pq_codewords):
                        mask = assignments == c
                        if mask.any():
                            centroids[c] = subspace_samples[mask].mean(dim=0)
                
                codebook[s] = centroids.to(codebook.dtype)
            except RuntimeError:
                # fallback to random sampling
                chunk = subspace_samples[: self.cfg.pq_codewords]
                codebook[s] = chunk.to(codebook.dtype)

    def _merge_and_reseed(self, timestamp: int) -> None:
        active_mask = self.mass > 0
        indices = active_mask.nonzero(as_tuple=False).view(-1)
        if indices.numel() < 2:
            return

        centers_k = self.key_centers[indices].float()
        centers_v = self.value_centers[indices].float()
        dist_k = torch.cdist(centers_k, centers_k, p=2)
        dist_v = torch.cdist(centers_v, centers_v, p=2)
        dist_k.fill_diagonal_(float("inf"))

        min_val, min_idx = dist_k.min(dim=-1)
        pair_val, pair_source = min_val.min(dim=0)
        if pair_val.item() > self.cfg.merge_threshold_k:
            return

        src = indices[int(pair_source.item())]
        dst = indices[int(min_idx[int(pair_source.item())].item())]

        if torch.norm(self.value_centers[src] - self.value_centers[dst]).item() > self.cfg.merge_threshold_v:
            return

        mass_src = self.mass[src].item()
        mass_dst = self.mass[dst].item()
        if mass_dst > mass_src:
            src, dst = dst, src
            mass_src, mass_dst = mass_dst, mass_src

        total_mass = max(mass_src + mass_dst, 1.0)
        weight_src = mass_src / total_mass
        weight_dst = mass_dst / total_mass

        self.key_centers[src] = F.normalize(
            weight_src * self.key_centers[src] + weight_dst * self.key_centers[dst], dim=-1
        )
        self.value_centers[src] = weight_src * self.value_centers[src] + weight_dst * self.value_centers[dst]
        self.mu[src] = weight_src * self.mu[src] + weight_dst * self.mu[dst]
        self.sigma[src] = weight_src * self.sigma[src] + weight_dst * self.sigma[dst]

        self.hist_k[src] += self.hist_k[dst]
        self.hist_v[src] += self.hist_v[dst]
        self.mass[src] = torch.tensor(total_mass, device=self.device)
        self.last_used[src] = timestamp

        self._reset_prototypes(self.mass == 0)
        self._reseed_prototype(int(dst.item()), timestamp)

    def _reseed_prototype(self, proto_idx: int, timestamp: int) -> None:
        if self.reseed_buffer:
            key, value, coord = self.reseed_buffer.pop()
            key = key.to(self.device, self.dtype)
            value = value.to(self.device, self.dtype)
            coord = coord.to(self.device, self.dtype)
            self.key_centers[proto_idx] = F.normalize(key, dim=-1)
            self.value_centers[proto_idx] = value
            self.mu[proto_idx] = coord
            self.sigma[proto_idx] = torch.ones_like(self.sigma[proto_idx])
            self.mass[proto_idx] = torch.tensor(1.0, device=self.device)
            self.hist_k[proto_idx] = 0
            self.hist_v[proto_idx] = 0
            self.last_used[proto_idx] = timestamp
        else:
            mask = torch.zeros_like(self.mass, dtype=torch.bool)
            mask[proto_idx] = True
            self._reset_prototypes(mask)


class ProtoTrackContextManager:
    """Context manager implementing ProtoTrack-KV for streaming attention."""

    def __init__(
        self,
        position_embedding,
        proto_config: Dict,
        fattn: bool = False,
    ) -> None:
        self.position_embedding = position_embedding
        self.config = ProtoTrackConfig.from_dict(proto_config)
        self.Attn, _ = get_multi_stage_dot_production_attention(False if self.config.expansion > 1 else fattn)

        self.batch_size = 0
        self.num_heads = 0
        self.num_heads_kv = 0
        self.dim_head = 0
        self.device: Optional[torch.device] = None
        self.dtype: Optional[torch.dtype] = None

        self.recent_k: Optional[torch.Tensor] = None
        self.recent_v: Optional[torch.Tensor] = None
        self.recent_coords: Optional[torch.Tensor] = None

        self.prototype_banks: List[List[PrototypeBank]] = []

        self.initialized = False
        self.to_retrieve = False
        self.retrieved_block_indices = None
        self.stream_position = 0
        self.token_cursor = 0

    # ------------------------------------------------------------------
    # Interface methods matching ContextManager
    # ------------------------------------------------------------------
    def init(
        self,
        local_q: torch.Tensor,
        local_k: torch.Tensor,
        local_v: torch.Tensor,
        global_q: torch.Tensor,
        global_k: torch.Tensor,
        global_v: torch.Tensor,
    ) -> None:
        self.batch_size = local_q.size(0)
        self.num_heads = local_q.size(1)
        self.num_heads_kv = local_k.size(1)
        self.dim_head = local_q.size(-1)
        self.device = local_q.device
        self.dtype = local_q.dtype

        cfg = self.config
        window = max(1, cfg.window_size)
        self.recent_k = torch.empty((self.batch_size, self.num_heads_kv, 0, self.dim_head), device=self.device, dtype=local_k.dtype)
        self.recent_v = torch.empty((self.batch_size, self.num_heads_kv, 0, self.dim_head), device=self.device, dtype=local_v.dtype)
        self.recent_coords = torch.empty((self.batch_size, self.num_heads_kv, 0, 2), device=self.device, dtype=local_q.dtype)

        self.prototype_banks = [
            [
                PrototypeBank(cfg, self.dim_head, self.dim_head, self.device, local_k.dtype)
                for _ in range(self.num_heads_kv)
            ]
            for _ in range(self.batch_size)
        ]

        self.initialized = True

    def append(
        self,
        local_q: torch.Tensor,
        local_k: torch.Tensor,
        local_v: torch.Tensor,
        global_q: torch.Tensor,
        global_k: torch.Tensor,
        global_v: torch.Tensor,
    ) -> torch.Tensor:
        if not self.initialized:
            self.init(local_q, local_k, local_v, global_q, global_k, global_v)

        length = local_q.size(-2)
        self._update_recent_cache(local_k, local_v, length)

        near_k = self.recent_k
        near_v = self.recent_v
        far_k, far_v, far_bias = self._build_far_pseudo()

        near_k_full = self._from_group_kv(near_k)
        near_v_full = self._from_group_kv(near_v)
        far_k_full = self._from_group_kv(far_k)
        far_v_full = self._from_group_kv(far_v)
        far_bias_full = self._expand_bias(far_bias)

        # Attention over near window only (simplified for now)
        local_h_q, local_h_k = self.position_embedding(local_q, local_k)
        local_h_v = local_v
        
        # Use only recent cache for attention (skip far pseudo-tokens temporarily)
        attn = self.Attn(local_q.shape, local_q.dtype, local_q.device)
        attn.append(
            local_h_q,
            local_h_k,
            local_h_v,
            sliding_window=self.config.window_size,
            end=True,
        )

        output, _ = attn.get_result()
        output = output.view(self.batch_size, self.num_heads, length, self.dim_head)
        return output.contiguous()

    def get_retrieved_kv(self, query=None):
        far_keys, far_values = self._build_far_replay()
        combined_k = torch.cat([self.recent_k, far_keys], dim=-2)
        combined_v = torch.cat([self.recent_v, far_values], dim=-2)
        return combined_k, combined_v

    def set_retrieval(self):
        self.to_retrieve = True

    def reset_retrieval(self):
        self.to_retrieve = False
        self.retrieved_block_indices = None

    def set_retrieved_block_indices(self, retrieved_block_indices):
        self.retrieved_block_indices = retrieved_block_indices

    def size(self, *args, **kwargs):
        return self.stream_position

    def calculate_cpu_memory(self):
        total = 0
        for banks in self.prototype_banks:
            for bank in banks:
                total += bank.key_centers.numel() * bank.key_centers.element_size()
                total += bank.value_centers.numel() * bank.value_centers.element_size()
                total += bank.hist_k.numel() * bank.hist_k.element_size()
                total += bank.hist_v.numel() * bank.hist_v.element_size()
        return total

    # ------------------------------------------------------------------
    # Helper routines
    # ------------------------------------------------------------------
    def _update_recent_cache(self, new_k: torch.Tensor, new_v: torch.Tensor, length: int) -> None:
        assert self.recent_k is not None and self.recent_v is not None
        coords = self._make_time_coords(length, new_k.device, new_k.dtype)

        self.recent_k = torch.cat([self.recent_k, new_k], dim=-2)
        self.recent_v = torch.cat([self.recent_v, new_v], dim=-2)
        self.recent_coords = torch.cat([self.recent_coords, coords], dim=-2)

        overflow = max(0, self.recent_k.size(-2) - self.config.window_size)
        if overflow > 0:
            old_k = self.recent_k[:, :, :overflow, :]
            old_v = self.recent_v[:, :, :overflow, :]
            old_c = self.recent_coords[:, :, :overflow, :]
            self._absorb_far_tokens(old_k, old_v, old_c)
            self.recent_k = self.recent_k[:, :, overflow:, :]
            self.recent_v = self.recent_v[:, :, overflow:, :]
            self.recent_coords = self.recent_coords[:, :, overflow:, :]

        self.stream_position += length

    def _absorb_far_tokens(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        coords: torch.Tensor,
    ) -> None:
        timestamp = self.stream_position
        for b in range(self.batch_size):
            for h in range(self.num_heads_kv):
                bank = self.prototype_banks[b][h]
                bank.ingest(keys[b, h], values[b, h], coords[b, h], timestamp)

    def _build_far_pseudo(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.device
        dtype = self.dtype
        assert device is not None and dtype is not None

        per_proto = max(1, self.config.expansion)
        total = self.config.bank_size * per_proto

        k_tensor = torch.zeros(
            (self.batch_size, self.num_heads_kv, total, self.dim_head),
            device=device,
            dtype=dtype,
        )
        v_tensor = torch.zeros_like(k_tensor)
        b_tensor = torch.full(
            (self.batch_size, self.num_heads_kv, total),
            float("-inf"),
            device=device,
            dtype=dtype,
        )

        for b in range(self.batch_size):
            for h in range(self.num_heads_kv):
                bank = self.prototype_banks[b][h]
                k_proto, v_proto, bias = bank.build_pseudo_tokens()
                k_tensor[b, h] = k_proto.view(total, self.dim_head)
                v_tensor[b, h] = v_proto.view(total, self.dim_head)
                b_tensor[b, h] = bias.view(total)

        b_tensor = b_tensor.unsqueeze(2)  # (B, H_kv, 1, total)
        return k_tensor, v_tensor, b_tensor

    def _build_far_replay(self) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.device
        dtype = self.dtype
        assert device is not None and dtype is not None

        max_tokens = max(1, self.config.bank_size * max(1, self.config.replay_cap))
        k_tensor = torch.zeros(
            (self.batch_size, self.num_heads_kv, max_tokens, self.dim_head),
            device=device,
            dtype=dtype,
        )
        v_tensor = torch.zeros_like(k_tensor)
        lengths = torch.zeros((self.batch_size, self.num_heads_kv), device=device, dtype=torch.long)

        for b in range(self.batch_size):
            for h in range(self.num_heads_kv):
                bank = self.prototype_banks[b][h]
                replay_k, replay_v = bank.build_replay_tokens()
                length = replay_k.size(0)
                length = min(length, max_tokens)
                if length > 0:
                    k_tensor[b, h, :length] = replay_k[:length]
                    v_tensor[b, h, :length] = replay_v[:length]
                    lengths[b, h] = length

        max_len = int(lengths.max().item())
        if max_len == 0:
            return (
                torch.zeros((self.batch_size, self.num_heads_kv, 0, self.dim_head), device=device, dtype=dtype),
                torch.zeros((self.batch_size, self.num_heads_kv, 0, self.dim_head), device=device, dtype=dtype),
            )

        return k_tensor[:, :, :max_len, :], v_tensor[:, :, :max_len, :]

    def _make_time_coords(
        self,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if length == 0:
            return torch.empty((self.batch_size, self.num_heads_kv, 0, 2), device=device, dtype=dtype)

        start = self.stream_position
        timeline = torch.arange(start, start + length, device=device, dtype=dtype)
        norm = max(float(start + length), 1.0)
        coords = torch.zeros((self.batch_size, self.num_heads_kv, length, 2), device=device, dtype=dtype)
        coords[..., 0] = timeline / norm
        return coords

    def _from_group_kv(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.size(1) == self.num_heads:
            return tensor
        num_group = self.num_heads // self.num_heads_kv
        tensor = tensor.view(self.batch_size, self.num_heads_kv, 1, tensor.size(2), self.dim_head)
        tensor = tensor.expand(self.batch_size, self.num_heads_kv, num_group, tensor.size(3), tensor.size(4))
        tensor = tensor.reshape(self.batch_size, self.num_heads, tensor.size(3), tensor.size(4))
        return tensor

    def _expand_bias(self, bias: torch.Tensor) -> torch.Tensor:
        if bias.numel() == 0:
            return bias
        bias = bias.to(self.device)
        if bias.size(1) == self.num_heads:
            return bias
        num_group = self.num_heads // self.num_heads_kv
        bias = bias.expand(self.batch_size, self.num_heads_kv, num_group, bias.size(-1))
        bias = bias.reshape(self.batch_size, self.num_heads, 1, bias.size(-1))
        return bias
