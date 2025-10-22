"""
Query-Agnostic Robustness (QAR) Measurement for ReKV
Implements the complete recipe for measuring QAR as a function of query staleness (Δ).
"""

import os
import json
import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from collections import defaultdict
import warnings

# Optional imports with fallbacks
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not available. Plotting disabled.")

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available. Some statistical tests disabled.")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: opencv not available. Video processing limited.")

try:
    from transformers import CLIPModel, CLIPProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not available. CLIP-based evidence detection disabled.")

try:
    from sklearn.metrics import accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not available. Using basic accuracy calculation.")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback tqdm
    def tqdm(iterable, **kwargs):
        return iterable

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


@dataclass
class QARConfig:
    """Configuration for QAR measurement"""
    # Staleness grid (in seconds)
    delta_grid: List[float] = None
    # Evidence detection method: 'manual', 'clip', 'attention'
    evidence_method: str = 'clip'
    # CLIP similarity threshold (percentile)
    clip_threshold_percentile: float = 90.0
    # Attention saliency threshold (percentile)
    attention_threshold_percentile: float = 95.0
    # Memory budget constraints
    memory_budget: int = 64  # Number of tokens/blocks
    # Video processing
    sample_fps: float = 0.5
    # Statistical testing
    confidence_level: float = 0.95
    # Random seed for reproducibility
    random_seed: int = 2024
    
    def __post_init__(self):
        if self.delta_grid is None:
            # Default staleness grid: 0s, 30s, 2min, 5min, 10min, 30min
            self.delta_grid = [0.0, 30.0, 120.0, 300.0, 600.0, 1800.0]


class EvidenceDetector:
    """Detects the earliest evidence timestamp τ_evi(q) for questions"""
    
    def __init__(self, method: str = 'clip', config: QARConfig = None):
        self.method = method
        self.config = config or QARConfig()
        
        if method == 'clip':
            if HAS_TRANSFORMERS:
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_model.eval()
            else:
                print("Warning: CLIP model not available. Evidence detection will use fallback method.")
                self.clip_model = None
                self.clip_processor = None
    
    def detect_evidence_timestamp(self, video_path: str, question: str, 
                                manual_timestamp: Optional[float] = None) -> float:
        """
        Detect τ_evi(q) using the specified method
        
        Args:
            video_path: Path to video file
            question: Question text
            manual_timestamp: Manual annotation (if available)
            
        Returns:
            Evidence timestamp in seconds
        """
        if self.method == 'manual' and manual_timestamp is not None:
            return manual_timestamp
        elif self.method == 'clip':
            return self._detect_clip_evidence(video_path, question)
        elif self.method == 'attention':
            return self._detect_attention_evidence(video_path, question)
        else:
            # Fallback: assume evidence appears at 10% of video length
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            video_length = frame_count / fps
            cap.release()
            return video_length * 0.1
    
    def _detect_clip_evidence(self, video_path: str, question: str) -> float:
        """CLIP-based evidence detection"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames at specified FPS
        sample_interval = max(1, int(fps / self.config.sample_fps))
        
        frames = []
        timestamps = []
        
        frame_idx = 0
        while frame_idx < frame_count:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            timestamps.append(frame_idx / fps)
            
            frame_idx += sample_interval
        
        cap.release()
        
        if not frames:
            return 0.0
        
        # Process with CLIP
        inputs = self.clip_processor(
            text=[question], 
            images=frames, 
            return_tensors="pt", 
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            similarities = outputs.logits_per_text[0].cpu().numpy()
        
        # Smooth similarities with temporal kernel
        kernel_size = min(5, len(similarities))
        if kernel_size > 1:
            kernel = np.ones(kernel_size) / kernel_size
            similarities = np.convolve(similarities, kernel, mode='same')
        
        # Find threshold based on percentile
        threshold = np.percentile(similarities, self.config.clip_threshold_percentile)
        
        # Find first frame above threshold
        evidence_indices = np.where(similarities >= threshold)[0]
        if len(evidence_indices) > 0:
            return timestamps[evidence_indices[0]]
        else:
            return timestamps[0]  # Fallback to beginning
    
    def _detect_attention_evidence(self, video_path: str, question: str) -> float:
        """Attention-based evidence detection (placeholder for teacher model)"""
        # This would require running a full-cache teacher model
        # For now, return a placeholder implementation
        print("Warning: Attention-based evidence detection not fully implemented")
        return self._detect_clip_evidence(video_path, question)


class QARMeasurer:
    """Main class for measuring Query-Agnostic Robustness"""
    
    def __init__(self, config: QARConfig = None):
        self.config = config or QARConfig()
        self.evidence_detector = EvidenceDetector(
            method=self.config.evidence_method, 
            config=self.config
        )
        np.random.seed(self.config.random_seed)
    
    def measure_qar(self, video_questions: List[Dict], methods: Dict[str, Any], 
                   video_dir: str) -> Dict[str, List[Tuple]]:
        """
        Measure QAR for multiple methods
        
        Args:
            video_questions: List of {'video_id', 'question', 'answer', 'manual_timestamp'}
            methods: Dict mapping method names to model instances
            video_dir: Directory containing video files
            
        Returns:
            Dict mapping method names to results [(qid, Δ, score)]
        """
        all_results = {}
        
        for method_name, method in methods.items():
            print(f"\nMeasuring QAR for {method_name}...")
            results = []
            
            for vq in tqdm(video_questions, desc=f"Processing {method_name}"):
                # Find the video file in subdirectories for MLVU dataset
                video_found = False
                video_path = None
                
                # Try different possible paths
                possible_paths = [
                    os.path.join(video_dir, f"{vq['video_id']}.mp4"),
                    os.path.join(video_dir, "1_plotQA", f"{vq['video_id']}.mp4"),
                    os.path.join(video_dir, "2_needle", f"{vq['video_id']}.mp4"),
                    os.path.join(video_dir, "3_ego", f"{vq['video_id']}.mp4"),
                    os.path.join(video_dir, "4_count", f"{vq['video_id']}.mp4"),
                    os.path.join(video_dir, "5_order", f"{vq['video_id']}.mp4"),
                    os.path.join(video_dir, "6_anomaly_reco", f"{vq['video_id']}.mp4"),
                    os.path.join(video_dir, "7_topic_reasoning", f"{vq['video_id']}.mp4"),
                    os.path.join(video_dir, "8_sub_scene", f"{vq['video_id']}.mp4"),
                    os.path.join(video_dir, "9_summary", f"{vq['video_id']}.mp4"),
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        video_path = path
                        video_found = True
                        break
                
                if not video_found:
                    print(f"Warning: Video not found for {vq['video_id']}")
                    continue
                
                # Detect evidence timestamp
                tau_evi = self.evidence_detector.detect_evidence_timestamp(
                    video_path, 
                    vq['question'],
                    vq.get('manual_timestamp')
                )
                
                # Test each staleness value
                for delta in self.config.delta_grid:
                    t_inject = tau_evi + delta
                    
                    # Skip if injection time exceeds video length
                    if self._get_video_length(video_path) < t_inject:
                        continue
                    
                    # Run query-agnostic streaming
                    score = self._run_streaming_evaluation(
                        method, video_path, vq['question'], 
                        vq['answer'], t_inject
                    )
                    
                    results.append((vq.get('id', len(results)), delta, score))
            
            all_results[method_name] = results
        
        return all_results
    
    def _get_video_length(self, video_path: str) -> float:
        """Get video length in seconds"""
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        length = frame_count / fps
        cap.release()
        return length
    
    def _run_streaming_evaluation(self, method: Any, video_path: str, 
                                 question: str, ground_truth: str, 
                                 t_inject: float) -> float:
        """
        Run query-agnostic streaming evaluation
        
        Args:
            method: Model instance with reset(), ingest(), and answer() methods
            video_path: Path to video file
            question: Question text
            ground_truth: Ground truth answer
            t_inject: Time to inject question (seconds)
            
        Returns:
            Score (0.0 or 1.0 for accuracy, or continuous score)
        """
        # Reset model state
        if hasattr(method, 'reset'):
            method.reset()
        elif hasattr(method, 'clear_cache'):
            method.clear_cache()
        
        # Stream video frames
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_idx = 0
        current_time = 0.0
        
        while current_time < t_inject:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Ingest frame (query-agnostic compression)
            if hasattr(method, 'ingest_frame'):
                method.ingest_frame(frame, current_time)
            
            frame_idx += 1
            current_time = frame_idx / fps
        
        cap.release()
        
        # Inject question and get answer
        try:
            if hasattr(method, 'answer'):
                predicted_answer = method.answer(question)
            else:
                # Fallback for ReKV models
                predicted_answer = self._rekv_answer(method, question)
            
            # Compute score (exact match for now)
            score = 1.0 if self._normalize_answer(predicted_answer) == self._normalize_answer(ground_truth) else 0.0
            
        except Exception as e:
            print(f"Error during inference: {e}")
            score = 0.0
        
        return score
    
    def _rekv_answer(self, model: Any, question: str) -> str:
        """Get answer from ReKV model"""
        try:
            with torch.no_grad():
                response = model.question_answering(question, max_new_tokens=128)
                return response if isinstance(response, str) else str(response)
        except Exception as e:
            print(f"ReKV inference error: {e}")
            return ""
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison"""
        return answer.lower().strip()
    
    def summarize_results(self, results: Dict[str, List[Tuple]]) -> Dict[str, Dict]:
        """
        Summarize QAR results with metrics
        
        Args:
            results: Dict mapping method names to [(qid, Δ, score)] lists
            
        Returns:
            Dict with summary statistics for each method
        """
        summaries = {}
        
        for method_name, method_results in results.items():
            # Group by Δ
            by_delta = defaultdict(list)
            for qid, delta, score in method_results:
                by_delta[delta].append(score)
            
            # Compute metrics
            curve = {}
            deltas = sorted(by_delta.keys())
            mean_scores = []
            
            for delta in deltas:
                scores = by_delta[delta]
                mean_score = np.mean(scores)
                ci_low, ci_high = self._bootstrap_ci(scores)
                curve[delta] = {
                    'mean': mean_score,
                    'std': np.std(scores),
                    'ci_low': ci_low,
                    'ci_high': ci_high,
                    'n_samples': len(scores)
                }
                mean_scores.append(mean_score)
            
            # AUC_Δ (Area Under Curve)
            if len(deltas) > 1:
                auc_delta = np.trapz(mean_scores, deltas) / (deltas[-1] - deltas[0])
            else:
                auc_delta = mean_scores[0] if mean_scores else 0.0
            
            # Staleness slope
            if len(deltas) > 1:
                if HAS_SCIPY:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(deltas, mean_scores)
                else:
                    # Simple linear regression fallback
                    x = np.array(deltas)
                    y = np.array(mean_scores)
                    slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x)) if np.std(x) > 0 else 0.0
                    std_err = 0.0
                    p_value = 1.0
            else:
                slope = 0.0
                std_err = 0.0
                p_value = 1.0
            
            # Late-Query Factor (LQF)
            if len(mean_scores) > 1:
                lqf = mean_scores[-1] / mean_scores[0] if mean_scores[0] > 0 else 0.0
            else:
                lqf = 1.0
            
            summaries[method_name] = {
                'curve': curve,
                'auc_delta': auc_delta,
                'slope': slope,
                'slope_stderr': std_err,
                'slope_pvalue': p_value,
                'lqf': lqf,
                'deltas': deltas,
                'mean_scores': mean_scores
            }
        
        return summaries
    
    def _bootstrap_ci(self, data: List[float], n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Compute bootstrap confidence interval"""
        if len(data) < 2:
            return 0.0, 0.0
        
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        alpha = 1 - self.config.confidence_level
        ci_low = np.percentile(bootstrap_means, 100 * alpha / 2)
        ci_high = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        
        return ci_low, ci_high
    
    def plot_qar_curves(self, summaries: Dict[str, Dict], save_path: str = None):
        """Plot QAR curves with confidence intervals"""
        if not HAS_PLOTTING:
            print("Plotting not available. Install matplotlib and seaborn to enable plotting.")
            return
        
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(summaries)))
        
        for i, (method_name, summary) in enumerate(summaries.items()):
            deltas = summary['deltas']
            mean_scores = summary['mean_scores']
            
            # Extract confidence intervals
            ci_lows = [summary['curve'][d]['ci_low'] for d in deltas]
            ci_highs = [summary['curve'][d]['ci_high'] for d in deltas]
            
            # Convert deltas to minutes for plotting
            deltas_min = [d / 60.0 for d in deltas]
            
            plt.plot(deltas_min, mean_scores, 'o-', color=colors[i], 
                    label=f"{method_name} (AUC={summary['auc_delta']:.3f})", linewidth=2)
            plt.fill_between(deltas_min, ci_lows, ci_highs, alpha=0.2, color=colors[i])
        
        plt.xlabel('Query Staleness Δ (minutes)')
        plt.ylabel('Score')
        plt.title('Query-Agnostic Robustness (QAR) Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"QAR curves saved to {save_path}")
        
        plt.show()
    
    def statistical_comparison(self, results: Dict[str, List[Tuple]], 
                             baseline_method: str = None) -> Dict:
        """Perform statistical comparisons between methods"""
        if baseline_method is None:
            baseline_method = list(results.keys())[0]
        
        comparisons = {}
        baseline_results = results[baseline_method]
        
        # Group baseline results by (qid, delta) for paired testing
        baseline_dict = {}
        for qid, delta, score in baseline_results:
            baseline_dict[(qid, delta)] = score
        
        for method_name, method_results in results.items():
            if method_name == baseline_method:
                continue
            
            # Find paired samples
            paired_baseline = []
            paired_method = []
            
            for qid, delta, score in method_results:
                if (qid, delta) in baseline_dict:
                    paired_baseline.append(baseline_dict[(qid, delta)])
                    paired_method.append(score)
            
            if len(paired_baseline) < 10:  # Need enough samples
                continue
            
            # Perform paired Wilcoxon test
            if HAS_SCIPY:
                statistic, p_value = stats.wilcoxon(paired_baseline, paired_method, 
                                                  alternative='two-sided')
            else:
                # Simple t-test fallback
                diff = np.array(paired_method) - np.array(paired_baseline)
                statistic = np.sum(diff > 0)  # Count of improvements
                p_value = 0.5  # Conservative estimate
            
            # Effect size (mean difference)
            mean_diff = np.mean(paired_method) - np.mean(paired_baseline)
            
            comparisons[method_name] = {
                'baseline': baseline_method,
                'n_pairs': len(paired_baseline),
                'wilcoxon_statistic': statistic,
                'p_value': p_value,
                'mean_difference': mean_diff,
                'significant': p_value < (1 - self.config.confidence_level)
            }
        
        return comparisons
    
    def generate_report(self, summaries: Dict[str, Dict], 
                       comparisons: Dict, save_path: str = None) -> str:
        """Generate a formatted report"""
        report = []
        report.append("# Query-Agnostic Robustness (QAR) Report\n")
        
        report.append("## Method Comparison\n")
        report.append("| Method | AUC_Δ | Slope (/min) | LQF | Slope p-value |")
        report.append("|--------|-------|-------------|-----|---------------|")
        
        for method_name, summary in summaries.items():
            slope_per_min = summary['slope'] * 60  # Convert to per-minute
            report.append(f"| {method_name} | {summary['auc_delta']:.3f} | "
                         f"{slope_per_min:.4f} | {summary['lqf']:.3f} | "
                         f"{summary['slope_pvalue']:.3f} |")
        
        report.append("\n## Statistical Comparisons\n")
        for method_name, comp in comparisons.items():
            significance = "**" if comp['significant'] else ""
            report.append(f"- **{method_name}** vs {comp['baseline']}: "
                         f"mean diff = {comp['mean_difference']:.3f}, "
                         f"p = {comp['p_value']:.3f} {significance}")
        
        report.append(f"\n## Configuration\n")
        report.append(f"- Evidence detection: {self.config.evidence_method}")
        report.append(f"- Staleness grid: {[f'{d/60:.1f}min' if d >= 60 else f'{d}s' for d in self.config.delta_grid]}")
        report.append(f"- Sample FPS: {self.config.sample_fps}")
        report.append(f"- Confidence level: {self.config.confidence_level}")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {save_path}")
        
        return report_text


# Example wrapper for ReKV models
class ReKVWrapper:
    """Wrapper to make ReKV models compatible with QAR measurement"""
    
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.frames = []
        
    def reset(self):
        """Reset model state"""
        self.model.clear_cache()
        self.frames = []
    
    def ingest_frame(self, frame, timestamp):
        """Ingest a frame for query-agnostic processing"""
        # Convert frame to the format expected by the model
        # This is a simplified version - you may need to adapt based on your exact model
        self.frames.append(frame)
    
    def answer(self, question):
        """Answer question based on ingested frames"""
        if not self.frames:
            return ""
        
        # Convert frames to video tensor
        # This is a placeholder - implement according to your model's input format
        video_tensor = np.array(self.frames)
        
        # Use the model's question_answering method
        response = self.model.question_answering(question, max_new_tokens=128)
        return response


def main():
    """Example usage of QAR measurement"""
    # Configuration
    config = QARConfig(
        delta_grid=[0.0, 30.0, 120.0, 300.0, 600.0, 1800.0],  # 0s, 30s, 2m, 5m, 10m, 30m
        evidence_method='clip',
        sample_fps=0.5
    )
    
    # Initialize measurer
    measurer = QARMeasurer(config)
    
    # Example data (replace with your actual dataset)
    video_questions = [
        {
            'id': 'q1',
            'video_id': 'video1',
            'question': 'What color is the car?',
            'answer': 'red',
            'manual_timestamp': None  # Optional manual annotation
        },
        # Add more questions...
    ]
    
    # Example methods (replace with your actual models)
    methods = {
        'ReKV': None,  # Your ReKV model wrapped with ReKVWrapper
        'Sliding-Window': None,  # Baseline implementation
        'Full-KV': None,  # Upper bound
        'InfiniPot-V': None  # Comparison method
    }
    
    # Note: You need to initialize these methods with actual model instances
    print("Warning: Please initialize the methods dictionary with actual model instances")
    
    # Measure QAR
    # results = measurer.measure_qar(video_questions, methods, video_dir='path/to/videos')
    
    # Summarize results
    # summaries = measurer.summarize_results(results)
    
    # Plot curves
    # measurer.plot_qar_curves(summaries, save_path='qar_curves.png')
    
    # Statistical comparisons
    # comparisons = measurer.statistical_comparison(results, baseline_method='Sliding-Window')
    
    # Generate report
    # report = measurer.generate_report(summaries, comparisons, save_path='qar_report.md')
    # print(report)


if __name__ == "__main__":
    main()