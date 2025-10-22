"""
QAR Evaluation Script for ReKV
Integrates QAR measurement with existing ReKV evaluation framework
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from typing import Dict, List, Any
import pandas as pd
from pathlib import Path

# Add the project root to Python path
sys.path.append('/home/minh/research/ReKV')

from qar_measurement import QARMeasurer, QARConfig, ReKVWrapper

# Optional model imports - only import what's needed
def load_llava_ov(*args, **kwargs):
    try:
        from model.llava_onevision_rekv import load_model
        return load_model(*args, **kwargs)
    except ImportError as e:
        raise ImportError(f"LLaVA OneVision model not available: {e}")

def load_video_llava(*args, **kwargs):
    try:
        from model.video_llava_rekv import load_model
        return load_model(*args, **kwargs)
    except ImportError as e:
        raise ImportError(f"Video-LLaVA model not available: {e}")

def load_longva(*args, **kwargs):
    try:
        from model.longva_rekv import load_model
        return load_model(*args, **kwargs)
    except ImportError as e:
        raise ImportError(f"LongVA model not available: {e}")

def load_flash_vstream(*args, **kwargs):
    try:
        from model.flash_vstream_rekv import load_model
        return load_model(*args, **kwargs)
    except ImportError as e:
        raise ImportError(f"Flash-VStream model not available: {e}")


class SlidingWindowBaseline:
    """Sliding window baseline for comparison"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.tokens = []
        
    def reset(self):
        self.tokens = []
    
    def ingest_frame(self, frame, timestamp):
        # Simulate token ingestion
        frame_tokens = np.random.randn(196)  # Placeholder
        self.tokens.extend(frame_tokens)
        
        # Keep only recent tokens (sliding window)
        if len(self.tokens) > self.window_size:
            self.tokens = self.tokens[-self.window_size:]
    
    def answer(self, question):
        # Placeholder implementation
        return "placeholder_answer"


class FullKVBaseline:
    """Full KV cache baseline (upper bound)"""
    
    def __init__(self):
        self.tokens = []
        
    def reset(self):
        self.tokens = []
    
    def ingest_frame(self, frame, timestamp):
        # Keep all tokens (no compression)
        frame_tokens = np.random.randn(196)  # Placeholder
        self.tokens.extend(frame_tokens)
    
    def answer(self, question):
        # Placeholder implementation
        return "placeholder_answer"


def load_rekv_model(model_name: str, n_local: int, topk: int) -> tuple:
    """Load ReKV model based on name"""
    if model_name == 'llava_ov_7b':
        return load_llava_ov(
            model_path='model_zoo/llava-onevision-qwen2-7b-ov-hf',
            n_local=n_local,
            topk=topk
        )
    elif model_name == 'video_llava_7b':
        return load_video_llava(
            model_path='model_zoo/Video-LLaVA-7B-hf',
            n_local=n_local,
            topk=topk
        )
    elif model_name == 'longva_7b':
        return load_longva(
            model_path='model_zoo/LongVA-7B',
            n_local=n_local,
            topk=topk
        )
    elif model_name == 'flash_vstream':
        return load_flash_vstream(
            n_local=n_local,
            topk=topk
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def load_evaluation_data(dataset_name: str, data_dir: str = 'data') -> List[Dict]:
    """Load evaluation dataset"""
    video_questions = []
    
    if dataset_name == 'mlvu':
        anno_path = os.path.join(data_dir, 'mlvu', 'dev_debug_mc.json')
        video_dir = os.path.join(data_dir, 'mlvu', 'videos')
        
        with open(anno_path, 'r') as f:
            data = json.load(f)
        
        for item in data:
            # Handle MLVU format with conversations
            for i, conv in enumerate(item.get('conversations', [])):
                video_questions.append({
                    'id': f"{item['video_id']}_{i}",
                    'video_id': item['video_id'],
                    'question': conv['question'],
                    'answer': conv.get('answer', ''),
                    'choices': conv.get('choices', []),
                    'manual_timestamp': None  # No manual timestamps in MLVU
                })
    
    elif dataset_name == 'egoschema':
        anno_path = os.path.join(data_dir, 'egoschema', 'full.json')
        video_dir = os.path.join(data_dir, 'egoschema', 'videos')
        
        with open(anno_path, 'r') as f:
            data = json.load(f)
        
        for item in data:
            video_questions.append({
                'id': item.get('q_uid', len(video_questions)),
                'video_id': item['video_uid'],
                'question': item['question'],
                'answer': item.get('answer', ''),
                'manual_timestamp': None
            })
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return video_questions, video_dir


def run_qar_evaluation(args):
    """Run QAR evaluation"""
    print("Starting QAR Evaluation...")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"n_local: {args.n_local}")
    print(f"topk: {args.topk}")
    
    # Configuration
    config = QARConfig(
        delta_grid=[0.0, 30.0, 120.0, 300.0, 600.0, 1800.0],  # 0s, 30s, 2min, 5min, 10min, 30min
        evidence_method=args.evidence_method,
        sample_fps=args.sample_fps,
        memory_budget=args.topk,
        random_seed=2024
    )
    
    # Load evaluation data
    video_questions, video_dir = load_evaluation_data(args.dataset, args.data_dir)
    
    # Limit to subset for testing
    if args.max_questions > 0:
        video_questions = video_questions[:args.max_questions]
    
    print(f"Loaded {len(video_questions)} questions")
    
    # Initialize methods
    methods = {}
    
    # ReKV model
    if args.include_rekv:
        try:
            rekv_model, rekv_processor = load_rekv_model(args.model, args.n_local, args.topk)
            methods['ReKV'] = ReKVWrapper(rekv_model, rekv_processor)
            print("✓ ReKV model loaded")
        except Exception as e:
            print(f"✗ Failed to load ReKV model: {e}")
    
    # Baseline methods
    if args.include_baselines:
        methods['Sliding-Window'] = SlidingWindowBaseline(window_size=args.n_local)
        methods['Full-KV'] = FullKVBaseline()
        print("✓ Baseline methods initialized")
    
    if not methods:
        print("No methods to evaluate!")
        return
    
    # Initialize QAR measurer
    measurer = QARMeasurer(config)
    
    # Run evaluation
    print("\nRunning QAR measurement...")
    results = measurer.measure_qar(video_questions, methods, video_dir)
    
    # Summarize results
    print("\nSummarizing results...")
    summaries = measurer.summarize_results(results)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw results
    results_path = output_dir / 'qar_raw_results.json'
    with open(results_path, 'w') as f:
        # Convert tuples to lists for JSON serialization
        json_results = {}
        for method, method_results in results.items():
            json_results[method] = [[qid, delta, score] for qid, delta, score in method_results]
        json.dump(json_results, f, indent=2)
    print(f"Raw results saved to {results_path}")
    
    # Save summaries
    summaries_path = output_dir / 'qar_summaries.json'
    with open(summaries_path, 'w') as f:
        json.dump(summaries, f, indent=2, default=str)
    print(f"Summaries saved to {summaries_path}")
    
    # Plot curves
    curves_path = output_dir / 'qar_curves.png'
    measurer.plot_qar_curves(summaries, save_path=str(curves_path))
    
    # Statistical comparisons
    if len(methods) > 1:
        baseline = 'Sliding-Window' if 'Sliding-Window' in methods else list(methods.keys())[0]
        comparisons = measurer.statistical_comparison(results, baseline_method=baseline)
        
        # Save comparisons
        comp_path = output_dir / 'qar_comparisons.json'
        with open(comp_path, 'w') as f:
            json.dump(comparisons, f, indent=2, default=str)
        print(f"Statistical comparisons saved to {comp_path}")
    else:
        comparisons = {}
    
    # Generate report
    report_path = output_dir / 'qar_report.md'
    report = measurer.generate_report(summaries, comparisons, save_path=str(report_path))
    
    # Print summary
    print("\n" + "="*50)
    print("QAR EVALUATION SUMMARY")
    print("="*50)
    
    for method_name, summary in summaries.items():
        slope_per_min = summary['slope'] * 60
        print(f"\n{method_name}:")
        print(f"  AUC_Δ: {summary['auc_delta']:.3f}")
        print(f"  Slope: {slope_per_min:.4f} score/min (p={summary['slope_pvalue']:.3f})")
        print(f"  LQF: {summary['lqf']:.3f}")
    
    if comparisons:
        print(f"\nStatistical Comparisons (vs {baseline}):")
        for method, comp in comparisons.items():
            sig = "***" if comp['p_value'] < 0.001 else "**" if comp['p_value'] < 0.01 else "*" if comp['p_value'] < 0.05 else ""
            print(f"  {method}: Δ={comp['mean_difference']:.3f}, p={comp['p_value']:.3f} {sig}")
    
    print(f"\nAll results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='QAR Evaluation for ReKV')
    
    # Model and dataset
    parser.add_argument('--model', type=str, default='llava_ov_7b',
                       choices=['llava_ov_7b', 'video_llava_7b', 'longva_7b', 'flash_vstream'],
                       help='Model to evaluate')
    parser.add_argument('--dataset', type=str, default='mlvu',
                       choices=['mlvu', 'egoschema'],
                       help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Data directory')
    
    # Model parameters
    parser.add_argument('--n_local', type=int, default=15000,
                       help='Local window size')
    parser.add_argument('--topk', type=int, default=64,
                       help='Number of retrieved tokens')
    
    # QAR parameters
    parser.add_argument('--evidence_method', type=str, default='clip',
                       choices=['manual', 'clip', 'attention'],
                       help='Evidence detection method')
    parser.add_argument('--sample_fps', type=float, default=0.5,
                       help='Video sampling FPS')
    
    # Evaluation options
    parser.add_argument('--include_rekv', action='store_true', default=True,
                       help='Include ReKV model')
    parser.add_argument('--include_baselines', action='store_true', default=True,
                       help='Include baseline methods')
    parser.add_argument('--max_questions', type=int, default=50,
                       help='Maximum number of questions to evaluate (0 for all)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='results/qar_evaluation',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Run evaluation
    run_qar_evaluation(args)


if __name__ == "__main__":
    main()