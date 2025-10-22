#!/usr/bin/env python3
"""
Quick ProtoTrack-KV QAR Test
Simple test script to measure QAR on ProtoTrack-KV
"""

import sys
import os
sys.path.append('/home/minh/research/ReKV')

from run_prototrack_qar import run_prototrack_qar_evaluation
import argparse

def main():
    # Simple test configuration for ProtoTrack-KV
    class Args:
        model = 'llava_ov_7b'
        dataset = 'mlvu'
        data_dir = 'data'
        
        # ProtoTrack parameters (matching config from codebase)
        n_local = 8000
        topk = 32
        
        # QAR parameters
        evidence_method = 'manual'  # Fastest for testing
        sample_fps = 0.5
        
        # Test with small subset first
        max_questions = 5  # Start small for testing
        
        # Evaluation options
        include_default = True
        include_prototrack_variants = False  # Skip variants for quick test
        include_baseline = True
        
        # Output
        output_dir = 'results/prototrack_test'
    
    args = Args()
    
    print("ProtoTrack-KV QAR Quick Test")
    print("=" * 40)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Questions: {args.max_questions}")
    print(f"ProtoTrack params: n_local={args.n_local}, topk={args.topk}")
    print()
    
    # Run the evaluation
    success = run_prototrack_qar_evaluation(args)
    
    if success:
        print("\n🎉 ProtoTrack-KV QAR test completed!")
        print(f"Check results in: {args.output_dir}/")
    else:
        print("\n❌ ProtoTrack-KV QAR test failed.")

if __name__ == "__main__":
    main()