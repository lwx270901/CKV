#!/usr/bin/env python3
"""
Full MLVU Dataset Evaluation Script
===================================

Run complete MLVU evaluation as described in README.md with ProtoTrack-KV integration.
This script follows the exact evaluation protocol from the paper.
"""

import os
import subprocess
import time
import json
from pathlib import Path

def check_mlvu_dataset():
    """Check if MLVU dataset is properly set up"""
    print("🔍 Checking MLVU dataset setup...")
    
    mlvu_data_dir = Path("data/mlvu")
    mlvu_annotation = mlvu_data_dir / "dev_debug_mc.json"
    mlvu_videos = mlvu_data_dir / "videos"
    
    if not mlvu_data_dir.exists():
        print("❌ MLVU data directory not found: data/mlvu/")
        print("   Please download MLVU dataset as described in README.md")
        return False
    
    if not mlvu_annotation.exists():
        print("❌ MLVU annotation file not found: data/mlvu/dev_debug_mc.json")
        print("   Please download MLVU-dev-mc from HuggingFace")
        return False
    
    if not mlvu_videos.exists():
        print("❌ MLVU videos directory not found: data/mlvu/videos/")
        print("   Please download MLVU videos")
        return False
    
    # Count available samples
    try:
        with open(mlvu_annotation, 'r') as f:
            data = json.load(f)
        sample_count = len(data)
        print(f"✅ MLVU dataset found: {sample_count} samples")
        
        # Check for extremely long videos warning from README
        long_videos = []
        for item in data:
            if 'duration' in item and item.get('duration', 0) > 3600:  # > 1 hour
                long_videos.append(item.get('video_name', 'unknown'))
        
        if long_videos:
            print(f"⚠️  Warning: Found {len(long_videos)} extremely long videos (>1hr)")
            print("   README mentions MLVU has ~9hr videos that may need removal for RAM constraints")
            print(f"   Long videos: {long_videos[:3]}..." if len(long_videos) > 3 else f"   Long videos: {long_videos}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading MLVU annotation: {e}")
        return False

def check_model_availability():
    """Check if required model is available"""
    print("🔍 Checking model availability...")
    
    model_dir = Path("model_zoo/llava-onevision-qwen2-7b-ov-hf")
    if not model_dir.exists():
        print("❌ LLaVA OneVision 7B model not found")
        print("   Please download llava-onevision-qwen2-7b-ov-hf to model_zoo/")
        return False
    
    # Check for key model files
    required_files = ["config.json", "model.safetensors.index.json"]
    for file in required_files:
        if not (model_dir / file).exists():
            print(f"❌ Missing model file: {file}")
            return False
    
    print("✅ LLaVA OneVision 7B model found")
    return True

def check_system_requirements():
    """Check system requirements"""
    print("🔍 Checking system requirements...")
    
    # Check GPU availability
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ NVIDIA GPU not available")
            return False
        
        # Parse GPU count and memory
        gpu_lines = [line for line in result.stdout.split('\n') if 'MiB' in line and '/' in line]
        gpu_count = len(gpu_lines)
        print(f"✅ Found {gpu_count} NVIDIA GPU(s)")
        
        # Check memory map limit
        with open('/proc/sys/vm/max_map_count', 'r') as f:
            max_map_count = int(f.read().strip())
        
        if max_map_count < 262144:
            print(f"⚠️  Warning: vm.max_map_count is {max_map_count}, should be ≥262144")
            print("   Run: sudo sysctl -w vm.max_map_count=262144")
        else:
            print(f"✅ Memory map limit OK: {max_map_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking system: {e}")
        return False

def run_full_mlvu_evaluation():
    """Run the full MLVU evaluation following README instructions"""
    print("\n" + "="*80)
    print("🚀 STARTING FULL MLVU DATASET EVALUATION")
    print("="*80)
    
    # Pre-flight checks
    if not check_mlvu_dataset():
        return False
    
    if not check_model_availability():
        return False
    
    if not check_system_requirements():
        return False
    
    print("\n✅ All prerequisites satisfied! Starting evaluation...")
    
    # Set evaluation parameters following README
    num_chunks = 4  # Adjust based on available GPUs
    model = "llava_ov_7b"
    dataset = "mlvu"
    sample_fps = 0.5
    n_local = 15000
    retrieve_size = 64
    
    print(f"\n📋 Evaluation Configuration:")
    print(f"   Model: {model}")
    print(f"   Dataset: {dataset}")
    print(f"   Parallel chunks: {num_chunks}")
    print(f"   Sample FPS: {sample_fps}")
    print(f"   Local cache size: {n_local}")
    print(f"   Retrieval size: {retrieve_size}")
    
    # Create results directory
    results_dir = Path(f"results/{model}/{dataset}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct the evaluation command from README
    eval_command = [
        "python", "-m", "video_qa.run_eval",
        "--num_chunks", str(num_chunks),
        "--model", model,
        "--dataset", dataset,
        "--sample_fps", str(sample_fps),
        "--n_local", str(n_local),
        "--retrieve_size", str(retrieve_size)
    ]
    
    print(f"\n🎯 Running evaluation command:")
    print(f"   {' '.join(eval_command)}")
    
    # Start evaluation
    start_time = time.time()
    
    try:
        print("\n" + "="*80)
        print("📊 EVALUATION IN PROGRESS...")
        print("="*80)
        
        # Run the evaluation
        process = subprocess.Popen(
            eval_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output in real-time
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())
        
        process.wait()
        
        if process.returncode == 0:
            elapsed_time = time.time() - start_time
            print("\n" + "="*80)
            print("🎉 FULL MLVU EVALUATION COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"⏱️  Total evaluation time: {elapsed_time/60:.1f} minutes")
            print(f"📁 Results saved to: {results_dir}")
            
            # Check for results files
            result_files = list(results_dir.glob("*.json"))
            if result_files:
                print(f"📊 Generated {len(result_files)} result files:")
                for file in result_files:
                    print(f"   • {file.name}")
            
            return True
        else:
            print(f"\n❌ Evaluation failed with return code {process.returncode}")
            return False
            
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        if 'process' in locals():
            process.terminate()
        return False
    except Exception as e:
        print(f"\n❌ Evaluation error: {e}")
        return False

def main():
    """Main function"""
    print("=" * 80)
    print("FULL MLVU DATASET EVALUATION - ReKV Paper Protocol")
    print("=" * 80)
    print("Following evaluation setup from README.md")
    print("This will run the complete MLVU dataset evaluation")
    print("with the official ReKV implementation.")
    
    success = run_full_mlvu_evaluation()
    
    if success:
        print("\n🎊 Evaluation completed successfully!")
        print("📈 Check the results/ directory for detailed output")
        print("🔬 Compare with paper benchmarks for validation")
    else:
        print("\n💥 Evaluation encountered issues")
        print("🔧 Check the error messages above for troubleshooting")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()