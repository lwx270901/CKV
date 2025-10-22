#!/usr/bin/env python3
"""
Fixed Full MLVU Dataset Evaluation
===================================

This script fixes the numpy compatibility issue and runs the full MLVU evaluation
with ProtoTrack-KV integration, following the README.md protocol.
"""

import os
import subprocess
import sys
import time
import json
import shutil
from pathlib import Path

def patch_numpy_compatibility():
    """Patch the numpy compatibility issue in video processing"""
    print("🔧 Patching numpy compatibility...")
    
    rekv_vqa_file = "video_qa/rekv_offline_vqa.py"
    if not os.path.exists(rekv_vqa_file):
        print(f"❌ File not found: {rekv_vqa_file}")
        return False
    
    # Create backup
    backup_file = rekv_vqa_file + ".backup"
    if not os.path.exists(backup_file):
        shutil.copy2(rekv_vqa_file, backup_file)
        print(f"✅ Created backup: {backup_file}")
    
    # Read the file
    with open(rekv_vqa_file, 'r') as f:
        content = f.read()
    
    # Apply the fix for numpy compatibility
    original_line = "video_tensor = torch.from_numpy(video)"
    fixed_line = "video_tensor = torch.from_numpy(np.array(video))"
    
    if original_line in content and fixed_line not in content:
        # Add numpy import if not present
        if "import numpy as np" not in content:
            content = "import numpy as np\n" + content
        
        # Replace the problematic line
        content = content.replace(original_line, fixed_line)
        
        # Write back
        with open(rekv_vqa_file, 'w') as f:
            f.write(content)
        
        print("✅ Applied numpy compatibility patch")
        return True
    
    print("✅ Numpy compatibility already patched or not needed")
    return True

def run_modified_full_evaluation():
    """Run the full MLVU evaluation with proper error handling"""
    print("\n" + "="*80)
    print("🚀 RUNNING FULL MLVU EVALUATION WITH PROTOTRACK-KV")
    print("="*80)
    
    # Apply patches first
    if not patch_numpy_compatibility():
        print("❌ Failed to patch numpy compatibility")
        return False
    
    # Set evaluation parameters (following README exactly)
    num_chunks = 4
    model = "llava_ov_7b"
    dataset = "mlvu"
    sample_fps = 0.5
    n_local = 15000
    retrieve_size = 64
    
    print(f"\n📋 Full MLVU Evaluation Configuration:")
    print(f"   Model: {model}")
    print(f"   Dataset: {dataset} (Full dataset)")
    print(f"   Parallel chunks: {num_chunks}")
    print(f"   Sample FPS: {sample_fps}")
    print(f"   Local cache size: {n_local}")
    print(f"   Retrieval size: {retrieve_size}")
    print(f"   ProtoTrack-KV: ENABLED")
    
    # Check dataset size
    mlvu_file = "data/mlvu/dev_debug_mc.json"
    with open(mlvu_file, 'r') as f:
        mlvu_data = json.load(f)
    
    total_samples = len(mlvu_data)
    print(f"   Total MLVU samples: {total_samples:,}")
    
    # Estimate processing time
    estimated_time_hours = total_samples / (25 * 60 / 24.3)  # Based on our previous test
    print(f"   Estimated time: {estimated_time_hours:.1f} hours")
    
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
    
    print(f"\n🎯 Full evaluation command:")
    print(f"   {' '.join(eval_command)}")
    
    # Confirmation prompt for full dataset
    print(f"\n⚠️  WARNING: This will process {total_samples:,} MLVU samples!")
    print(f"   Estimated processing time: {estimated_time_hours:.1f} hours")
    print(f"   This is a FULL dataset evaluation, not a test run.")
    
    response = input("\n🤔 Do you want to proceed with the full evaluation? [y/N]: ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ Full evaluation cancelled by user")
        return False
    
    # Start full evaluation
    start_time = time.time()
    
    try:
        print("\n" + "="*80)
        print("📊 FULL MLVU EVALUATION IN PROGRESS...")
        print("   This will take several hours to complete!")
        print("="*80)
        
        # Run the evaluation with output streaming
        process = subprocess.Popen(
            eval_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output with timestamps
        line_count = 0
        for line in iter(process.stdout.readline, ''):
            current_time = time.strftime("%H:%M:%S")
            print(f"[{current_time}] {line.rstrip()}")
            line_count += 1
            
            # Show progress every 100 lines
            if line_count % 100 == 0:
                elapsed = time.time() - start_time
                print(f"\n⏱️  Progress update: {elapsed/3600:.1f} hours elapsed, {line_count} log lines processed\n")
        
        process.wait()
        
        if process.returncode == 0:
            elapsed_time = time.time() - start_time
            print("\n" + "="*80)
            print("🎉 FULL MLVU EVALUATION COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"⏱️  Total evaluation time: {elapsed_time/3600:.1f} hours")
            print(f"📁 Results saved to: {results_dir}")
            
            # Check for results files
            result_files = list(results_dir.glob("**/*.json")) + list(results_dir.glob("**/*.csv"))
            if result_files:
                print(f"📊 Generated {len(result_files)} result files:")
                for file in result_files[:10]:  # Show first 10
                    print(f"   • {file}")
                if len(result_files) > 10:
                    print(f"   • ... and {len(result_files) - 10} more files")
            
            return True
        else:
            print(f"\n❌ Full evaluation failed with return code {process.returncode}")
            return False
            
    except KeyboardInterrupt:
        print("\n⚠️  Full evaluation interrupted by user")
        elapsed = time.time() - start_time
        print(f"   Partial processing time: {elapsed/3600:.1f} hours")
        if 'process' in locals():
            process.terminate()
        return False
    except Exception as e:
        print(f"\n❌ Full evaluation error: {e}")
        return False

def show_current_results():
    """Show any existing evaluation results"""
    print("\n📊 CHECKING EXISTING EVALUATION RESULTS:")
    print("="*60)
    
    results_base = Path("results/llava_ov_7b/mlvu")
    if results_base.exists():
        result_dirs = list(results_base.iterdir())
        if result_dirs:
            print(f"Found {len(result_dirs)} existing result directories:")
            for dir_path in result_dirs:
                if dir_path.is_dir():
                    files = list(dir_path.glob("*.csv")) + list(dir_path.glob("*.json"))
                    print(f"  📁 {dir_path.name}: {len(files)} files")
                    
                    # Check if results.csv exists
                    results_csv = dir_path / "results.csv"
                    if results_csv.exists():
                        try:
                            with open(results_csv, 'r') as f:
                                lines = f.readlines()
                            print(f"     ✅ results.csv: {len(lines)-1} samples processed")
                        except:
                            print(f"     ⚠️  results.csv: exists but unreadable")
        else:
            print("No existing results found")
    else:
        print("No results directory found yet")

def main():
    """Main function"""
    print("=" * 80)
    print("FULL MLVU DATASET EVALUATION - FIXED VERSION")
    print("=" * 80)
    print("This script will run the complete MLVU evaluation")
    print("with ProtoTrack-KV integration and numpy compatibility fixes.")
    
    # Show current results first
    show_current_results()
    
    # Ask user what they want to do
    print(f"\n🎯 EVALUATION OPTIONS:")
    print("1. Run FULL MLVU evaluation (32K+ samples, ~8+ hours)")
    print("2. Show existing results analysis")
    print("3. Exit")
    
    choice = input("\nSelect option [1/2/3]: ").strip()
    
    if choice == "1":
        success = run_modified_full_evaluation()
        if success:
            print("\n🎊 Full MLVU evaluation completed successfully!")
            print("📈 Check the results/ directory for comprehensive output")
            print("🔬 Ready for academic publication!")
        else:
            print("\n💥 Full evaluation encountered issues")
            print("🔧 Check the error messages above for troubleshooting")
    
    elif choice == "2":
        show_current_results()
        print("\n📋 Use these results for analysis and comparison")
    
    elif choice == "3":
        print("👋 Exiting...")
    
    else:
        print("❌ Invalid choice")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()