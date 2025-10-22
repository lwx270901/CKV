"""
Simplified QAR evaluation script that handles missing model dependencies gracefully
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.append('/home/minh/research/ReKV')

from qar_measurement import QARMeasurer, QARConfig


def check_model_availability(model_name: str) -> bool:
    """Check if a model is available for loading"""
    try:
        if model_name == 'llava_ov_7b':
            from model.llava_onevision_rekv import load_model
            return True
        elif model_name == 'video_llava_7b':
            from model.video_llava_rekv import load_model
            return True
        elif model_name == 'longva_7b':
            from model.longva_rekv import load_model
            return True
        elif model_name == 'flash_vstream':
            from model.flash_vstream_rekv import load_model
            return True
        else:
            return False
    except ImportError:
        return False


def check_dataset_availability(dataset_name: str, data_dir: str = 'data') -> bool:
    """Check if a dataset is available"""
    if dataset_name == 'mlvu':
        anno_path = os.path.join(data_dir, 'mlvu', 'dev_debug_mc.json')
        video_dir = os.path.join(data_dir, 'mlvu', 'videos')
        return os.path.exists(anno_path) and os.path.exists(video_dir)
    elif dataset_name == 'egoschema':
        anno_path = os.path.join(data_dir, 'egoschema', 'full.json')
        video_dir = os.path.join(data_dir, 'egoschema', 'videos')
        return os.path.exists(anno_path) and os.path.exists(video_dir)
    else:
        return False


def run_availability_check():
    """Check availability of models and datasets"""
    print("QAR Evaluation - Availability Check")
    print("=" * 50)
    
    # Check models
    models = ['llava_ov_7b', 'video_llava_7b', 'longva_7b', 'flash_vstream']
    print("Model Availability:")
    available_models = []
    
    for model in models:
        available = check_model_availability(model)
        status = "✓" if available else "✗"
        print(f"  {status} {model}")
        if available:
            available_models.append(model)
    
    # Check datasets
    datasets = ['mlvu', 'egoschema']
    print("\nDataset Availability:")
    available_datasets = []
    
    for dataset in datasets:
        available = check_dataset_availability(dataset)
        status = "✓" if available else "✗"
        print(f"  {status} {dataset}")
        if available:
            available_datasets.append(dataset)
    
    # Dependencies check
    print("\nDependency Check:")
    dependencies = [
        ('numpy', 'Core numerical computing'),
        ('torch', 'PyTorch deep learning'),
        ('transformers', 'Hugging Face transformers'),
        ('opencv-python', 'Video processing'),
        ('matplotlib', 'Plotting'),
        ('scipy', 'Statistical analysis'),
        ('sklearn', 'Machine learning utilities')
    ]
    
    missing_deps = []
    for dep_name, description in dependencies:
        try:
            __import__(dep_name.replace('-', '_'))
            print(f"  ✓ {dep_name} - {description}")
        except ImportError:
            print(f"  ✗ {dep_name} - {description}")
            missing_deps.append(dep_name)
    
    # Summary and recommendations
    print("\n" + "=" * 50)
    print("AVAILABILITY SUMMARY")
    print("=" * 50)
    
    if available_models:
        print(f"✓ Available models: {', '.join(available_models)}")
    else:
        print("✗ No models available")
    
    if available_datasets:
        print(f"✓ Available datasets: {', '.join(available_datasets)}")
    else:
        print("✗ No datasets available")
    
    if missing_deps:
        print(f"✗ Missing dependencies: {', '.join(missing_deps)}")
        print("\nTo install missing dependencies:")
        print("bash install_qar_deps.sh")
    else:
        print("✓ All dependencies available")
    
    # Provide specific recommendations
    print("\nRecommendations:")
    
    if not available_models and not available_datasets:
        print("1. Run mock test first: python test_qar_mock.py")
        print("2. Install dependencies: bash install_qar_deps.sh")
        print("3. Set up ReKV models and datasets")
    elif available_models and not available_datasets:
        print("1. Download and prepare datasets (MLVU or EgoSchema)")
        print("2. Update data paths in the evaluation script")
    elif available_datasets and not available_models:
        print("1. Install ReKV model dependencies")
        print("2. Download model checkpoints")
    else:
        print("✓ Ready for QAR evaluation!")
        if available_models and available_datasets:
            model = available_models[0]
            dataset = available_datasets[0]
            print(f"Try: python run_qar_evaluation.py --model {model} --dataset {dataset} --max_questions 10")


def main():
    parser = argparse.ArgumentParser(description='QAR Evaluation Checker')
    parser.add_argument('--check-only', action='store_true', 
                       help='Only check availability, do not run evaluation')
    
    args = parser.parse_args()
    
    if args.check_only:
        run_availability_check()
    else:
        # Run availability check first
        run_availability_check()
        
        print("\n" + "=" * 50)
        print("For now, use the mock test to validate QAR implementation:")
        print("python test_qar_mock.py")
        print("\nOnce dependencies are installed, use:")
        print("python run_qar_evaluation.py --model <model> --dataset <dataset>")


if __name__ == "__main__":
    main()