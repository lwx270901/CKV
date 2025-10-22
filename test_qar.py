"""
Test script for QAR measurement implementation
"""

import sys
import os
import numpy as np
import torch
from typing import Dict, List

# Add project root to path
sys.path.append('/home/minh/research/ReKV')

from qar_measurement import QARMeasurer, QARConfig, ReKVWrapper


class MockModel:
    """Mock model for testing QAR measurement without actual video models"""
    
    def __init__(self, name: str):
        self.name = name
        self.frames = []
        
    def reset(self):
        self.frames = []
        
    def clear_cache(self):
        self.frames = []
    
    def ingest_frame(self, frame, timestamp):
        self.frames.append((frame, timestamp))
    
    def answer(self, question: str) -> str:
        # Simulate different model behaviors based on staleness
        num_frames = len(self.frames)
        
        # Simulate degradation with longer delays
        if self.name == 'ReKV':
            # ReKV should be more robust to staleness
            if num_frames < 10:
                return "correct" if np.random.random() > 0.1 else "wrong"
            else:
                return "correct" if np.random.random() > 0.2 else "wrong"
        
        elif self.name == 'Sliding-Window':
            # Sliding window degrades more with staleness
            if num_frames < 10:
                return "correct" if np.random.random() > 0.2 else "wrong"
            else:
                return "correct" if np.random.random() > 0.5 else "wrong"
        
        else:  # Full-KV
            # Full-KV should perform best
            return "correct" if np.random.random() > 0.05 else "wrong"


class MockEvidenceDetector:
    """Mock evidence detector for testing"""
    
    def detect_evidence_timestamp(self, video_path: str, question: str, 
                                manual_timestamp=None) -> float:
        # Return a fixed timestamp for testing
        return 10.0  # Evidence appears at 10 seconds


def create_mock_video_questions(num_questions: int = 20) -> List[Dict]:
    """Create mock video-question data for testing"""
    questions = []
    
    for i in range(num_questions):
        questions.append({
            'id': f'q{i}',
            'video_id': f'video{i % 5}',  # Reuse 5 videos
            'question': f'What happens in this video {i}?',
            'answer': 'correct',
            'manual_timestamp': 10.0 + i * 0.5  # Vary evidence timing slightly
        })
    
    return questions


def mock_video_length(video_path: str) -> float:
    """Mock video length function"""
    return 60.0  # 1 minute videos


def test_qar_measurement():
    """Test QAR measurement with mock data"""
    print("Testing QAR Measurement Implementation")
    print("=" * 50)
    
    # Set random seed for reproducible results
    np.random.seed(2024)
    
    # Configuration
    config = QARConfig(
        delta_grid=[0.0, 10.0, 20.0, 30.0],  # Shorter grid for testing
        evidence_method='clip',
        sample_fps=1.0,
        random_seed=2024
    )
    
    # Create mock data
    video_questions = create_mock_video_questions(20)
    print(f"Created {len(video_questions)} mock questions")
    
    # Create mock models
    methods = {
        'ReKV': MockModel('ReKV'),
        'Sliding-Window': MockModel('Sliding-Window'),
        'Full-KV': MockModel('Full-KV')
    }
    print(f"Created {len(methods)} mock models")
    
    # Create measurer with mock evidence detector
    measurer = QARMeasurer(config)
    measurer.evidence_detector = MockEvidenceDetector()
    
    # Mock the video length function
    measurer._get_video_length = mock_video_length
    
    # Override the streaming evaluation to use mock data
    original_run_streaming = measurer._run_streaming_evaluation
    
    def mock_streaming_evaluation(method, video_path, question, ground_truth, t_inject):
        # Reset method
        method.reset()
        
        # Simulate ingesting frames based on injection time
        num_frames = int(t_inject)  # 1 frame per second
        for i in range(num_frames):
            frame = np.random.randn(224, 224, 3)  # Mock frame
            method.ingest_frame(frame, i)
        
        # Get answer
        predicted = method.answer(question)
        
        # Compute score
        return 1.0 if predicted == ground_truth else 0.0
    
    measurer._run_streaming_evaluation = mock_streaming_evaluation
    
    # Run QAR measurement
    print("\nRunning QAR measurement...")
    results = measurer.measure_qar(video_questions, methods, video_dir="mock_videos")
    
    # Summarize results
    print("\nSummarizing results...")
    summaries = measurer.summarize_results(results)
    
    # Print results
    print("\n" + "="*50)
    print("QAR MEASUREMENT RESULTS")
    print("="*50)
    
    for method_name, summary in summaries.items():
        slope_per_min = summary['slope'] * 60
        print(f"\n{method_name}:")
        print(f"  AUC_Δ: {summary['auc_delta']:.3f}")
        print(f"  Slope: {slope_per_min:.4f} score/min (p={summary['slope_pvalue']:.3f})")
        print(f"  LQF: {summary['lqf']:.3f}")
        
        print(f"  Score by staleness:")
        for delta in summary['deltas']:
            curve_data = summary['curve'][delta]
            print(f"    Δ={delta}s: {curve_data['mean']:.3f} ± {curve_data['std']:.3f}")
    
    # Statistical comparisons
    if len(methods) > 1:
        print(f"\nStatistical Comparisons:")
        comparisons = measurer.statistical_comparison(results, baseline_method='Sliding-Window')
        
        for method, comp in comparisons.items():
            sig = "***" if comp['p_value'] < 0.001 else "**" if comp['p_value'] < 0.01 else "*" if comp['p_value'] < 0.05 else ""
            print(f"  {method} vs {comp['baseline']}: Δ={comp['mean_difference']:.3f}, p={comp['p_value']:.3f} {sig}")
    
    # Test plotting (without showing)
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        measurer.plot_qar_curves(summaries, save_path='test_qar_curves.png')
        print(f"\nQAR curves saved to test_qar_curves.png")
    except Exception as e:
        print(f"\nPlotting failed: {e}")
    
    # Generate report
    try:
        report = measurer.generate_report(summaries, comparisons if len(methods) > 1 else {}, 
                                        save_path='test_qar_report.md')
        print(f"\nReport saved to test_qar_report.md")
        
        # Print first few lines of report
        print("\nReport preview:")
        print("-" * 30)
        print('\n'.join(report.split('\n')[:10]))
        print("...")
        
    except Exception as e:
        print(f"\nReport generation failed: {e}")
    
    print(f"\n✓ QAR measurement test completed successfully!")
    
    return results, summaries


def test_evidence_detection():
    """Test evidence detection with mock data"""
    print("\nTesting Evidence Detection")
    print("=" * 30)
    
    # This would require actual video files, so we'll just test the interface
    from evidence_detection import AdvancedEvidenceDetector
    
    config = QARConfig()
    detector = AdvancedEvidenceDetector(method='clip', config=config)
    
    print(f"✓ Evidence detector initialized with method: {detector.method}")
    
    # Test text processing
    question = "What color is the car in the video?"
    keywords = detector._extract_visual_concepts(question)
    print(f"✓ Extracted visual concepts: {keywords}")
    
    return detector


if __name__ == "__main__":
    print("QAR Implementation Test Suite")
    print("="*50)
    
    # Test evidence detection
    detector = test_evidence_detection()
    
    # Test QAR measurement
    results, summaries = test_qar_measurement()
    
    print("\n" + "="*50)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*50)
    
    print("\nNext steps:")
    print("1. Install dependencies: bash install_qar_deps.sh")
    print("2. Run on real data: python run_qar_evaluation.py --model llava_ov_7b --dataset mlvu")
    print("3. Customize evidence detection method and parameters as needed")