# Query-Agnostic Robustness (QAR) Measurement for ReKV

This implementation provides a complete, reproducible recipe for measuring **Query-Agnostic Robustness (QAR)** as a function of **query staleness** (Δ) for the ReKV video question-answering system.

## Overview

QAR measures how well a video QA model performs when questions are asked at different delays after the relevant evidence first appears in the video stream. This is critical for evaluating streaming video systems that must maintain performance even when questions arrive late.

## Key Concepts

- **Query Staleness (Δ)**: Time delay between when evidence first appears and when the question is asked
- **Evidence Timestamp (τ_evi)**: Earliest time when minimal evidence for answering the question appears
- **Query-Agnostic Processing**: Model processes video without seeing the question until injection time

## Files

- `qar_measurement.py`: Core QAR measurement implementation
- `evidence_detection.py`: Advanced evidence detection methods (CLIP, attention-based)
- `run_qar_evaluation.py`: Integration with ReKV evaluation framework
- `test_qar.py`: Test suite with mock data
- `install_qar_deps.sh`: Dependency installation script

## Installation

```bash
# Install dependencies
bash install_qar_deps.sh

# Or install manually:
pip install opencv-python nltk spacy scikit-learn matplotlib seaborn
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Quick Start

### 1. Test the Implementation

```bash
# Run tests with mock data
python test_qar.py
```

### 2. Run QAR Evaluation

```bash
# Basic evaluation
python run_qar_evaluation.py --model llava_ov_7b --dataset mlvu --max_questions 50

# Full evaluation with all baselines
python run_qar_evaluation.py \
    --model llava_ov_7b \
    --dataset mlvu \
    --include_rekv \
    --include_baselines \
    --evidence_method clip \
    --max_questions 100 \
    --output_dir results/qar_llava_ov_7b
```

### 3. Advanced Usage

```python
from qar_measurement import QARMeasurer, QARConfig

# Configure QAR measurement
config = QARConfig(
    delta_grid=[0.0, 30.0, 120.0, 300.0, 600.0, 1800.0],  # 0s, 30s, 2min, 5min, 10min, 30min
    evidence_method='clip',  # 'clip', 'attention', or 'manual'
    sample_fps=0.5,
    memory_budget=64
)

# Initialize measurer
measurer = QARMeasurer(config)

# Prepare methods to compare
methods = {
    'ReKV': rekv_model_wrapper,
    'Sliding-Window': sliding_window_baseline,
    'Full-KV': full_kv_baseline
}

# Run measurement
results = measurer.measure_qar(video_questions, methods, video_dir)
summaries = measurer.summarize_results(results)

# Generate report
report = measurer.generate_report(summaries, comparisons)
```

## Evidence Detection Methods

### 1. CLIP-based Detection (Recommended)

- Extracts visual concepts from questions
- Computes frame-text similarities using CLIP
- Finds first frame above similarity threshold
- **Pros**: Fully automatic, works well for visual questions
- **Cons**: May miss temporal or audio cues

```python
config = QARConfig(evidence_method='clip', clip_threshold_percentile=90.0)
```

### 2. Teacher Attention-based Detection

- Uses full-cache teacher model to find attended frames
- Aggregates cross-attention weights over video frames
- **Pros**: Most faithful to model's actual evidence usage
- **Cons**: Requires running teacher model for each question

```python
config = QARConfig(evidence_method='attention', attention_threshold_percentile=95.0)
```

### 3. Manual Annotation

- Human annotators mark evidence timestamps
- Most accurate but labor-intensive
- Use for validation and small high-confidence datasets

```python
# Provide manual timestamps in your data
video_questions = [
    {
        'question': 'What color is the car?',
        'manual_timestamp': 15.2,  # Evidence appears at 15.2 seconds
        ...
    }
]
```

## Metrics Reported

### Primary Metrics

1. **Score vs. Staleness Curve**: Performance across different Δ values
2. **AUC_Δ (Area Under Curve)**: Overall robustness measure (higher = better)
3. **Staleness Slope**: Rate of degradation per minute (flatter = better)
4. **Late-Query Factor (LQF)**: Ratio of performance at max vs. min staleness

### Statistical Tests

- Paired Wilcoxon tests between methods
- Bootstrap confidence intervals
- Significance testing for slopes and AUC differences

## Output Files

Each evaluation produces:

- `qar_raw_results.json`: Raw (question_id, Δ, score) tuples
- `qar_summaries.json`: Computed metrics and statistics
- `qar_curves.png`: Visualization of score vs. staleness
- `qar_comparisons.json`: Statistical comparison results
- `qar_report.md`: Formatted summary report

## Configuration Options

```python
@dataclass
class QARConfig:
    # Staleness grid (in seconds)
    delta_grid: List[float] = [0.0, 30.0, 120.0, 300.0, 600.0, 1800.0]
    
    # Evidence detection
    evidence_method: str = 'clip'  # 'manual', 'clip', 'attention'
    clip_threshold_percentile: float = 90.0
    attention_threshold_percentile: float = 95.0
    
    # Video processing
    sample_fps: float = 0.5
    
    # Evaluation
    memory_budget: int = 64
    confidence_level: float = 0.95
    random_seed: int = 2024
```

## Integration with ReKV Models

The implementation provides a wrapper class to make ReKV models compatible:

```python
class ReKVWrapper:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.frames = []
        
    def reset(self):
        """Reset model state for new evaluation"""
        self.model.clear_cache()
        self.frames = []
    
    def ingest_frame(self, frame, timestamp):
        """Ingest frame for query-agnostic processing"""
        self.frames.append(frame)
    
    def answer(self, question):
        """Answer question based on ingested frames"""
        return self.model.question_answering(question)
```

## Validation and Quality Controls

### Recommended Practices

1. **Same memory budget** across all methods
2. **Deterministic preprocessing** (fixed FPS, resolution)
3. **Anchor validation**: Manually verify evidence timestamps for 20-30 examples
4. **No query-conditioned compression**: Ensure questions don't influence cache before injection

### Validation Script

```python
from evidence_detection import validate_evidence_detection

# Validate evidence detection against manual annotations
validation_results = validate_evidence_detection(
    detector, video_questions, video_dir, manual_annotations
)
print(f"Mean Absolute Error: {validation_results['mean_mae']:.2f} seconds")
print(f"Agreement within 30s: {validation_results['agreement_rate']:.1%}")
```

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Run `bash install_qar_deps.sh`
2. **Video loading errors**: Ensure OpenCV is properly installed
3. **Memory issues**: Reduce `max_questions` or `sample_fps`
4. **Model loading errors**: Check model paths and GPU memory

### Performance Tips

1. **Precompute features**: Cache CLIP embeddings for repeated evaluations
2. **Parallel processing**: Use multiple GPUs for different Δ values
3. **Subset evaluation**: Start with `--max_questions 20` for quick tests

## Paper Integration

### Reporting Template

```markdown
**Query-Agnostic Robustness.** We anchor each question q at the earliest evidence 
time τ_evi(q) using CLIP-based visual similarity (90th percentile threshold). 
We then stream the video without queries and inject q at delays 
Δ∈{0s,30s,2min,5min,10min,30min}. Figure X plots score vs. Δ with 95% CIs. 
Our method achieves AUC_Δ=0.85 and slope -0.02/min (vs. Sliding-Window -0.08/min, 
p<0.01). The Late-Query Factor at 30min is 0.92, indicating strong staleness-invariance.
```

### Key Results to Report

1. **AUC_Δ comparison** with baselines
2. **Slope analysis** with confidence intervals
3. **Statistical significance** of differences
4. **Late-Query Factor** for long delays

## Extension Points

### Custom Evidence Detection

```python
class CustomEvidenceDetector:
    def detect_evidence_timestamp(self, video_path, question, **kwargs):
        # Implement your custom logic
        return timestamp

measurer.evidence_detector = CustomEvidenceDetector()
```

### Custom Baselines

```python
class CustomBaseline:
    def reset(self): pass
    def ingest_frame(self, frame, timestamp): pass
    def answer(self, question): return "answer"

methods['Custom-Method'] = CustomBaseline()
```

### Dataset Integration

Add support for new datasets by extending `load_evaluation_data()` in `run_qar_evaluation.py`.

## Citation

If you use this QAR measurement implementation, please cite:

```bibtex
@misc{rekv-qar-2024,
  title={Query-Agnostic Robustness Measurement for Video Question Answering},
  author={ReKV Team},
  year={2024},
  note={Implementation for ReKV evaluation}
}
```

## License

This implementation is part of the ReKV project and follows the same license terms.