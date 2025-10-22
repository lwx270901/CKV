
# QAR Measurement Installation Guide

## Quick Setup (Recommended)

1. Install core dependencies:
```bash
pip install numpy matplotlib seaborn scipy scikit-learn pandas torch transformers
```

2. Install optional video/text processing:
```bash
pip install opencv-python nltk spacy
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

3. Test installation:
```bash
python test_qar.py
```

## Minimal Setup (Text-only)

If you only need basic functionality without video processing:

```bash
pip install numpy matplotlib scipy scikit-learn transformers
```

## Docker Setup (Alternative)

```dockerfile
FROM python:3.9

RUN pip install numpy matplotlib seaborn scipy scikit-learn pandas torch transformers
RUN pip install opencv-python nltk spacy
RUN python -m spacy download en_core_web_sm

COPY . /app
WORKDIR /app
```

## Troubleshooting

- **Import errors**: Install missing packages with pip
- **Video errors**: Install opencv-python
- **Text processing errors**: Install nltk and spacy
- **Memory issues**: Reduce sample_fps or max_questions

## Usage Examples

### Basic QAR evaluation:
```python
from qar_measurement import QARMeasurer, QARConfig

config = QARConfig(evidence_method='manual')
measurer = QARMeasurer(config)
```

### With video processing:
```python
config = QARConfig(evidence_method='clip', sample_fps=0.5)
```
