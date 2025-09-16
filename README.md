# AC-Activation-Prediction

Neuroimaging machine learning project for predicting auditory cortex activation patterns during language tasks.

## Quick Start

```bash
# Setup environment
uv venv ac-prediction
source ac-prediction/bin/activate
uv pip install -r requirements.txt

# Run analysis
cd src && python RidgeVert.py
```

## Project Structure

```
.
├── src/
│   ├── RidgeVert.py         # Main optimized pipeline
│   └── utlis/               # Utility modules
├── raw/                     # Input data
│   ├── LANGUAGE/           # Task activation data
│   ├── PAC_Features/       # Brain features
│   └── subjs/              # Subject metadata
├── results/Ridge_766/       # Output predictions
└── archive/                 # Old versions (archived)
```

## Key Features

- **Ridge Regression**: Vertex-level prediction with nested cross-validation
- **Multi-Modal Integration**: Combines functional, structural, and diffusion features
- **Optimized Performance**: Parallel processing, progress tracking, and efficient memory usage
- **766 HCP Subjects**: Large-scale analysis of Human Connectome Project data

## Documentation

- `CLAUDE.md`: Detailed technical documentation for development
- `CONSISTENCY_*.md`: Algorithm validation reports

## Requirements

- Python 3.9+
- 16GB+ RAM recommended
- See `requirements.txt` for dependencies