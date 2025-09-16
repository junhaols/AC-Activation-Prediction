# AC-Activation-Prediction

A machine learning project for predicting auditory cortex (AC) activation patterns during language tasks using Ridge regression on neuroimaging data from the Human Connectome Project.

## Overview

This project implements vertex-level prediction of brain activation in the auditory cortex during language processing tasks. Using multi-modal neuroimaging features from 766 subjects, the model combines functional connectivity, structural, and diffusion imaging data to predict task-evoked brain activation patterns.

## Quick Start

### Environment Setup
```bash
# Create and activate virtual environment
conda create -n ac-prediction python=3.9
conda activate ac-prediction

# Or using uv
uv venv ac-prediction
source ac-prediction/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis
```bash
# Run main prediction pipeline
cd src && python RidgeVert.py

# Run weight extraction (for model interpretation)
cd src && python -c "from RidgeVert import run_local_weights; run_local_weights()"
```

## Project Structure

```
AC-Activation-Prediction/
├── src/
│   ├── RidgeVert.py         # Main optimized Ridge regression pipeline
│   └── utlis/
│       └── io_.py           # Neuroimaging I/O and utility functions
├── raw/                     # Input data directory
│   ├── LANGUAGE/           # Task activation data (.pkl files)
│   ├── PAC_Features/       # Brain feature data (.pkl files)
│   ├── subjs/              # Subject metadata (.mat files)
│   └── stat_data/          # Statistical summary data
├── results/                 # Output directory
│   └── Ridge_766/          # Prediction results and model weights
└── requirements.txt         # Python dependencies
```

## Methodology

### Data Sources
- **Subjects**: 766 participants from Human Connectome Project (HCP)
- **Tasks**: Language processing tasks (Story-Math, Mean activation)
- **Brain Regions**: Left and Right Primary Auditory Cortex (LPAC/RPAC)

### Features
- **Functional Connectivity**: Fisher-Z transformed correlation matrices (`fisherZ`)
- **Structural Features**: Cortical area, thickness, myelination (`area`, `thick`, `myelin`)
- **Diffusion Features**: NODDI metrics (`NDI`, `ODI`, `ISO`)
- **FC Strength**: Functional connectivity strength measures (`FCs`)

### Machine Learning Pipeline
1. **Data Integration**: Combines multi-modal features at vertex level
2. **Cross-Validation**: 5-fold nested CV with subject sorting by mean activation
3. **Hyperparameter Optimization**: Ridge regression alpha tuning (2^-10 to 2^5)
4. **Evaluation**: Correlation, MAE, R², NRMSE metrics per subject

## Key Features

- **Vertex-Level Analysis**: Fine-grained prediction at brain surface vertices
- **Multi-Modal Integration**: Combines functional, structural, and diffusion MRI data
- **Nested Cross-Validation**: Robust model evaluation with hyperparameter optimization
- **Optimized Performance**: Memory-efficient processing with progress tracking
- **Reproducible Results**: Consistent subject sorting and random state management

## Results

The model generates:
- **Prediction Accuracy**: Subject-level correlation and error metrics
- **Model Weights**: Spatial patterns of feature importance
- **Cross-Validation Performance**: Robust evaluation across data splits

Output files are saved in `results/Ridge_766/` with metrics including:
- Correlation between predicted and actual activation
- Mean Absolute Error (MAE)
- R-squared values
- Normalized Root Mean Square Error (NRMSE)

## Requirements

### System Requirements
- Python 3.9+
- 16GB+ RAM recommended
- Multi-core CPU (for parallel processing)

### Dependencies
See `requirements.txt` for complete list:
- pandas >= 1.5.0
- numpy >= 1.20.0
- scipy >= 1.8.0
- scikit-learn >= 1.1.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0

## Usage Examples

### Basic Prediction
```python
# Run with default parameters
cd src && python RidgeVert.py
```

### Custom Feature Combination
```python
# Modify features in RidgeVert.py
features_all = {
    'FCMap': ['fisherZ'],
    'Structs': ['area', 'thick', 'myelin', 'NDI', 'ODI', 'ISO'],
    'Combined': ['fisherZ', 'area', 'thick', 'myelin']
}
```

## Data Requirements

Due to data sharing restrictions, neuroimaging data files are not included in this repository. To run the analysis, you need:

1. **HCP Language Task Data**: Preprocessed activation maps
2. **PAC Feature Data**: Extracted functional and structural features
3. **Subject Metadata**: Subject IDs and sorting indices

Contact the authors for data access procedures following HCP data sharing policies.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{ac-activation-prediction,
  title={AC-Activation-Prediction: Ridge Regression for Auditory Cortex Activation Prediction},
  author={Your Name},
  year={2025},
  url={https://github.com/junhaols/AC-Activation-Prediction}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions about the code or methodology, please open an issue or contact the authors.