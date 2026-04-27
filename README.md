# AC-Activation-Prediction

A neuroimaging machine-learning pipeline that predicts auditory-cortex (AC) activation patterns during language tasks using Ridge regression on Human Connectome Project data, plus a complete R plotting pipeline that reproduces every figure in the paper.

## Overview

Vertex-level prediction of task-evoked brain activation in the auditory cortex. Multi-modal features (functional connectivity, cortical structure, diffusion MRI) from 766 HCP subjects are combined and passed through nested cross-validated Ridge regression. The repo contains:

- **Python pipeline** (`src/`) — feature integration, nested CV, prediction, weight extraction.
- **R plotting pipeline** (`scripts/plots/`) — 12 scripts that reproduce all main and supplementary figures from pre-aggregated CSV statistics.

## Project Structure

```
AC-Activation-Prediction/
├── src/
│   ├── RidgeVert.py             # Main Ridge regression pipeline (parallel, progress-tracked)
│   └── utlis/io_.py             # Neuroimaging I/O and utilities
├── scripts/
│   └── plots/                   # R figure-reproduction scripts (Fig 2-7, Fig S1-S4)
│       ├── Plot-Fig2-B.R / Plot-Fig2-D.R
│       ├── Plot-Fig3-A.R / Plot-Fig3-B.R
│       ├── Plot-Fig5.R / Plot-Fig5-circle.R
│       ├── Plot-Fig6.R          # uses geom_paired_raincloud.R helper
│       ├── Plot-Fig7.R
│       ├── Plot-Fig_S1.R ... Plot-Fig_S4.R
│       └── geom_paired_raincloud.R
├── raw/                         # Input data (gitignored except .gitkeeps)
│   ├── LANGUAGE/                # Task activation data (.pkl)
│   ├── PAC_Features/            # Brain features (.pkl)
│   ├── subjs/                   # Subject metadata (.mat)
│   ├── stat_data/               # Aggregated stats (DataForStats766.csv)
│   └── figs_csv/                # Per-figure CSV inputs for R scripts
├── results/                     # Python pipeline outputs (gitignored)
│   └── Ridge_766/               # Predictions, weights, metrics
├── papers/figures/              # Final figure artifacts (gitignored)
├── tests/                       # Plot-script smoke-test outputs (gitignored)
├── requirements.txt             # Python dependencies
└── README.md
```

## Quick Start

### 1. Python pipeline

```bash
# Create venv and install
uv venv ac-prediction && source ac-prediction/bin/activate
uv pip install -r requirements.txt

# Run prediction
cd src && python RidgeVert.py

# Extract model weights for interpretation
cd src && python -c "from RidgeVert import run_local_weights; run_local_weights()"
```

### 2. R figure pipeline

```bash
# From project root, run any of the 12 scripts:
Rscript scripts/plots/Plot-Fig3-A.R

# Or run them all (mirrors the test-harness used during development):
for f in scripts/plots/Plot-*.R; do Rscript "$f"; done
```

Each script auto-resolves `PROJECT_ROOT`, reads from `raw/figs_csv/`, and writes PDFs/PNGs/TIFFs to `papers/figures/raw/Fig{N}/`.

## Methodology

### Data Sources

- **Subjects**: 766 participants from the Human Connectome Project (HCP).
- **Tasks**: Language tasks (Story-Math contrast, Mean speech-vs-baseline activation).
- **Regions**: Left and Right peri-Sylvian / Primary Auditory Cortex (LPAC, RPAC).

### Features

| Modality | Features |
|---|---|
| Functional connectivity | Fisher-Z FC maps (`fisherZ`), FC strength (`FCs`) |
| Structural | Cortical area, thickness, myelin (`area`, `thick`, `myelin`) |
| Diffusion (NODDI) | `NDI`, `ODI`, `ISO` |

### Machine Learning Pipeline

1. **Data integration** — multi-modal features concatenated at vertex level per subject.
2. **Cross-validation** — 5-fold nested CV with subjects sorted by mean activation.
3. **Hyperparameter search** — Ridge `alpha` ∈ `2^-10 … 2^5` (inner loop).
4. **Evaluation** — correlation, MAE, R², NRMSE per subject (outer loop).

## Figure Reproduction

| Figure | Script | Output(s) |
|---|---|---|
| Fig 2B | `Plot-Fig2-B.R` | HCP raincloud (4 contrasts, paired t-tests) |
| Fig 2D | `Plot-Fig2-D.R` | Validation paired violin (run-1 / run-2) |
| Fig 3A | `Plot-Fig3-A.R` | Feature-combination boxplot (FCMap / Structs / FCMap+Structs) |
| Fig 3B | `Plot-Fig3-B.R` | Cross-task ridge plots (own-correlation; 3 variants) |
| Fig 5  | `Plot-Fig5.R` | Brain-measure scatter plots + behaviour scatter (PicVocab, g) |
| Fig 5  | `Plot-Fig5-circle.R` | Circular bar chart of measure-PS correlations |
| Fig 6  | `Plot-Fig6.R` | LI raincloud + LI scatters + LI circular bar |
| Fig 7  | `Plot-Fig7.R` | 4-group panel (cognition / averages / LIs) |
| Fig S1 | `Plot-Fig_S1.R` | Per-fold scatter plots (10 folds) |
| Fig S2 | `Plot-Fig_S2.R` | Per-run validation scatter plots |
| Fig S3 | `Plot-Fig_S3.R` | DiceAUC raincloud across 4 contrasts |
| Fig S4 | `Plot-Fig_S4.R` | Cross-task ridge plots (language-based correlation) |

### R dependencies

`tidyverse`, `ggpubr`, `ggridges`, `ggdist`, `ggsci`, `ggstatsplot`, `rstatix`, `viridis`, `hrbrthemes`, `geomtextpath`, `cowplot`, `see`, `svglite`, `showtext`.

## Requirements

- Python 3.9+, 16 GB+ RAM (parallel CV is memory-bound)
- R 4.2+ for the plotting pipeline
- Dependencies: see `requirements.txt` (Python) and the section above (R)

## Data Access

Raw neuroimaging data are not included in the repository (HCP data-sharing terms). To reproduce results you need:

1. **HCP language-task activation maps** (preprocessed)
2. **PAC feature data** — functional, structural, diffusion at vertex level
3. **Subject metadata** — subject IDs and sort indices
4. **Pre-aggregated CSV statistics** for the R pipeline (drop into `raw/figs_csv/`)

Contact the authors for access procedures aligned with HCP data-sharing policy.

## Citation

```bibtex
@misc{ac-activation-prediction,
  title  = {AC-Activation-Prediction: Ridge Regression for Auditory Cortex Activation Prediction},
  author = {Luo, Junhao},
  year   = {2026},
  url    = {https://github.com/junhaols/AC-Activation-Prediction},
  note   = {v1.0.0}
}
```

## License

MIT — see `LICENSE`.

## Contact

Issues and pull requests are welcome. For other questions: `junhaol@mail.bnu.edu.cn`.
