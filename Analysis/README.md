# Road Environment and Aggressive Driving

This repository provides anonymized code and derived data supporting the analyses in the manuscript  
**“Road Environment Differently Shapes Urban Aggressive Driving”** (under review).

The repository is intended to ensure transparency and reproducibility of the analytical workflow, while respecting data privacy and confidentiality constraints.

---

## Overview

This project investigates how multilayered road environments—operational, streetscape, and structural factors—shape aggressive driving behaviour at the urban scale.

Using large-scale ride-hailing trajectory data and street-view imagery from a major Chinese city, the study quantifies aggressive driving through a jerk-based index and applies explainable machine-learning models to uncover nonlinear and context-dependent effects.

This repository contains:
- The full data-processing and modelling pipeline
- Anonymized, analysis-ready datasets
- Scripts to reproduce all main and supplementary results reported in the paper

---

## Repository structure
    Analysis/
    ├── main_analysis.ipynb # Main entry point for reproducing results
    ├── modules/ # Core data processing, modelling, and SHAP utilities
    ├── data.csv # Input dataset (not included / to be provided separately)
    ├── outputs/ # Automatically generated results
    │ └── <condition>/ # e.g. night, night+motorway
    │ ├── figures/ # SHAP plots
    │ ├── tables/ # Performance metrics and SHAP importance tables
    │ └── models/ # Trained XGBoost model (final model only)
    └── README.md

---

## Requirements

The analysis is implemented in Python. The main dependencies include:

- Python ≥ 3.8
- numpy
- pandas
- scikit-learn
- xgboost
- shap
- matplotlib

All experiments were run and tested under a standard scientific Python environment.

---

## Reproducibility

To reproduce the results reported in the paper:

1. Place the input dataset as `data.csv` in the root directory of this repository.
2. Open `main_analysis.ipynb`.
3. Modify the following variables in **Cell 1** to select the experimental condition:
   - `PERIOD` (`"day"` or `"night"`)
   - `TYPE_FILTER` (`None` or a specific road type, e.g. `"motorway"`)
4. Run the notebook from top to bottom.

All outputs (figures, tables, and the trained model) will be automatically saved to the corresponding subdirectory under `./outputs/`.

---


## License

This code is provided for academic research and reproducibility purposes.
