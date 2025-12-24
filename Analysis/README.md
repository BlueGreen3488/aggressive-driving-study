# Analysis

This folder contains the complete analysis workflow used to reproduce the results reported in the paper.

The analysis is organised as a single executable notebook (`main_analysis.ipynb`) together with a set of supporting Python modules for data processing, modelling, and SHAP-based interpretation.

---

## Files in this folder

- `main_analysis.ipynb`  
  Main analysis notebook. Running this notebook from top to bottom reproduces the modelling results, SHAP stability analysis, and visualisations reported in the paper.

- `modules/`  
  Supporting Python modules used by the notebook:
  - `config.py`: fixed configuration, including feature definitions, data types, and model parameter presets.
  - `data_utils.py`: data loading and preprocessing.
  - `model_utils.py`: model training and cross-validation.
  - `shap_utils.py`: SHAP value computation.
  - `shap_stability_utils.py`: SHAP stability assessment via repeated sampling.
  - `draw_figure.py`: functions for generating SHAP summary and dependence plots.
  - `io_utils.py`: utilities for output directory and file naming.

---

## How to run the analysis

1. Place the input dataset as `data.csv` in this folder.  
   (The dataset is not included in this repository.)

2. Open `main_analysis.ipynb`.

3. In **Cell 1**, specify the experimental condition by setting:
   - `PERIOD` (`"day"` or `"night"`)
   - `TYPE_FILTER` (`None` or a specific road type, e.g. `"motorway"`)

4. Run the notebook sequentially from top to bottom.

All results will be generated automatically during execution.

---

## Outputs

During execution, the notebook creates an `outputs/` directory containing condition-specific subfolders (e.g. `night`, `night+motorway`). These folders store:

- SHAP visualisations (`figures/`)
- Model performance metrics and SHAP importance tables (`tables/`)
- The final trained XGBoost model (`models/`, final model only)

The `outputs/` directory is generated at runtime and is not part of the repository.

---

## Notes on reproducibility

- Cross-validation is used to assess model robustness under the selected configuration.
- A single train–validation–test split is used to train the final model for SHAP interpretation.
- In rare cases, an unfavourable random split may lead to premature early stopping. Adjusting the `random_state` parameter and re-running the notebook is sufficient to recover stable performance.
- Trained models are saved solely to ensure reproducibility of downstream SHAP analyses and are not intended for deployment.

---

## Dependencies

The analysis requires a standard scientific Python environment, including:

- Python ≥ 3.8
- numpy
- pandas
- scikit-learn
- xgboost
- shap
- matplotlib
