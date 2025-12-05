# Running Both Dataset Analyses

## Quick Start

### Run General Diabetes Analysis
```bash
python main_general.py
```
- **Output:** `models/general_diabetes/` and `results/general_diabetes/`
- **Dataset:** 96,146 samples, 8 features
- **Expected Accuracy:** ~97% (Random Forest)

### Run Pima Diabetes Analysis
```bash
python main_pima.py
```
- **Output:** `models/pima_diabetes/` and `results/pima_diabetes/`
- **Dataset:** 768 samples, 8 features
- **Expected Accuracy:** ~75% (Random Forest)

### Run Both Sequentially
```bash
python main_general.py && python main_pima.py
```

## Results Location

### General Dataset Results
- **Models:** `models/general_diabetes/`
  - `logistic_regression_model.pkl`
  - `random_forest_model.pkl`
  - `svm_model.pkl`

- **Visualizations:** `results/general_diabetes/`
  - ROC curves, confusion matrices, feature importance plots
  - Model comparison charts
  - `model_comparison_summary.csv`

### Pima Dataset Results
- **Models:** `models/pima_diabetes/`
  - `logistic_regression_model.pkl`
  - `random_forest_model.pkl`
  - `svm_model.pkl`

- **Visualizations:** `results/pima_diabetes/`
  - ROC curves, confusion matrices, feature importance plots
  - Model comparison charts
  - `model_comparison_summary.csv`

## Performance Summary

| Dataset | Best Model | Accuracy | ROC-AUC | Training Time |
|---------|-----------|----------|---------|---------------|
| **General** | Random Forest | 97.12% | 0.9680 | 0.69s |
| **Pima** | Random Forest | 75.32% | 0.8085 | 0.33s |

## Key Differences

- **General:** Large dataset (96K), very high accuracy, severe class imbalance
- **Pima:** Small dataset (768), moderate accuracy, realistic messy data

See [DATASET_COMPARISON.md](DATASET_COMPARISON.md) for detailed comparative analysis.

## Workflow Visualization

For a visual representation of the complete ML pipeline used for both datasets:
- View the workflow diagram: `workflow/ml_pipeline_workflow.png`
- Regenerate diagram: `python workflow/generate_workflow_diagram.py`
