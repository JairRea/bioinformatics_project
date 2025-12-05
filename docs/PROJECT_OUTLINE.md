# Bioinformatics Project: Diabetes Prediction Using Machine Learning

**GitHub Repository:** [https://github.com/JairRea/bioinformatics_project](https://github.com/JairRea/bioinformatics_project)

---

## Computational and Statistical Approach

This project implements a supervised binary classification pipeline to predict type 2 diabetes status using clinical and demographic biomarkers from the Kaggle Diabetes Prediction Dataset (100,000 patient records). The computational workflow encompasses comprehensive data preprocessing, three distinct machine learning algorithms, rigorous evaluation metrics, and extensive visualization capabilities, all implemented in Python 3.12.3 within a reproducible virtual environment.

---

## Pipeline and Workflow Design

### Custom Python Pipeline
The analysis employs a modular, object-oriented Python pipeline with the following components:

**Core Modules:**
- `data_preprocessing.py`: Data loading, cleaning, encoding, and scaling
- `logistic_regression_model.py`: Logistic regression implementation
- `random_forest_model.py`: Random forest classifier implementation  
- `svm_model.py`: Support vector machine implementation
- `visualizations.py`: Plotting and visualization functions
- `main.py`: Orchestration script coordinating the entire workflow

**Environment Management:**
- Python virtual environment (`.venv/`) with pinned dependencies
- `requirements.txt` for reproducible package installation
- Python version: 3.12.3

**Workflow Execution:**
```bash
# Setup
pip install -r requirements.txt

# Full pipeline execution
python main.py
```

The pipeline automatically handles:
1. Dataset loading and validation
2. Data quality checks and duplicate removal
3. Feature engineering and encoding
4. Train-test splitting with stratification
5. Model training for all three algorithms
6. Comprehensive evaluation and metrics computation
7. Visualization generation
8. Model serialization and result persistence

---

## Data Preprocessing

### Tools and Versions
- **pandas** (2.3.3): Data manipulation and exploratory analysis
- **numpy** (2.3.5): Numerical computations and array operations
- **scikit-learn** (1.7.2): Preprocessing transformers

### Preprocessing Steps

**1. Data Quality Control:**
- Duplicate removal: 3,854 duplicates identified and removed from 100,000 records
- Missing value detection and imputation (median for numerical, mode for categorical)
- Dataset validation and integrity checks

**2. Feature Engineering:**
- **Categorical encoding:**
  - Gender: Female=0, Male=1, Other=2
  - Smoking history: Ordinal mapping across 6 categories (never=0, No Info=1, current=2, former=3, ever=4, not current=5)
- **Numerical features:** Age, BMI, HbA1c level, blood glucose level retained as continuous variables
- **Binary features:** Hypertension, heart disease as 0/1 indicators

**3. Feature Scaling:**
- StandardScaler (z-score normalization): μ=0, σ=1
- Fit on training set, transform on both training and test sets to prevent data leakage

**4. Data Partitioning:**
- Train-test split: 80/20 ratio (76,916 train / 19,230 test samples)
- Stratified sampling to maintain class distribution (91.5% negative / 8.5% positive)
- Random state: 42 (for reproducibility)

**5. Class Imbalance Handling:**
- **imbalanced-learn** (0.14.0) available for SMOTE or other resampling techniques
- Current implementation relies on model-native handling of imbalanced data

---

## Modeling and Analysis Approach

### Algorithm 1: Logistic Regression
**Implementation:** `sklearn.linear_model.LogisticRegression`

**Hyperparameters:**
- Solver: LBFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno)
- Maximum iterations: 1,000
- Random state: 42

**Methodology:**
- Linear probabilistic classifier modeling log-odds as linear combination of features
- Closed-form optimization via quasi-Newton method
- Provides interpretable coefficients for feature importance

**Performance:**
- Accuracy: 95.92%
- Precision: 86.83%
- Recall: 63.38%
- F1-Score: 73.28%
- ROC-AUC: 0.9595
- Training time: 0.64 seconds

**Feature Importance (Top 3):**
1. HbA1c level (coefficient: 2.486)
2. Blood glucose level (coefficient: 1.363)
3. Age (coefficient: 1.064)

---

### Algorithm 2: Random Forest Classifier
**Implementation:** `sklearn.ensemble.RandomForestClassifier`

**Hyperparameters:**
- Number of estimators: 100 decision trees
- Minimum samples per split: 5
- Minimum samples per leaf: 2
- Parallelization: n_jobs=-1 (all CPU cores)
- Random state: 42

**Methodology:**
- Ensemble learning via bootstrap aggregating (bagging)
- Gini impurity-based feature importance
- Majority voting for final prediction
- Built-in protection against overfitting via randomization

**Performance:**
- Accuracy: 97.12% ⭐ **Best overall**
- Precision: 98.39%
- Recall: 68.46%
- F1-Score: 80.74%
- ROC-AUC: 0.9680
- Training time: 0.69 seconds

**Feature Importance (Top 3):**
1. HbA1c level (importance: 0.421)
2. Blood glucose level (importance: 0.357)
3. Age (importance: 0.087)

---

### Algorithm 3: Support Vector Machine (SVM)
**Implementation:** `sklearn.svm.SVC`

**Hyperparameters:**
- Kernel: RBF (Radial Basis Function)
- Regularization parameter (C): 1.0
- Gamma: 'scale' (1 / (n_features × variance))
- Probability estimates: Enabled

**Methodology:**
- Maximum margin classifier with kernel transformation
- Maps features to higher-dimensional space for non-linear separation
- Dual optimization via quadratic programming
- Support vectors: 3,626 (class 0) + 3,427 (class 1) = 7,053 total

**Performance:**
- Accuracy: 96.26%
- Precision: 97.20%
- Recall: 59.26%
- F1-Score: 73.63%
- ROC-AUC: 0.9282
- Training time: 106.10 seconds

---

### Hyperparameter Tuning (Optional)
**Implementation:** `sklearn.model_selection.GridSearchCV`

Grid search with cross-validation is implemented but disabled by default for computational efficiency. Can be activated via:
```python
TUNE_HYPERPARAMETERS = True  # in main.py
```

**Random Forest search space:**
- n_estimators: [50, 100, 200]
- max_depth: [10, 20, 30, None]
- min_samples_split: [2, 5, 10]
- min_samples_leaf: [1, 2, 4]

**SVM search space:**
- kernel: ['linear', 'rbf']
- C: [0.1, 1, 10, 100]
- gamma: ['scale', 'auto', 0.001, 0.01, 0.1]

**Cross-validation:** 3-5 folds
**Scoring metric:** ROC-AUC

---

## Evaluation and Validation

### Performance Metrics
All metrics computed using **scikit-learn.metrics** module:

**Binary Classification Metrics:**
1. **Accuracy:** Overall correctness = (TP + TN) / (TP + TN + FP + FN)
2. **Precision:** Positive predictive value = TP / (TP + FP)
3. **Recall (Sensitivity):** True positive rate = TP / (TP + FN)
4. **F1-Score:** Harmonic mean of precision and recall = 2 × (P × R) / (P + R)
5. **ROC-AUC:** Area under receiver operating characteristic curve (discrimination across all thresholds)

**Confusion Matrix Analysis:**
- True Positives (TP): Correct diabetes predictions
- True Negatives (TN): Correct non-diabetes predictions
- False Positives (FP): Type I errors
- False Negatives (FN): Type II errors (clinically critical)

**Classification Reports:**
- Per-class precision, recall, F1-score
- Macro and weighted averages
- Support (sample counts per class)

### Model Comparison
Systematic comparison across all metrics to identify optimal algorithm:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time |
|-------|----------|-----------|--------|----------|---------|---------------|
| Logistic Regression | 95.92% | 86.83% | 63.38% | 73.28% | 0.9595 | 0.64s |
| Random Forest | **97.12%** | **98.39%** | **68.46%** | **80.74%** | **0.9680** | 0.69s |
| SVM | 96.26% | 97.20% | 59.26% | 73.63% | 0.9282 | 106.10s |

**Winner:** Random Forest excels in all metrics while maintaining computational efficiency.

### Model Persistence
**Tool:** joblib (1.5.2) for efficient model serialization

Trained models saved to `models/` directory:
- `logistic_regression_model.pkl`
- `random_forest_model.pkl`
- `svm_model.pkl`

Enables deployment and inference without retraining.

---

## Visualization

### Tools and Versions
- **matplotlib** (3.10.7): Low-level plotting library
- **seaborn** (0.13.2): Statistical visualization and enhanced aesthetics

### Generated Visualizations

**1. ROC Curves**
- Individual curves for each model showing TPR vs FPR trade-offs
- Comparative plot overlaying all three models
- AUC values annotated for quantitative comparison
- Files: `*_roc_curve.png`, `model_comparison_roc_curves.png`

**2. Confusion Matrices**
- Normalized heatmaps showing prediction patterns
- True/false positive/negative counts
- Individual matrices per model
- Composite view showing all three models
- Files: `*_confusion_matrix.png`, `all_confusion_matrices.png`

**3. Feature Importance Plots**
- **Logistic Regression:** Horizontal bar chart of absolute coefficient magnitudes
- **Random Forest:** Gini importance-based feature rankings
- Color-coded by importance magnitude
- Files: `logistic_regression_coefficients.png`, `random_forest_feature_importance.png`

**4. Model Performance Comparison**
- Grouped bar chart comparing all five metrics across models
- Side-by-side visualization for rapid assessment
- File: `model_metrics_comparison.png`

**5. Summary Tables**
- CSV export of all metrics for external analysis
- File: `model_comparison_summary.csv`

All visualizations saved to `results/` directory with high-resolution PNG format.

---

## Software Dependencies

### Core Machine Learning Stack
```
numpy>=1.24.0              (installed: 2.3.5)
pandas>=2.0.0              (installed: 2.3.3)
scikit-learn>=1.3.0        (installed: 1.7.2)
imbalanced-learn>=0.11.0   (installed: 0.14.0)
```

### Visualization Libraries
```
matplotlib>=3.7.0          (installed: 3.10.7)
seaborn>=0.12.0            (installed: 0.13.2)
```

### Utility Packages
```
jupyter>=1.0.0             (installed: 1.1.1)
joblib                     (installed: 1.5.2)
scipy                      (installed: 1.16.3)
```

### Python Environment
```
Python: 3.12.3
Virtual environment: .venv/
```

---

## GitHub Repository

**URL:** [https://github.com/JairRea/bioinformatics_project](https://github.com/JairRea/bioinformatics_project)

### Repository Structure
```
bioinformatics_project/
├── README.md                          # Comprehensive project documentation
├── requirements.txt                   # Pinned dependencies for reproducibility
├── data/
│   └── diabetes_prediction_dataset.csv    # Kaggle dataset (100k records)
├── models/
│   ├── logistic_regression_model.pkl      # Serialized LR model
│   ├── random_forest_model.pkl            # Serialized RF model
│   └── svm_model.pkl                      # Serialized SVM model
├── results/
│   ├── *_roc_curve.png                    # ROC visualizations
│   ├── *_confusion_matrix.png             # Confusion matrices
│   ├── *_feature_importance.png           # Feature analysis plots
│   ├── model_comparison_*.png             # Comparative visualizations
│   └── model_comparison_summary.csv       # Metrics table
├── data_preprocessing.py              # Data pipeline implementation
├── logistic_regression_model.py       # LR classifier class
├── random_forest_model.py             # RF classifier class
├── svm_model.py                       # SVM classifier class
├── visualizations.py                  # Plotting functions
└── main.py                            # Workflow orchestration script
```

### README.md Contents
The repository includes a comprehensive README.md with:
- Project overview and objectives
- Complete file structure catalog
- Setup and installation instructions
- Usage guide for running the analysis
- Model descriptions and methodologies
- Evaluation metrics explanation
- Results directory contents

### Usage Instructions
```bash
# Clone repository
git clone https://github.com/JairRea/bioinformatics_project.git
cd bioinformatics_project

# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Download dataset from Kaggle and place in data/

# Run complete analysis pipeline
python main.py

# Results will be saved to results/ and models/ directories
```

---

## Project Timeline

### ✅ Completed Work (Phase 1: November 2025)

**Week 1: Data Acquisition and Preprocessing**
- [x] Dataset sourced from Kaggle (100,000 patient records)
- [x] Exploratory data analysis and quality assessment
- [x] Data cleaning pipeline implemented (duplicate removal, missing value handling)
- [x] Feature engineering and categorical encoding
- [x] Standardization and train-test splitting with stratification

**Week 2: Model Development**
- [x] Logistic Regression implementation and training (0.64s training time)
- [x] Random Forest implementation and training (0.69s training time)
- [x] SVM implementation and training (106.10s training time)
- [x] Hyperparameter tuning framework via GridSearchCV (optional mode)
- [x] Model serialization and persistence

**Week 3: Evaluation and Visualization**
- [x] Comprehensive evaluation metrics computation (accuracy, precision, recall, F1, ROC-AUC)
- [x] Confusion matrix analysis for all models
- [x] Feature importance extraction (coefficients for LR, Gini for RF)
- [x] ROC curve generation (individual and comparative)
- [x] Confusion matrix heatmaps
- [x] Feature importance bar charts
- [x] Model comparison visualizations
- [x] Results export to CSV

**Week 4: Documentation and Version Control**
- [x] Modular code architecture with OOP design patterns
- [x] Comprehensive inline documentation and docstrings
- [x] README.md creation with full project catalog
- [x] requirements.txt with versioned dependencies
- [x] GitHub repository initialization and first commit
- [x] Complete codebase push to remote repository

**Current Status:** Core pipeline fully functional with three trained models achieving 95-97% accuracy. All results and visualizations generated and saved.

---

### 🔄 In Progress (Phase 2: December 2025)

**Week 5: Advanced Analysis**
- [ ] Cross-validation implementation for robust performance estimation
- [ ] Hyperparameter tuning optimization (activate GridSearchCV)
- [ ] Statistical significance testing (paired t-tests, McNemar's test)
- [ ] Learning curve analysis to assess bias-variance trade-off
- [ ] Threshold optimization for clinical deployment scenarios

**Week 6: Model Enhancement**
- [ ] Ensemble methods (stacking, voting classifiers)
- [ ] Class imbalance handling via SMOTE or class weighting
- [ ] Feature selection experiments (recursive feature elimination)
- [ ] Additional algorithms (XGBoost, Neural Networks)
- [ ] Calibration analysis (reliability diagrams, Brier scores)

---

### 📋 Planned Work (Phase 3: January-February 2026)

**Week 7-8: Clinical Interpretation**
- [ ] SHAP (SHapley Additive exPlanations) values for model explainability
- [ ] Partial dependence plots for feature relationship visualization
- [ ] Subgroup analysis (age groups, gender, comorbidity patterns)
- [ ] Clinical decision threshold analysis (cost-benefit optimization)
- [ ] Literature review comparing performance to published studies

**Week 9: Deployment Preparation**
- [ ] Model API development (FastAPI or Flask)
- [ ] Docker containerization for reproducible deployment
- [ ] Web interface for interactive predictions (Streamlit or Gradio)
- [ ] Model monitoring and drift detection framework
- [ ] Documentation for clinical end-users

**Week 10: Validation and Testing**
- [ ] External validation on independent dataset (if available)
- [ ] Sensitivity analysis for feature perturbations
- [ ] Edge case testing and error analysis
- [ ] Performance benchmarking against baseline methods
- [ ] Code review and refactoring for production readiness

**Week 11-12: Final Documentation and Presentation**
- [ ] Comprehensive technical report (methodology, results, discussion)
- [ ] Scientific poster or slide deck for presentation
- [ ] Jupyter notebook tutorial for educational purposes
- [ ] Video demonstration of pipeline execution
- [ ] Publication-ready manuscript draft (optional)

---

### 🎯 Project Milestones

| Milestone | Target Date | Status |
|-----------|-------------|--------|
| Core pipeline implementation | November 20, 2025 | ✅ **Complete** |
| Model training and evaluation | November 20, 2025 | ✅ **Complete** |
| Visualization suite | November 20, 2025 | ✅ **Complete** |
| GitHub repository setup | November 20, 2025 | ✅ **Complete** |
| Cross-validation and tuning | December 5, 2025 | 🔄 In Progress |
| Advanced model development | December 20, 2025 | 📋 Planned |
| Explainability analysis | January 15, 2026 | 📋 Planned |
| Deployment prototype | February 1, 2026 | 📋 Planned |
| Final documentation | February 15, 2026 | 📋 Planned |
| Project presentation | February 28, 2026 | 📋 Planned |

---

### 🚀 Success Metrics

**Technical Achievements:**
- ✅ 97.12% accuracy with Random Forest (exceeds baseline)
- ✅ 0.9680 ROC-AUC demonstrating excellent discrimination
- ✅ Sub-second training time for LR and RF (production-ready)
- ✅ Fully reproducible pipeline with version-controlled code
- ✅ Comprehensive evaluation across 5 key metrics

**Project Deliverables:**
- ✅ 5 Python modules (1,205 lines of code)
- ✅ 3 trained and serialized ML models
- ✅ 11 publication-quality visualizations
- ✅ Complete GitHub repository with documentation
- ✅ Reproducible environment with pinned dependencies

**Learning Outcomes:**
- ✅ Hands-on experience with scikit-learn ML algorithms
- ✅ Data preprocessing and feature engineering skills
- ✅ Model evaluation and comparison methodologies
- ✅ Scientific visualization with matplotlib/seaborn
- ✅ Version control and collaborative development practices

---

## References and Resources

**Dataset:**
- Diabetes Prediction Dataset (Kaggle): https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

**Documentation:**
- scikit-learn: https://scikit-learn.org/stable/
- pandas: https://pandas.pydata.org/docs/
- matplotlib: https://matplotlib.org/stable/contents.html
- seaborn: https://seaborn.pydata.org/

**GitHub Repository:**
- Project URL: https://github.com/JairRea/bioinformatics_project

---

## Contact Information

**Developer:** Jair Rea  
**Email:** jair.rea@live.com  
**GitHub:** [@jairrea](https://github.com/jairrea)  
**Repository:** [bioinformatics_project](https://github.com/JairRea/bioinformatics_project)

---

*Last Updated: November 20, 2025*
