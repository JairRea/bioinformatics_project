# Bioinformatics Project: Diabetes Prediction Using Machine Learning

Final project for Bioinformatics. Machine Learning using a Diabetes dataset.

## Project Overview
This project uses biomarkers from the Diabetes Prediction Dataset (Kaggle) to predict type 2 diabetes onset using three machine learning models:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

## Project Structure
```
Bio_ML_Project/
├── data/                          # Dataset folder
│   ├── general_diabetes.csv       # General diabetes dataset (Kaggle)
│   └── pima_diabetes.csv          # Pima Indians diabetes dataset
├── models/                        # Trained model files
│   ├── general_diabetes/          # General dataset models
│   └── pima_diabetes/             # Pima dataset models
├── results/                       # Evaluation metrics and plots
│   ├── general_diabetes/          # General dataset results
│   └── pima_diabetes/             # Pima dataset results
├── docs/                          # Documentation files
│   ├── RUN_BOTH_DATASETS.md       # Quick start guide
│   ├── DATASET_COMPARISON.md      # Comparative analysis
│   ├── PROJECT_OUTLINE.md         # Detailed methodology
│   ├── NAMING_CONVENTION_UPDATE.md # File organization guide
│   └── # Code Citations.md        # APA citations
├── general_preprocessing.py       # General dataset preprocessing
├── pima_preprocessing.py          # Pima dataset preprocessing
├── logistic_regression_model.py   # Logistic regression implementation
├── random_forest_model.py         # Random forest implementation
├── svm_model.py                   # SVM implementation
├── visualizations.py              # Plotting and visualization functions
├── main_general.py                # General dataset analysis script
├── main_pima.py                   # Pima dataset analysis script
├── requirements.txt               # Project dependencies
├── README.md                      # Project documentation
└── ai_usage.md                    # AI usage documentation
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
Download the General Diabetes Prediction Dataset from Kaggle and place the CSV file in the `data/` folder as `general_diabetes.csv`.

### 3. Run the Analysis
```bash
python main_general.py  # For General Diabetes dataset
python main_pima.py     # For Pima Indians dataset
```

## Models

### Logistic Regression
- Simple, interpretable baseline model
- Provides probability estimates for diabetes risk
- Fast training and prediction

### Random Forest
- Ensemble model with feature importance analysis
- Handles non-linear relationships
- Robust to overfitting

### Support Vector Machine (SVM)
- Powerful for high-dimensional data
- Multiple kernel options (linear, RBF)
- Good generalization performance

## Evaluation Metrics
- Accuracy
- Precision, Recall, F1-Score
- ROC-AUC Score
- Confusion Matrix
- Feature Importance (for Random Forest)

## Results
All results, including model performance metrics and visualizations, will be saved in the `results/` folder.

## Documentation
Additional documentation is available in the `docs/` folder:
- **[Quick Start Guide](docs/RUN_BOTH_DATASETS.md)** - Instructions for running both datasets
- **[Dataset Comparison](docs/DATASET_COMPARISON.md)** - Comprehensive analysis comparing General vs Pima datasets
- **[Project Outline](docs/PROJECT_OUTLINE.md)** - Detailed methodology, timeline, and approach
- **[Naming Convention](docs/NAMING_CONVENTION_UPDATE.md)** - File naming standards and organization
- **[Citations](docs/# Code Citations.md)** - APA citations for datasets, tools, and algorithms
