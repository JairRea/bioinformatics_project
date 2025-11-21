# Diabetes Prediction Using Machine Learning

## Project Overview
This project uses biomarkers from the Diabetes Prediction Dataset (Kaggle) to predict type 2 diabetes onset using three machine learning models:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

## Project Structure
```
Bio_ML_Project/
├── data/                          # Dataset folder
├── models/                        # Trained model files
├── results/                       # Evaluation metrics and plots
├── data_preprocessing.py          # Data loading and preprocessing
├── logistic_regression_model.py   # Logistic regression implementation
├── random_forest_model.py         # Random forest implementation
├── svm_model.py                   # SVM implementation
├── visualizations.py              # Plotting and visualization functions
├── main.py                        # Main execution script
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
Download the Diabetes Prediction Dataset from Kaggle and place the CSV file in the `data/` folder.

### 3. Run the Analysis
```bash
python main.py
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
