import numpy as np
import pandas as pd
import time
from datetime import datetime

# Import preprocessing
from general_preprocessing import get_general_preprocessed_data

# Import models
from logistic_regression_model import LogisticRegressionModel
from random_forest_model import RandomForestModel
from svm_model import SVMModel

# Import visualization
from visualizations import (
    plot_confusion_matrix, plot_roc_curve, plot_multiple_roc_curves,
    plot_feature_importance, plot_model_coefficients, plot_metrics_comparison,
    create_summary_table, plot_all_confusion_matrices
)


def print_section_header(title):
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")


def train_all_models(X_train, X_test, y_train, y_test, feature_names, 
                     save_path_models='models/general_diabetes/',
                     tune_hyperparameters=False):
    results = {}
    
    # ==================== LOGISTIC REGRESSION ====================
    print_section_header("LOGISTIC REGRESSION")
    start_time = time.time()
    
    lr_model = LogisticRegressionModel()
    lr_model.train(X_train, y_train)
    lr_metrics = lr_model.evaluate(X_test, y_test)
    lr_model.print_evaluation(lr_metrics)
    lr_coefficients = lr_model.get_coefficients(feature_names)
    
    print("\nTop 10 Most Important Features (by coefficient magnitude):")
    print(lr_coefficients.head(10))
    
    lr_model.save_model(filepath=f'{save_path_models}logistic_regression_model.pkl')
    lr_time = time.time() - start_time
    
    results['Logistic Regression'] = {
        'model': lr_model,
        'metrics': lr_metrics,
        'coefficients': lr_coefficients,
        'training_time': lr_time,
        'predictions': lr_model.predict(X_test),
        'probabilities': lr_model.predict_proba(X_test)[:, 1]
    }
    
    print(f"\nTraining time: {lr_time:.2f} seconds")
    
    # ==================== RANDOM FOREST ====================
    print_section_header("RANDOM FOREST")
    start_time = time.time()
    
    rf_model = RandomForestModel()
    
    if tune_hyperparameters:
        rf_model.hyperparameter_tuning(X_train, y_train, cv=3)
    else:
        rf_model.train(X_train, y_train)
    
    rf_metrics = rf_model.evaluate(X_test, y_test)
    rf_model.print_evaluation(rf_metrics)
    rf_feature_importance = rf_model.get_feature_importance(feature_names)
    
    print("\nTop 10 Most Important Features:")
    print(rf_feature_importance.head(10))
    
    rf_model.save_model(filepath=f'{save_path_models}random_forest_model.pkl')
    rf_time = time.time() - start_time
    
    results['Random Forest'] = {
        'model': rf_model,
        'metrics': rf_metrics,
        'feature_importance': rf_feature_importance,
        'training_time': rf_time,
        'predictions': rf_model.predict(X_test),
        'probabilities': rf_model.predict_proba(X_test)[:, 1]
    }
    
    print(f"\nTraining time: {rf_time:.2f} seconds")
    
    # ==================== SUPPORT VECTOR MACHINE ====================
    print_section_header("SUPPORT VECTOR MACHINE (SVM)")
    start_time = time.time()
    
    svm_model = SVMModel()
    
    if tune_hyperparameters:
        svm_model.hyperparameter_tuning(X_train, y_train, cv=3)
    else:
        svm_model.train(X_train, y_train)
    
    svm_metrics = svm_model.evaluate(X_test, y_test)
    svm_model.print_evaluation(svm_metrics)
    
    sv_info = svm_model.get_support_vectors_info()
    print(f"\nNumber of support vectors: {sv_info['n_support_vectors']}")
    print(f"Kernel: {sv_info['kernel']}")
    print(f"C: {sv_info['C']}")
    print(f"Gamma: {sv_info['gamma']}")
    
    svm_model.save_model(filepath=f'{save_path_models}svm_model.pkl')
    svm_time = time.time() - start_time
    
    results['SVM'] = {
        'model': svm_model,
        'metrics': svm_metrics,
        'support_vector_info': sv_info,
        'training_time': svm_time,
        'predictions': svm_model.predict(X_test),
        'probabilities': svm_model.predict_proba(X_test)[:, 1]
    }
    
    print(f"\nTraining time: {svm_time:.2f} seconds")
    
    return results


def generate_visualizations(results, y_test, save_path='results/general_diabetes/'):
    print_section_header("GENERATING VISUALIZATIONS")
    
    # Individual ROC curves and confusion matrices
    for model_name, model_results in results.items():
        print(f"Creating visualizations for {model_name}...")
        
        # ROC curve
        plot_roc_curve(y_test, model_results['probabilities'], model_name, save_path)
        
        # Confusion matrix
        plot_confusion_matrix(model_results['metrics']['confusion_matrix'], 
                            model_name, save_path)
    
    # Feature importance/coefficients
    if 'coefficients' in results['Logistic Regression']:
        plot_model_coefficients(results['Logistic Regression']['coefficients'],
                              'Logistic Regression', save_path=save_path)
    
    if 'feature_importance' in results['Random Forest']:
        plot_feature_importance(results['Random Forest']['feature_importance'],
                              'Random Forest', save_path=save_path)
    
    # Comparison plots
    print("Creating comparison visualizations...")
    
    # Multiple ROC curves
    roc_data = {name: (y_test, res['probabilities']) 
                for name, res in results.items()}
    plot_multiple_roc_curves(roc_data, save_path)
    
    # Metrics comparison
    metrics_data = {name: res['metrics'] for name, res in results.items()}
    plot_metrics_comparison(metrics_data, save_path)
    
    # All confusion matrices
    plot_all_confusion_matrices(metrics_data, save_path)
    
    print("All visualizations generated successfully!")


def compare_models(results):
    print_section_header("MODEL COMPARISON SUMMARY")
    
    # Extract metrics
    metrics_data = {name: res['metrics'] for name, res in results.items()}
    
    # Create and display summary table
    summary_df = create_summary_table(metrics_data, save_path='results/general_diabetes/')
    
    # Training time comparison
    print("\n" + "="*70)
    print("TRAINING TIME COMPARISON")
    print("="*70)
    for model_name, model_results in results.items():
        print(f"{model_name:25s}: {model_results['training_time']:8.2f} seconds")
    print("="*70)
    
    # Identify best model for each metric
    print("\n" + "="*70)
    print("BEST MODEL BY METRIC")
    print("="*70)
    
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    for metric in metrics_to_compare:
        scores = {name: res['metrics'][metric] for name, res in results.items()}
        best_model = max(scores, key=scores.get)
        print(f"{metric.replace('_', ' ').title():20s}: {best_model:25s} ({scores[best_model]:.4f})")
    print("="*70)


def main():
    print("\n" + "="*80)
    print(" GENERAL DIABETES PREDICTION USING MACHINE LEARNING ".center(80, "="))
    print("="*80)
    print(f"\nAnalysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nModels: Logistic Regression, Random Forest, SVM")
    print("Dataset: General Diabetes Prediction Dataset (Kaggle)")
    
    # Configuration
    TUNE_HYPERPARAMETERS = False  # Set to True for hyperparameter tuning (slower)
    
    # Load and preprocess data
    print_section_header("DATA LOADING AND PREPROCESSING")
    try:
        X_train, X_test, y_train, y_test, feature_names = get_general_preprocessed_data(explore=True)
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nPlease download the General Diabetes Prediction Dataset from Kaggle:")
        print("https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset")
        print("\nPlace the CSV file in the 'data/' folder as 'general_diabetes.csv'")
        return
    
    # Train all models
    results = train_all_models(X_train, X_test, y_train, y_test, feature_names,
                              save_path_models='models/general_diabetes/',
                              tune_hyperparameters=TUNE_HYPERPARAMETERS)
    
    # Generate visualizations
    generate_visualizations(results, y_test, save_path='results/general_diabetes/')
    
    # Compare models
    compare_models(results)
    
    # Final message
    print_section_header("ANALYSIS COMPLETE")
    print("All models have been trained and evaluated successfully!")
    print("Results and visualizations have been saved to the 'results/general_diabetes/' folder.")
    print("Trained models have been saved to the 'models/general_diabetes/' folder.")
    print(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
