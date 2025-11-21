import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import joblib
import os


class SVMModel:
    
    def __init__(self, random_state=42, kernel='rbf', C=1.0, gamma='scale'):
        self.model = SVC(
            random_state=random_state,
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True
        )
        self.random_state = random_state
        self.is_trained = False
        self.best_params = None
        
    def train(self, X_train, y_train):
        print("\n=== Training SVM Model ===")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Training completed successfully!")
        
    def hyperparameter_tuning(self, X_train, y_train, cv=5):
        print("\n=== Performing Hyperparameter Tuning ===")
        
        param_grid = {
            'kernel': ['linear', 'rbf'],
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
        }
        
        grid_search = GridSearchCV(
            SVC(random_state=self.random_state, probability=True),
            param_grid,
            cv=cv,
            scoring='roc_auc',
            verbose=1,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.is_trained = True
        
        print(f"\nBest parameters: {self.best_params}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        return metrics
    
    def print_evaluation(self, metrics):
        print("\n=== SVM Model Evaluation ===")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        print("\nClassification Report:")
        print(metrics['classification_report'])
        
        if self.best_params:
            print(f"\nBest Hyperparameters: {self.best_params}")
    
    def get_support_vectors_info(self):
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        info = {
            'n_support_vectors': self.model.n_support_,
            'support_vector_indices': self.model.support_,
            'kernel': self.model.kernel,
            'C': self.model.C,
            'gamma': self.model.gamma
        }
        
        return info
    
    def save_model(self, filepath='models/svm_model.pkl'):
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/svm_model.pkl'):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")


def train_and_evaluate_svm(X_train, X_test, y_train, y_test, tune_hyperparameters=False):
    # Create model
    model = SVMModel()
    
    # Train with or without hyperparameter tuning
    if tune_hyperparameters:
        model.hyperparameter_tuning(X_train, y_train)
    else:
        model.train(X_train, y_train)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    model.print_evaluation(metrics)
    
    # Get support vector information
    sv_info = model.get_support_vectors_info()
    print("\n=== Support Vector Information ===")
    print(f"Number of support vectors: {sv_info['n_support_vectors']}")
    print(f"Kernel: {sv_info['kernel']}")
    print(f"C: {sv_info['C']}")
    print(f"Gamma: {sv_info['gamma']}")
    
    # Save model
    model.save_model()
    
    return model, metrics


if __name__ == "__main__":
    # Test the model with preprocessed data
    from data_preprocessing import get_preprocessed_data
    
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, feature_names = get_preprocessed_data()
    
    # Train and evaluate (set tune_hyperparameters=True for hyperparameter tuning)
    model, metrics = train_and_evaluate_svm(
        X_train, X_test, y_train, y_test, tune_hyperparameters=False
    )
