import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import joblib
import os


class LogisticRegressionModel:
    
    def __init__(self, random_state=42):
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            solver='lbfgs'
        )
        self.random_state = random_state
        self.is_trained = False
        
    def train(self, X_train, y_train):
        print("\n=== Training Logistic Regression Model ===")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Training completed successfully!")
        
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
        print("\n=== Logistic Regression Model Evaluation ===")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        print("\nClassification Report:")
        print(metrics['classification_report'])
    
    def get_coefficients(self, feature_names):
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        coefficients = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': self.model.coef_[0]
        })
        coefficients['Abs_Coefficient'] = np.abs(coefficients['Coefficient'])
        coefficients = coefficients.sort_values('Abs_Coefficient', ascending=False)
        
        return coefficients
    
    def save_model(self, filepath='models/logistic_regression_model.pkl'):
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/logistic_regression_model.pkl'):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")


def train_and_evaluate_logistic_regression(X_train, X_test, y_train, y_test, feature_names):
    # Create and train model
    model = LogisticRegressionModel()
    model.train(X_train, y_train)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    model.print_evaluation(metrics)
    
    # Get coefficients
    coefficients = model.get_coefficients(feature_names)
    print("\n=== Top 10 Most Important Features ===")
    print(coefficients.head(10))
    
    # Save model
    model.save_model()
    
    return model, metrics, coefficients


if __name__ == "__main__":
    # Test the model with preprocessed data
    from data_preprocessing import get_preprocessed_data
    
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, feature_names = get_preprocessed_data()
    
    # Train and evaluate
    model, metrics, coefficients = train_and_evaluate_logistic_regression(
        X_train, X_test, y_train, y_test, feature_names
    )
