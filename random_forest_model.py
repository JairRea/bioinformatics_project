import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import joblib
import os


class RandomForestModel:
    
    def __init__(self, random_state=42, n_estimators=100, max_depth=None):
        self.model = RandomForestClassifier(
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1
        )
        self.random_state = random_state
        self.is_trained = False
        self.best_params = None
        
    def train(self, X_train, y_train):
        print("\n=== Training Random Forest Model ===")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Training completed successfully!")
        
    def hyperparameter_tuning(self, X_train, y_train, cv=5):
        print("\n=== Performing Hyperparameter Tuning ===")
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
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
        print("\n=== Random Forest Model Evaluation ===")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        print("\nClassification Report:")
        print(metrics['classification_report'])
    
    def get_feature_importance(self, feature_names):
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.model.feature_importances_
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath='models/random_forest_model.pkl'):
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/random_forest_model.pkl'):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")


def train_and_evaluate_random_forest(X_train, X_test, y_train, y_test, feature_names, tune_hyperparameters=False):
    # Create model
    model = RandomForestModel()
    
    # Train with or without hyperparameter tuning
    if tune_hyperparameters:
        model.hyperparameter_tuning(X_train, y_train)
    else:
        model.train(X_train, y_train)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    model.print_evaluation(metrics)
    
    # Get feature importance
    feature_importance = model.get_feature_importance(feature_names)
    print("\n=== Top 10 Most Important Features ===")
    print(feature_importance.head(10))
    
    # Save model
    model.save_model()
    
    return model, metrics, feature_importance


if __name__ == "__main__":
    # Test the model with preprocessed data
    from data_preprocessing import get_preprocessed_data
    
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, feature_names = get_preprocessed_data()
    
    # Train and evaluate (set tune_hyperparameters=True for hyperparameter tuning)
    model, metrics, feature_importance = train_and_evaluate_random_forest(
        X_train, X_test, y_train, y_test, feature_names, tune_hyperparameters=False
    )
