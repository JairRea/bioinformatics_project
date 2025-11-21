import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os


# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def plot_confusion_matrix(cm, model_name, save_path='results/'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(save_path, f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix to {filename}")
    plt.close()


def plot_roc_curve(y_test, y_pred_proba, model_name, save_path='results/'):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(save_path, f'{model_name.lower().replace(" ", "_")}_roc_curve.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved ROC curve to {filename}")
    plt.close()


def plot_multiple_roc_curves(models_data, save_path='results/'):
    plt.figure(figsize=(10, 8))
    
    colors = ['darkorange', 'green', 'red', 'purple', 'brown']
    
    for idx, (model_name, (y_test, y_pred_proba)) in enumerate(models_data.items()):
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2,
                label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Model Comparison')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(save_path, 'model_comparison_roc_curves.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved comparison ROC curves to {filename}")
    plt.close()


def plot_feature_importance(feature_importance_df, model_name, top_n=15, save_path='results/'):
    plt.figure(figsize=(10, 8))
    top_features = feature_importance_df.head(top_n)
    
    sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
    plt.title(f'Top {top_n} Feature Importance - {model_name}')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(save_path, f'{model_name.lower().replace(" ", "_")}_feature_importance.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved feature importance plot to {filename}")
    plt.close()


def plot_model_coefficients(coefficients_df, model_name, top_n=15, save_path='results/'):
    plt.figure(figsize=(10, 8))
    top_features = coefficients_df.head(top_n)
    
    colors = ['green' if x > 0 else 'red' for x in top_features['Coefficient']]
    sns.barplot(x='Coefficient', y='Feature', data=top_features, palette=colors)
    plt.title(f'Top {top_n} Feature Coefficients - {model_name}')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Features')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(save_path, f'{model_name.lower().replace(" ", "_")}_coefficients.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved coefficients plot to {filename}")
    plt.close()


def plot_metrics_comparison(metrics_dict, save_path='results/'):
    # Extract metrics for comparison
    models = list(metrics_dict.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    # Create DataFrame for plotting
    data = []
    for model_name, metrics in metrics_dict.items():
        for metric_name in metrics_names:
            data.append({
                'Model': model_name,
                'Metric': metric_name.replace('_', ' ').title(),
                'Score': metrics[metric_name]
            })
    
    df = pd.DataFrame(data)
    
    # Create grouped bar plot
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Metric', y='Score', hue='Model', data=df, palette='Set2')
    plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Metric', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.ylim([0, 1.0])
    plt.legend(title='Model', loc='lower right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(save_path, 'model_metrics_comparison.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved metrics comparison to {filename}")
    plt.close()


def create_summary_table(metrics_dict, save_path='results/'):
    # Create summary DataFrame
    summary_data = []
    for model_name, metrics in metrics_dict.items():
        summary_data.append({
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1_score']:.4f}",
            'ROC-AUC': f"{metrics['roc_auc']:.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    os.makedirs(save_path, exist_ok=True)
    csv_filename = os.path.join(save_path, 'model_comparison_summary.csv')
    summary_df.to_csv(csv_filename, index=False)
    print(f"\nSaved summary table to {csv_filename}")
    
    # Print to console
    print("\n" + "="*70)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*70)
    print(summary_df.to_string(index=False))
    print("="*70)
    
    return summary_df


def plot_all_confusion_matrices(models_metrics, save_path='results/'):
    n_models = len(models_metrics)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, metrics) in enumerate(models_metrics.items()):
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=['No Diabetes', 'Diabetes'],
                    yticklabels=['No Diabetes', 'Diabetes'],
                    ax=axes[idx])
        axes[idx].set_title(f'{model_name}')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(save_path, 'all_confusion_matrices.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved all confusion matrices to {filename}")
    plt.close()


if __name__ == "__main__":
    print("Visualization module loaded successfully!")
    print("This module provides functions for visualizing model performance.")
