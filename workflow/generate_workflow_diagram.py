"""
Generate workflow diagram for diabetes prediction ML pipeline
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_workflow_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Color scheme
    data_color = '#E3F2FD'  # Light blue
    process_color = '#FFF9C4'  # Light yellow
    model_color = '#F3E5F5'  # Light purple
    result_color = '#E8F5E9'  # Light green
    edge_color = '#424242'
    
    # Title
    ax.text(5, 11.5, 'Diabetes Prediction ML Pipeline', 
            ha='center', va='center', fontsize=18, fontweight='bold')
    
    # Step 1: Data Input (Two datasets)
    box1a = FancyBboxPatch((0.5, 9.5), 1.8, 1, boxstyle="round,pad=0.1", 
                           edgecolor=edge_color, facecolor=data_color, linewidth=2)
    ax.add_patch(box1a)
    ax.text(1.4, 10.2, 'General\nDiaabetes', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(1.4, 9.8, '100K samples', ha='center', va='center', fontsize=8)
    
    box1b = FancyBboxPatch((2.5, 9.5), 1.8, 1, boxstyle="round,pad=0.1",
                           edgecolor=edge_color, facecolor=data_color, linewidth=2)
    ax.add_patch(box1b)
    ax.text(3.4, 10.2, 'Pima Indians\nDiabetes', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(3.4, 9.8, '768 samples', ha='center', va='center', fontsize=8)
    
    # Arrow down
    arrow1 = FancyArrowPatch((2.4, 9.5), (2.4, 8.8),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color=edge_color)
    ax.add_patch(arrow1)
    
    # Step 2: Preprocessing
    box2 = FancyBboxPatch((0.8, 7.5), 3.2, 1.2, boxstyle="round,pad=0.1",
                          edgecolor=edge_color, facecolor=process_color, linewidth=2)
    ax.add_patch(box2)
    ax.text(2.4, 8.4, 'Data Preprocessing', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(2.4, 8.05, '• Handle missing values', ha='center', va='center', fontsize=8)
    ax.text(2.4, 7.8, '• Remove duplicates', ha='center', va='center', fontsize=8)
    
    # Arrow down
    arrow2 = FancyArrowPatch((2.4, 7.5), (2.4, 6.8),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color=edge_color)
    ax.add_patch(arrow2)
    
    # Step 3: Feature Engineering
    box3 = FancyBboxPatch((0.8, 5.5), 3.2, 1.2, boxstyle="round,pad=0.1",
                          edgecolor=edge_color, facecolor=process_color, linewidth=2)
    ax.add_patch(box3)
    ax.text(2.4, 6.4, 'Feature Engineering', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(2.4, 6.05, '• Encode categorical variables', ha='center', va='center', fontsize=8)
    ax.text(2.4, 5.8, '• StandardScaler normalization', ha='center', va='center', fontsize=8)
    
    # Arrow down
    arrow3 = FancyArrowPatch((2.4, 5.5), (2.4, 4.8),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color=edge_color)
    ax.add_patch(arrow3)
    
    # Step 4: Train/Test Split
    box4 = FancyBboxPatch((0.8, 3.8), 3.2, 0.8, boxstyle="round,pad=0.1",
                          edgecolor=edge_color, facecolor=process_color, linewidth=2)
    ax.add_patch(box4)
    ax.text(2.4, 4.2, 'Train/Test Split (80/20)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows to models area
    arrow4 = FancyArrowPatch((2.4, 3.8), (2.4, 2.8),
                             arrowstyle='->', mutation_scale=20, linewidth=2, color=edge_color)
    ax.add_patch(arrow4)
    
    # Step 5: Models - Single unified box containing all three models
    box5 = FancyBboxPatch((0.5, 1.5), 3.8, 1.2, boxstyle="round,pad=0.1",
                          edgecolor=edge_color, facecolor=model_color, linewidth=2)
    ax.add_patch(box5)
    
    # Model names and details
    ax.text(1.4, 2.3, 'Logistic\nRegression', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(1.4, 1.8, 'LBFGS solver', ha='center', va='center', fontsize=7)
    
    ax.text(2.4, 2.3, 'Random\nForest', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(2.4, 1.8, '100 estimators', ha='center', va='center', fontsize=7)
    
    ax.text(3.4, 2.3, 'Support Vector\nMachine', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(3.4, 1.8, 'RBF kernel', ha='center', va='center', fontsize=7)
    
    # Arrows to evaluation
    arrow5 = FancyArrowPatch((2.4, 1.5), (2.4, 0.8),
                             arrowstyle='->', mutation_scale=20, linewidth=2, color=edge_color)
    ax.add_patch(arrow5)
    
    # Step 6: Evaluation
    box6 = FancyBboxPatch((5.2, 5.5), 4, 5, boxstyle="round,pad=0.15",
                          edgecolor=edge_color, facecolor=result_color, linewidth=2)
    ax.add_patch(box6)
    ax.text(7.2, 10.2, 'Model Evaluation & Results', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    
    # Evaluation metrics
    metrics = [
        ('Performance Metrics:', 9.6, True),
        ('• Accuracy', 9.2, False),
        ('• Precision', 8.9, False),
        ('• Recall', 8.6, False),
        ('• F1-Score', 8.3, False),
        ('• ROC-AUC', 8.0, False),
        ('', 7.7, False),
        ('Visualizations:', 7.4, True),
        ('• Confusion matrices', 7.0, False),
        ('• ROC curves', 6.7, False),
        ('• Feature importance', 6.4, False),
        ('• Model comparison', 6.1, False),
    ]
    
    for text, y, is_header in metrics:
        if is_header:
            ax.text(7.2, y, text, ha='center', va='center', fontsize=9, fontweight='bold')
        else:
            ax.text(7.2, y, text, ha='center', va='center', fontsize=8)
    
    # Arrow from models to evaluation
    arrow6 = FancyArrowPatch((4.3, 2.4), (5.2, 7.5),
                            arrowstyle='->', mutation_scale=20, linewidth=2, 
                            color=edge_color, connectionstyle="arc3,rad=0.3")
    ax.add_patch(arrow6)
    
    # Add legend/notes at bottom
    ax.text(5, 0.3, 'Pipeline applies to both General and Pima datasets independently', 
            ha='center', va='center', fontsize=9, style='italic', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('workflow/ml_pipeline_workflow.png', dpi=300, bbox_inches='tight')
    print("Workflow diagram saved to: workflow/ml_pipeline_workflow.png")
    plt.close()

if __name__ == "__main__":
    create_workflow_diagram()
