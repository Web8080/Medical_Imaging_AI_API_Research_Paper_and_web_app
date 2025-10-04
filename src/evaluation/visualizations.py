"""
Training and evaluation visualizations for medical imaging AI.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

logger = logging.getLogger(__name__)


class TrainingVisualizer:
    """Visualizer for training progress and model evaluation."""
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        plt.style.use(style)
        sns.set_palette("husl")
        
        # Configure matplotlib for medical imaging
        plt.rcParams.update({
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 9
        })
    
    def plot_training_curves(self, training_history: Dict[str, List], 
                           save_path: Optional[Path] = None) -> None:
        """Plot training and validation curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(training_history['train_loss']) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, training_history['train_loss'], 'b-', label='Training Loss')
        axes[0, 0].plot(epochs, training_history['val_loss'], 'r-', label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        if training_history['train_metrics'] and 'accuracy' in training_history['train_metrics'][0]:
            train_acc = [m['accuracy'] for m in training_history['train_metrics']]
            val_acc = [m['accuracy'] for m in training_history['val_metrics']]
            
            axes[0, 1].plot(epochs, train_acc, 'b-', label='Training Accuracy')
            axes[0, 1].plot(epochs, val_acc, 'r-', label='Validation Accuracy')
            axes[0, 1].set_title('Training and Validation Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Dice Score curves (for segmentation)
        if training_history['train_metrics'] and 'dice_score' in training_history['train_metrics'][0]:
            train_dice = [m['dice_score'] for m in training_history['train_metrics']]
            val_dice = [m['dice_score'] for m in training_history['val_metrics']]
            
            axes[1, 0].plot(epochs, train_dice, 'b-', label='Training Dice Score')
            axes[1, 0].plot(epochs, val_dice, 'r-', label='Validation Dice Score')
            axes[1, 0].set_title('Training and Validation Dice Score')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Dice Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # F1 Score curves
        if training_history['train_metrics'] and 'f1_score' in training_history['train_metrics'][0]:
            train_f1 = [m['f1_score'] for m in training_history['train_metrics']]
            val_f1 = [m['f1_score'] for m in training_history['val_metrics']]
            
            axes[1, 1].plot(epochs, train_f1, 'b-', label='Training F1 Score')
            axes[1, 1].plot(epochs, val_f1, 'r-', label='Validation F1 Score')
            axes[1, 1].set_title('Training and Validation F1 Score')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('F1 Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Training curves saved to {save_path}")
        
        plt.show()
    
    def plot_metrics_comparison(self, training_history: Dict[str, List], 
                              save_path: Optional[Path] = None) -> None:
        """Plot comparison of different metrics."""
        if not training_history['val_metrics']:
            logger.warning("No validation metrics available for comparison")
            return
        
        # Extract metrics
        metrics_names = list(training_history['val_metrics'][0].keys())
        epochs = range(1, len(training_history['val_metrics']) + 1)
        
        # Create subplots
        n_metrics = len(metrics_names)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, metric_name in enumerate(metrics_names):
            row = i // n_cols
            col = i % n_cols
            
            if n_rows == 1:
                ax = axes[col] if n_cols > 1 else axes
            else:
                ax = axes[row, col]
            
            train_values = [m.get(metric_name, 0) for m in training_history['train_metrics']]
            val_values = [m.get(metric_name, 0) for m in training_history['val_metrics']]
            
            ax.plot(epochs, train_values, 'b-', label='Training', alpha=0.7)
            ax.plot(epochs, val_values, 'r-', label='Validation', alpha=0.7)
            ax.set_title(f'{metric_name.replace("_", " ").title()}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name.replace("_", " ").title())
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows == 1:
                axes[col].set_visible(False)
            else:
                axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Metrics comparison saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            class_names: Optional[List[str]] = None,
                            save_path: Optional[Path] = None) -> None:
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray, 
                      model_name: str = "Model", save_path: Optional[Path] = None) -> None:
        """Plot ROC curve."""
        from sklearn.metrics import auc
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'{model_name} (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_scores: np.ndarray,
                                  model_name: str = "Model", save_path: Optional[Path] = None) -> None:
        """Plot precision-recall curve."""
        from sklearn.metrics import auc
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'{model_name} (PR-AUC = {pr_auc:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Precision-recall curve saved to {save_path}")
        
        plt.show()
    
    def plot_dice_score_distribution(self, dice_scores: List[float], 
                                   title: str = "Dice Score Distribution",
                                   save_path: Optional[Path] = None) -> None:
        """Plot Dice score distribution."""
        plt.figure(figsize=(10, 6))
        
        # Create histogram
        n, bins, patches = plt.hist(dice_scores, bins=30, alpha=0.7, 
                                  color='skyblue', edgecolor='black')
        
        # Add statistics
        mean_score = np.mean(dice_scores)
        median_score = np.median(dice_scores)
        std_score = np.std(dice_scores)
        
        plt.axvline(mean_score, color='red', linestyle='--', 
                   label=f'Mean: {mean_score:.3f}')
        plt.axvline(median_score, color='green', linestyle='--', 
                   label=f'Median: {median_score:.3f}')
        
        plt.xlabel('Dice Score')
        plt.ylabel('Frequency')
        plt.title(f'{title}\nMean: {mean_score:.3f} ± {std_score:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Dice score distribution saved to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(self, model_results: Dict[str, Dict[str, float]],
                            metrics: List[str], save_path: Optional[Path] = None) -> None:
        """Plot comparison of multiple models."""
        model_names = list(model_results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics[:4]):  # Limit to 4 metrics
            values = [model_results[model].get(metric, 0) for model in model_names]
            
            bars = axes[i].bar(model_names, values, alpha=0.7)
            axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Hide unused subplots
        for i in range(len(metrics), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Model comparison saved to {save_path}")
        
        plt.show()
    
    def create_interactive_dashboard(self, training_history: Dict[str, List],
                                   model_results: Optional[Dict[str, Dict[str, float]]] = None,
                                   save_path: Optional[Path] = None) -> None:
        """Create interactive dashboard using Plotly."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Progress', 'Metrics Comparison', 
                          'Model Performance', 'Loss Curves'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        epochs = list(range(1, len(training_history['train_loss']) + 1))
        
        # Training progress
        fig.add_trace(
            go.Scatter(x=epochs, y=training_history['train_loss'], 
                      name='Train Loss', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=training_history['val_loss'], 
                      name='Val Loss', line=dict(color='red')),
            row=1, col=1
        )
        
        # Metrics comparison
        if training_history['val_metrics'] and 'accuracy' in training_history['val_metrics'][0]:
            train_acc = [m['accuracy'] for m in training_history['train_metrics']]
            val_acc = [m['accuracy'] for m in training_history['val_metrics']]
            
            fig.add_trace(
                go.Scatter(x=epochs, y=train_acc, name='Train Accuracy', 
                          line=dict(color='blue', dash='dash')),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=epochs, y=val_acc, name='Val Accuracy', 
                          line=dict(color='red', dash='dash')),
                row=1, col=2
            )
        
        # Model performance
        if model_results:
            model_names = list(model_results.keys())
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            
            for metric in metrics:
                values = [model_results[model].get(metric, 0) for model in model_names]
                fig.add_trace(
                    go.Bar(x=model_names, y=values, name=metric),
                    row=2, col=1
                )
        
        # Loss curves (detailed)
        fig.add_trace(
            go.Scatter(x=epochs, y=training_history['train_loss'], 
                      name='Train Loss Detail', line=dict(color='blue', width=3)),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=training_history['val_loss'], 
                      name='Val Loss Detail', line=dict(color='red', width=3)),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Medical Imaging AI Training Dashboard",
            showlegend=True,
            height=800
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Accuracy", row=1, col=2)
        fig.update_xaxes(title_text="Model", row=2, col=1)
        fig.update_yaxes(title_text="Score", row=2, col=1)
        fig.update_xaxes(title_text="Epoch", row=2, col=2)
        fig.update_yaxes(title_text="Loss", row=2, col=2)
        
        if save_path:
            fig.write_html(str(save_path))
            logger.info(f"Interactive dashboard saved to {save_path}")
        
        fig.show()
    
    def plot_attention_maps(self, image: np.ndarray, attention_maps: List[np.ndarray],
                          layer_names: List[str], save_path: Optional[Path] = None) -> None:
        """Plot attention maps for model interpretability."""
        n_layers = len(attention_maps)
        fig, axes = plt.subplots(2, n_layers + 1, figsize=(4 * (n_layers + 1), 8))
        
        # Original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Attention maps
        for i, (attention_map, layer_name) in enumerate(zip(attention_maps, layer_names)):
            # Raw attention map
            im = axes[0, i + 1].imshow(attention_map, cmap='hot')
            axes[0, i + 1].set_title(f'{layer_name} Attention')
            axes[0, i + 1].axis('off')
            plt.colorbar(im, ax=axes[0, i + 1])
            
            # Overlay
            axes[1, i + 1].imshow(image, cmap='gray')
            axes[1, i + 1].imshow(attention_map, alpha=0.5, cmap='hot')
            axes[1, i + 1].set_title(f'{layer_name} Overlay')
            axes[1, i + 1].axis('off')
        
        # Hide unused subplot
        axes[1, 0].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Attention maps saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, feature_names: List[str], importances: List[float],
                              title: str = "Feature Importance", 
                              save_path: Optional[Path] = None) -> None:
        """Plot feature importance."""
        # Sort by importance
        sorted_indices = np.argsort(importances)[::-1]
        sorted_names = [feature_names[i] for i in sorted_indices]
        sorted_importances = [importances[i] for i in sorted_indices]
        
        plt.figure(figsize=(10, 8))
        plt.bar(range(len(importances)), sorted_importances)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title(title)
        plt.xticks(range(len(importances)), sorted_names, rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Feature importance saved to {save_path}")
        
        plt.show()
