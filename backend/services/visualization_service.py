"""
Visualization service for medical imaging AI model evaluation.
"""

import base64
import io
import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    classification_report
)

from ..schemas.api import DetectionResult, ProcessingResult

logger = logging.getLogger(__name__)


class VisualizationService:
    """Service for generating evaluation visualizations."""
    
    def __init__(self):
        # Set matplotlib style for medical imaging
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Configure matplotlib for better medical imaging plots
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
    
    def create_confusion_matrix(
        self, 
        y_true: List[int], 
        y_pred: List[int], 
        class_names: List[str],
        title: str = "Confusion Matrix"
    ) -> str:
        """Create confusion matrix visualization."""
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=class_names, 
                yticklabels=class_names,
                ax=ax
            )
            
            ax.set_title(title)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating confusion matrix: {e}")
            return None
    
    def create_roc_curve(
        self, 
        y_true: List[int], 
        y_scores: List[float],
        model_name: str = "Model"
    ) -> str:
        """Create ROC curve visualization."""
        try:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, color='darkorange', lw=2, 
                   label=f'{model_name} (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                   label='Random Classifier')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating ROC curve: {e}")
            return None
    
    def create_precision_recall_curve(
        self, 
        y_true: List[int], 
        y_scores: List[float],
        model_name: str = "Model"
    ) -> str:
        """Create precision-recall curve visualization."""
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            pr_auc = auc(recall, precision)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(recall, precision, color='blue', lw=2,
                   label=f'{model_name} (PR-AUC = {pr_auc:.3f})')
            
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating precision-recall curve: {e}")
            return None
    
    def create_dice_score_distribution(
        self, 
        dice_scores: List[float],
        title: str = "Dice Score Distribution"
    ) -> str:
        """Create Dice score distribution histogram."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create histogram
            n, bins, patches = ax.hist(dice_scores, bins=30, alpha=0.7, 
                                     color='skyblue', edgecolor='black')
            
            # Add statistics
            mean_score = np.mean(dice_scores)
            median_score = np.median(dice_scores)
            std_score = np.std(dice_scores)
            
            ax.axvline(mean_score, color='red', linestyle='--', 
                      label=f'Mean: {mean_score:.3f}')
            ax.axvline(median_score, color='green', linestyle='--', 
                      label=f'Median: {median_score:.3f}')
            
            ax.set_xlabel('Dice Score')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{title}\nMean: {mean_score:.3f} ± {std_score:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating Dice score distribution: {e}")
            return None
    
    def create_metrics_comparison(
        self, 
        models_data: Dict[str, Dict[str, float]],
        metrics: List[str]
    ) -> str:
        """Create metrics comparison bar chart."""
        try:
            model_names = list(models_data.keys())
            n_metrics = len(metrics)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.ravel()
            
            for i, metric in enumerate(metrics[:4]):  # Limit to 4 metrics
                values = [models_data[model].get(metric, 0) for model in model_names]
                
                bars = axes[i].bar(model_names, values, alpha=0.7)
                axes[i].set_title(f'{metric} Comparison')
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    axes[i].text(bar.get_x() + bar.get_width()/2, 
                               bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
            
            # Hide unused subplots
            for i in range(n_metrics, 4):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating metrics comparison: {e}")
            return None
    
    def create_detection_confidence_distribution(
        self, 
        confidences: List[float], 
        labels: List[int],
        title: str = "Detection Confidence Distribution"
    ) -> str:
        """Create detection confidence distribution plot."""
        try:
            # Separate true positives and false positives
            tp_conf = [c for c, l in zip(confidences, labels) if l == 1]
            fp_conf = [c for c, l in zip(confidences, labels) if l == 0]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Combined histogram
            ax1.hist(tp_conf, bins=30, alpha=0.7, label='True Positives', 
                    color='green', density=True)
            ax1.hist(fp_conf, bins=30, alpha=0.7, label='False Positives', 
                    color='red', density=True)
            ax1.set_xlabel('Confidence Score')
            ax1.set_ylabel('Density')
            ax1.set_title('Confidence Distribution by Label')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Overall histogram
            ax2.hist(confidences, bins=30, alpha=0.7, color='skyblue', 
                    edgecolor='black', density=True)
            ax2.set_xlabel('Confidence Score')
            ax2.set_ylabel('Density')
            ax2.set_title('Overall Confidence Distribution')
            ax2.grid(True, alpha=0.3)
            
            plt.suptitle(title)
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating confidence distribution: {e}")
            return None
    
    def create_correlation_plot(
        self, 
        predicted: List[float], 
        actual: List[float],
        title: str = "Predicted vs Actual"
    ) -> str:
        """Create correlation scatter plot."""
        try:
            from scipy.stats import pearsonr
            
            correlation, p_value = pearsonr(predicted, actual)
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(actual, predicted, alpha=0.6, s=50)
            
            # Add perfect correlation line
            min_val = min(min(predicted), min(actual))
            max_val = max(max(predicted), max(actual))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
                   label='Perfect Correlation', linewidth=2)
            
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'{title}\nCorrelation: {correlation:.3f} (p={p_value:.3f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating correlation plot: {e}")
            return None
    
    def create_segmentation_overlay(
        self, 
        image: np.ndarray, 
        ground_truth: np.ndarray, 
        prediction: np.ndarray,
        slice_idx: int = 0,
        title: str = "Segmentation Comparison"
    ) -> str:
        """Create segmentation overlay visualization."""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(image[slice_idx], cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Ground truth
            axes[1].imshow(image[slice_idx], cmap='gray')
            axes[1].imshow(ground_truth[slice_idx], alpha=0.5, cmap='Reds')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            # Prediction
            axes[2].imshow(image[slice_idx], cmap='gray')
            axes[2].imshow(prediction[slice_idx], alpha=0.5, cmap='Blues')
            axes[2].set_title('Prediction')
            axes[2].axis('off')
            
            plt.suptitle(title)
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating segmentation overlay: {e}")
            return None
    
    def create_processing_result_summary(
        self, 
        result: ProcessingResult
    ) -> Dict[str, str]:
        """Create comprehensive visualization summary for processing results."""
        try:
            visualizations = {}
            
            # Extract data for visualizations
            detections = result.detections
            if not detections:
                return {"message": "No detections to visualize"}
            
            # Confidence scores
            confidences = [det.class_confidence for det in detections]
            labels = [1] * len(confidences)  # All are positive detections
            
            # Create confidence distribution
            if confidences:
                visualizations["confidence_distribution"] = self.create_detection_confidence_distribution(
                    confidences, labels, "Detection Confidence Distribution"
                )
            
            # Volume measurements (if available)
            volumes = [det.metrics.volume_mm3 for det in detections if det.metrics and det.metrics.volume_mm3]
            if len(volumes) > 1:
                # Create volume distribution
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(volumes, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
                ax.set_xlabel('Volume (mm³)')
                ax.set_ylabel('Frequency')
                ax.set_title('Tumor Volume Distribution')
                ax.grid(True, alpha=0.3)
                visualizations["volume_distribution"] = self._fig_to_base64(fig)
            
            # Dice scores (if available in results)
            if hasattr(result, 'dice_scores') and result.dice_scores:
                visualizations["dice_distribution"] = self.create_dice_score_distribution(
                    result.dice_scores, "Dice Score Distribution"
                )
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Error creating processing result summary: {e}")
            return {"error": str(e)}
    
    def create_evaluation_dashboard(
        self, 
        evaluation_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """Create comprehensive evaluation dashboard."""
        try:
            dashboard = {}
            
            # ROC Curve
            if 'y_true' in evaluation_data and 'y_scores' in evaluation_data:
                dashboard["roc_curve"] = self.create_roc_curve(
                    evaluation_data['y_true'], 
                    evaluation_data['y_scores'],
                    evaluation_data.get('model_name', 'Model')
                )
            
            # Precision-Recall Curve
            if 'y_true' in evaluation_data and 'y_scores' in evaluation_data:
                dashboard["pr_curve"] = self.create_precision_recall_curve(
                    evaluation_data['y_true'], 
                    evaluation_data['y_scores'],
                    evaluation_data.get('model_name', 'Model')
                )
            
            # Confusion Matrix
            if 'y_true' in evaluation_data and 'y_pred' in evaluation_data:
                class_names = evaluation_data.get('class_names', ['Negative', 'Positive'])
                dashboard["confusion_matrix"] = self.create_confusion_matrix(
                    evaluation_data['y_true'], 
                    evaluation_data['y_pred'],
                    class_names
                )
            
            # Metrics Comparison
            if 'models_metrics' in evaluation_data:
                metrics = evaluation_data.get('metrics', ['accuracy', 'precision', 'recall', 'f1_score'])
                dashboard["metrics_comparison"] = self.create_metrics_comparison(
                    evaluation_data['models_metrics'], 
                    metrics
                )
            
            # Dice Score Distribution
            if 'dice_scores' in evaluation_data:
                dashboard["dice_distribution"] = self.create_dice_score_distribution(
                    evaluation_data['dice_scores']
                )
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error creating evaluation dashboard: {e}")
            return {"error": str(e)}
    
    def _fig_to_base64(self, fig: Figure) -> str:
        """Convert matplotlib figure to base64 string."""
        try:
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)  # Close figure to free memory
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            logger.error(f"Error converting figure to base64: {e}")
            return None
    
    def get_visualization_types(self) -> List[Dict[str, str]]:
        """Get list of available visualization types."""
        return [
            {
                "type": "confusion_matrix",
                "name": "Confusion Matrix",
                "description": "Shows classification performance by comparing predicted vs actual labels"
            },
            {
                "type": "roc_curve",
                "name": "ROC Curve",
                "description": "Shows the trade-off between sensitivity and specificity"
            },
            {
                "type": "precision_recall_curve",
                "name": "Precision-Recall Curve",
                "description": "Shows the relationship between precision and recall"
            },
            {
                "type": "dice_score_distribution",
                "name": "Dice Score Distribution",
                "description": "Shows the distribution of Dice scores across cases"
            },
            {
                "type": "metrics_comparison",
                "name": "Metrics Comparison",
                "description": "Compares multiple models across different metrics"
            },
            {
                "type": "confidence_distribution",
                "name": "Confidence Distribution",
                "description": "Shows the distribution of detection confidence scores"
            },
            {
                "type": "correlation_plot",
                "name": "Correlation Plot",
                "description": "Shows correlation between predicted and actual values"
            },
            {
                "type": "segmentation_overlay",
                "name": "Segmentation Overlay",
                "description": "Visual comparison of predicted vs ground truth segmentations"
            }
        ]
