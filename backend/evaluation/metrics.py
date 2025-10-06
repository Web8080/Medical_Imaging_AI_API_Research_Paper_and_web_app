"""
Comprehensive evaluation metrics for medical imaging AI.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve
)

logger = logging.getLogger(__name__)


class MedicalMetrics:
    """Comprehensive metrics for medical imaging evaluation."""
    
    def __init__(self):
        self.metrics_history = []
    
    def calculate_metrics(self, y_true: Union[np.ndarray, List], 
                         y_pred: Union[np.ndarray, List],
                         y_prob: Optional[Union[np.ndarray, List]] = None) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for medical imaging tasks.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            
        Returns:
            Dictionary of calculated metrics
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if y_prob is not None:
            y_prob = np.array(y_prob)
        
        metrics = {}
        
        # Classification metrics
        metrics.update(self._calculate_classification_metrics(y_true, y_pred, y_prob))
        
        # Segmentation metrics (if applicable)
        if self._is_segmentation_task(y_true, y_pred):
            metrics.update(self._calculate_segmentation_metrics(y_true, y_pred))
        
        # Detection metrics (if applicable)
        if self._is_detection_task(y_true, y_pred):
            metrics.update(self._calculate_detection_metrics(y_true, y_pred, y_prob))
        
        return metrics
    
    def _calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                        y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate classification metrics."""
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Binary classification specific metrics
        if len(np.unique(y_true)) == 2:
            metrics['precision_binary'] = precision_score(y_true, y_pred, zero_division=0)
            metrics['recall_binary'] = recall_score(y_true, y_pred, zero_division=0)
            metrics['f1_binary'] = f1_score(y_true, y_pred, zero_division=0)
            
            # Specificity (True Negative Rate)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # ROC AUC
            if y_prob is not None:
                if y_prob.ndim > 1 and y_prob.shape[1] > 1:
                    # Multi-class probabilities
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
                else:
                    # Binary probabilities
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
                    metrics['average_precision'] = average_precision_score(y_true, y_prob)
        
        return metrics
    
    def _calculate_segmentation_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate segmentation-specific metrics."""
        metrics = {}
        
        # Ensure binary masks
        y_true_binary = (y_true > 0.5).astype(np.uint8)
        y_pred_binary = (y_pred > 0.5).astype(np.uint8)
        
        # Dice Score
        metrics['dice_score'] = self._calculate_dice_score(y_true_binary, y_pred_binary)
        
        # Jaccard Index (IoU)
        metrics['jaccard_index'] = self._calculate_jaccard_index(y_true_binary, y_pred_binary)
        
        # Hausdorff Distance
        metrics['hausdorff_distance'] = self._calculate_hausdorff_distance(y_true_binary, y_pred_binary)
        
        # Volume metrics
        volume_metrics = self._calculate_volume_metrics(y_true_binary, y_pred_binary)
        metrics.update(volume_metrics)
        
        # Surface metrics
        surface_metrics = self._calculate_surface_metrics(y_true_binary, y_pred_binary)
        metrics.update(surface_metrics)
        
        return metrics
    
    def _calculate_detection_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate detection-specific metrics."""
        metrics = {}
        
        # Convert to binary for detection
        y_true_binary = (y_true > 0).astype(np.uint8)
        y_pred_binary = (y_pred > 0).astype(np.uint8)
        
        # Detection accuracy
        metrics['detection_accuracy'] = accuracy_score(y_true_binary, y_pred_binary)
        
        # Precision and recall for detection
        metrics['detection_precision'] = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        metrics['detection_recall'] = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        metrics['detection_f1'] = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        # Average Precision (AP)
        if y_prob is not None:
            metrics['average_precision'] = average_precision_score(y_true_binary, y_prob)
        
        return metrics
    
    def _calculate_dice_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Dice coefficient."""
        intersection = np.logical_and(y_true, y_pred).sum()
        union = y_true.sum() + y_pred.sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return 2.0 * intersection / union
    
    def _calculate_jaccard_index(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Jaccard index (IoU)."""
        intersection = np.logical_and(y_true, y_pred).sum()
        union = np.logical_or(y_true, y_pred).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return intersection / union
    
    def _calculate_hausdorff_distance(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Hausdorff distance."""
        try:
            # Get coordinates of non-zero pixels
            true_coords = np.column_stack(np.where(y_true > 0))
            pred_coords = np.column_stack(np.where(y_pred > 0))
            
            if len(true_coords) == 0 or len(pred_coords) == 0:
                return float('inf')
            
            # Calculate directed Hausdorff distances
            h1 = directed_hausdorff(true_coords, pred_coords)[0]
            h2 = directed_hausdorff(pred_coords, true_coords)[0]
            
            return max(h1, h2)
        
        except Exception as e:
            logger.warning(f"Error calculating Hausdorff distance: {e}")
            return float('inf')
    
    def _calculate_volume_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate volume-related metrics."""
        metrics = {}
        
        true_volume = y_true.sum()
        pred_volume = y_pred.sum()
        
        if true_volume > 0:
            metrics['volume_difference'] = abs(true_volume - pred_volume)
            metrics['volume_relative_error'] = abs(true_volume - pred_volume) / true_volume
            metrics['volume_overlap'] = np.logical_and(y_true, y_pred).sum() / true_volume
        else:
            metrics['volume_difference'] = pred_volume
            metrics['volume_relative_error'] = float('inf') if pred_volume > 0 else 0.0
            metrics['volume_overlap'] = 0.0
        
        return metrics
    
    def _calculate_surface_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate surface-related metrics."""
        metrics = {}
        
        try:
            # Calculate surface areas using morphological operations
            true_surface = self._calculate_surface_area(y_true)
            pred_surface = self._calculate_surface_area(y_pred)
            
            if true_surface > 0:
                metrics['surface_difference'] = abs(true_surface - pred_surface)
                metrics['surface_relative_error'] = abs(true_surface - pred_surface) / true_surface
            else:
                metrics['surface_difference'] = pred_surface
                metrics['surface_relative_error'] = float('inf') if pred_surface > 0 else 0.0
            
            # Surface overlap
            intersection_surface = self._calculate_surface_area(np.logical_and(y_true, y_pred))
            union_surface = self._calculate_surface_area(np.logical_or(y_true, y_pred))
            
            if union_surface > 0:
                metrics['surface_overlap'] = intersection_surface / union_surface
            else:
                metrics['surface_overlap'] = 0.0
        
        except Exception as e:
            logger.warning(f"Error calculating surface metrics: {e}")
            metrics['surface_difference'] = float('inf')
            metrics['surface_relative_error'] = float('inf')
            metrics['surface_overlap'] = 0.0
        
        return metrics
    
    def _calculate_surface_area(self, mask: np.ndarray) -> float:
        """Calculate surface area of a binary mask."""
        if mask.ndim == 2:
            # 2D case
            eroded = ndimage.binary_erosion(mask)
            surface = np.logical_xor(mask, eroded).sum()
        elif mask.ndim == 3:
            # 3D case
            surface = 0
            for i in range(mask.shape[0]):
                slice_mask = mask[i]
                eroded = ndimage.binary_erosion(slice_mask)
                surface += np.logical_xor(slice_mask, eroded).sum()
        else:
            surface = 0
        
        return float(surface)
    
    def _is_segmentation_task(self, y_true: np.ndarray, y_pred: np.ndarray) -> bool:
        """Check if this is a segmentation task."""
        # Segmentation tasks typically have continuous values or binary masks
        return (y_true.dtype in [np.float32, np.float64] or 
                np.all(np.isin(y_true, [0, 1])) and np.all(np.isin(y_pred, [0, 1])))
    
    def _is_detection_task(self, y_true: np.ndarray, y_pred: np.ndarray) -> bool:
        """Check if this is a detection task."""
        # Detection tasks typically have binary labels
        return (len(np.unique(y_true)) == 2 and len(np.unique(y_pred)) == 2 and
                np.all(np.isin(y_true, [0, 1])) and np.all(np.isin(y_pred, [0, 1])))
    
    def calculate_ensemble_metrics(self, predictions_list: List[np.ndarray], 
                                 y_true: np.ndarray) -> Dict[str, float]:
        """Calculate metrics for ensemble predictions."""
        # Average predictions
        ensemble_pred = np.mean(predictions_list, axis=0)
        
        # Calculate metrics for ensemble
        ensemble_metrics = self.calculate_metrics(y_true, ensemble_pred)
        
        # Calculate individual model metrics
        individual_metrics = []
        for pred in predictions_list:
            individual_metrics.append(self.calculate_metrics(y_true, pred))
        
        # Calculate ensemble improvement
        best_individual = max(individual_metrics, key=lambda x: x.get('dice_score', x.get('accuracy', 0)))
        
        ensemble_metrics['ensemble_improvement'] = (
            ensemble_metrics.get('dice_score', ensemble_metrics.get('accuracy', 0)) - 
            best_individual.get('dice_score', best_individual.get('accuracy', 0))
        )
        
        return ensemble_metrics
    
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                     class_names: Optional[List[str]] = None) -> str:
        """Generate detailed classification report."""
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(np.unique(y_true)))]
        
        return classification_report(y_true, y_pred, target_names=class_names)
    
    def calculate_confidence_intervals(self, metrics: Dict[str, float], 
                                     n_samples: int, confidence: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for metrics."""
        from scipy import stats
        
        alpha = 1 - confidence
        z_score = stats.norm.ppf(1 - alpha/2)
        
        confidence_intervals = {}
        
        for metric_name, metric_value in metrics.items():
            if metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'dice_score', 'jaccard_index']:
                # For proportion-based metrics
                se = np.sqrt(metric_value * (1 - metric_value) / n_samples)
                ci_lower = max(0, metric_value - z_score * se)
                ci_upper = min(1, metric_value + z_score * se)
                confidence_intervals[metric_name] = (ci_lower, ci_upper)
            else:
                # For other metrics, assume normal distribution
                se = metric_value / np.sqrt(n_samples)
                ci_lower = metric_value - z_score * se
                ci_upper = metric_value + z_score * se
                confidence_intervals[metric_name] = (ci_lower, ci_upper)
        
        return confidence_intervals
