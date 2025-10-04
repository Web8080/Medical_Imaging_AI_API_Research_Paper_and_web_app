# Medical Imaging AI Model Evaluation Visualizations

## Overview

This guide covers the essential visual charts and graphs used for analyzing and evaluating medical imaging AI model results. These visualizations help researchers, clinicians, and developers understand model performance, identify issues, and make informed decisions about model deployment.

## 1. Classification Performance Metrics

### 1.1 Confusion Matrix
**Purpose**: Shows the performance of classification models by comparing predicted vs actual labels.

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
```

**Medical Imaging Applications**:
- Tumor classification (benign vs malignant)
- Disease detection (normal vs abnormal)
- Multi-class organ classification

### 1.2 ROC Curve (Receiver Operating Characteristic)
**Purpose**: Shows the trade-off between sensitivity and specificity at different threshold values.

```python
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_scores, model_name="Model"):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'{model_name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
```

### 1.3 Precision-Recall Curve
**Purpose**: Shows the relationship between precision and recall, especially useful for imbalanced datasets.

```python
from sklearn.metrics import precision_recall_curve, auc

def plot_precision_recall_curve(y_true, y_scores, model_name="Model"):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'{model_name} (PR-AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()
```

## 2. Segmentation Performance Metrics

### 2.1 Dice Score Distribution
**Purpose**: Shows the distribution of Dice scores across different cases or regions.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_dice_distribution(dice_scores, title="Dice Score Distribution"):
    plt.figure(figsize=(10, 6))
    plt.hist(dice_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(dice_scores), color='red', linestyle='--', 
                label=f'Mean: {np.mean(dice_scores):.3f}')
    plt.axvline(np.median(dice_scores), color='green', linestyle='--', 
                label=f'Median: {np.median(dice_scores):.3f}')
    plt.xlabel('Dice Score')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

### 2.2 Hausdorff Distance Box Plot
**Purpose**: Shows the distribution of Hausdorff distances (surface distance errors).

```python
import seaborn as sns

def plot_hausdorff_boxplot(hausdorff_distances, model_names):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=hausdorff_distances, x='Model', y='Hausdorff_Distance')
    plt.title('Hausdorff Distance Comparison')
    plt.ylabel('Hausdorff Distance (mm)')
    plt.xticks(rotation=45)
    plt.show()
```

### 2.3 Segmentation Overlay Visualization
**Purpose**: Visual comparison of predicted vs ground truth segmentations.

```python
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_segmentation_overlay(image, ground_truth, prediction, slice_idx=0):
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
    
    plt.tight_layout()
    plt.show()
```

## 3. Detection Performance Metrics

### 3.1 Precision-Recall Curve for Detection
**Purpose**: Shows detection performance at different confidence thresholds.

```python
def plot_detection_pr_curve(precisions, recalls, thresholds):
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, 'b-', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Detection Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    
    # Add threshold annotations
    for i, (rec, prec, thresh) in enumerate(zip(recalls, precisions, thresholds)):
        if i % 5 == 0:  # Annotate every 5th point
            plt.annotate(f'{thresh:.2f}', (rec, prec), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
    plt.show()
```

### 3.2 FROC Curve (Free-Response ROC)
**Purpose**: Shows detection performance considering false positives per image.

```python
def plot_froc_curve(sensitivities, fps_per_image):
    plt.figure(figsize=(10, 6))
    plt.plot(fps_per_image, sensitivities, 'b-', linewidth=2, marker='o')
    plt.xlabel('False Positives per Image')
    plt.ylabel('Sensitivity')
    plt.title('FROC Curve')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, max(fps_per_image)])
    plt.ylim([0, 1])
    plt.show()
```

### 3.3 Detection Confidence Distribution
**Purpose**: Shows the distribution of detection confidence scores.

```python
def plot_confidence_distribution(confidences, labels, title="Detection Confidence Distribution"):
    plt.figure(figsize=(12, 5))
    
    # Separate true positives and false positives
    tp_conf = [c for c, l in zip(confidences, labels) if l == 1]
    fp_conf = [c for c, l in zip(confidences, labels) if l == 0]
    
    plt.subplot(1, 2, 1)
    plt.hist(tp_conf, bins=30, alpha=0.7, label='True Positives', color='green')
    plt.hist(fp_conf, bins=30, alpha=0.7, label='False Positives', color='red')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Confidence Distribution by Label')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(confidences, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Overall Confidence Distribution')
    
    plt.tight_layout()
    plt.show()
```

## 4. Regression and Measurement Metrics

### 4.1 Bland-Altman Plot
**Purpose**: Shows agreement between predicted and actual measurements.

```python
def plot_bland_altman(predicted, actual, title="Bland-Altman Plot"):
    mean_values = (predicted + actual) / 2
    differences = predicted - actual
    
    plt.figure(figsize=(10, 6))
    plt.scatter(mean_values, differences, alpha=0.6)
    
    # Calculate limits of agreement
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    upper_limit = mean_diff + 1.96 * std_diff
    lower_limit = mean_diff - 1.96 * std_diff
    
    plt.axhline(mean_diff, color='red', linestyle='-', label=f'Mean: {mean_diff:.2f}')
    plt.axhline(upper_limit, color='red', linestyle='--', label=f'Upper Limit: {upper_limit:.2f}')
    plt.axhline(lower_limit, color='red', linestyle='--', label=f'Lower Limit: {lower_limit:.2f}')
    
    plt.xlabel('Mean of Predicted and Actual')
    plt.ylabel('Difference (Predicted - Actual)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

### 4.2 Correlation Scatter Plot
**Purpose**: Shows correlation between predicted and actual values.

```python
from scipy.stats import pearsonr

def plot_correlation(predicted, actual, title="Predicted vs Actual"):
    correlation, p_value = pearsonr(predicted, actual)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(actual, predicted, alpha=0.6)
    
    # Add perfect correlation line
    min_val = min(min(predicted), min(actual))
    max_val = max(max(predicted), max(actual))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', 
             label='Perfect Correlation')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{title}\nCorrelation: {correlation:.3f} (p={p_value:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

## 5. Model Comparison and Analysis

### 5.1 Performance Metrics Comparison
**Purpose**: Compare multiple models across different metrics.

```python
def plot_metrics_comparison(models_data, metrics):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        model_names = list(models_data.keys())
        values = [models_data[model][metric] for model in model_names]
        
        bars = axes[i].bar(model_names, values, alpha=0.7)
        axes[i].set_title(f'{metric} Comparison')
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
```

### 5.2 Learning Curves
**Purpose**: Shows model performance vs training data size.

```python
def plot_learning_curves(train_sizes, train_scores, val_scores, title="Learning Curves"):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='blue')
    
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

### 5.3 Error Analysis Heatmap
**Purpose**: Shows where models make errors across different categories.

```python
def plot_error_heatmap(error_matrix, categories, title="Error Analysis"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(error_matrix, annot=True, fmt='.2f', cmap='Reds',
                xticklabels=categories, yticklabels=categories)
    plt.title(title)
    plt.xlabel('Predicted Category')
    plt.ylabel('Actual Category')
    plt.show()
```

## 6. Clinical and Statistical Analysis

### 6.1 Survival Analysis (Kaplan-Meier)
**Purpose**: Shows survival curves for different patient groups.

```python
from lifelines import KaplanMeierFitter

def plot_survival_curves(survival_data, groups, title="Survival Analysis"):
    plt.figure(figsize=(10, 6))
    
    for group in groups:
        kmf = KaplanMeierFitter()
        group_data = survival_data[survival_data['group'] == group]
        kmf.fit(group_data['time'], group_data['event'])
        kmf.plot(label=f'Group {group}')
    
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

### 6.2 Calibration Plot
**Purpose**: Shows how well predicted probabilities match actual frequencies.

```python
from sklearn.calibration import calibration_curve

def plot_calibration(y_true, y_prob, n_bins=10):
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins)
    
    plt.figure(figsize=(8, 8))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="Model")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Plot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

## 7. Advanced Visualizations

### 7.1 Attention Maps
**Purpose**: Shows which parts of the image the model focuses on.

```python
def plot_attention_maps(image, attention_map, title="Attention Map"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Attention map
    im = axes[1].imshow(attention_map, cmap='hot')
    axes[1].set_title('Attention Map')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    # Overlay
    axes[2].imshow(image, cmap='gray')
    axes[2].imshow(attention_map, alpha=0.5, cmap='hot')
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
```

### 7.2 Feature Importance
**Purpose**: Shows which features contribute most to predictions.

```python
def plot_feature_importance(feature_names, importances, title="Feature Importance"):
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 8))
    plt.bar(range(len(importances)), importances[indices])
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(title)
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], 
               rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
```

### 7.3 Uncertainty Visualization
**Purpose**: Shows model uncertainty in predictions.

```python
def plot_uncertainty(image, prediction, uncertainty, title="Prediction Uncertainty"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Prediction
    axes[1].imshow(prediction, cmap='viridis')
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    
    # Uncertainty
    im = axes[2].imshow(uncertainty, cmap='Reds')
    axes[2].set_title('Uncertainty')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
```

## 8. Interactive Visualizations

### 8.1 3D Volume Rendering
**Purpose**: Interactive 3D visualization of medical volumes and segmentations.

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_3d_volume(volume, segmentation=None, title="3D Volume"):
    fig = go.Figure()
    
    # Add volume
    fig.add_trace(go.Volume(
        x=volume[0].flatten(),
        y=volume[1].flatten(),
        z=volume[2].flatten(),
        value=volume[3].flatten(),
        opacity=0.1,
        surface_count=20,
        name='Volume'
    ))
    
    # Add segmentation if provided
    if segmentation is not None:
        fig.add_trace(go.Volume(
            x=segmentation[0].flatten(),
            y=segmentation[1].flatten(),
            z=segmentation[2].flatten(),
            value=segmentation[3].flatten(),
            opacity=0.3,
            surface_count=20,
            name='Segmentation'
        ))
    
    fig.update_layout(title=title)
    fig.show()
```

### 8.2 Interactive Dashboard
**Purpose**: Comprehensive dashboard for model evaluation.

```python
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_evaluation_dashboard(metrics_data):
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('ROC Curve', 'Precision-Recall', 'Confusion Matrix', 'Metrics Comparison'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "heatmap"}, {"type": "bar"}]]
    )
    
    # Add ROC curve
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC'), row=1, col=1)
    
    # Add PR curve
    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='PR'), row=1, col=2)
    
    # Add confusion matrix
    fig.add_trace(go.Heatmap(z=confusion_matrix, name='CM'), row=2, col=1)
    
    # Add metrics comparison
    fig.add_trace(go.Bar(x=model_names, y=metric_values, name='Metrics'), row=2, col=2)
    
    fig.update_layout(height=800, title_text="Model Evaluation Dashboard")
    fig.show()
```

## 9. Best Practices for Medical Imaging Visualizations

### 9.1 Color Schemes
- **Use colorblind-friendly palettes** (viridis, plasma, inferno)
- **Maintain consistency** across all visualizations
- **Use appropriate colormaps** for medical data (grayscale for images, diverging for differences)

### 9.2 Statistical Significance
- **Include confidence intervals** where appropriate
- **Show p-values** for statistical tests
- **Use appropriate statistical tests** for medical data

### 9.3 Clinical Relevance
- **Include clinical thresholds** (e.g., diagnostic accuracy requirements)
- **Show practical significance** alongside statistical significance
- **Provide clinical interpretation** of results

### 9.4 Reproducibility
- **Include all parameters** used in visualizations
- **Provide code** for generating plots
- **Document data preprocessing** steps

## 10. Implementation in Our API

To integrate these visualizations into our Medical Imaging AI API, we can add:

```python
# Add to our API endpoints
@router.get("/results/{job_id}/visualizations")
async def get_result_visualizations(job_id: str):
    """Generate visualization plots for processing results."""
    # Implementation for generating various plots
    pass

@router.get("/results/{job_id}/dashboard")
async def get_evaluation_dashboard(job_id: str):
    """Generate interactive evaluation dashboard."""
    # Implementation for interactive dashboard
    pass
```

This comprehensive set of visualizations will help you thoroughly evaluate and analyze your medical imaging AI models, providing insights into performance, errors, and clinical relevance.
