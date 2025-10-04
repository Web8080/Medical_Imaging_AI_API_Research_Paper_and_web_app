"""
Phase 4: Methodology (Model Development)
Goal: Build and validate tumor detection/segmentation models with advanced architectures.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import wandb
from tqdm import tqdm

# Medical imaging libraries
try:
    import monai
    from monai.data import Dataset as MonaiDataset, DataLoader as MonaiDataLoader
    from monai.losses import DiceLoss, DiceCELoss, FocalLoss, TverskyLoss
    from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
    from monai.transforms import (
        Compose, LoadImaged, AddChanneld, Spacingd, Orientationd,
        ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
        RandRotate90d, RandFlipd, RandShiftIntensityd, RandGaussianNoised,
        ToTensord, EnsureChannelFirstd, Resized, NormalizeIntensityd
    )
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False

try:
    import torchio as tio
    TORCHIO_AVAILABLE = True
except ImportError:
    TORCHIO_AVAILABLE = False

from ..models.advanced_models import create_model, create_ensemble
from ..evaluation.metrics import MedicalMetrics
from ..evaluation.visualizations import TrainingVisualizer

logger = logging.getLogger(__name__)


class Phase4Trainer:
    """
    Phase 4: Advanced model training with state-of-the-art methodologies.
    
    Features:
    - Baseline models (U-Net, nnU-Net, Mask R-CNN)
    - Advanced architectures (Attention U-Net, Vision Transformer, EfficientNet)
    - Ensemble methods
    - Uncertainty quantification
    - Experiment tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.metrics = MedicalMetrics()
        self.visualizer = TrainingVisualizer()
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        # Setup logging and experiment tracking
        self.setup_logging()
        self.setup_experiment_tracking()
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.config.get('log_dir', 'logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
    
    def setup_experiment_tracking(self):
        """Setup experiment tracking with Weights & Biases."""
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('project_name', 'medical-imaging-ai'),
                config=self.config,
                name=self.config.get('experiment_name', f'experiment_{int(time.time())}')
            )
            logger.info("Weights & Biases tracking initialized")
    
    def create_baseline_model(self, model_type: str) -> nn.Module:
        """Create baseline models as specified in Phase 4."""
        
        if model_type == 'unet_2d':
            return self._create_2d_unet()
        elif model_type == 'unet_3d':
            return self._create_3d_unet()
        elif model_type == 'nnunet':
            return self._create_nnunet()
        elif model_type == 'mask_rcnn':
            return self._create_mask_rcnn()
        else:
            raise ValueError(f"Unknown baseline model type: {model_type}")
    
    def _create_2d_unet(self) -> nn.Module:
        """Create 2D U-Net baseline model."""
        if MONAI_AVAILABLE:
            from monai.networks.nets import UNet
            return UNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            )
        else:
            # Fallback to custom implementation
            from ..models.advanced_models import AttentionUNet
            return AttentionUNet(in_channels=1, out_channels=1, base_features=64)
    
    def _create_3d_unet(self) -> nn.Module:
        """Create 3D U-Net baseline model."""
        if MONAI_AVAILABLE:
            from monai.networks.nets import UNet
            return UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            )
        else:
            # Fallback to custom implementation
            from ..models.advanced_models import AttentionUNet
            return AttentionUNet(in_channels=1, out_channels=1, base_features=64)
    
    def _create_nnunet(self) -> nn.Module:
        """Create nnU-Net model."""
        if MONAI_AVAILABLE:
            from monai.networks.nets import DynUNet
            return DynUNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                kernel_size=[3, 3, 3, 3, 3],
                strides=[1, 2, 2, 2, 2],
                upsample_kernel_size=[2, 2, 2, 2],
                filters=[32, 64, 128, 256, 512],
                norm_name='instance',
                act_name='leakyrelu',
                deep_supervision=True,
                deep_supr_num=3,
            )
        else:
            logger.warning("MONAI not available. Using 3D U-Net as nnU-Net substitute.")
            return self._create_3d_unet()
    
    def _create_mask_rcnn(self) -> nn.Module:
        """Create Mask R-CNN for detection tasks."""
        try:
            import torchvision.models as models
            from torchvision.models.detection import maskrcnn_resnet50_fpn
            
            # Load pre-trained Mask R-CNN
            model = maskrcnn_resnet50_fpn(pretrained=True)
            
            # Modify for medical imaging
            model.backbone.body.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.roi_heads.mask_predictor.mask_fcn_logits = nn.Conv2d(256, 1, kernel_size=1)
            
            return model
        except ImportError:
            logger.warning("torchvision not available. Using custom detection model.")
            return self._create_custom_detection_model()
    
    def _create_custom_detection_model(self) -> nn.Module:
        """Create custom detection model."""
        from ..models.advanced_models import MedicalVisionTransformer
        return MedicalVisionTransformer(
            img_size=224, patch_size=16, in_channels=1, num_classes=2
        )
    
    def create_advanced_model(self, model_type: str) -> nn.Module:
        """Create advanced models as specified in Phase 4."""
        
        if model_type == 'attention_unet':
            from ..models.advanced_models import AttentionUNet
            return AttentionUNet(
                in_channels=self.config.get('in_channels', 1),
                out_channels=self.config.get('out_channels', 1),
                base_features=self.config.get('base_features', 64)
            )
        elif model_type == 'vit_unet':
            from ..models.advanced_models import VisionTransformerUNet
            return VisionTransformerUNet(
                img_size=self.config.get('img_size', 224),
                patch_size=self.config.get('patch_size', 16),
                in_channels=self.config.get('in_channels', 1),
                out_channels=self.config.get('out_channels', 1),
                embed_dim=self.config.get('embed_dim', 768),
                num_heads=self.config.get('num_heads', 12),
                num_layers=self.config.get('num_layers', 12)
            )
        elif model_type == 'efficientnet_unet':
            from ..models.advanced_models import EfficientNetUNet
            return EfficientNetUNet(
                model_name=self.config.get('efficientnet_model', 'efficientnet-b0'),
                in_channels=self.config.get('in_channels', 1),
                out_channels=self.config.get('out_channels', 1)
            )
        elif model_type == 'ensemble':
            return self._create_ensemble_model()
        else:
            raise ValueError(f"Unknown advanced model type: {model_type}")
    
    def _create_ensemble_model(self) -> nn.Module:
        """Create ensemble model."""
        models_config = self.config.get('ensemble_models', [
            {'type': 'attention_unet', 'params': {'in_channels': 1, 'out_channels': 1}, 'weight': 0.4},
            {'type': 'vit_unet', 'params': {'img_size': 224, 'in_channels': 1, 'out_channels': 1}, 'weight': 0.3},
            {'type': 'efficientnet_unet', 'params': {'in_channels': 1, 'out_channels': 1}, 'weight': 0.3}
        ])
        
        return create_ensemble(models_config)
    
    def create_loss_function(self, loss_type: str) -> nn.Module:
        """Create loss function as specified in Phase 4."""
        
        if MONAI_AVAILABLE:
            if loss_type == 'dice_bce':
                return DiceCELoss(
                    dice_weight=self.config.get('dice_weight', 0.5),
                    ce_weight=self.config.get('ce_weight', 0.5),
                    smooth_nr=1e-5,
                    smooth_dr=1e-5
                )
            elif loss_type == 'dice':
                return DiceLoss(
                    smooth_nr=1e-5,
                    smooth_dr=1e-5
                )
            elif loss_type == 'focal':
                return FocalLoss(
                    alpha=self.config.get('focal_alpha', 1.0),
                    gamma=self.config.get('focal_gamma', 2.0)
                )
            elif loss_type == 'tversky':
                return TverskyLoss(
                    alpha=self.config.get('tversky_alpha', 0.3),
                    beta=self.config.get('tversky_beta', 0.7),
                    smooth_nr=1e-5,
                    smooth_dr=1e-5
                )
            else:
                return DiceCELoss()
        else:
            # Fallback to custom loss functions
            from ..training.trainer import DiceBCELoss, FocalLoss, TverskyLoss
            
            if loss_type == 'dice_bce':
                return DiceBCELoss(
                    dice_weight=self.config.get('dice_weight', 0.5),
                    bce_weight=self.config.get('bce_weight', 0.5)
                )
            elif loss_type == 'focal':
                return FocalLoss(
                    alpha=self.config.get('focal_alpha', 1.0),
                    gamma=self.config.get('focal_gamma', 2.0)
                )
            elif loss_type == 'tversky':
                return TverskyLoss(
                    alpha=self.config.get('tversky_alpha', 0.3),
                    beta=self.config.get('tversky_beta', 0.7)
                )
            else:
                return DiceBCELoss()
    
    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create optimizer as specified in Phase 4 (AdamW)."""
        
        return optim.AdamW(
            model.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 1e-4),
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    def create_scheduler(self, optimizer: optim.Optimizer) -> Optional[Any]:
        """Create learning rate scheduler."""
        
        scheduler_type = self.config.get('scheduler_type', 'cosine')
        
        if scheduler_type == 'cosine':
            return CosineAnnealingLR(
                optimizer,
                T_max=self.config.get('epochs', 100),
                eta_min=self.config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'plateau':
            return ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=10,
                verbose=True
            )
        elif scheduler_type == 'onecycle':
            return OneCycleLR(
                optimizer,
                max_lr=self.config.get('learning_rate', 1e-4),
                epochs=self.config.get('epochs', 100),
                steps_per_epoch=self.config.get('steps_per_epoch', 100)
            )
        else:
            return None
    
    def create_metrics(self) -> Dict[str, Any]:
        """Create evaluation metrics as specified in Phase 4."""
        
        metrics = {}
        
        if MONAI_AVAILABLE:
            metrics['dice'] = DiceMetric(include_background=False, reduction='mean')
            metrics['hausdorff'] = HausdorffDistanceMetric(include_background=False, reduction='mean')
            metrics['surface_distance'] = SurfaceDistanceMetric(include_background=False, reduction='mean')
        
        return metrics
    
    def train_epoch(self, model: nn.Module, dataloader: DataLoader, 
                   optimizer: optim.Optimizer, criterion: nn.Module,
                   metrics: Dict[str, Any]) -> Dict[str, float]:
        """Train for one epoch."""
        
        model.train()
        total_loss = 0.0
        epoch_metrics = {}
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {self.current_epoch+1}/{self.config["epochs"]}')
        
        for batch_idx, batch_data in enumerate(progress_bar):
            # Handle different data formats
            if isinstance(batch_data, dict):
                inputs = batch_data['image'].to(self.device)
                targets = batch_data.get('label', batch_data.get('seg', None))
                if targets is not None:
                    targets = targets.to(self.device)
            else:
                inputs, targets = batch_data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            if isinstance(outputs, dict):
                # Multi-task model
                if 'segmentation' in outputs:
                    loss = criterion(outputs['segmentation'], targets)
                else:
                    loss = criterion(outputs['classification'], targets)
            else:
                loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip_norm'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['grad_clip_norm'])
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update metrics
            if MONAI_AVAILABLE and targets is not None:
                with torch.no_grad():
                    if isinstance(outputs, dict):
                        pred = outputs.get('segmentation', outputs.get('classification', outputs))
                    else:
                        pred = outputs
                    
                    # Apply sigmoid for segmentation
                    if pred.shape[1] == 1:  # Binary segmentation
                        pred = torch.sigmoid(pred)
                        pred = (pred > 0.5).float()
                    
                    for metric_name, metric_fn in metrics.items():
                        if hasattr(metric_fn, 'y_pred'):
                            metric_fn(y_pred=pred, y=targets)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(dataloader)
        epoch_metrics['loss'] = avg_loss
        
        if MONAI_AVAILABLE:
            for metric_name, metric_fn in metrics.items():
                if hasattr(metric_fn, 'aggregate'):
                    epoch_metrics[metric_name] = metric_fn.aggregate().item()
                    metric_fn.reset()
        
        return epoch_metrics
    
    def validate_epoch(self, model: nn.Module, dataloader: DataLoader, 
                      criterion: nn.Module, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Validate for one epoch."""
        
        model.eval()
        total_loss = 0.0
        epoch_metrics = {}
        
        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc='Validation'):
                # Handle different data formats
                if isinstance(batch_data, dict):
                    inputs = batch_data['image'].to(self.device)
                    targets = batch_data.get('label', batch_data.get('seg', None))
                    if targets is not None:
                        targets = targets.to(self.device)
                else:
                    inputs, targets = batch_data
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Calculate loss
                if isinstance(outputs, dict):
                    if 'segmentation' in outputs:
                        loss = criterion(outputs['segmentation'], targets)
                    else:
                        loss = criterion(outputs['classification'], targets)
                else:
                    loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                
                # Update metrics
                if MONAI_AVAILABLE and targets is not None:
                    if isinstance(outputs, dict):
                        pred = outputs.get('segmentation', outputs.get('classification', outputs))
                    else:
                        pred = outputs
                    
                    # Apply sigmoid for segmentation
                    if pred.shape[1] == 1:  # Binary segmentation
                        pred = torch.sigmoid(pred)
                        pred = (pred > 0.5).float()
                    
                    for metric_name, metric_fn in metrics.items():
                        if hasattr(metric_fn, 'y_pred'):
                            metric_fn(y_pred=pred, y=targets)
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(dataloader)
        epoch_metrics['loss'] = avg_loss
        
        if MONAI_AVAILABLE:
            for metric_name, metric_fn in metrics.items():
                if hasattr(metric_fn, 'aggregate'):
                    epoch_metrics[metric_name] = metric_fn.aggregate().item()
                    metric_fn.reset()
        
        return epoch_metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """Main training loop following Phase 4 methodology."""
        
        logger.info("Starting Phase 4 training...")
        
        # Create model
        model_type = self.config.get('model_type', 'attention_unet')
        if model_type in ['unet_2d', 'unet_3d', 'nnunet', 'mask_rcnn']:
            model = self.create_baseline_model(model_type)
        else:
            model = self.create_advanced_model(model_type)
        
        model = model.to(self.device)
        logger.info(f"Model created: {model.__class__.__name__}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create optimizer and scheduler
        optimizer = self.create_optimizer(model)
        scheduler = self.create_scheduler(optimizer)
        criterion = self.create_loss_function(self.config.get('loss_type', 'dice_bce'))
        metrics = self.create_metrics()
        
        # Training loop
        start_time = time.time()
        best_metric = 0.0
        patience_counter = 0
        early_stopping_patience = self.config.get('early_stopping_patience', 20)
        
        for epoch in range(self.config['epochs']):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(model, train_loader, optimizer, criterion, metrics)
            
            # Validate
            val_metrics = self.validate_epoch(model, val_loader, criterion, metrics)
            
            # Update scheduler
            if scheduler:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_metrics.get('dice', val_metrics.get('accuracy', 0)))
                else:
                    scheduler.step()
            
            # Log metrics
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_metrics'].append(train_metrics)
            self.training_history['val_metrics'].append(val_metrics)
            
            # Log to console
            logger.info(f"Epoch {epoch+1}/{self.config['epochs']}")
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            
            # Log specific metrics
            for metric_name, value in val_metrics.items():
                if metric_name != 'loss':
                    logger.info(f"Val {metric_name}: {value:.4f}")
            
            # Log to Weights & Biases
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    **{f'train_{k}': v for k, v in train_metrics.items() if k != 'loss'},
                    **{f'val_{k}': v for k, v in val_metrics.items() if k != 'loss'}
                })
            
            # Early stopping based on validation Dice coefficient
            current_metric = val_metrics.get('dice', val_metrics.get('accuracy', 0))
            if current_metric > best_metric:
                best_metric = current_metric
                patience_counter = 0
                self.save_model(model, epoch, val_metrics, is_best=True)
            else:
                patience_counter += 1
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                self.save_model(model, epoch, val_metrics, is_best=False)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Best metric: {best_metric:.4f}")
        
        # Generate final visualizations
        self.generate_training_visualizations()
        
        return {
            'model': model,
            'training_history': self.training_history,
            'best_metric': best_metric,
            'training_time': training_time
        }
    
    def save_model(self, model: nn.Module, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        save_dir = Path(self.config.get('save_dir', 'checkpoints'))
        save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'training_history': self.training_history
        }
        
        if is_best:
            torch.save(checkpoint, save_dir / 'best_model.pth')
            logger.info(f"Best model saved at epoch {epoch+1}")
        else:
            torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    def generate_training_visualizations(self):
        """Generate training visualizations."""
        logger.info("Generating training visualizations...")
        
        viz_dir = Path(self.config.get('viz_dir', 'visualizations'))
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate plots
        self.visualizer.plot_training_curves(
            self.training_history,
            save_path=viz_dir / 'training_curves.png'
        )
        
        self.visualizer.plot_metrics_comparison(
            self.training_history,
            save_path=viz_dir / 'metrics_comparison.png'
        )
        
        logger.info(f"Visualizations saved to {viz_dir}")
    
    def quantify_uncertainty(self, model: nn.Module, dataloader: DataLoader, 
                           n_samples: int = 10) -> Dict[str, np.ndarray]:
        """
        Quantify model uncertainty using Monte Carlo dropout.
        
        Args:
            model: Trained model
            dataloader: Data loader
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Dictionary with uncertainty metrics
        """
        model.train()  # Enable dropout
        
        uncertainties = {
            'predictions': [],
            'uncertainties': [],
            'entropies': []
        }
        
        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc='Uncertainty Quantification'):
                if isinstance(batch_data, dict):
                    inputs = batch_data['image'].to(self.device)
                else:
                    inputs, _ = batch_data
                    inputs = inputs.to(self.device)
                
                # Collect multiple predictions
                predictions = []
                for _ in range(n_samples):
                    outputs = model(inputs)
                    if isinstance(outputs, dict):
                        pred = outputs.get('segmentation', outputs.get('classification', outputs))
                    else:
                        pred = outputs
                    
                    if pred.shape[1] == 1:  # Binary segmentation
                        pred = torch.sigmoid(pred)
                    
                    predictions.append(pred.cpu().numpy())
                
                # Calculate uncertainty metrics
                predictions = np.array(predictions)  # Shape: (n_samples, batch_size, ...)
                
                # Mean prediction
                mean_pred = np.mean(predictions, axis=0)
                
                # Prediction uncertainty (variance)
                pred_uncertainty = np.var(predictions, axis=0)
                
                # Entropy (for classification)
                if len(predictions.shape) > 3:  # Multi-class
                    entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-8), axis=1)
                else:  # Binary
                    entropy = -(mean_pred * np.log(mean_pred + 1e-8) + 
                              (1 - mean_pred) * np.log(1 - mean_pred + 1e-8))
                
                uncertainties['predictions'].append(mean_pred)
                uncertainties['uncertainties'].append(pred_uncertainty)
                uncertainties['entropies'].append(entropy)
        
        # Concatenate all batches
        for key in uncertainties:
            uncertainties[key] = np.concatenate(uncertainties[key], axis=0)
        
        return uncertainties
