"""
Advanced training pipeline for medical imaging AI models.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import wandb
from tqdm import tqdm

from ..models.advanced_models import create_model, create_ensemble
from ..evaluation.metrics import MedicalMetrics
from ..evaluation.visualizations import TrainingVisualizer

logger = logging.getLogger(__name__)


class AdvancedTrainer:
    """Advanced trainer with state-of-the-art training methodologies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize metrics and visualizer
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
        
        # Setup logging
        self.setup_logging()
        
        # Initialize Weights & Biases if configured
        if config.get('use_wandb', False):
            wandb.init(
                project=config.get('project_name', 'medical-imaging-ai'),
                config=config,
                name=config.get('experiment_name', f'experiment_{int(time.time())}')
            )
    
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
    
    def create_model(self) -> nn.Module:
        """Create model based on configuration."""
        model_config = self.config['model']
        
        if model_config['type'] == 'ensemble':
            return create_ensemble(model_config['models'])
        else:
            return create_model(
                model_config['type'],
                **model_config.get('params', {})
            )
    
    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        opt_config = self.config['optimizer']
        
        if opt_config['type'] == 'adamw':
            return optim.AdamW(
                model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 0.01),
                betas=opt_config.get('betas', (0.9, 0.999))
            )
        elif opt_config['type'] == 'sgd':
            return optim.SGD(
                model.parameters(),
                lr=opt_config['lr'],
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=opt_config.get('weight_decay', 1e-4)
            )
        elif opt_config['type'] == 'adam':
            return optim.Adam(
                model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_config['type']}")
    
    def create_scheduler(self, optimizer: optim.Optimizer) -> Optional[Any]:
        """Create learning rate scheduler."""
        scheduler_config = self.config.get('scheduler')
        if not scheduler_config:
            return None
        
        if scheduler_config['type'] == 'cosine':
            return CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config.get('T_max', self.config['epochs']),
                eta_min=scheduler_config.get('eta_min', 0)
            )
        elif scheduler_config['type'] == 'plateau':
            return ReduceLROnPlateau(
                optimizer,
                mode=scheduler_config.get('mode', 'max'),
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 10),
                verbose=True
            )
        elif scheduler_config['type'] == 'onecycle':
            return OneCycleLR(
                optimizer,
                max_lr=scheduler_config.get('max_lr', self.config['optimizer']['lr']),
                epochs=self.config['epochs'],
                steps_per_epoch=scheduler_config.get('steps_per_epoch', 100)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_config['type']}")
    
    def create_loss_function(self) -> nn.Module:
        """Create loss function based on task type."""
        loss_config = self.config['loss']
        task_type = self.config.get('task_type', 'segmentation')
        
        if task_type == 'segmentation':
            if loss_config['type'] == 'dice_bce':
                return DiceBCELoss()
            elif loss_config['type'] == 'focal':
                return FocalLoss(alpha=loss_config.get('alpha', 1.0), 
                               gamma=loss_config.get('gamma', 2.0))
            elif loss_config['type'] == 'tversky':
                return TverskyLoss(alpha=loss_config.get('alpha', 0.3),
                                 beta=loss_config.get('beta', 0.7))
            else:
                return nn.BCEWithLogitsLoss()
        
        elif task_type == 'classification':
            if loss_config['type'] == 'focal':
                return FocalLoss(alpha=loss_config.get('alpha', 1.0),
                               gamma=loss_config.get('gamma', 2.0))
            elif loss_config['type'] == 'cross_entropy':
                return nn.CrossEntropyLoss()
            else:
                return nn.CrossEntropyLoss()
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def train_epoch(self, model: nn.Module, dataloader: DataLoader, 
                   optimizer: optim.Optimizer, criterion: nn.Module) -> Dict[str, float]:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {self.current_epoch+1}/{self.config["epochs"]}')
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            if isinstance(model, nn.ModuleDict) or hasattr(model, 'forward') and 'classification' in str(type(model)):
                # Multi-task model
                outputs = model(data)
                if isinstance(outputs, dict):
                    if 'segmentation' in outputs:
                        loss = criterion(outputs['segmentation'], target)
                    else:
                        loss = criterion(outputs['classification'], target)
                else:
                    loss = criterion(outputs, target)
            else:
                outputs = model(data)
                loss = criterion(outputs, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip_norm'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['grad_clip_norm'])
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # Collect predictions for metrics
            with torch.no_grad():
                if isinstance(outputs, dict):
                    if 'classification' in outputs:
                        pred = torch.argmax(outputs['classification'], dim=1)
                    else:
                        pred = (outputs['segmentation'] > 0.5).float()
                else:
                    if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                        pred = torch.argmax(outputs, dim=1)
                    else:
                        pred = (outputs > 0.5).float()
                
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        metrics = self.metrics.calculate_metrics(all_targets, all_predictions)
        
        return {
            'loss': avg_loss,
            **metrics
        }
    
    def validate_epoch(self, model: nn.Module, dataloader: DataLoader, 
                      criterion: nn.Module) -> Dict[str, float]:
        """Validate for one epoch."""
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in tqdm(dataloader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                outputs = model(data)
                
                if isinstance(outputs, dict):
                    if 'segmentation' in outputs:
                        loss = criterion(outputs['segmentation'], target)
                        pred = (outputs['segmentation'] > 0.5).float()
                        prob = outputs['segmentation']
                    else:
                        loss = criterion(outputs['classification'], target)
                        pred = torch.argmax(outputs['classification'], dim=1)
                        prob = torch.softmax(outputs['classification'], dim=1)
                else:
                    loss = criterion(outputs, target)
                    if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                        pred = torch.argmax(outputs, dim=1)
                        prob = torch.softmax(outputs, dim=1)
                    else:
                        pred = (outputs > 0.5).float()
                        prob = torch.sigmoid(outputs)
                
                total_loss += loss.item()
                
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(prob.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        metrics = self.metrics.calculate_metrics(all_targets, all_predictions, all_probabilities)
        
        return {
            'loss': avg_loss,
            **metrics
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """Main training loop."""
        logger.info("Starting training...")
        
        # Create model
        model = self.create_model().to(self.device)
        logger.info(f"Model created: {model.__class__.__name__}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create optimizer and scheduler
        optimizer = self.create_optimizer(model)
        scheduler = self.create_scheduler(optimizer)
        criterion = self.create_loss_function()
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(model, train_loader, optimizer, criterion)
            
            # Validate
            val_metrics = self.validate_epoch(model, val_loader, criterion)
            
            # Update scheduler
            if scheduler:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_metrics.get('dice_score', val_metrics.get('accuracy', 0)))
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
            
            # Log specific metrics based on task
            if 'dice_score' in val_metrics:
                logger.info(f"Val Dice Score: {val_metrics['dice_score']:.4f}")
            if 'accuracy' in val_metrics:
                logger.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            
            # Log to Weights & Biases
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    **{f'train_{k}': v for k, v in train_metrics.items() if k != 'loss'},
                    **{f'val_{k}': v for k, v in val_metrics.items() if k != 'loss'}
                })
            
            # Save best model
            current_metric = val_metrics.get('dice_score', val_metrics.get('accuracy', 0))
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.save_model(model, epoch, val_metrics, is_best=True)
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                self.save_model(model, epoch, val_metrics, is_best=False)
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Generate final visualizations
        self.generate_training_visualizations()
        
        return {
            'model': model,
            'training_history': self.training_history,
            'best_metric': self.best_metric,
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
        
        # Create visualizations directory
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


# Custom Loss Functions

class DiceLoss(nn.Module):
    """Dice loss for segmentation tasks."""
    
    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class DiceBCELoss(nn.Module):
    """Combined Dice and BCE loss."""
    
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super(DiceBCELoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
    
    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        bce = self.bce_loss(inputs, targets)
        return self.dice_weight * dice + self.bce_weight * bce


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class TverskyLoss(nn.Module):
    """Tversky loss for segmentation."""
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        true_pos = (inputs * targets).sum()
        false_neg = (targets * (1 - inputs)).sum()
        false_pos = ((1 - targets) * inputs).sum()
        
        tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth)
        
        return 1 - tversky
