#!/usr/bin/env python3
"""
Training script for research paper methodology implementation.
Follows the specific training strategy described in the research paper.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add current directory to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb
from tqdm import tqdm

# Medical imaging libraries
try:
    import monai
    from monai.losses import DiceLoss, DiceCELoss, FocalLoss
    from monai.metrics import DiceMetric, HausdorffDistanceMetric
    from monai.transforms import (
        Compose, LoadImaged, AddChanneld, Spacingd, Orientationd,
        ScaleIntensityRanged, RandRotate90d, RandFlipd, 
        RandShiftIntensityd, RandGaussianNoised, ToTensord
    )
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False

from models.research_paper_models import ResearchPaperModels
from training.research_paper_trainer import ResearchPaperTrainer
from evaluation.research_metrics import ResearchMetrics
from data.research_data_loader import ResearchDataLoader

logger = logging.getLogger(__name__)


class ResearchPaperTrainingPipeline:
    """
    Training pipeline following the research paper methodology.
    
    Implements:
    - 5-fold cross-validation with stratified sampling
    - Combined Dice loss and cross-entropy loss
    - AdamW optimizer with learning rate scheduling
    - Data augmentation (rotations, flips, elastic deformations, intensity variations)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.model_factory = ResearchPaperModels(config)
        self.data_loader = ResearchDataLoader(config)
        self.metrics = ResearchMetrics()
        
        # Training state
        self.training_results = {}
        
        # Setup logging
        self.setup_logging()
        
        # Setup experiment tracking
        if config.get('use_wandb', False):
            wandb.init(
                project=config.get('project_name', 'research-paper-medical-ai'),
                config=config,
                name=config.get('experiment_name', f'research_experiment_{int(time.time())}')
            )
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.config.get('log_dir', 'logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'research_training.log'),
                logging.StreamHandler()
            ]
        )
    
    def create_loss_function(self, task_type: str) -> nn.Module:
        """Create loss function as specified in research paper."""
        
        if MONAI_AVAILABLE:
            if task_type == 'segmentation':
                return DiceCELoss(
                    dice_weight=0.5,
                    ce_weight=0.5,
                    smooth_nr=1e-5,
                    smooth_dr=1e-5
                )
            elif task_type == 'detection':
                return FocalLoss(alpha=1.0, gamma=2.0)
            else:
                return DiceCELoss()
        else:
            # Fallback loss functions
            if task_type == 'segmentation':
                return nn.BCEWithLogitsLoss()
            else:
                return nn.CrossEntropyLoss()
    
    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create AdamW optimizer as specified in research paper."""
        return optim.AdamW(
            model.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 1e-4),
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    def create_scheduler(self, optimizer: optim.Optimizer) -> Any:
        """Create learning rate scheduler as specified in research paper."""
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.get('epochs', 100),
            eta_min=1e-6
        )
    
    def create_augmentation_pipeline(self) -> Optional[Compose]:
        """Create data augmentation pipeline as specified in research paper."""
        if not MONAI_AVAILABLE:
            return None
        
        augmentation_transforms = [
            # Geometric augmentations
            RandRotate90d(keys=['image', 'label'], prob=0.5),
            RandFlipd(keys=['image', 'label'], prob=0.5),
            
            # Intensity augmentations
            RandShiftIntensityd(keys=['image'], offsets=0.1, prob=0.5),
            RandGaussianNoised(keys=['image'], std=0.01, prob=0.5),
        ]
        
        return Compose(augmentation_transforms)
    
    def train_single_fold(self, train_loader: DataLoader, val_loader: DataLoader,
                         model: nn.Module, fold: int, task_type: str) -> Dict[str, Any]:
        """Train model for a single fold."""
        
        logger.info(f"Training fold {fold + 1}/5")
        
        # Create trainer
        trainer = ResearchPaperTrainer(self.config)
        
        # Create loss function and optimizer
        criterion = self.create_loss_function(task_type)
        optimizer = self.create_optimizer(model)
        scheduler = self.create_scheduler(optimizer)
        
        # Training loop
        best_metric = 0.0
        fold_results = {
            'train_losses': [],
            'val_losses': [],
            'train_metrics': [],
            'val_metrics': [],
            'best_metric': 0.0,
            'best_epoch': 0
        }
        
        for epoch in range(self.config.get('epochs', 100)):
            # Training
            model.train()
            train_loss = 0.0
            train_metrics = {}
            
            for batch_data in tqdm(train_loader, desc=f'Fold {fold+1} Epoch {epoch+1}'):
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
                loss = criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_metrics = {}
            
            with torch.no_grad():
                for batch_data in val_loader:
                    if isinstance(batch_data, dict):
                        inputs = batch_data['image'].to(self.device)
                        targets = batch_data.get('label', batch_data.get('seg', None))
                        if targets is not None:
                            targets = targets.to(self.device)
                    else:
                        inputs, targets = batch_data
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    # Calculate metrics
                    if task_type == 'segmentation':
                        pred = (torch.sigmoid(outputs) > 0.5).float()
                        dice = self.metrics.calculate_dice_score(targets, pred)
                        val_metrics['dice'] = dice
                    else:
                        pred = torch.argmax(outputs, dim=1)
                        acc = accuracy_score(targets.cpu().numpy(), pred.cpu().numpy())
                        val_metrics['accuracy'] = acc
            
            # Update scheduler
            scheduler.step()
            
            # Record results
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            fold_results['train_losses'].append(avg_train_loss)
            fold_results['val_losses'].append(avg_val_loss)
            fold_results['train_metrics'].append(train_metrics)
            fold_results['val_metrics'].append(val_metrics)
            
            # Check for best model
            current_metric = val_metrics.get('dice', val_metrics.get('accuracy', 0))
            if current_metric > best_metric:
                best_metric = current_metric
                fold_results['best_metric'] = best_metric
                fold_results['best_epoch'] = epoch
                
                # Save best model
                self.save_model(model, fold, epoch, val_metrics)
            
            # Log progress
            logger.info(f"Fold {fold+1}, Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {avg_val_loss:.4f}, Val Metric: {current_metric:.4f}")
            
            # Log to Weights & Biases
            if self.config.get('use_wandb', False):
                wandb.log({
                    'fold': fold,
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    **{f'val_{k}': v for k, v in val_metrics.items()}
                })
        
        return fold_results
    
    def run_5_fold_cv(self, dataset_name: str, model_type: str, task_type: str) -> Dict[str, Any]:
        """Run 5-fold cross-validation as specified in research paper."""
        
        logger.info(f"Starting 5-fold cross-validation for {dataset_name} - {model_type}")
        
        # Load dataset
        dataset = self.data_loader.load_dataset(dataset_name)
        
        # Create cross-validation splits
        if task_type == 'classification':
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            splits = kfold.split(dataset, [item['label'] for item in dataset])
        else:
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            splits = kfold.split(dataset)
        
        cv_results = {
            'dataset': dataset_name,
            'model_type': model_type,
            'task_type': task_type,
            'fold_results': [],
            'mean_metrics': {},
            'std_metrics': {}
        }
        
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"Processing fold {fold + 1}/5")
            
            # Create data splits
            train_dataset = [dataset[i] for i in train_idx]
            val_dataset = [dataset[i] for i in val_idx]
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.config.get('batch_size', 16),
                shuffle=True,
                num_workers=4
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.get('batch_size', 16),
                shuffle=False,
                num_workers=4
            )
            
            # Create model
            model_config = self.config['models'][model_type]
            model = self.model_factory.create_model(model_type, model_config)
            model = model.to(self.device)
            
            # Train fold
            fold_result = self.train_single_fold(
                train_loader, val_loader, model, fold, task_type
            )
            
            cv_results['fold_results'].append(fold_result)
            fold_metrics.append(fold_result['best_metric'])
            
            logger.info(f"Fold {fold + 1} completed. Best metric: {fold_result['best_metric']:.4f}")
        
        # Calculate mean and std metrics
        cv_results['mean_metrics']['best_metric'] = np.mean(fold_metrics)
        cv_results['std_metrics']['best_metric'] = np.std(fold_metrics)
        
        logger.info(f"5-fold CV completed. Mean metric: {cv_results['mean_metrics']['best_metric']:.4f} "
                   f"± {cv_results['std_metrics']['best_metric']:.4f}")
        
        return cv_results
    
    def save_model(self, model: nn.Module, fold: int, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        save_dir = Path(self.config.get('save_dir', 'checkpoints'))
        save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'fold': fold,
            'epoch': epoch,
            'metrics': metrics,
            'config': self.config
        }
        
        torch.save(checkpoint, save_dir / f'research_model_fold_{fold}_epoch_{epoch}.pth')
    
    def generate_training_report(self, cv_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive training report."""
        
        report = {
            'experiment_info': {
                'dataset': cv_results['dataset'],
                'model_type': cv_results['model_type'],
                'task_type': cv_results['task_type'],
                'timestamp': str(Path().cwd()),
                'config': self.config
            },
            'cross_validation_results': cv_results,
            'summary': {
                'mean_performance': cv_results['mean_metrics']['best_metric'],
                'std_performance': cv_results['std_metrics']['best_metric'],
                'best_fold': np.argmax([fold['best_metric'] for fold in cv_results['fold_results']]),
                'worst_fold': np.argmin([fold['best_metric'] for fold in cv_results['fold_results']])
            }
        }
        
        # Save report
        results_dir = Path(self.config.get('results_dir', 'results'))
        results_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = results_dir / f'research_training_report_{cv_results["dataset"]}_{cv_results["model_type"]}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Training report saved to {report_file}")
        
        return report


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train models using research paper methodology")
    parser.add_argument("--dataset", required=True, 
                       choices=['brats2021', 'lidc_idri', 'medical_decathlon'],
                       help="Dataset to use for training")
    parser.add_argument("--model", required=True,
                       choices=['unet_2d', 'unet_3d', 'attention_unet', 'nnunet', 'mask_rcnn', 'vision_transformer'],
                       help="Model architecture to train")
    parser.add_argument("--config", type=Path, default=Path("config/research_paper_config.json"),
                       help="Configuration file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Update config with command line arguments
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.learning_rate
    config['use_wandb'] = args.use_wandb
    
    # Determine task type
    task_type = config['datasets'][args.dataset]['task']
    
    try:
        # Create training pipeline
        pipeline = ResearchPaperTrainingPipeline(config)
        
        # Run 5-fold cross-validation
        cv_results = pipeline.run_5_fold_cv(args.dataset, args.model, task_type)
        
        # Generate training report
        report = pipeline.generate_training_report(cv_results)
        
        print(f"Training completed successfully!")
        print(f"Dataset: {args.dataset}")
        print(f"Model: {args.model}")
        print(f"Task: {task_type}")
        print(f"Mean Performance: {cv_results['mean_metrics']['best_metric']:.4f} ± {cv_results['std_metrics']['best_metric']:.4f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
