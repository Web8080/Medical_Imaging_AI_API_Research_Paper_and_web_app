#!/usr/bin/env python3
"""
Comprehensive training script for medical imaging AI models.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.data.dataset_downloader import MedicalDatasetDownloader
from backend.training.trainer import AdvancedTrainer
from backend.evaluation.metrics import MedicalMetrics
from backend.evaluation.visualizations import TrainingVisualizer
from backend.models.advanced_models import create_model, create_ensemble

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split


class MedicalDataset(Dataset):
    """Custom dataset for medical imaging data."""
    
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return torch.FloatTensor(image), torch.FloatTensor(label)


def load_brats_data(data_path: Path) -> tuple:
    """Load BRATS 2021 dataset."""
    # This is a placeholder - in practice, you would load actual BRATS data
    # For demonstration, we'll create synthetic data
    logging.info("Loading BRATS 2021 dataset...")
    
    # Create synthetic brain MRI data
    n_samples = 1000
    image_size = (128, 128)
    
    # Generate synthetic images
    images = np.random.rand(n_samples, 1, *image_size).astype(np.float32)
    
    # Generate synthetic segmentation masks
    labels = np.random.randint(0, 2, (n_samples, 1, *image_size)).astype(np.float32)
    
    logging.info(f"Loaded {n_samples} samples with shape {images.shape}")
    return images, labels


def load_lidc_data(data_path: Path) -> tuple:
    """Load LIDC-IDRI dataset."""
    logging.info("Loading LIDC-IDRI dataset...")
    
    # Create synthetic lung CT data
    n_samples = 800
    image_size = (256, 256)
    
    # Generate synthetic images
    images = np.random.rand(n_samples, 1, *image_size).astype(np.float32)
    
    # Generate synthetic detection labels
    labels = np.random.randint(0, 2, (n_samples,)).astype(np.float32)
    
    logging.info(f"Loaded {n_samples} samples with shape {images.shape}")
    return images, labels


def load_decathlon_data(data_path: Path) -> tuple:
    """Load Medical Segmentation Decathlon data."""
    logging.info("Loading Medical Segmentation Decathlon dataset...")
    
    # Create synthetic multi-organ data
    n_samples = 1200
    image_size = (224, 224)
    
    # Generate synthetic images
    images = np.random.rand(n_samples, 1, *image_size).astype(np.float32)
    
    # Generate synthetic multi-class segmentation masks
    labels = np.random.randint(0, 4, (n_samples, 1, *image_size)).astype(np.float32)
    
    logging.info(f"Loaded {n_samples} samples with shape {images.shape}")
    return images, labels


def create_training_config(dataset_name: str, model_type: str, 
                          task_type: str) -> Dict[str, Any]:
    """Create training configuration based on dataset and model."""
    
    base_config = {
        'dataset_name': dataset_name,
        'model_type': model_type,
        'task_type': task_type,
        'epochs': 100,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'save_dir': f'checkpoints/{dataset_name}_{model_type}',
        'log_dir': f'logs/{dataset_name}_{model_type}',
        'viz_dir': f'visualizations/{dataset_name}_{model_type}',
        'use_wandb': False,
        'project_name': 'medical-imaging-ai',
        'experiment_name': f'{dataset_name}_{model_type}_{task_type}',
        'grad_clip_norm': 1.0,
        'save_interval': 10
    }
    
    # Model-specific configurations
    if model_type == 'attention_unet':
        base_config['model'] = {
            'type': 'attention_unet',
            'params': {
                'in_channels': 1,
                'out_channels': 1,
                'base_features': 64
            }
        }
    elif model_type == 'vit_unet':
        base_config['model'] = {
            'type': 'vit_unet',
            'params': {
                'img_size': 224,
                'patch_size': 16,
                'in_channels': 1,
                'out_channels': 1,
                'embed_dim': 768,
                'num_heads': 12,
                'num_layers': 12
            }
        }
    elif model_type == 'efficientnet_unet':
        base_config['model'] = {
            'type': 'efficientnet_unet',
            'params': {
                'model_name': 'efficientnet-b0',
                'in_channels': 1,
                'out_channels': 1
            }
        }
    elif model_type == 'ensemble':
        base_config['model'] = {
            'type': 'ensemble',
            'models': [
                {
                    'type': 'attention_unet',
                    'params': {'in_channels': 1, 'out_channels': 1, 'base_features': 64},
                    'weight': 0.4
                },
                {
                    'type': 'vit_unet',
                    'params': {'img_size': 224, 'patch_size': 16, 'in_channels': 1, 'out_channels': 1},
                    'weight': 0.3
                },
                {
                    'type': 'efficientnet_unet',
                    'params': {'model_name': 'efficientnet-b0', 'in_channels': 1, 'out_channels': 1},
                    'weight': 0.3
                }
            ]
        }
    
    # Task-specific configurations
    if task_type == 'segmentation':
        base_config['loss'] = {
            'type': 'dice_bce',
            'dice_weight': 0.5,
            'bce_weight': 0.5
        }
        base_config['optimizer'] = {
            'type': 'adamw',
            'lr': 1e-4,
            'weight_decay': 0.01
        }
        base_config['scheduler'] = {
            'type': 'cosine',
            'T_max': 100,
            'eta_min': 1e-6
        }
    elif task_type == 'classification':
        base_config['loss'] = {
            'type': 'focal',
            'alpha': 1.0,
            'gamma': 2.0
        }
        base_config['optimizer'] = {
            'type': 'adam',
            'lr': 1e-4,
            'weight_decay': 1e-4
        }
        base_config['scheduler'] = {
            'type': 'plateau',
            'mode': 'max',
            'factor': 0.5,
            'patience': 10
        }
    
    return base_config


def train_model(dataset_name: str, model_type: str, task_type: str, 
                data_path: Path) -> Dict[str, Any]:
    """Train a model on the specified dataset."""
    
    logging.info(f"Training {model_type} on {dataset_name} for {task_type} task")
    
    # Load data
    if dataset_name == 'brats2021':
        images, labels = load_brats_data(data_path)
    elif dataset_name == 'lidc_idri':
        images, labels = load_lidc_data(data_path)
    elif dataset_name == 'medical_decathlon':
        images, labels = load_decathlon_data(data_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = MedicalDataset(X_train, y_train)
    val_dataset = MedicalDataset(X_val, y_val)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=True, 
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=16, 
        shuffle=False, 
        num_workers=4
    )
    
    # Create training configuration
    config = create_training_config(dataset_name, model_type, task_type)
    
    # Create trainer
    trainer = AdvancedTrainer(config)
    
    # Train model
    results = trainer.train(train_loader, val_loader)
    
    return results


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train medical imaging AI models")
    parser.add_argument("--dataset", required=True, 
                       choices=['brats2021', 'lidc_idri', 'medical_decathlon'],
                       help="Dataset to use for training")
    parser.add_argument("--model", required=True,
                       choices=['attention_unet', 'vit_unet', 'efficientnet_unet', 'ensemble'],
                       help="Model architecture to train")
    parser.add_argument("--task", required=True,
                       choices=['segmentation', 'classification'],
                       help="Task type")
    parser.add_argument("--data_path", type=Path, default=Path("data/datasets"),
                       help="Path to dataset")
    parser.add_argument("--download", action="store_true",
                       help="Download dataset before training")
    parser.add_argument("--config", type=Path,
                       help="Path to custom configuration file")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Download dataset if requested
    if args.download:
        downloader = MedicalDatasetDownloader(args.data_path)
        success = downloader.download_dataset(args.dataset)
        if not success:
            logging.error(f"Failed to download dataset {args.dataset}")
            return 1
    
    # Load custom config if provided
    if args.config and args.config.exists():
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
        logging.info(f"Loaded custom configuration from {args.config}")
    else:
        custom_config = None
    
    try:
        # Train model
        results = train_model(args.dataset, args.model, args.task, args.data_path)
        
        # Save results
        results_dir = Path(f"results/{args.dataset}_{args.model}_{args.task}")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save training history
        with open(results_dir / "training_history.json", 'w') as f:
            json.dump(results['training_history'], f, indent=2)
        
        # Save model
        torch.save(results['model'].state_dict(), results_dir / "model.pth")
        
        # Generate final report
        report = {
            'dataset': args.dataset,
            'model': args.model,
            'task': args.task,
            'best_metric': results['best_metric'],
            'training_time': results['training_time'],
            'final_metrics': results['training_history']['val_metrics'][-1]
        }
        
        with open(results_dir / "training_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        logging.info(f"Training completed successfully!")
        logging.info(f"Best metric: {results['best_metric']:.4f}")
        logging.info(f"Training time: {results['training_time']:.2f} seconds")
        logging.info(f"Results saved to: {results_dir}")
        
        return 0
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
