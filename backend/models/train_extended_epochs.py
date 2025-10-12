#!/usr/bin/env python3
"""
Extended Training Script for Research Paper - 50-100 Epochs
This script provides comprehensive training with proper validation, checkpointing,
and results collection for scientific publication.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import medmnist
from medmnist import ChestMNIST, DermaMNIST, OCTMNIST
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_extended.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SimpleCNN(nn.Module):
    """Simple CNN for baseline comparisons."""
    
    def __init__(self, num_classes: int, input_channels: int = 1):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class AdvancedCNN(nn.Module):
    """Advanced CNN with attention mechanisms and residual connections."""
    
    def __init__(self, num_classes: int, input_channels: int = 1):
        super(AdvancedCNN, self).__init__()
        
        # Feature extraction with residual blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual blocks
        self.res_block1 = self._make_residual_block(64, 128, stride=2)
        self.res_block2 = self._make_residual_block(128, 256, stride=2)
        self.res_block3 = self._make_residual_block(256, 512, stride=2)
        
        # Attention mechanism (Squeeze-and-Excitation)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def _make_residual_block(self, in_channels: int, out_channels: int, stride: int = 1):
        """Create a residual block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        # Apply attention
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        x = self.classifier(x)
        return x


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution (MBConv) block for EfficientNet."""
    
    def __init__(self, in_channels: int, out_channels: int, expand_ratio: int = 6, stride: int = 1):
        super(MBConvBlock, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        
        layers = []
        if expand_ratio != 1:
            # Expansion phase
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, 
                     padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])
        
        # Squeeze and Excitation
        layers.extend([
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, hidden_dim, kernel_size=1),
            nn.Sigmoid()
        ])
        
        # Projection phase
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class EfficientNetInspired(nn.Module):
    """EfficientNet-inspired architecture with MBConv blocks."""
    
    def __init__(self, num_classes: int, input_channels: int = 1):
        super(EfficientNetInspired, self).__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        
        # MBConv blocks
        self.blocks = nn.Sequential(
            MBConvBlock(32, 64, expand_ratio=1, stride=1),
            MBConvBlock(64, 128, expand_ratio=6, stride=2),
            MBConvBlock(128, 128, expand_ratio=6, stride=1),
            MBConvBlock(128, 256, expand_ratio=6, stride=2),
            MBConvBlock(256, 256, expand_ratio=6, stride=1),
            MBConvBlock(256, 512, expand_ratio=6, stride=2),
        )
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


class ExtendedTrainer:
    """Comprehensive trainer with proper validation, checkpointing, and logging."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create results directory
        self.results_dir = Path(config['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create checkpoints directory
        self.checkpoints_dir = Path(config['checkpoints_dir'])
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Best metrics
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def load_data(self, dataset_name: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load MedMNIST dataset with proper splits."""
        logger.info(f"Loading {dataset_name} dataset...")
        
        # Define transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Load dataset
        if dataset_name == 'chestmnist':
            train_dataset = ChestMNIST(split='train', transform=transform, download=True)
            val_dataset = ChestMNIST(split='val', transform=transform, download=True)
            test_dataset = ChestMNIST(split='test', transform=transform, download=True)
            num_classes = 14  # Multi-label
            task = 'multi-label'
        elif dataset_name == 'dermamnist':
            train_dataset = DermaMNIST(split='train', transform=transform, download=True)
            val_dataset = DermaMNIST(split='val', transform=transform, download=True)
            test_dataset = DermaMNIST(split='test', transform=transform, download=True)
            num_classes = 7
            task = 'single-label'
        elif dataset_name == 'octmnist':
            train_dataset = OCTMNIST(split='train', transform=transform, download=True)
            val_dataset = OCTMNIST(split='val', transform=transform, download=True)
            test_dataset = OCTMNIST(split='test', transform=transform, download=True)
            num_classes = 4
            task = 'single-label'
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Val samples: {len(val_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")
        logger.info(f"Num classes: {num_classes}, Task: {task}")
        
        return train_loader, val_loader, test_loader, num_classes, task
    
    def create_model(self, model_type: str, num_classes: int, input_channels: int) -> nn.Module:
        """Create model based on type."""
        logger.info(f"Creating {model_type} model...")
        
        if model_type == 'simple':
            model = SimpleCNN(num_classes, input_channels)
        elif model_type == 'advanced':
            model = AdvancedCNN(num_classes, input_channels)
        elif model_type == 'efficientnet':
            model = EfficientNetInspired(num_classes, input_channels)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model = model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   criterion: nn.Module, optimizer: optim.Optimizer, task: str) -> Tuple[float, float]:
        """Train for one epoch."""
        model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Handle multi-label vs single-label
            if task == 'multi-label':
                labels = labels.float()
            else:
                labels = labels.squeeze().long()
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            
            if task == 'multi-label':
                preds = (torch.sigmoid(outputs) > 0.5).float()
            else:
                preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(train_loader)
        
        # Calculate accuracy
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        if task == 'multi-label':
            epoch_acc = accuracy_score(all_targets, all_preds)
        else:
            epoch_acc = accuracy_score(all_targets, all_preds)
        
        return epoch_loss, epoch_acc
    
    def validate(self, model: nn.Module, val_loader: DataLoader, 
                criterion: nn.Module, task: str) -> Tuple[float, float]:
        """Validate the model."""
        model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validating'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Handle multi-label vs single-label
                if task == 'multi-label':
                    labels = labels.float()
                else:
                    labels = labels.squeeze().long()
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                
                if task == 'multi-label':
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                else:
                    preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        
        val_loss = running_loss / len(val_loader)
        
        # Calculate accuracy
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        if task == 'multi-label':
            val_acc = accuracy_score(all_targets, all_preds)
        else:
            val_acc = accuracy_score(all_targets, all_preds)
        
        return val_loss, val_acc
    
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, 
                       scheduler: Any, epoch: int, val_acc: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'val_acc': val_acc,
            'history': self.history,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoints_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoints_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
    
    def plot_training_history(self):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.history['train_acc'], label='Train Acc')
        axes[0, 1].plot(self.history['val_acc'], label='Val Acc')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        axes[1, 0].plot(self.history['learning_rates'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Overfitting gap
        overfitting_gap = np.array(self.history['train_acc']) - np.array(self.history['val_acc'])
        axes[1, 1].plot(overfitting_gap)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Train - Val Accuracy')
        axes[1, 1].set_title('Overfitting Gap')
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = self.results_dir / f"training_history_{self.config['experiment_name']}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training history plot: {plot_path}")
        plt.close()
    
    def evaluate_test(self, model: nn.Module, test_loader: DataLoader, task: str) -> Dict[str, Any]:
        """Comprehensive evaluation on test set."""
        logger.info("Evaluating on test set...")
        model.eval()
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Testing'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Handle multi-label vs single-label
                if task == 'multi-label':
                    labels = labels.float()
                else:
                    labels = labels.squeeze().long()
                
                # Forward pass
                outputs = model(images)
                
                if task == 'multi-label':
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float()
                else:
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        results = {}
        
        if task == 'multi-label':
            results['accuracy'] = accuracy_score(all_targets, all_preds)
            # Per-label metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_targets, all_preds, average='samples', zero_division=0
            )
            results['precision'] = precision
            results['recall'] = recall
            results['f1_score'] = f1
        else:
            results['accuracy'] = accuracy_score(all_targets, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_targets, all_preds, average='weighted', zero_division=0
            )
            results['precision'] = precision
            results['recall'] = recall
            results['f1_score'] = f1
            
            # Confusion matrix
            cm = confusion_matrix(all_targets, all_preds)
            results['confusion_matrix'] = cm.tolist()
            
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            cm_path = self.results_dir / f"confusion_matrix_{self.config['experiment_name']}.png"
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix: {cm_path}")
            plt.close()
        
        logger.info(f"Test Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Test Precision: {results['precision']:.4f}")
        logger.info(f"Test Recall: {results['recall']:.4f}")
        logger.info(f"Test F1-Score: {results['f1_score']:.4f}")
        
        return results
    
    def train(self, dataset_name: str, model_type: str):
        """Main training loop."""
        logger.info("="*50)
        logger.info(f"Starting training: {dataset_name} - {model_type}")
        logger.info(f"Configuration: {self.config}")
        logger.info("="*50)
        
        # Load data
        train_loader, val_loader, test_loader, num_classes, task = self.load_data(dataset_name)
        
        # Determine input channels
        sample_image, _ = train_loader.dataset[0]
        input_channels = sample_image.shape[0]
        logger.info(f"Input channels: {input_channels}")
        
        # Create model
        model = self.create_model(model_type, num_classes, input_channels)
        
        # Create loss function
        if task == 'multi-label':
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Create optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Create scheduler
        if self.config['scheduler'] == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.1,
                patience=10,
                verbose=True
            )
        elif self.config['scheduler'] == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.config['epochs'],
                eta_min=1e-6
            )
        else:
            scheduler = None
        
        # Training loop
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            logger.info(f"\nEpoch {epoch + 1}/{self.config['epochs']}")
            logger.info("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer, task)
            
            # Validate
            val_loss, val_acc = self.validate(model, val_loader, criterion, task)
            
            # Update learning rate
            current_lr = optimizer.param_groups[0]['lr']
            if scheduler:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_acc)
                else:
                    scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Log results
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            logger.info(f"Learning Rate: {current_lr:.6f}")
            
            # Check for best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                patience_counter = 0
                logger.info(f"New best validation accuracy: {val_acc:.4f}")
            else:
                patience_counter += 1
            
            # Save checkpoint
            if (epoch + 1) % self.config['save_frequency'] == 0 or is_best:
                self.save_checkpoint(model, optimizer, scheduler, epoch + 1, val_acc, is_best)
            
            # Early stopping
            if self.config['early_stopping'] and patience_counter >= self.config['early_stopping_patience']:
                logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break
        
        training_time = time.time() - start_time
        logger.info(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
        
        # Load best model for final evaluation
        best_checkpoint = torch.load(self.checkpoints_dir / "best_model.pth")
        model.load_state_dict(best_checkpoint['model_state_dict'])
        
        # Final evaluation on test set
        test_results = self.evaluate_test(model, test_loader, task)
        
        # Plot training history
        self.plot_training_history()
        
        # Save results
        final_results = {
            'experiment_name': self.config['experiment_name'],
            'dataset': dataset_name,
            'model_type': model_type,
            'task': task,
            'num_classes': num_classes,
            'config': self.config,
            'training_time': training_time,
            'best_epoch': self.best_epoch,
            'best_val_acc': self.best_val_acc,
            'test_results': test_results,
            'training_history': self.history
        }
        
        results_file = self.results_dir / f"results_{self.config['experiment_name']}.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        logger.info(f"Saved results to: {results_file}")
        
        return final_results


def main():
    parser = argparse.ArgumentParser(description='Extended Training Script for Research Paper')
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['chestmnist', 'dermamnist', 'octmnist'],
                       help='Dataset to train on')
    parser.add_argument('--model', type=str, required=True,
                       choices=['simple', 'advanced', 'efficientnet'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    parser.add_argument('--scheduler', type=str, default='plateau',
                       choices=['plateau', 'cosine', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--early_stopping', action='store_true',
                       help='Enable early stopping')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                       help='Early stopping patience (default: 15)')
    parser.add_argument('--save_frequency', type=int, default=10,
                       help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--results_dir', type=str, default='training_results_extended',
                       help='Results directory')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints_extended',
                       help='Checkpoints directory')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Create experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.experiment_name = f"{args.dataset}_{args.model}_{timestamp}"
    
    # Create configuration
    config = {
        'dataset': args.dataset,
        'model_type': args.model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'scheduler': args.scheduler,
        'early_stopping': args.early_stopping,
        'early_stopping_patience': args.early_stopping_patience,
        'save_frequency': args.save_frequency,
        'results_dir': args.results_dir,
        'checkpoints_dir': args.checkpoints_dir,
        'experiment_name': args.experiment_name
    }
    
    # Create trainer and run training
    trainer = ExtendedTrainer(config)
    results = trainer.train(args.dataset, args.model)
    
    logger.info("\n" + "="*50)
    logger.info("Training completed successfully!")
    logger.info(f"Results saved to: {trainer.results_dir}")
    logger.info(f"Best model saved to: {trainer.checkpoints_dir / 'best_model.pth'}")
    logger.info("="*50)


if __name__ == '__main__':
    main()



