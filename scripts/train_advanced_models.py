#!/usr/bin/env python3
"""
Advanced training script with multiple model architectures and datasets.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import medmnist
from medmnist import ChestMNIST, DermaMNIST, OCTMNIST

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        
        # Attention mechanism
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
        
        # Residual blocks with skip connections
        residual = x
        x = self.res_block1(x)
        if x.shape[1] != residual.shape[1]:
            residual = nn.functional.adaptive_avg_pool2d(residual, x.shape[2:])
            residual = nn.functional.pad(residual, (0, 0, 0, 0, 0, x.shape[1] - residual.shape[1]))
        x = x + residual
        x = nn.functional.relu(x)
        
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        # Apply attention
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        x = self.classifier(x)
        return x

class EfficientNet(nn.Module):
    """EfficientNet-inspired architecture for medical imaging."""
    
    def __init__(self, num_classes: int, input_channels: int = 1):
        super(EfficientNet, self).__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # MBConv blocks
        self.mbconv1 = self._make_mbconv(32, 16, 1, 1)
        self.mbconv2 = self._make_mbconv(16, 24, 6, 2)
        self.mbconv3 = self._make_mbconv(24, 40, 6, 2)
        self.mbconv4 = self._make_mbconv(40, 80, 6, 2)
        self.mbconv5 = self._make_mbconv(80, 112, 6, 1)
        self.mbconv6 = self._make_mbconv(112, 192, 6, 2)
        self.mbconv7 = self._make_mbconv(192, 320, 6, 1)
        
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )
    
    def _make_mbconv(self, in_channels: int, out_channels: int, expand_ratio: int, stride: int):
        """Create MBConv block."""
        return nn.Sequential(
            # Expansion
            nn.Conv2d(in_channels, in_channels * expand_ratio, kernel_size=1),
            nn.BatchNorm2d(in_channels * expand_ratio),
            nn.SiLU(inplace=True),
            # Depthwise
            nn.Conv2d(in_channels * expand_ratio, in_channels * expand_ratio, 
                     kernel_size=3, stride=stride, padding=1, groups=in_channels * expand_ratio),
            nn.BatchNorm2d(in_channels * expand_ratio),
            nn.SiLU(inplace=True),
            # Squeeze and excitation
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * expand_ratio, in_channels * expand_ratio // 4, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels * expand_ratio // 4, in_channels * expand_ratio, kernel_size=1),
            nn.Sigmoid(),
            # Projection
            nn.Conv2d(in_channels * expand_ratio, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.mbconv6(x)
        x = self.mbconv7(x)
        x = self.head(x)
        return x

class AdvancedTrainer:
    """Advanced trainer with multiple architectures."""
    
    def __init__(self, dataset_name: str, model_type: str = 'advanced', epochs: int = 5, batch_size: int = 32):
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load dataset
        self.load_dataset()
        
        # Initialize model
        self.model = self.create_model()
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        logger.info(f"Initialized {model_type} trainer for {dataset_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def load_dataset(self):
        """Load the specified MedMNIST dataset."""
        logger.info(f"Loading {self.dataset_name} dataset...")
        
        # Define transforms with augmentation
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        if self.dataset_name == 'chestmnist':
            self.train_dataset = ChestMNIST(split='train', download=True, transform=transform)
            self.val_dataset = ChestMNIST(split='val', download=True, transform=transform)
            self.test_dataset = ChestMNIST(split='test', download=True, transform=transform)
            self.classes = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
                          'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
                          'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
            self.input_channels = 1
        elif self.dataset_name == 'dermamnist':
            self.train_dataset = DermaMNIST(split='train', download=True, transform=transform)
            self.val_dataset = DermaMNIST(split='val', download=True, transform=transform)
            self.test_dataset = DermaMNIST(split='test', download=True, transform=transform)
            self.classes = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis-like lesions',
                          'Dermatofibroma', 'Melanoma', 'Melanocytic nevi', 'Vascular lesions']
            self.input_channels = 3  # RGB images
        elif self.dataset_name == 'octmnist':
            self.train_dataset = OCTMNIST(split='train', download=True, transform=transform)
            self.val_dataset = OCTMNIST(split='val', download=True, transform=transform)
            self.test_dataset = OCTMNIST(split='test', download=True, transform=transform)
            self.classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
            self.input_channels = 1
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        # Create data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        
        logger.info(f"Dataset loaded: {len(self.train_dataset)} train, {len(self.val_dataset)} val, {len(self.test_dataset)} test")
        logger.info(f"Number of classes: {len(self.classes)}, Input channels: {self.input_channels}")
    
    def create_model(self):
        """Create the specified model architecture."""
        if self.model_type == 'advanced':
            return AdvancedCNN(len(self.classes), self.input_channels).to(self.device)
        elif self.model_type == 'efficientnet':
            return EfficientNet(len(self.classes), self.input_channels).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Handle multi-label case for chest X-rays
            if self.dataset_name == 'chestmnist':
                target = target.float()
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                target = target.long().squeeze()
                self.criterion = nn.CrossEntropyLoss()
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            if self.dataset_name == 'chestmnist':
                pred = (torch.sigmoid(output) > 0.5).float()
                correct += (pred == target).all(dim=1).sum().item()
            else:
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
            
            total += target.size(0)
            
            if batch_idx % 50 == 0:
                logger.info(f'Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """Validate for one epoch."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                if self.dataset_name == 'chestmnist':
                    target = target.float()
                    self.criterion = nn.BCEWithLogitsLoss()
                else:
                    target = target.long().squeeze()
                    self.criterion = nn.CrossEntropyLoss()
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                
                if self.dataset_name == 'chestmnist':
                    pred = (torch.sigmoid(output) > 0.5).float()
                    correct += (pred == target).all(dim=1).sum().item()
                else:
                    pred = output.argmax(dim=1)
                    correct += (pred == target).sum().item()
                
                total += target.size(0)
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self):
        """Train the model."""
        logger.info(f"Starting training for {self.epochs} epochs...")
        
        best_val_acc = 0.0
        
        for epoch in range(self.epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate_epoch()
            
            # Update scheduler
            self.scheduler.step()
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            epoch_time = time.time() - start_time
            
            logger.info(f'Epoch {epoch+1}/{self.epochs}:')
            logger.info(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            logger.info(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            logger.info(f'  LR: {self.scheduler.get_last_lr()[0]:.6f}, Time: {epoch_time:.2f}s')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(f'best_{self.model_type}_{self.dataset_name}.pth')
        
        logger.info(f'Training completed. Best validation accuracy: {best_val_acc:.2f}%')
    
    def evaluate(self):
        """Evaluate on test set."""
        logger.info("Evaluating on test set...")
        
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                if self.dataset_name == 'chestmnist':
                    target = target.float()
                    output = self.model(data)
                    pred = (torch.sigmoid(output) > 0.5).float()
                else:
                    target = target.long().squeeze()
                    output = self.model(data)
                    pred = output.argmax(dim=1)
                
                all_preds.append(pred.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        # Concatenate all predictions
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Calculate metrics
        if self.dataset_name == 'chestmnist':
            test_acc = (all_preds == all_targets).all(axis=1).mean() * 100
            logger.info(f'Test Accuracy: {test_acc:.2f}%')
        else:
            test_acc = (all_preds == all_targets).mean() * 100
            logger.info(f'Test Accuracy: {test_acc:.2f}%')
            
            # Classification report
            report = classification_report(all_targets, all_preds, target_names=self.classes, output_dict=True)
            logger.info("Classification Report:")
            for class_name, metrics in report.items():
                if isinstance(metrics, dict):
                    logger.info(f"  {class_name}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        return test_acc, all_preds, all_targets
    
    def save_model(self, path: str):
        """Save model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'dataset_name': self.dataset_name,
            'model_type': self.model_type,
            'num_classes': len(self.classes),
            'classes': self.classes,
            'input_channels': self.input_channels
        }, path)
        logger.info(f"Model saved to {path}")

def main():
    parser = argparse.ArgumentParser(description='Train advanced models on MedMNIST datasets')
    parser.add_argument('--datasets', nargs='+', default=['dermamnist', 'octmnist'],
                       help='Datasets to train on')
    parser.add_argument('--models', nargs='+', default=['advanced', 'efficientnet'],
                       help='Model architectures to use')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='results/advanced_training',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_summary = {}
    
    for dataset_name in args.datasets:
        for model_type in args.models:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {model_type.upper()} on {dataset_name.upper()}")
            logger.info(f"{'='*60}")
            
            try:
                # Initialize trainer
                trainer = AdvancedTrainer(
                    dataset_name=dataset_name,
                    model_type=model_type,
                    epochs=args.epochs,
                    batch_size=args.batch_size
                )
                
                # Train model
                trainer.train()
                
                # Evaluate
                test_acc, preds, targets = trainer.evaluate()
                
                # Save results
                results_summary[f"{dataset_name}_{model_type}"] = {
                    'test_accuracy': float(test_acc),
                    'model_type': model_type,
                    'dataset': dataset_name,
                    'num_classes': len(trainer.classes),
                    'input_channels': trainer.input_channels,
                    'model_parameters': sum(p.numel() for p in trainer.model.parameters())
                }
                
                # Save model
                trainer.save_model(str(output_path / f"{model_type}_{dataset_name}_model.pth"))
                
            except Exception as e:
                logger.error(f"Error training {model_type} on {dataset_name}: {str(e)}")
                results_summary[f"{dataset_name}_{model_type}"] = {'error': str(e)}
    
    # Save summary
    with open(output_path / 'advanced_training_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("ADVANCED TRAINING SUMMARY")
    logger.info(f"{'='*60}")
    for key, results in results_summary.items():
        if 'error' not in results:
            logger.info(f"{key}: {results['test_accuracy']:.2f}% accuracy ({results['model_parameters']:,} params)")
        else:
            logger.info(f"{key}: ERROR - {results['error']}")
    
    logger.info(f"\nAll results saved to: {output_path}")

if __name__ == '__main__':
    main()
