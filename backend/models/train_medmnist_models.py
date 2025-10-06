#!/usr/bin/env python3
"""
Training script for MedMNIST datasets using real medical imaging data.
"""

import argparse
import json
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Any
import time

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
from medmnist import ChestMNIST, DermaMNIST, OCTMNIST, INFO

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleCNN(nn.Module):
    """Simple CNN for MedMNIST classification."""
    
    def __init__(self, num_classes: int, input_channels: int = 1):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class MedMNISTTrainer:
    """Trainer for MedMNIST datasets."""
    
    def __init__(self, dataset_name: str, epochs: int = 10, batch_size: int = 32, lr: float = 0.001):
        self.dataset_name = dataset_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load dataset
        self.load_dataset()
        
        # Initialize model
        self.model = SimpleCNN(
            num_classes=len(self.classes),
            input_channels=1 if self.dataset_name in ['chestmnist', 'dermamnist', 'octmnist'] else 3
        ).to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        logger.info(f"Initialized trainer for {dataset_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def load_dataset(self):
        """Load the specified MedMNIST dataset."""
        logger.info(f"Loading {self.dataset_name} dataset...")
        
        # Define transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        ])
        
        if self.dataset_name == 'chestmnist':
            self.train_dataset = ChestMNIST(split='train', download=True, transform=transform)
            self.val_dataset = ChestMNIST(split='val', download=True, transform=transform)
            self.test_dataset = ChestMNIST(split='test', download=True, transform=transform)
            self.classes = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
                          'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
                          'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        elif self.dataset_name == 'dermamnist':
            self.train_dataset = DermaMNIST(split='train', download=True, transform=transform)
            self.val_dataset = DermaMNIST(split='val', download=True, transform=transform)
            self.test_dataset = DermaMNIST(split='test', download=True, transform=transform)
            self.classes = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis-like lesions',
                          'Dermatofibroma', 'Melanoma', 'Melanocytic nevi', 'Vascular lesions']
        elif self.dataset_name == 'octmnist':
            self.train_dataset = OCTMNIST(split='train', download=True, transform=transform)
            self.val_dataset = OCTMNIST(split='val', download=True, transform=transform)
            self.test_dataset = OCTMNIST(split='test', download=True, transform=transform)
            self.classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        # Create data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        
        logger.info(f"Dataset loaded: {len(self.train_dataset)} train, {len(self.val_dataset)} val, {len(self.test_dataset)} test")
        logger.info(f"Number of classes: {len(self.classes)}")
    
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
            
            if batch_idx % 100 == 0:
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
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            epoch_time = time.time() - start_time
            
            logger.info(f'Epoch {epoch+1}/{self.epochs}:')
            logger.info(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            logger.info(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            logger.info(f'  Time: {epoch_time:.2f}s')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(f'best_model_{self.dataset_name}.pth')
        
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
            # Multi-label classification
            test_acc = (all_preds == all_targets).all(axis=1).mean() * 100
            logger.info(f'Test Accuracy: {test_acc:.2f}%')
        else:
            # Single-label classification
            test_acc = (all_preds == all_targets).mean() * 100
            logger.info(f'Test Accuracy: {test_acc:.2f}%')
            
            # Classification report
            report = classification_report(all_targets, all_preds, target_names=self.classes, output_dict=True)
            logger.info("Classification Report:")
            for class_name, metrics in report.items():
                if isinstance(metrics, dict):
                    logger.info(f"  {class_name}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        return test_acc, all_preds, all_targets
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{self.dataset_name.upper()} - Training Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accs, label='Train Accuracy')
        ax2.plot(self.val_accs, label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title(f'{self.dataset_name.upper()} - Training Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, preds, targets, save_path: str = None):
        """Plot confusion matrix."""
        if self.dataset_name == 'chestmnist':
            logger.info("Skipping confusion matrix for multi-label classification")
            return
        
        cm = confusion_matrix(targets, preds)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.classes, yticklabels=self.classes)
        plt.title(f'{self.dataset_name.upper()} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def save_model(self, path: str):
        """Save model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'dataset_name': self.dataset_name,
            'num_classes': len(self.classes),
            'classes': self.classes
        }, path)
        logger.info(f"Model saved to {path}")
    
    def save_results(self, output_dir: str):
        """Save training results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'dataset_name': self.dataset_name,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'lr': self.lr
        }
        
        with open(output_path / f'{self.dataset_name}_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # Save model
        self.save_model(str(output_path / f'{self.dataset_name}_final_model.pth'))
        
        # Plot and save visualizations
        self.plot_training_history(str(output_path / f'{self.dataset_name}_training_history.png'))
        
        logger.info(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Train MedMNIST models')
    parser.add_argument('--datasets', nargs='+', default=['chestmnist', 'dermamnist', 'octmnist'],
                       help='Datasets to train on')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='results/medmnist_training',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_summary = {}
    
    for dataset_name in args.datasets:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {dataset_name.upper()}")
        logger.info(f"{'='*50}")
        
        try:
            # Initialize trainer
            trainer = MedMNISTTrainer(
                dataset_name=dataset_name,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr
            )
            
            # Train model
            trainer.train()
            
            # Evaluate
            test_acc, preds, targets = trainer.evaluate()
            
            # Save results
            trainer.save_results(args.output_dir)
            
            # Plot confusion matrix
            trainer.plot_confusion_matrix(preds, targets, 
                                        str(output_path / f'{dataset_name}_confusion_matrix.png'))
            
            results_summary[dataset_name] = {
                'test_accuracy': float(test_acc),
                'num_classes': len(trainer.classes),
                'num_train_samples': len(trainer.train_dataset),
                'num_val_samples': len(trainer.val_dataset),
                'num_test_samples': len(trainer.test_dataset)
            }
            
        except Exception as e:
            logger.error(f"Error training {dataset_name}: {str(e)}")
            results_summary[dataset_name] = {'error': str(e)}
    
    # Save summary
    with open(output_path / 'training_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info(f"\n{'='*50}")
    logger.info("TRAINING SUMMARY")
    logger.info(f"{'='*50}")
    for dataset, results in results_summary.items():
        if 'error' not in results:
            logger.info(f"{dataset.upper()}: {results['test_accuracy']:.2f}% accuracy")
        else:
            logger.info(f"{dataset.upper()}: ERROR - {results['error']}")
    
    logger.info(f"\nAll results saved to: {output_path}")

if __name__ == '__main__':
    main()
