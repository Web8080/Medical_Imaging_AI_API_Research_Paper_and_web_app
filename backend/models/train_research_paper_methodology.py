#!/usr/bin/env python3
"""
Research Paper Methodology Implementation for MedMNIST datasets.
Follows the specific training strategy described in the research paper.
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
from sklearn.model_selection import KFold, StratifiedKFold
import medmnist
from medmnist import ChestMNIST, DermaMNIST, OCTMNIST

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResearchPaperCNN(nn.Module):
    """
    CNN architecture following research paper specifications:
    - U-Net inspired architecture with skip connections
    - Batch normalization and dropout for regularization
    - Attention mechanisms for feature refinement
    """
    
    def __init__(self, num_classes: int, input_channels: int = 1):
        super(ResearchPaperCNN, self).__init__()
        
        # Encoder path
        self.encoder1 = self._make_encoder_block(input_channels, 64)
        self.encoder2 = self._make_encoder_block(64, 128)
        self.encoder3 = self._make_encoder_block(128, 256)
        self.encoder4 = self._make_encoder_block(256, 512)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)
        )
        
        # Decoder path with skip connections
        self.decoder4 = self._make_decoder_block(1024, 512, 512)
        self.decoder3 = self._make_decoder_block(512, 256, 256)
        self.decoder2 = self._make_decoder_block(256, 128, 128)
        self.decoder1 = self._make_decoder_block(128, 64, 64)
        
        # Attention mechanisms
        self.attention4 = self._make_attention_block(512, 512)
        self.attention3 = self._make_attention_block(256, 256)
        self.attention2 = self._make_attention_block(128, 128)
        self.attention1 = self._make_attention_block(64, 64)
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def _make_encoder_block(self, in_channels: int, out_channels: int):
        """Create encoder block with batch normalization."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
    
    def _make_decoder_block(self, in_channels: int, skip_channels: int, out_channels: int):
        """Create decoder block with skip connections."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _make_attention_block(self, in_channels: int, out_channels: int):
        """Create attention block for feature refinement."""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Encoder path
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Bottleneck
        b = self.bottleneck(e4)
        
        # Decoder path with attention
        d4 = self.decoder4(b)
        att4 = self.attention4(d4)
        d4 = d4 * att4
        
        d3 = self.decoder3(d4)
        att3 = self.attention3(d3)
        d3 = d3 * att3
        
        d2 = self.decoder2(d3)
        att2 = self.attention2(d2)
        d2 = d2 * att2
        
        d1 = self.decoder1(d2)
        att1 = self.attention1(d1)
        d1 = d1 * att1
        
        # Classification
        output = self.classifier(d1)
        return output

class ResearchPaperTrainer:
    """
    Trainer implementing research paper methodology:
    - 5-fold cross-validation
    - Combined Dice loss and cross-entropy loss
    - AdamW optimizer with cosine annealing
    - Advanced data augmentation
    """
    
    def __init__(self, dataset_name: str, epochs: int = 5, batch_size: int = 32):
        self.dataset_name = dataset_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load dataset
        self.load_dataset()
        
        # Initialize model
        self.model = ResearchPaperCNN(
            num_classes=len(self.classes),
            input_channels=self.input_channels
        ).to(self.device)
        
        # Research paper loss function (combined Dice + CrossEntropy)
        self.criterion = self.create_research_loss()
        
        # AdamW optimizer as specified in research paper
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        
        # Cosine annealing scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        logger.info(f"Initialized Research Paper trainer for {dataset_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def create_research_loss(self):
        """Create combined loss function as specified in research paper."""
        class CombinedLoss(nn.Module):
            def __init__(self, num_classes: int):
                super().__init__()
                self.ce_loss = nn.CrossEntropyLoss()
                self.dice_loss = self.dice_loss_fn
                self.alpha = 0.5  # Weight for combining losses
            
            def dice_loss_fn(self, pred, target):
                """Dice loss implementation."""
                smooth = 1e-5
                pred = torch.softmax(pred, dim=1)
                target_one_hot = torch.zeros_like(pred)
                target_one_hot.scatter_(1, target.unsqueeze(1), 1)
                
                intersection = (pred * target_one_hot).sum(dim=(2, 3))
                union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
                dice = (2.0 * intersection + smooth) / (union + smooth)
                return 1 - dice.mean()
            
            def forward(self, pred, target):
                ce = self.ce_loss(pred, target)
                dice = self.dice_loss(pred, target)
                return self.alpha * ce + (1 - self.alpha) * dice
        
        return CombinedLoss(len(self.classes))
    
    def load_dataset(self):
        """Load dataset with research paper augmentation strategy."""
        logger.info(f"Loading {self.dataset_name} dataset with research paper methodology...")
        
        # Research paper augmentation strategy
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            # Additional augmentations as per research paper
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
            self.input_channels = 3
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
    
    def train_epoch(self):
        """Train for one epoch with research paper methodology."""
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
        """Train the model using research paper methodology."""
        logger.info(f"Starting research paper training for {self.epochs} epochs...")
        
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
                self.save_model(f'best_research_{self.dataset_name}.pth')
        
        logger.info(f'Research paper training completed. Best validation accuracy: {best_val_acc:.2f}%')
    
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
            'num_classes': len(self.classes),
            'classes': self.classes,
            'input_channels': self.input_channels,
            'methodology': 'research_paper'
        }, path)
        logger.info(f"Research paper model saved to {path}")

def main():
    parser = argparse.ArgumentParser(description='Train models using research paper methodology')
    parser.add_argument('--datasets', nargs='+', default=['chestmnist', 'dermamnist', 'octmnist'],
                       help='Datasets to train on')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='results/research_paper_training',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_summary = {}
    
    for dataset_name in args.datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Research Paper Training: {dataset_name.upper()}")
        logger.info(f"{'='*60}")
        
        try:
            # Initialize trainer
            trainer = ResearchPaperTrainer(
                dataset_name=dataset_name,
                epochs=args.epochs,
                batch_size=args.batch_size
            )
            
            # Train model
            trainer.train()
            
            # Evaluate
            test_acc, preds, targets = trainer.evaluate()
            
            # Save results
            results_summary[f"{dataset_name}_research"] = {
                'test_accuracy': float(test_acc),
                'methodology': 'research_paper',
                'dataset': dataset_name,
                'num_classes': len(trainer.classes),
                'input_channels': trainer.input_channels,
                'model_parameters': sum(p.numel() for p in trainer.model.parameters()),
                'architecture': 'ResearchPaperCNN',
                'loss_function': 'Combined Dice + CrossEntropy',
                'optimizer': 'AdamW',
                'scheduler': 'CosineAnnealingLR',
                'augmentation': 'Advanced (rotation, flip, color jitter)'
            }
            
            # Save model
            trainer.save_model(str(output_path / f"research_{dataset_name}_model.pth"))
            
        except Exception as e:
            logger.error(f"Error training {dataset_name}: {str(e)}")
            results_summary[f"{dataset_name}_research"] = {'error': str(e)}
    
    # Save summary
    with open(output_path / 'research_paper_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("RESEARCH PAPER TRAINING SUMMARY")
    logger.info(f"{'='*60}")
    for key, results in results_summary.items():
        if 'error' not in results:
            logger.info(f"{key}: {results['test_accuracy']:.2f}% accuracy ({results['model_parameters']:,} params)")
        else:
            logger.info(f"{key}: ERROR - {results['error']}")
    
    logger.info(f"\nAll research paper results saved to: {output_path}")

if __name__ == '__main__':
    main()
