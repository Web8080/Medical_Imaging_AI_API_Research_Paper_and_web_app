#!/usr/bin/env python3
"""
Model Loader for Medical Imaging AI API
Handles loading and management of trained models.
"""

import torch
import torch.nn as nn
import logging
from pathlib import Path
from typing import Dict, Optional, Any
import json

logger = logging.getLogger(__name__)

class ModelLoader:
    """Handles loading and management of trained models."""
    
    def __init__(self):
        self.models = {}
        self.model_info = {}
        self.class_names = {
            'chest': ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
                     'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
                     'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'],
            'derma': ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis-like lesions',
                     'Dermatofibroma', 'Melanoma', 'Melanocytic nevi', 'Vascular lesions'],
            'oct': ['CNV', 'DME', 'DRUSEN', 'NORMAL']
        }
        
        # Model paths
        self.model_paths = {
            'chest': 'results/real_medmnist_training/chestmnist_final_model.pth',
            'derma': 'training_results/advanced_models/dermamnist_advanced_cnn_model.pth',
            'oct': 'training_results/advanced_models/octmnist_advanced_cnn_model.pth'
        }
    
    async def load_all_models(self):
        """Load all available models."""
        logger.info("Loading all models...")
        
        for model_type, model_path in self.model_paths.items():
            try:
                await self.load_model(model_type, model_path)
                logger.info(f"Successfully loaded {model_type} model")
            except Exception as e:
                logger.error(f"Failed to load {model_type} model: {str(e)}")
    
    async def load_model(self, model_type: str, model_path: str):
        """Load a specific model."""
        try:
            # Check if model file exists
            if not Path(model_path).exists():
                logger.warning(f"Model file not found: {model_path}")
                return False
            
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Create model architecture based on type
            if model_type == 'chest':
                model = self._create_chest_model(checkpoint)
            elif model_type == 'derma':
                model = self._create_derma_model(checkpoint)
            elif model_type == 'oct':
                model = self._create_oct_model(checkpoint)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Load model state
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            self.models[model_type] = model
            
            # Store model info
            self.model_info[model_type] = {
                'path': model_path,
                'num_classes': len(self.class_names[model_type]),
                'input_channels': checkpoint.get('input_channels', 1),
                'parameters': sum(p.numel() for p in model.parameters()),
                'loaded_at': str(torch.datetime.now())
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_type}: {str(e)}")
            return False
    
    def _create_chest_model(self, checkpoint):
        """Create chest X-ray model."""
        # Simple CNN for chest X-rays
        class ChestCNN(nn.Module):
            def __init__(self, num_classes=14, input_channels=1):
                super(ChestCNN, self).__init__()
                self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.dropout = nn.Dropout(0.5)
                self.fc1 = nn.Linear(128 * 3 * 3, 256)
                self.fc2 = nn.Linear(256, num_classes)
                
            def forward(self, x):
                x = self.pool(torch.relu(self.conv1(x)))
                x = self.pool(torch.relu(self.conv2(x)))
                x = self.pool(torch.relu(self.conv3(x)))
                x = x.view(x.size(0), -1)
                x = self.dropout(torch.relu(self.fc1(x)))
                x = self.fc2(x)
                return x
        
        num_classes = len(self.class_names['chest'])
        input_channels = checkpoint.get('input_channels', 1)
        return ChestCNN(num_classes, input_channels)
    
    def _create_derma_model(self, checkpoint):
        """Create dermatology model."""
        # Advanced CNN for dermatology
        class DermaCNN(nn.Module):
            def __init__(self, num_classes=7, input_channels=3):
                super(DermaCNN, self).__init__()
                self.conv1 = nn.Conv2d(input_channels, 64, 3, padding=1)
                self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
                self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
                self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.dropout = nn.Dropout(0.5)
                self.fc1 = nn.Linear(512 * 1 * 1, 256)
                self.fc2 = nn.Linear(256, num_classes)
                
            def forward(self, x):
                x = self.pool(torch.relu(self.conv1(x)))
                x = self.pool(torch.relu(self.conv2(x)))
                x = self.pool(torch.relu(self.conv3(x)))
                x = self.pool(torch.relu(self.conv4(x)))
                x = x.view(x.size(0), -1)
                x = self.dropout(torch.relu(self.fc1(x)))
                x = self.fc2(x)
                return x
        
        num_classes = len(self.class_names['derma'])
        input_channels = checkpoint.get('input_channels', 3)
        return DermaCNN(num_classes, input_channels)
    
    def _create_oct_model(self, checkpoint):
        """Create OCT model."""
        # Advanced CNN for OCT
        class OCTCNN(nn.Module):
            def __init__(self, num_classes=4, input_channels=1):
                super(OCTCNN, self).__init__()
                self.conv1 = nn.Conv2d(input_channels, 64, 3, padding=1)
                self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
                self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
                self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.dropout = nn.Dropout(0.5)
                self.fc1 = nn.Linear(512 * 1 * 1, 256)
                self.fc2 = nn.Linear(256, num_classes)
                
            def forward(self, x):
                x = self.pool(torch.relu(self.conv1(x)))
                x = self.pool(torch.relu(self.conv2(x)))
                x = self.pool(torch.relu(self.conv3(x)))
                x = self.pool(torch.relu(self.conv4(x)))
                x = x.view(x.size(0), -1)
                x = self.dropout(torch.relu(self.fc1(x)))
                x = self.fc2(x)
                return x
        
        num_classes = len(self.class_names['oct'])
        input_channels = checkpoint.get('input_channels', 1)
        return OCTCNN(num_classes, input_channels)
    
    def get_model(self, model_type: str):
        """Get a loaded model."""
        return self.models.get(model_type)
    
    def get_loaded_models(self):
        """Get list of loaded models."""
        return list(self.models.keys())
    
    def get_model_info(self):
        """Get information about loaded models."""
        return self.model_info
    
    def get_class_names(self, model_type: str):
        """Get class names for a model type."""
        return self.class_names.get(model_type, [])
    
    async def reload_model(self, model_type: str):
        """Reload a specific model."""
        if model_type in self.model_paths:
            model_path = self.model_paths[model_type]
            return await self.load_model(model_type, model_path)
        return False
