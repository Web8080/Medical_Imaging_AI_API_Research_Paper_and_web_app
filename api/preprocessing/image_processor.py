#!/usr/bin/env python3
"""
Image Processor for Medical Imaging AI API
Handles image preprocessing and preparation for model inference.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import logging
from typing import Union, Tuple
import io

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Handles image preprocessing for medical imaging models."""
    
    def __init__(self):
        self.image_size = 28  # MedMNIST standard size
        self.normalize_mean = [0.5]
        self.normalize_std = [0.5]
        
        # Define transforms for different model types
        self.transforms = {
            'chest': transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)
            ]),
            'derma': transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]),
            'oct': transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)
            ])
        }
    
    async def process_image(self, file_content: bytes, filename: str) -> Image.Image:
        """
        Process uploaded image file.
        
        Args:
            file_content: Raw file content
            filename: Original filename
            
        Returns:
            Processed PIL Image
        """
        try:
            # Determine file type and process accordingly
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                return await self._process_standard_image(file_content)
            elif filename.lower().endswith(('.dcm', '.dicom')):
                return await self._process_dicom_image(file_content)
            elif filename.lower().endswith(('.nii', '.nii.gz')):
                return await self._process_nifti_image(file_content)
            else:
                # Try to process as standard image
                return await self._process_standard_image(file_content)
                
        except Exception as e:
            logger.error(f"Error processing image {filename}: {str(e)}")
            raise ValueError(f"Failed to process image: {str(e)}")
    
    async def _process_standard_image(self, file_content: bytes) -> Image.Image:
        """Process standard image formats."""
        try:
            # Load image from bytes
            image = Image.open(io.BytesIO(file_content))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB' and image.mode != 'L':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            logger.error(f"Error processing standard image: {str(e)}")
            raise
    
    async def _process_dicom_image(self, file_content: bytes) -> Image.Image:
        """Process DICOM image."""
        try:
            # For now, treat as standard image
            # In production, you would use pydicom library
            logger.warning("DICOM processing not fully implemented, treating as standard image")
            return await self._process_standard_image(file_content)
            
        except Exception as e:
            logger.error(f"Error processing DICOM image: {str(e)}")
            raise
    
    async def _process_nifti_image(self, file_content: bytes) -> Image.Image:
        """Process NIfTI image."""
        try:
            # For now, treat as standard image
            # In production, you would use nibabel library
            logger.warning("NIfTI processing not fully implemented, treating as standard image")
            return await self._process_standard_image(file_content)
            
        except Exception as e:
            logger.error(f"Error processing NIfTI image: {str(e)}")
            raise
    
    def prepare_for_model(self, image: Image.Image, model_type: str) -> torch.Tensor:
        """
        Prepare image for model inference.
        
        Args:
            image: PIL Image
            model_type: Type of model ('chest', 'derma', 'oct')
            
        Returns:
            Preprocessed tensor ready for model input
        """
        try:
            # Apply appropriate transforms
            if model_type in self.transforms:
                transform = self.transforms[model_type]
            else:
                # Default transform
                transform = self.transforms['chest']
            
            # Apply transforms
            tensor = transform(image)
            
            # Add batch dimension
            tensor = tensor.unsqueeze(0)
            
            return tensor
            
        except Exception as e:
            logger.error(f"Error preparing image for model: {str(e)}")
            raise
    
    def preprocess_for_classification(self, image: Image.Image, model_type: str) -> torch.Tensor:
        """Preprocess image for classification."""
        return self.prepare_for_model(image, model_type)
    
    def preprocess_for_segmentation(self, image: Image.Image, model_type: str) -> torch.Tensor:
        """Preprocess image for segmentation."""
        # For segmentation, we might need different preprocessing
        return self.prepare_for_model(image, model_type)
    
    def postprocess_classification(self, predictions: torch.Tensor, model_type: str) -> dict:
        """
        Postprocess classification predictions.
        
        Args:
            predictions: Model predictions
            model_type: Type of model
            
        Returns:
            Processed predictions
        """
        try:
            # Apply softmax to get probabilities
            probabilities = torch.softmax(predictions, dim=1)
            
            # Get top prediction
            confidence, predicted_class = torch.max(probabilities, 1)
            
            return {
                'predicted_class': predicted_class.item(),
                'confidence': confidence.item(),
                'probabilities': probabilities[0].tolist()
            }
            
        except Exception as e:
            logger.error(f"Error postprocessing classification: {str(e)}")
            raise
    
    def postprocess_segmentation(self, predictions: torch.Tensor, model_type: str) -> dict:
        """
        Postprocess segmentation predictions.
        
        Args:
            predictions: Model predictions
            model_type: Type of model
            
        Returns:
            Processed segmentation mask
        """
        try:
            # Apply sigmoid for binary segmentation
            if predictions.shape[1] == 1:
                mask = torch.sigmoid(predictions)
            else:
                mask = torch.softmax(predictions, dim=1)
            
            # Convert to numpy
            mask_np = mask[0].cpu().numpy()
            
            return {
                'segmentation_mask': mask_np,
                'binary_mask': (mask_np > 0.5).astype(np.uint8)
            }
            
        except Exception as e:
            logger.error(f"Error postprocessing segmentation: {str(e)}")
            raise
    
    def resize_image(self, image: Image.Image, size: Tuple[int, int]) -> Image.Image:
        """Resize image to specified size."""
        return image.resize(size, Image.Resampling.LANCZOS)
    
    def normalize_image(self, image: np.ndarray, mean: float = 0.5, std: float = 0.5) -> np.ndarray:
        """Normalize image array."""
        return (image - mean) / std
    
    def denormalize_image(self, image: np.ndarray, mean: float = 0.5, std: float = 0.5) -> np.ndarray:
        """Denormalize image array."""
        return image * std + mean
    
    def apply_data_augmentation(self, image: Image.Image, augment: bool = True) -> Image.Image:
        """Apply data augmentation if enabled."""
        if not augment:
            return image
        
        # Simple augmentation for inference
        # In production, you might want more sophisticated augmentation
        return image
    
    def validate_image(self, image: Image.Image) -> bool:
        """Validate image quality and format."""
        try:
            # Check image size
            if image.size[0] < 10 or image.size[1] < 10:
                return False
            
            # Check if image is not corrupted
            image.verify()
            
            return True
            
        except Exception:
            return False
