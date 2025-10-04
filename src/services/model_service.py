"""
Model serving and inference service.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import cv2
from scipy import ndimage
from skimage import measure, morphology

from ..core.config import settings
from ..schemas.api import DetectionResult, BoundingBox, SegmentationMask, TumorMetrics

logger = logging.getLogger(__name__)


class ModelService:
    """Service for model inference and result processing."""
    
    def __init__(self):
        self.models = {}
        self.model_cache = {}
        self.cache_size = settings.MODEL_CACHE_SIZE
    
    def load_model(self, model_id: str, model_path: str, config: Dict[str, Any]) -> bool:
        """
        Load a model for inference.
        
        Args:
            model_id: Unique model identifier
            model_path: Path to model file
            config: Model configuration
            
        Returns:
            True if model loaded successfully
        """
        try:
            # Load model based on type
            if config.get("type") == "pytorch":
                model = self._load_pytorch_model(model_path, config)
            elif config.get("type") == "onnx":
                model = self._load_onnx_model(model_path, config)
            else:
                raise ValueError(f"Unsupported model type: {config.get('type')}")
            
            self.models[model_id] = {
                "model": model,
                "config": config,
                "loaded_at": time.time()
            }
            
            logger.info(f"Model {model_id} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            return False
    
    def _load_pytorch_model(self, model_path: str, config: Dict[str, Any]) -> nn.Module:
        """Load PyTorch model."""
        # This is a placeholder - in practice, you would load your trained models
        # For now, we'll create a simple UNet-like architecture
        
        class SimpleUNet(nn.Module):
            def __init__(self, in_channels=1, out_channels=1):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(in_channels, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
                
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(128, 64, 2, stride=2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, out_channels, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
        
        model = SimpleUNet()
        
        # Load weights if available
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
        except Exception as e:
            logger.warning(f"Could not load model weights: {e}")
            # Initialize with random weights for demo purposes
        
        return model
    
    def _load_onnx_model(self, model_path: str, config: Dict[str, Any]):
        """Load ONNX model."""
        import onnxruntime as ort
        
        providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')
        
        session = ort.InferenceSession(model_path, providers=providers)
        return session
    
    def predict(self, model_id: str, image_data: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference on image data.
        
        Args:
            model_id: Model identifier
            image_data: Preprocessed image data
            config: Inference configuration
            
        Returns:
            Dictionary containing predictions and metadata
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not loaded")
        
        model_info = self.models[model_id]
        model = model_info["model"]
        model_config = model_info["config"]
        
        start_time = time.time()
        
        try:
            # Prepare input
            input_tensor = self._prepare_input(image_data, model_config)
            
            # Run inference
            if model_config.get("type") == "pytorch":
                predictions = self._pytorch_inference(model, input_tensor)
            elif model_config.get("type") == "onnx":
                predictions = self._onnx_inference(model, input_tensor, model_config)
            else:
                raise ValueError(f"Unsupported model type: {model_config.get('type')}")
            
            # Post-process results
            processed_results = self._postprocess_predictions(
                predictions, image_data, model_config
            )
            
            inference_time = time.time() - start_time
            
            return {
                "predictions": processed_results,
                "inference_time": inference_time,
                "model_id": model_id,
                "input_shape": list(image_data.shape),
                "output_shape": [list(pred.shape) for pred in predictions] if isinstance(predictions, list) else [list(predictions.shape)]
            }
            
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise ValueError(f"Inference failed: {e}")
    
    def _prepare_input(self, image_data: np.ndarray, config: Dict[str, Any]) -> torch.Tensor:
        """Prepare input tensor for model inference."""
        # Ensure correct dimensions
        if len(image_data.shape) == 2:
            # Add channel dimension
            image_data = np.expand_dims(image_data, axis=0)
        elif len(image_data.shape) == 3:
            # Assume (H, W, C) and convert to (C, H, W)
            if image_data.shape[-1] <= 4:  # Likely channels last
                image_data = np.transpose(image_data, (2, 0, 1))
            else:  # Likely (D, H, W) - add channel dimension
                image_data = np.expand_dims(image_data, axis=0)
        
        # Add batch dimension
        image_data = np.expand_dims(image_data, axis=0)
        
        # Convert to tensor
        input_tensor = torch.from_numpy(image_data.astype(np.float32))
        
        return input_tensor
    
    def _pytorch_inference(self, model: nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run PyTorch model inference."""
        with torch.no_grad():
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
                model = model.cuda()
            
            output = model(input_tensor)
            
            if torch.cuda.is_available():
                output = output.cpu()
            
            return output
    
    def _onnx_inference(self, session, input_tensor: torch.Tensor, config: Dict[str, Any]) -> np.ndarray:
        """Run ONNX model inference."""
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Convert tensor to numpy
        input_numpy = input_tensor.numpy()
        
        # Run inference
        outputs = session.run([output_name], {input_name: input_numpy})
        
        return outputs[0]
    
    def _postprocess_predictions(self, predictions: Any, original_image: np.ndarray, config: Dict[str, Any]) -> List[DetectionResult]:
        """Post-process model predictions into standardized format."""
        results = []
        
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.numpy()
        
        # Handle different prediction types
        if config.get("task_type") == "segmentation":
            results = self._process_segmentation_predictions(predictions, original_image, config)
        elif config.get("task_type") == "detection":
            results = self._process_detection_predictions(predictions, original_image, config)
        else:
            # Default to segmentation
            results = self._process_segmentation_predictions(predictions, original_image, config)
        
        return results
    
    def _process_segmentation_predictions(self, predictions: np.ndarray, original_image: np.ndarray, config: Dict[str, Any]) -> List[DetectionResult]:
        """Process segmentation predictions."""
        results = []
        
        # Remove batch dimension if present
        if len(predictions.shape) == 4:
            predictions = predictions[0]
        
        # Apply threshold
        threshold = config.get("threshold", 0.5)
        binary_mask = (predictions > threshold).astype(np.uint8)
        
        # Find connected components
        labeled_mask, num_components = ndimage.label(binary_mask)
        
        for i in range(1, num_components + 1):
            component_mask = (labeled_mask == i).astype(np.uint8)
            
            # Calculate bounding box
            bbox = self._calculate_bounding_box(component_mask)
            
            # Calculate metrics
            metrics = self._calculate_tumor_metrics(component_mask, original_image)
            
            # Create segmentation mask
            segmentation_mask = SegmentationMask(
                mask_data=self._encode_mask(component_mask),
                format="numpy",
                dimensions=list(component_mask.shape),
                confidence=float(np.mean(predictions[component_mask > 0]))
            )
            
            # Create detection result
            detection = DetectionResult(
                detection_id=f"seg_{i}",
                class_name="tumor",
                class_confidence=float(np.mean(predictions[component_mask > 0])),
                bounding_box=bbox,
                segmentation_mask=segmentation_mask,
                metrics=metrics
            )
            
            results.append(detection)
        
        return results
    
    def _process_detection_predictions(self, predictions: np.ndarray, original_image: np.ndarray, config: Dict[str, Any]) -> List[DetectionResult]:
        """Process detection predictions (bounding boxes)."""
        results = []
        
        # This is a simplified implementation
        # In practice, you would parse YOLO, R-CNN, or other detection format outputs
        
        # For demo purposes, create a simple detection
        if np.max(predictions) > config.get("threshold", 0.5):
            # Create a bounding box covering the center region
            h, w = original_image.shape[:2]
            center_x, center_y = w // 2, h // 2
            box_size = min(w, h) // 4
            
            bbox = BoundingBox(
                x_min=float(center_x - box_size // 2),
                y_min=float(center_y - box_size // 2),
                x_max=float(center_x + box_size // 2),
                y_max=float(center_y + box_size // 2),
                confidence=float(np.max(predictions))
            )
            
            detection = DetectionResult(
                detection_id="det_1",
                class_name="tumor",
                class_confidence=float(np.max(predictions)),
                bounding_box=bbox
            )
            
            results.append(detection)
        
        return results
    
    def _calculate_bounding_box(self, mask: np.ndarray) -> BoundingBox:
        """Calculate bounding box from segmentation mask."""
        coords = np.where(mask > 0)
        
        if len(coords[0]) == 0:
            return BoundingBox(x_min=0, y_min=0, x_max=0, y_max=0, confidence=0.0)
        
        y_min, y_max = np.min(coords[0]), np.max(coords[0])
        x_min, x_max = np.min(coords[1]), np.max(coords[1])
        
        # Handle 3D case
        if len(mask.shape) == 3:
            z_min, z_max = np.min(coords[2]), np.max(coords[2])
            return BoundingBox(
                x_min=float(x_min), y_min=float(y_min), z_min=float(z_min),
                x_max=float(x_max), y_max=float(y_max), z_max=float(z_max),
                confidence=1.0
            )
        else:
            return BoundingBox(
                x_min=float(x_min), y_min=float(y_min),
                x_max=float(x_max), y_max=float(y_max),
                confidence=1.0
            )
    
    def _calculate_tumor_metrics(self, mask: np.ndarray, original_image: np.ndarray) -> TumorMetrics:
        """Calculate tumor metrics from segmentation mask."""
        # Calculate volume (assuming voxel size of 1mm³ for simplicity)
        volume_mm3 = float(np.sum(mask))
        
        # Calculate surface area
        if len(mask.shape) == 3:
            surface_area_mm2 = float(np.sum(mask) - np.sum(ndimage.binary_erosion(mask)))
        else:
            surface_area_mm2 = float(np.sum(mask) - np.sum(ndimage.binary_erosion(mask)))
        
        # Calculate maximum diameter
        coords = np.where(mask > 0)
        if len(coords[0]) > 0:
            distances = []
            for i in range(len(coords[0])):
                for j in range(i + 1, len(coords[0])):
                    dist = np.sqrt(sum((coords[k][i] - coords[k][j])**2 for k in range(len(coords))))
                    distances.append(dist)
            diameter_mm = float(max(distances)) if distances else 0.0
        else:
            diameter_mm = 0.0
        
        # Calculate shape metrics
        if len(mask.shape) == 3:
            # 3D shape metrics
            sphericity = self._calculate_sphericity_3d(mask)
            elongation = self._calculate_elongation_3d(mask)
            compactness = self._calculate_compactness_3d(mask)
        else:
            # 2D shape metrics
            sphericity = self._calculate_sphericity_2d(mask)
            elongation = self._calculate_elongation_2d(mask)
            compactness = self._calculate_compactness_2d(mask)
        
        return TumorMetrics(
            volume_mm3=volume_mm3,
            surface_area_mm2=surface_area_mm2,
            diameter_mm=diameter_mm,
            sphericity=sphericity,
            elongation=elongation,
            compactness=compactness
        )
    
    def _calculate_sphericity_3d(self, mask: np.ndarray) -> float:
        """Calculate 3D sphericity."""
        volume = np.sum(mask)
        surface_area = np.sum(mask) - np.sum(ndimage.binary_erosion(mask))
        
        if surface_area > 0:
            sphericity = (36 * np.pi * volume**2)**(1/3) / surface_area
            return float(min(1.0, sphericity))
        return 0.0
    
    def _calculate_sphericity_2d(self, mask: np.ndarray) -> float:
        """Calculate 2D sphericity (circularity)."""
        area = np.sum(mask)
        perimeter = np.sum(mask) - np.sum(ndimage.binary_erosion(mask))
        
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter**2)
            return float(min(1.0, circularity))
        return 0.0
    
    def _calculate_elongation_3d(self, mask: np.ndarray) -> float:
        """Calculate 3D elongation."""
        coords = np.where(mask > 0)
        if len(coords[0]) < 2:
            return 0.0
        
        # Calculate principal components
        points = np.column_stack(coords)
        centered = points - np.mean(points, axis=0)
        cov_matrix = np.cov(centered.T)
        eigenvalues = np.linalg.eigvals(cov_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        if eigenvalues[0] > 0:
            elongation = eigenvalues[1] / eigenvalues[0]
            return float(elongation)
        return 0.0
    
    def _calculate_elongation_2d(self, mask: np.ndarray) -> float:
        """Calculate 2D elongation."""
        coords = np.where(mask > 0)
        if len(coords[0]) < 2:
            return 0.0
        
        points = np.column_stack(coords)
        centered = points - np.mean(points, axis=0)
        cov_matrix = np.cov(centered.T)
        eigenvalues = np.linalg.eigvals(cov_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        if eigenvalues[0] > 0:
            elongation = eigenvalues[1] / eigenvalues[0]
            return float(elongation)
        return 0.0
    
    def _calculate_compactness_3d(self, mask: np.ndarray) -> float:
        """Calculate 3D compactness."""
        volume = np.sum(mask)
        surface_area = np.sum(mask) - np.sum(ndimage.binary_erosion(mask))
        
        if volume > 0:
            compactness = volume / (surface_area**(3/2))
            return float(compactness)
        return 0.0
    
    def _calculate_compactness_2d(self, mask: np.ndarray) -> float:
        """Calculate 2D compactness."""
        area = np.sum(mask)
        perimeter = np.sum(mask) - np.sum(ndimage.binary_erosion(mask))
        
        if perimeter > 0:
            compactness = area / (perimeter**2)
            return float(compactness)
        return 0.0
    
    def _encode_mask(self, mask: np.ndarray) -> str:
        """Encode mask as base64 string."""
        import base64
        import io
        
        # Convert to bytes
        buffer = io.BytesIO()
        np.save(buffer, mask)
        mask_bytes = buffer.getvalue()
        
        # Encode as base64
        encoded = base64.b64encode(mask_bytes).decode('utf-8')
        
        return encoded
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models."""
        models = []
        for model_id, model_info in self.models.items():
            models.append({
                "model_id": model_id,
                "config": model_info["config"],
                "loaded_at": model_info["loaded_at"]
            })
        return models
    
    def unload_model(self, model_id: str) -> bool:
        """Unload a model to free memory."""
        if model_id in self.models:
            del self.models[model_id]
            logger.info(f"Model {model_id} unloaded")
            return True
        return False
