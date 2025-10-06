"""
DICOM processing service for handling medical imaging data.
"""

import hashlib
import io
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError
import nibabel as nib
from PIL import Image
import cv2

from ..core.config import settings
from ..core.security import anonymize_dicom_data, validate_file_upload

logger = logging.getLogger(__name__)


class DICOMProcessor:
    """Service for processing DICOM and other medical imaging formats."""
    
    def __init__(self):
        self.supported_formats = settings.SUPPORTED_FORMATS
        self.max_file_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    
    def process_upload(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Process uploaded medical imaging file.
        
        Args:
            file_content: Raw file content
            filename: Original filename
            
        Returns:
            Dictionary containing processed data and metadata
        """
        # Validate file
        if not validate_file_upload(file_content, filename):
            raise ValueError(f"Invalid file: {filename}")
        
        # Determine file format
        file_format = self._detect_format(filename, file_content)
        
        # Generate file hash for deduplication
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        # Process based on format
        if file_format.upper() == "DICOM":
            return self._process_dicom(file_content, filename, file_hash)
        elif file_format.upper() == "NIFTI":
            return self._process_nifti(file_content, filename, file_hash)
        elif file_format.upper() in ["JPEG", "PNG"]:
            return self._process_image(file_content, filename, file_hash)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    
    def _detect_format(self, filename: str, content: bytes) -> str:
        """Detect file format from filename and content."""
        filename_lower = filename.lower()
        
        # Check by extension first
        if any(ext in filename_lower for ext in ['.dcm', '.dicom']):
            return "DICOM"
        elif any(ext in filename_lower for ext in ['.nii', '.nii.gz']):
            return "NIfTI"
        elif any(ext in filename_lower for ext in ['.jpg', '.jpeg']):
            return "JPEG"
        elif '.png' in filename_lower:
            return "PNG"
        
        # Check by content (magic bytes)
        if content.startswith(b'\x00\x00\x01\x00') or content.startswith(b'DICM'):
            return "DICOM"
        elif content.startswith(b'\x93\x4E\x49\x49\x1A\x0A\x00\x00'):
            return "NIfTI"
        elif content.startswith(b'\xFF\xD8\xFF'):
            return "JPEG"
        elif content.startswith(b'\x89PNG\r\n\x1a\n'):
            return "PNG"
        
        raise ValueError(f"Unable to detect format for file: {filename}")
    
    def _process_dicom(self, content: bytes, filename: str, file_hash: str) -> Dict[str, Any]:
        """Process DICOM file."""
        try:
            # Read DICOM file
            dicom_file = pydicom.dcmread(io.BytesIO(content))
            
            # Extract and anonymize metadata
            metadata = self._extract_dicom_metadata(dicom_file)
            anonymized_metadata = anonymize_dicom_data(metadata)
            
            # Extract pixel data
            pixel_array = self._extract_pixel_data(dicom_file)
            
            # Calculate image properties
            image_properties = self._calculate_image_properties(pixel_array, dicom_file)
            
            return {
                "format": "DICOM",
                "filename": filename,
                "file_hash": file_hash,
                "file_size": len(content),
                "metadata": anonymized_metadata,
                "pixel_data": pixel_array,
                "image_properties": image_properties,
                "processing_status": "success"
            }
            
        except InvalidDicomError as e:
            logger.error(f"Invalid DICOM file {filename}: {e}")
            raise ValueError(f"Invalid DICOM file: {e}")
        except Exception as e:
            logger.error(f"Error processing DICOM file {filename}: {e}")
            raise ValueError(f"Error processing DICOM file: {e}")
    
    def _process_nifti(self, content: bytes, filename: str, file_hash: str) -> Dict[str, Any]:
        """Process NIfTI file."""
        try:
            # Read NIfTI file
            nifti_file = nib.load(io.BytesIO(content))
            
            # Extract metadata
            metadata = self._extract_nifti_metadata(nifti_file)
            
            # Extract image data
            image_data = nifti_file.get_fdata()
            
            # Calculate image properties
            image_properties = self._calculate_image_properties(image_data, nifti_file)
            
            return {
                "format": "NIfTI",
                "filename": filename,
                "file_hash": file_hash,
                "file_size": len(content),
                "metadata": metadata,
                "pixel_data": image_data,
                "image_properties": image_properties,
                "processing_status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error processing NIfTI file {filename}: {e}")
            raise ValueError(f"Error processing NIfTI file: {e}")
    
    def _process_image(self, content: bytes, filename: str, file_hash: str) -> Dict[str, Any]:
        """Process standard image file (JPEG, PNG)."""
        try:
            # Read image
            image = Image.open(io.BytesIO(content))
            
            # Convert to numpy array
            image_array = np.array(image)
            
            # Extract metadata
            metadata = {
                "format": image.format,
                "mode": image.mode,
                "size": image.size,
                "filename": filename
            }
            
            # Calculate image properties
            image_properties = self._calculate_image_properties(image_array, None)
            
            return {
                "format": image.format or "UNKNOWN",
                "filename": filename,
                "file_hash": file_hash,
                "file_size": len(content),
                "metadata": metadata,
                "pixel_data": image_array,
                "image_properties": image_properties,
                "processing_status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error processing image file {filename}: {e}")
            raise ValueError(f"Error processing image file: {e}")
    
    def _extract_dicom_metadata(self, dicom_file: Dataset) -> Dict[str, Any]:
        """Extract metadata from DICOM file."""
        metadata = {}
        
        # Common DICOM tags
        tags_to_extract = [
            "PatientName", "PatientID", "PatientBirthDate", "PatientSex",
            "StudyDate", "StudyTime", "StudyDescription", "StudyID",
            "SeriesDescription", "SeriesNumber", "InstanceNumber",
            "Modality", "Manufacturer", "ManufacturerModelName",
            "SliceThickness", "PixelSpacing", "ImagePositionPatient",
            "ImageOrientationPatient", "Rows", "Columns", "BitsAllocated",
            "BitsStored", "HighBit", "PhotometricInterpretation",
            "SamplesPerPixel", "PlanarConfiguration"
        ]
        
        for tag in tags_to_extract:
            if hasattr(dicom_file, tag):
                value = getattr(dicom_file, tag)
                # Convert to serializable format
                if hasattr(value, '__iter__') and not isinstance(value, str):
                    metadata[tag] = list(value)
                else:
                    metadata[tag] = str(value) if value is not None else None
        
        return metadata
    
    def _extract_nifti_metadata(self, nifti_file: nib.Nifti1Image) -> Dict[str, Any]:
        """Extract metadata from NIfTI file."""
        header = nifti_file.header
        affine = nifti_file.affine
        
        metadata = {
            "dimensions": list(header.get_data_shape()),
            "voxel_sizes": list(header.get_zooms()),
            "data_type": str(header.get_data_dtype()),
            "affine_matrix": affine.tolist(),
            "qform_code": int(header['qform_code']),
            "sform_code": int(header['sform_code']),
            "xyzt_units": int(header['xyzt_units']),
            "cal_min": float(header['cal_min']),
            "cal_max": float(header['cal_max']),
            "description": header['descrip'].tobytes().decode('utf-8', errors='ignore').strip('\x00')
        }
        
        return metadata
    
    def _extract_pixel_data(self, dicom_file: Dataset) -> np.ndarray:
        """Extract pixel data from DICOM file."""
        try:
            pixel_array = dicom_file.pixel_array
            
            # Handle different photometric interpretations
            if hasattr(dicom_file, 'PhotometricInterpretation'):
                if dicom_file.PhotometricInterpretation == 'MONOCHROME1':
                    # Invert if needed
                    pixel_array = np.max(pixel_array) - pixel_array
            
            return pixel_array
            
        except Exception as e:
            logger.error(f"Error extracting pixel data: {e}")
            raise ValueError(f"Error extracting pixel data: {e}")
    
    def _calculate_image_properties(self, image_data: np.ndarray, file_obj: Any) -> Dict[str, Any]:
        """Calculate image properties and quality metrics."""
        properties = {
            "shape": list(image_data.shape),
            "dtype": str(image_data.dtype),
            "min_value": float(np.min(image_data)),
            "max_value": float(np.max(image_data)),
            "mean_value": float(np.mean(image_data)),
            "std_value": float(np.std(image_data)),
            "size_bytes": image_data.nbytes
        }
        
        # Calculate quality metrics
        properties["quality_score"] = self._calculate_quality_score(image_data)
        
        # Calculate spatial properties if available
        if file_obj and hasattr(file_obj, 'PixelSpacing'):
            properties["pixel_spacing"] = list(file_obj.PixelSpacing)
        elif file_obj and hasattr(file_obj, 'get_zooms'):
            properties["voxel_sizes"] = list(file_obj.get_zooms())
        
        return properties
    
    def _calculate_quality_score(self, image_data: np.ndarray) -> float:
        """Calculate image quality score (0-1, higher is better)."""
        try:
            # Normalize image to 0-1 range
            normalized = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data) + 1e-8)
            
            # Calculate contrast (standard deviation)
            contrast = np.std(normalized)
            
            # Calculate sharpness (Laplacian variance)
            if len(image_data.shape) == 2:
                laplacian = cv2.Laplacian(normalized.astype(np.float32), cv2.CV_64F)
                sharpness = np.var(laplacian)
            else:
                # For 3D images, calculate sharpness for each slice and average
                sharpness_values = []
                for i in range(image_data.shape[0]):
                    slice_img = normalized[i].astype(np.float32)
                    laplacian = cv2.Laplacian(slice_img, cv2.CV_64F)
                    sharpness_values.append(np.var(laplacian))
                sharpness = np.mean(sharpness_values)
            
            # Combine metrics (normalize to 0-1)
            quality_score = min(1.0, (contrast * 2 + sharpness * 0.1) / 3)
            
            return float(quality_score)
            
        except Exception as e:
            logger.warning(f"Error calculating quality score: {e}")
            return 0.5  # Default moderate quality score
    
    def preprocess_for_model(self, processed_data: Dict[str, Any], model_requirements: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess image data for model inference.
        
        Args:
            processed_data: Output from process_upload
            model_requirements: Model-specific preprocessing requirements
            
        Returns:
            Preprocessed image array ready for model inference
        """
        image_data = processed_data["pixel_data"]
        
        # Apply standard preprocessing steps
        preprocessed = self._normalize_intensity(image_data)
        preprocessed = self._resize_to_target(preprocessed, model_requirements.get("input_size"))
        preprocessed = self._apply_augmentation(preprocessed, model_requirements.get("augmentation", False))
        
        return preprocessed
    
    def _normalize_intensity(self, image_data: np.ndarray) -> np.ndarray:
        """Normalize image intensity values."""
        # Z-score normalization
        mean = np.mean(image_data)
        std = np.std(image_data)
        
        if std > 0:
            normalized = (image_data - mean) / std
        else:
            normalized = image_data - mean
        
        return normalized
    
    def _resize_to_target(self, image_data: np.ndarray, target_size: Optional[List[int]]) -> np.ndarray:
        """Resize image to target dimensions."""
        if target_size is None:
            return image_data
        
        if len(image_data.shape) == 2:
            # 2D image
            resized = cv2.resize(image_data, (target_size[1], target_size[0]))
        elif len(image_data.shape) == 3:
            # 3D image
            resized = np.zeros((target_size[0], target_size[1], target_size[2]))
            for i in range(min(image_data.shape[0], target_size[0])):
                slice_resized = cv2.resize(image_data[i], (target_size[2], target_size[1]))
                resized[i] = slice_resized
        else:
            resized = image_data
        
        return resized
    
    def _apply_augmentation(self, image_data: np.ndarray, apply_augmentation: bool) -> np.ndarray:
        """Apply data augmentation if requested."""
        if not apply_augmentation:
            return image_data
        
        # Simple augmentation - random noise
        noise = np.random.normal(0, 0.01, image_data.shape)
        augmented = image_data + noise
        
        return augmented
