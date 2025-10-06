"""
Phase 3: Data Preprocessing Pipeline
Goal: Standardize data for model readiness with de-identification and normalization.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError
import nibabel as nib
from scipy import ndimage
from scipy.ndimage import zoom
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

# Medical imaging libraries
try:
    import monai
    from monai.transforms import (
        Compose, LoadImaged, AddChanneld, Spacingd, Orientationd,
        ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
        RandRotate90d, RandFlipd, RandShiftIntensityd, RandGaussianNoised,
        ToTensord, EnsureChannelFirstd, Resized, NormalizeIntensityd
    )
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    warnings.warn("MONAI not available. Some advanced preprocessing features will be limited.")

try:
    import torchio as tio
    TORCHIO_AVAILABLE = True
except ImportError:
    TORCHIO_AVAILABLE = False
    warnings.warn("TorchIO not available. Some advanced preprocessing features will be limited.")

logger = logging.getLogger(__name__)


class MedicalDataPreprocessor:
    """
    Comprehensive medical data preprocessing pipeline following Phase 3 requirements.
    
    Features:
    - De-identification (HIPAA/GDPR compliance)
    - Resampling & normalization
    - Data augmentation
    - Dataset splitting
    - Quality control
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.anonymization_tags = self._get_anonymization_tags()
        
    def _get_default_config(self) -> Dict:
        """Get default preprocessing configuration."""
        return {
            # De-identification
            'anonymize': True,
            'remove_private_tags': True,
            
            # Resampling & normalization
            'target_spacing': [1.0, 1.0, 1.0],  # mm
            'target_size': [128, 128, 128],  # voxels
            'normalization_method': 'z_score',  # 'z_score', 'min_max', 'percentile'
            'intensity_range': [0.0, 1.0],
            
            # Augmentation
            'enable_augmentation': True,
            'augmentation_prob': 0.5,
            'rotation_range': [-15, 15],  # degrees
            'flip_prob': 0.5,
            'noise_std': 0.01,
            'elastic_deformation': True,
            
            # Dataset splitting
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'stratify_by': 'tumor_size',  # 'tumor_size', 'tumor_type', None
            
            # Quality control
            'min_volume': 100,  # minimum voxels
            'max_volume': 1000000,  # maximum voxels
            'quality_threshold': 0.5,  # minimum quality score
        }
    
    def _get_anonymization_tags(self) -> List[str]:
        """Get list of DICOM tags to anonymize for HIPAA/GDPR compliance."""
        return [
            # Patient identification
            'PatientName', 'PatientID', 'PatientBirthDate', 'PatientSex',
            'PatientAge', 'PatientWeight', 'PatientSize', 'PatientAddress',
            'PatientTelephoneNumbers', 'PatientMotherBirthName',
            
            # Study information
            'StudyDate', 'StudyTime', 'StudyDescription', 'StudyID',
            'AccessionNumber', 'ReferringPhysicianName', 'PerformingPhysicianName',
            'StudyComments', 'StudyPriorityID',
            
            # Series information
            'SeriesDescription', 'SeriesNumber', 'SeriesDate', 'SeriesTime',
            'SeriesComments', 'SeriesPerformingPhysicianName',
            
            # Institution information
            'InstitutionName', 'InstitutionAddress', 'InstitutionDepartmentName',
            'StationName', 'Manufacturer', 'ManufacturerModelName',
            'DeviceSerialNumber', 'SoftwareVersions',
            
            # Operator information
            'OperatorName', 'PerformingPhysicianName', 'PhysicianOfRecord',
            
            # Private tags (remove all)
            'PrivateCreator', 'PrivateTagData',
        ]
    
    def deidentify_dicom(self, dicom_path: Union[str, Path], 
                        output_path: Optional[Union[str, Path]] = None) -> Path:
        """
        De-identify DICOM file for HIPAA/GDPR compliance.
        
        Args:
            dicom_path: Path to input DICOM file
            output_path: Path to save anonymized DICOM (optional)
            
        Returns:
            Path to anonymized DICOM file
        """
        try:
            # Load DICOM file
            ds = pydicom.dcmread(dicom_path)
            
            # Create anonymized copy
            anonymized_ds = ds.copy()
            
            # Remove or anonymize specified tags
            for tag in self.anonymization_tags:
                if hasattr(anonymized_ds, tag):
                    if tag in ['PatientID', 'StudyID', 'AccessionNumber']:
                        # Replace with anonymized ID
                        anonymized_ds[tag].value = f"ANON_{np.random.randint(100000, 999999)}"
                    elif tag in ['StudyDate', 'PatientBirthDate']:
                        # Replace with anonymized date (keep year for age calculation)
                        original_date = getattr(anonymized_ds, tag).value
                        if original_date and len(str(original_date)) >= 4:
                            anonymized_ds[tag].value = f"1900{str(original_date)[4:]}"
                    else:
                        # Remove or replace with generic value
                        anonymized_ds[tag].value = "ANONYMIZED"
            
            # Remove private tags
            if self.config['remove_private_tags']:
                anonymized_ds.remove_private_tags()
            
            # Set anonymization flag
            anonymized_ds.add_new(0x00120010, 'CS', 'YES')  # Patient Identity Removed
            anonymized_ds.add_new(0x00120062, 'CS', 'YES')  # De-identification Method
            
            # Save anonymized file
            if output_path is None:
                output_path = dicom_path.parent / f"anonymized_{dicom_path.name}"
            
            anonymized_ds.save_as(output_path)
            
            logger.info(f"DICOM file anonymized: {output_path}")
            return Path(output_path)
            
        except InvalidDicomError as e:
            logger.error(f"Invalid DICOM file {dicom_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error anonymizing DICOM file {dicom_path}: {e}")
            raise
    
    def resample_image(self, image: np.ndarray, 
                      original_spacing: List[float],
                      target_spacing: List[float]) -> np.ndarray:
        """
        Resample image to target voxel spacing.
        
        Args:
            image: Input image array
            original_spacing: Original voxel spacing [x, y, z]
            target_spacing: Target voxel spacing [x, y, z]
            
        Returns:
            Resampled image array
        """
        try:
            # Calculate zoom factors
            zoom_factors = [orig / target for orig, target in zip(original_spacing, target_spacing)]
            
            # Resample using scipy
            resampled = zoom(image, zoom_factors, order=1, mode='nearest')
            
            logger.debug(f"Resampled image from {image.shape} to {resampled.shape}")
            return resampled
            
        except Exception as e:
            logger.error(f"Error resampling image: {e}")
            raise
    
    def normalize_intensity(self, image: np.ndarray, 
                          method: str = 'z_score') -> np.ndarray:
        """
        Normalize image intensity values.
        
        Args:
            image: Input image array
            method: Normalization method ('z_score', 'min_max', 'percentile')
            
        Returns:
            Normalized image array
        """
        try:
            if method == 'z_score':
                # Z-score normalization
                mean = np.mean(image)
                std = np.std(image)
                if std > 0:
                    normalized = (image - mean) / std
                else:
                    normalized = image - mean
                    
            elif method == 'min_max':
                # Min-max normalization
                min_val = np.min(image)
                max_val = np.max(image)
                if max_val > min_val:
                    normalized = (image - min_val) / (max_val - min_val)
                else:
                    normalized = np.zeros_like(image)
                    
            elif method == 'percentile':
                # Percentile normalization (robust to outliers)
                p2, p98 = np.percentile(image, [2, 98])
                if p98 > p2:
                    normalized = np.clip((image - p2) / (p98 - p2), 0, 1)
                else:
                    normalized = np.zeros_like(image)
                    
            else:
                raise ValueError(f"Unknown normalization method: {method}")
            
            logger.debug(f"Normalized image using {method} method")
            return normalized.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error normalizing image: {e}")
            raise
    
    def create_augmentation_pipeline(self) -> Optional[Compose]:
        """
        Create data augmentation pipeline using MONAI.
        
        Returns:
            MONAI Compose transform pipeline
        """
        if not MONAI_AVAILABLE:
            logger.warning("MONAI not available. Skipping advanced augmentation.")
            return None
        
        if not self.config['enable_augmentation']:
            return None
        
        augmentation_transforms = [
            # Geometric augmentations
            RandRotate90d(keys=['image', 'label'], prob=self.config['augmentation_prob']),
            RandFlipd(keys=['image', 'label'], prob=self.config['flip_prob']),
            
            # Intensity augmentations
            RandShiftIntensityd(
                keys=['image'], 
                offsets=0.1, 
                prob=self.config['augmentation_prob']
            ),
            RandGaussianNoised(
                keys=['image'], 
                std=self.config['noise_std'], 
                prob=self.config['augmentation_prob']
            ),
        ]
        
        return Compose(augmentation_transforms)
    
    def create_preprocessing_pipeline(self) -> Compose:
        """
        Create preprocessing pipeline using MONAI.
        
        Returns:
            MONAI Compose transform pipeline
        """
        if not MONAI_AVAILABLE:
            logger.warning("MONAI not available. Using basic preprocessing.")
            return None
        
        preprocessing_transforms = [
            # Load and add channel dimension
            LoadImaged(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),
            EnsureChannelFirstd(keys=['image', 'label']),
            
            # Orientation and spacing
            Orientationd(keys=['image', 'label'], axcodes='RAS'),
            Spacingd(
                keys=['image', 'label'], 
                pixdim=self.config['target_spacing'], 
                mode=['bilinear', 'nearest']
            ),
            
            # Intensity normalization
            ScaleIntensityRanged(
                keys=['image'],
                a_min=-1000, a_max=1000,  # CT window
                b_min=0.0, b_max=1.0,
                clip=True
            ),
            
            # Resize to target size
            Resized(
                keys=['image', 'label'],
                spatial_size=self.config['target_size'],
                mode=['bilinear', 'nearest']
            ),
            
            # Convert to tensor
            ToTensord(keys=['image', 'label']),
        ]
        
        return Compose(preprocessing_transforms)
    
    def split_dataset(self, data_list: List[Dict], 
                     stratify_key: Optional[str] = None) -> Tuple[List, List, List]:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            data_list: List of data dictionaries
            stratify_key: Key to stratify by (optional)
            
        Returns:
            Tuple of (train_list, val_list, test_list)
        """
        try:
            from sklearn.model_selection import train_test_split
            
            if stratify_key and stratify_key in data_list[0]:
                # Stratified split
                stratify_values = [item[stratify_key] for item in data_list]
                
                # First split: train vs (val + test)
                train_list, temp_list, train_strat, temp_strat = train_test_split(
                    data_list, stratify_values,
                    test_size=(self.config['val_ratio'] + self.config['test_ratio']),
                    stratify=stratify_values,
                    random_state=42
                )
                
                # Second split: val vs test
                val_ratio = self.config['val_ratio'] / (self.config['val_ratio'] + self.config['test_ratio'])
                val_list, test_list, _, _ = train_test_split(
                    temp_list, temp_strat,
                    test_size=(1 - val_ratio),
                    stratify=temp_strat,
                    random_state=42
                )
            else:
                # Random split
                train_list, temp_list = train_test_split(
                    data_list,
                    test_size=(self.config['val_ratio'] + self.config['test_ratio']),
                    random_state=42
                )
                
                val_ratio = self.config['val_ratio'] / (self.config['val_ratio'] + self.config['test_ratio'])
                val_list, test_list = train_test_split(
                    temp_list,
                    test_size=(1 - val_ratio),
                    random_state=42
                )
            
            logger.info(f"Dataset split: Train={len(train_list)}, Val={len(val_list)}, Test={len(test_list)}")
            return train_list, val_list, test_list
            
        except Exception as e:
            logger.error(f"Error splitting dataset: {e}")
            raise
    
    def quality_control(self, image: np.ndarray, 
                       label: Optional[np.ndarray] = None) -> Dict[str, any]:
        """
        Perform quality control on image and label.
        
        Args:
            image: Input image array
            label: Input label array (optional)
            
        Returns:
            Dictionary with quality metrics and pass/fail status
        """
        try:
            quality_metrics = {}
            
            # Image quality metrics
            quality_metrics['image_shape'] = image.shape
            quality_metrics['image_min'] = float(np.min(image))
            quality_metrics['image_max'] = float(np.max(image))
            quality_metrics['image_mean'] = float(np.mean(image))
            quality_metrics['image_std'] = float(np.std(image))
            
            # Check for empty or constant images
            quality_metrics['is_empty'] = np.all(image == 0)
            quality_metrics['is_constant'] = np.std(image) == 0
            
            # Calculate image quality score (simple metric)
            if not quality_metrics['is_constant']:
                # Contrast-based quality score
                contrast = np.std(image) / (np.mean(image) + 1e-8)
                quality_metrics['contrast_score'] = float(contrast)
            else:
                quality_metrics['contrast_score'] = 0.0
            
            # Label quality metrics (if provided)
            if label is not None:
                quality_metrics['label_shape'] = label.shape
                quality_metrics['label_volume'] = int(np.sum(label > 0))
                quality_metrics['label_unique_values'] = int(len(np.unique(label)))
                
                # Check volume constraints
                quality_metrics['volume_valid'] = (
                    self.config['min_volume'] <= quality_metrics['label_volume'] <= self.config['max_volume']
                )
            else:
                quality_metrics['volume_valid'] = True
            
            # Overall quality assessment
            quality_metrics['quality_score'] = quality_metrics['contrast_score']
            quality_metrics['passes_quality'] = (
                not quality_metrics['is_empty'] and
                not quality_metrics['is_constant'] and
                quality_metrics['volume_valid'] and
                quality_metrics['quality_score'] >= self.config['quality_threshold']
            )
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error in quality control: {e}")
            return {'passes_quality': False, 'error': str(e)}
    
    def preprocess_dataset(self, input_dir: Path, output_dir: Path,
                          dataset_type: str = 'brats') -> Dict[str, any]:
        """
        Complete preprocessing pipeline for a dataset.
        
        Args:
            input_dir: Input directory containing raw data
            output_dir: Output directory for processed data
            dataset_type: Type of dataset ('brats', 'lidc', 'decathlon')
            
        Returns:
            Dictionary with preprocessing results and statistics
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create preprocessing pipeline
            preprocessing_pipeline = self.create_preprocessing_pipeline()
            augmentation_pipeline = self.create_augmentation_pipeline()
            
            # Process files
            processed_files = []
            quality_reports = []
            
            # Find all data files
            if dataset_type == 'brats':
                data_files = self._find_brats_files(input_dir)
            elif dataset_type == 'lidc':
                data_files = self._find_lidc_files(input_dir)
            elif dataset_type == 'decathlon':
                data_files = self._find_decathlon_files(input_dir)
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")
            
            logger.info(f"Found {len(data_files)} files to process")
            
            for i, data_file in enumerate(data_files):
                try:
                    logger.info(f"Processing file {i+1}/{len(data_files)}: {data_file['image_path']}")
                    
                    # Load data
                    if data_file['format'] == 'dicom':
                        image = self._load_dicom_image(data_file['image_path'])
                    elif data_file['format'] == 'nifti':
                        image = self._load_nifti_image(data_file['image_path'])
                    else:
                        logger.warning(f"Unsupported format: {data_file['format']}")
                        continue
                    
                    # Load label if available
                    label = None
                    if data_file.get('label_path'):
                        if data_file['format'] == 'dicom':
                            label = self._load_dicom_image(data_file['label_path'])
                        elif data_file['format'] == 'nifti':
                            label = self._load_nifti_image(data_file['label_path'])
                    
                    # Quality control
                    quality_report = self.quality_control(image, label)
                    quality_reports.append(quality_report)
                    
                    if not quality_report['passes_quality']:
                        logger.warning(f"File failed quality control: {data_file['image_path']}")
                        continue
                    
                    # Preprocessing
                    if preprocessing_pipeline:
                        # Use MONAI pipeline
                        data_dict = {'image': image}
                        if label is not None:
                            data_dict['label'] = label
                        
                        processed_data = preprocessing_pipeline(data_dict)
                        processed_image = processed_data['image'].numpy()
                        processed_label = processed_data.get('label', label)
                    else:
                        # Basic preprocessing
                        processed_image = self.normalize_intensity(image)
                        processed_label = label
                    
                    # Save processed data
                    output_file = self._save_processed_data(
                        processed_image, processed_label, 
                        data_file, output_dir, i
                    )
                    
                    processed_files.append(output_file)
                    
                except Exception as e:
                    logger.error(f"Error processing file {data_file['image_path']}: {e}")
                    continue
            
            # Split dataset
            train_list, val_list, test_list = self.split_dataset(processed_files)
            
            # Save split information
            split_info = {
                'train': train_list,
                'val': val_list,
                'test': test_list
            }
            
            with open(output_dir / 'dataset_split.json', 'w') as f:
                import json
                json.dump(split_info, f, indent=2)
            
            # Generate summary report
            summary = {
                'total_files_processed': len(processed_files),
                'total_files_found': len(data_files),
                'quality_reports': quality_reports,
                'dataset_split': {
                    'train': len(train_list),
                    'val': len(val_list),
                    'test': len(test_list)
                },
                'preprocessing_config': self.config
            }
            
            with open(output_dir / 'preprocessing_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Preprocessing completed. Processed {len(processed_files)} files.")
            logger.info(f"Dataset split: Train={len(train_list)}, Val={len(val_list)}, Test={len(test_list)}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {e}")
            raise
    
    def _find_brats_files(self, input_dir: Path) -> List[Dict]:
        """Find BRATS dataset files."""
        files = []
        
        # Look for BRATS structure
        for case_dir in input_dir.glob('*'):
            if case_dir.is_dir():
                # Find image files
                image_files = list(case_dir.glob('*_t1.nii.gz')) + list(case_dir.glob('*_t1ce.nii.gz'))
                label_files = list(case_dir.glob('*_seg.nii.gz'))
                
                for image_file in image_files:
                    files.append({
                        'image_path': image_file,
                        'label_path': label_files[0] if label_files else None,
                        'format': 'nifti',
                        'case_id': case_dir.name
                    })
        
        return files
    
    def _find_lidc_files(self, input_dir: Path) -> List[Dict]:
        """Find LIDC-IDRI dataset files."""
        files = []
        
        # Look for LIDC structure
        for case_dir in input_dir.glob('*'):
            if case_dir.is_dir():
                # Find DICOM files
                dicom_files = list(case_dir.rglob('*.dcm'))
                
                for dicom_file in dicom_files:
                    files.append({
                        'image_path': dicom_file,
                        'label_path': None,  # LIDC has separate annotation files
                        'format': 'dicom',
                        'case_id': case_dir.name
                    })
        
        return files
    
    def _find_decathlon_files(self, input_dir: Path) -> List[Dict]:
        """Find Medical Segmentation Decathlon files."""
        files = []
        
        # Look for Decathlon structure
        for task_dir in input_dir.glob('Task*'):
            if task_dir.is_dir():
                images_dir = task_dir / 'imagesTr'
                labels_dir = task_dir / 'labelsTr'
                
                if images_dir.exists() and labels_dir.exists():
                    for image_file in images_dir.glob('*.nii.gz'):
                        label_file = labels_dir / image_file.name
                        if label_file.exists():
                            files.append({
                                'image_path': image_file,
                                'label_path': label_file,
                                'format': 'nifti',
                                'case_id': image_file.stem,
                                'task': task_dir.name
                            })
        
        return files
    
    def _load_dicom_image(self, dicom_path: Path) -> np.ndarray:
        """Load DICOM image."""
        try:
            ds = pydicom.dcmread(dicom_path)
            return ds.pixel_array.astype(np.float32)
        except Exception as e:
            logger.error(f"Error loading DICOM {dicom_path}: {e}")
            raise
    
    def _load_nifti_image(self, nifti_path: Path) -> np.ndarray:
        """Load NIfTI image."""
        try:
            nii = nib.load(nifti_path)
            return nii.get_fdata().astype(np.float32)
        except Exception as e:
            logger.error(f"Error loading NIfTI {nifti_path}: {e}")
            raise
    
    def _save_processed_data(self, image: np.ndarray, label: Optional[np.ndarray],
                           data_file: Dict, output_dir: Path, index: int) -> Dict:
        """Save processed data."""
        try:
            # Create output filename
            case_id = data_file.get('case_id', f'case_{index:04d}')
            output_file = output_dir / f'{case_id}_processed.npz'
            
            # Save as compressed numpy array
            save_dict = {'image': image}
            if label is not None:
                save_dict['label'] = label
            
            np.savez_compressed(output_file, **save_dict)
            
            return {
                'output_path': str(output_file),
                'case_id': case_id,
                'original_path': str(data_file['image_path']),
                'image_shape': image.shape,
                'label_shape': label.shape if label is not None else None
            }
            
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise


def main():
    """Command line interface for preprocessing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Medical Data Preprocessing Pipeline")
    parser.add_argument("--input_dir", type=Path, required=True, help="Input directory")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--dataset_type", choices=['brats', 'lidc', 'decathlon'], 
                       required=True, help="Dataset type")
    parser.add_argument("--config", type=Path, help="Configuration file")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level)
    
    # Load configuration
    config = {}
    if args.config and args.config.exists():
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Create preprocessor
    preprocessor = MedicalDataPreprocessor(config)
    
    # Run preprocessing
    try:
        results = preprocessor.preprocess_dataset(
            args.input_dir, args.output_dir, args.dataset_type
        )
        
        print(f"Preprocessing completed successfully!")
        print(f"Processed {results['total_files_processed']} files")
        print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
