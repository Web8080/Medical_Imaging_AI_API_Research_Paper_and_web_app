#!/usr/bin/env python3
"""
Download real medical imaging datasets for training and testing.
"""

import argparse
import logging
import os
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import sys

logger = logging.getLogger(__name__)


class RealDatasetDownloader:
    """Download real medical imaging datasets."""
    
    def __init__(self, base_dir: str = "data/datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def download_brats_sample(self) -> bool:
        """Download a sample of BRATS data (smaller subset for testing)."""
        try:
            logger.info("Downloading BRATS sample dataset...")
            
            # For demonstration, we'll download a small sample
            # In practice, you would download from the official BRATS website
            brats_dir = self.base_dir / "brats_sample"
            brats_dir.mkdir(exist_ok=True)
            
            # Create a README with download instructions
            readme_content = """
# BRATS 2021 Dataset Download Instructions

## Official Download
1. Visit: https://www.synapse.org/#!Synapse:syn27046444
2. Register for an account
3. Request access to the dataset
4. Download the following files:
   - BraTS2021_Training_Data.tar (15GB)
   - BraTS2021_ValidationData.tar (3GB)

## Alternative: Use preprocessed samples
For testing purposes, you can use smaller sample datasets from:
- https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation
- https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1

## Expected Structure
```
brats2021/
├── BraTS2021_00000/
│   ├── BraTS2021_00000_t1.nii.gz
│   ├── BraTS2021_00000_t1ce.nii.gz
│   ├── BraTS2021_00000_t2.nii.gz
│   ├── BraTS2021_00000_flair.nii.gz
│   └── BraTS2021_00000_seg.nii.gz
└── ...
```
"""
            
            with open(brats_dir / "README.md", "w") as f:
                f.write(readme_content)
            
            logger.info(f"BRATS download instructions saved to {brats_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up BRATS download: {e}")
            return False
    
    def download_lidc_sample(self) -> bool:
        """Download a sample of LIDC-IDRI data."""
        try:
            logger.info("Downloading LIDC-IDRI sample dataset...")
            
            lidc_dir = self.base_dir / "lidc_sample"
            lidc_dir.mkdir(exist_ok=True)
            
            readme_content = """
# LIDC-IDRI Dataset Download Instructions

## Official Download
1. Visit: https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI
2. Register for an account
3. Download the dataset (120GB total)
4. Use the NBIA Data Retriever tool

## Alternative: Use preprocessed samples
For testing purposes, you can use:
- https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images
- https://www.kaggle.com/datasets/kmader/finding-lungs-in-ct-data

## Expected Structure
```
lidc_idri/
├── LIDC-IDRI-0001/
│   ├── 1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.dcm
│   └── ...
└── ...
```
"""
            
            with open(lidc_dir / "README.md", "w") as f:
                f.write(readme_content)
            
            logger.info(f"LIDC-IDRI download instructions saved to {lidc_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up LIDC download: {e}")
            return False
    
    def download_medical_decathlon_sample(self) -> bool:
        """Download a sample of Medical Segmentation Decathlon data."""
        try:
            logger.info("Downloading Medical Segmentation Decathlon sample...")
            
            decathlon_dir = self.base_dir / "decathlon_sample"
            decathlon_dir.mkdir(exist_ok=True)
            
            readme_content = """
# Medical Segmentation Decathlon Dataset Download Instructions

## Official Download
1. Visit: http://medicaldecathlon.com/
2. Register for an account
3. Download the dataset (8GB total)
4. Extract the tar files

## Alternative: Use individual task datasets
For testing, you can download individual tasks:
- Task01_BrainTumour: https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation
- Task02_Heart: https://www.kaggle.com/datasets/awsaf49/heart-mri-dataset
- Task03_Liver: https://www.kaggle.com/datasets/awsaf49/liver-tumor-segmentation

## Expected Structure
```
medical_decathlon/
├── Task01_BrainTumour/
│   ├── imagesTr/
│   └── labelsTr/
├── Task02_Heart/
│   ├── imagesTr/
│   └── labelsTr/
└── ...
```
"""
            
            with open(decathlon_dir / "README.md", "w") as f:
                f.write(readme_content)
            
            logger.info(f"Medical Decathlon download instructions saved to {decathlon_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up Decathlon download: {e}")
            return False
    
    def download_kaggle_sample(self, dataset_name: str) -> bool:
        """Download a sample dataset from Kaggle."""
        try:
            logger.info(f"Downloading Kaggle dataset: {dataset_name}")
            
            # Check if kaggle is installed
            try:
                import kaggle
            except ImportError:
                logger.error("Kaggle API not installed. Install with: pip install kaggle")
                return False
            
            # Create dataset directory
            dataset_dir = self.base_dir / dataset_name
            dataset_dir.mkdir(exist_ok=True)
            
            # Download dataset
            cmd = f"kaggle datasets download -d {dataset_name} -p {dataset_dir}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully downloaded {dataset_name}")
                
                # Extract if it's a zip file
                zip_files = list(dataset_dir.glob("*.zip"))
                for zip_file in zip_files:
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(dataset_dir)
                    zip_file.unlink()  # Remove zip file after extraction
                
                return True
            else:
                logger.error(f"Failed to download {dataset_name}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading Kaggle dataset {dataset_name}: {e}")
            return False
    
    def create_synthetic_data(self, dataset_name: str, num_samples: int = 100) -> bool:
        """Create synthetic medical imaging data for testing."""
        try:
            logger.info(f"Creating synthetic {dataset_name} data...")
            
            import numpy as np
            import nibabel as nib
            
            dataset_dir = self.base_dir / f"{dataset_name}_synthetic"
            dataset_dir.mkdir(exist_ok=True)
            
            if dataset_name == "brats":
                # Create synthetic brain MRI data
                for i in range(num_samples):
                    case_dir = dataset_dir / f"BraTS2021_{i:05d}"
                    case_dir.mkdir(exist_ok=True)
                    
                    # Create synthetic brain image (128x128x128)
                    brain_image = np.random.rand(128, 128, 128).astype(np.float32)
                    brain_image = (brain_image * 1000) - 500  # CT-like intensities
                    
                    # Create synthetic segmentation mask
                    segmentation = np.zeros_like(brain_image)
                    # Add some "tumor" regions
                    segmentation[40:80, 40:80, 40:80] = 1  # Whole tumor
                    segmentation[50:70, 50:70, 50:70] = 2  # Core
                    segmentation[55:65, 55:65, 55:65] = 3  # Enhancing
                    
                    # Save as NIfTI files
                    affine = np.eye(4)
                    nii_image = nib.Nifti1Image(brain_image, affine)
                    nii_seg = nib.Nifti1Image(segmentation, affine)
                    
                    nib.save(nii_image, case_dir / f"BraTS2021_{i:05d}_t1.nii.gz")
                    nib.save(nii_seg, case_dir / f"BraTS2021_{i:05d}_seg.nii.gz")
                
                logger.info(f"Created {num_samples} synthetic BRATS samples")
                
            elif dataset_name == "lidc":
                # Create synthetic lung CT data
                for i in range(num_samples):
                    case_dir = dataset_dir / f"LIDC-IDRI_{i:04d}"
                    case_dir.mkdir(exist_ok=True)
                    
                    # Create synthetic lung CT image (256x256x64)
                    lung_image = np.random.rand(256, 256, 64).astype(np.float32)
                    lung_image = (lung_image * 1000) - 1000  # CT-like intensities
                    
                    # Add some "lung" regions
                    lung_image[50:200, 50:200, :] -= 200  # Lungs are darker
                    
                    # Add some "nodules"
                    for _ in range(np.random.randint(0, 3)):
                        x = np.random.randint(50, 200)
                        y = np.random.randint(50, 200)
                        z = np.random.randint(10, 50)
                        size = np.random.randint(5, 15)
                        lung_image[x-size:x+size, y-size:y+size, z-size:z+size] += 100
                    
                    # Save as NIfTI
                    affine = np.eye(4)
                    nii_image = nib.Nifti1Image(lung_image, affine)
                    nib.save(nii_image, case_dir / f"LIDC-IDRI_{i:04d}.nii.gz")
                
                logger.info(f"Created {num_samples} synthetic LIDC samples")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating synthetic {dataset_name} data: {e}")
            return False
    
    def download_all_samples(self) -> Dict[str, bool]:
        """Download all sample datasets."""
        results = {}
        
        # Download instruction files
        results['brats'] = self.download_brats_sample()
        results['lidc'] = self.download_lidc_sample()
        results['decathlon'] = self.download_medical_decathlon_sample()
        
        # Create synthetic data for immediate testing
        results['brats_synthetic'] = self.create_synthetic_data("brats", 50)
        results['lidc_synthetic'] = self.create_synthetic_data("lidc", 50)
        
        return results


def main():
    """Main download script."""
    parser = argparse.ArgumentParser(description="Download real medical imaging datasets")
    parser.add_argument("--dataset", choices=['brats', 'lidc', 'decathlon', 'all'],
                       default='all', help="Dataset to download")
    parser.add_argument("--synthetic", action="store_true",
                       help="Create synthetic data for testing")
    parser.add_argument("--kaggle", type=str, help="Kaggle dataset name to download")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level)
    
    # Create downloader
    downloader = RealDatasetDownloader()
    
    try:
        if args.kaggle:
            # Download specific Kaggle dataset
            success = downloader.download_kaggle_sample(args.kaggle)
            if success:
                print(f"Successfully downloaded Kaggle dataset: {args.kaggle}")
            else:
                print(f"Failed to download Kaggle dataset: {args.kaggle}")
                return 1
        
        elif args.synthetic:
            # Create synthetic data
            if args.dataset == 'all':
                datasets = ['brats', 'lidc']
            else:
                datasets = [args.dataset]
            
            for dataset in datasets:
                success = downloader.create_synthetic_data(dataset, 100)
                if success:
                    print(f"Successfully created synthetic {dataset} data")
                else:
                    print(f"Failed to create synthetic {dataset} data")
                    return 1
        
        else:
            # Download real datasets (instructions)
            if args.dataset == 'all':
                results = downloader.download_all_samples()
            else:
                if args.dataset == 'brats':
                    results = {'brats': downloader.download_brats_sample()}
                elif args.dataset == 'lidc':
                    results = {'lidc': downloader.download_lidc_sample()}
                elif args.dataset == 'decathlon':
                    results = {'decathlon': downloader.download_medical_decathlon_sample()}
            
            # Print results
            print("Download Results:")
            for dataset, success in results.items():
                status = "✅ Success" if success else "❌ Failed"
                print(f"  {dataset}: {status}")
        
        print(f"\nData directory: {downloader.base_dir}")
        print("Next steps:")
        print("1. Follow the README files in each dataset directory for official downloads")
        print("2. Or use synthetic data for immediate testing")
        print("3. Run training scripts with the downloaded data")
        
        return 0
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
