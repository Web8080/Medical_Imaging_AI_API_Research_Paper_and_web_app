#!/usr/bin/env python3
"""
Download real medical imaging datasets from publicly available sources.
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
import json

logger = logging.getLogger(__name__)


class RealMedicalDataDownloader:
    """Download real medical imaging datasets."""
    
    def __init__(self, base_dir: str = "data/datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def download_brats_sample_from_kaggle(self) -> bool:
        """Download BRATS sample from Kaggle."""
        try:
            logger.info("Downloading BRATS sample from Kaggle...")
            
            # Use a publicly available BRATS dataset on Kaggle
            dataset_name = "awsaf49/brats20-dataset-training-validation"
            output_dir = self.base_dir / "brats_kaggle"
            
            cmd = f"kaggle datasets download -d {dataset_name} -p {output_dir} --unzip"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully downloaded BRATS dataset to {output_dir}")
                return True
            else:
                logger.error(f"Failed to download BRATS: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading BRATS: {e}")
            return False
    
    def download_lung_ct_sample(self) -> bool:
        """Download lung CT sample dataset."""
        try:
            logger.info("Downloading lung CT sample dataset...")
            
            # Use a publicly available lung CT dataset
            dataset_name = "mohamedhanyyy/chest-ctscan-images"
            output_dir = self.base_dir / "lung_ct_kaggle"
            
            cmd = f"kaggle datasets download -d {dataset_name} -p {output_dir} --unzip"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully downloaded lung CT dataset to {output_dir}")
                return True
            else:
                logger.error(f"Failed to download lung CT: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading lung CT: {e}")
            return False
    
    def download_heart_mri_sample(self) -> bool:
        """Download heart MRI sample dataset."""
        try:
            logger.info("Downloading heart MRI sample dataset...")
            
            # Use a publicly available heart MRI dataset
            dataset_name = "awsaf49/heart-mri-dataset"
            output_dir = self.base_dir / "heart_mri_kaggle"
            
            cmd = f"kaggle datasets download -d {dataset_name} -p {output_dir} --unzip"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully downloaded heart MRI dataset to {output_dir}")
                return True
            else:
                logger.error(f"Failed to download heart MRI: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading heart MRI: {e}")
            return False
    
    def download_liver_tumor_sample(self) -> bool:
        """Download liver tumor segmentation sample."""
        try:
            logger.info("Downloading liver tumor sample dataset...")
            
            # Use a publicly available liver tumor dataset
            dataset_name = "awsaf49/liver-tumor-segmentation"
            output_dir = self.base_dir / "liver_tumor_kaggle"
            
            cmd = f"kaggle datasets download -d {dataset_name} -p {output_dir} --unzip"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully downloaded liver tumor dataset to {output_dir}")
                return True
            else:
                logger.error(f"Failed to download liver tumor: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading liver tumor: {e}")
            return False
    
    def download_medical_image_net_sample(self) -> bool:
        """Download Medical ImageNet sample."""
        try:
            logger.info("Downloading Medical ImageNet sample...")
            
            # Use a publicly available medical imaging dataset
            dataset_name = "awsaf49/medical-imagenet"
            output_dir = self.base_dir / "medical_imagenet_kaggle"
            
            cmd = f"kaggle datasets download -d {dataset_name} -p {output_dir} --unzip"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully downloaded Medical ImageNet to {output_dir}")
                return True
            else:
                logger.error(f"Failed to download Medical ImageNet: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading Medical ImageNet: {e}")
            return False
    
    def download_chest_xray_sample(self) -> bool:
        """Download chest X-ray sample dataset."""
        try:
            logger.info("Downloading chest X-ray sample dataset...")
            
            # Use a publicly available chest X-ray dataset
            dataset_name = "paultimothymooney/chest-xray-pneumonia"
            output_dir = self.base_dir / "chest_xray_kaggle"
            
            cmd = f"kaggle datasets download -d {dataset_name} -p {output_dir} --unzip"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully downloaded chest X-ray dataset to {output_dir}")
                return True
            else:
                logger.error(f"Failed to download chest X-ray: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading chest X-ray: {e}")
            return False
    
    def download_all_available_datasets(self) -> Dict[str, bool]:
        """Download all available real medical imaging datasets."""
        results = {}
        
        # Download various medical imaging datasets
        results['brats'] = self.download_brats_sample_from_kaggle()
        results['lung_ct'] = self.download_lung_ct_sample()
        results['heart_mri'] = self.download_heart_mri_sample()
        results['liver_tumor'] = self.download_liver_tumor_sample()
        results['medical_imagenet'] = self.download_medical_image_net_sample()
        results['chest_xray'] = self.download_chest_xray_sample()
        
        return results
    
    def create_dataset_summary(self, results: Dict[str, bool]) -> Dict[str, any]:
        """Create a summary of downloaded datasets."""
        summary = {
            'download_results': results,
            'successful_downloads': [k for k, v in results.items() if v],
            'failed_downloads': [k for k, v in results.items() if not v],
            'total_datasets': len(results),
            'successful_count': sum(results.values()),
            'datasets_info': {}
        }
        
        # Add information about each dataset
        for dataset_name, success in results.items():
            if success:
                dataset_dir = self.base_dir / f"{dataset_name}_kaggle"
                if dataset_dir.exists():
                    files = list(dataset_dir.rglob("*"))
                    summary['datasets_info'][dataset_name] = {
                        'path': str(dataset_dir),
                        'file_count': len(files),
                        'size_mb': sum(f.stat().st_size for f in files if f.is_file()) / (1024*1024)
                    }
        
        return summary


def main():
    """Main download script."""
    parser = argparse.ArgumentParser(description="Download real medical imaging datasets")
    parser.add_argument("--dataset", choices=['brats', 'lung_ct', 'heart_mri', 'liver_tumor', 
                                            'medical_imagenet', 'chest_xray', 'all'],
                       default='all', help="Dataset to download")
    parser.add_argument("--output_dir", type=str, default="data/datasets",
                       help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level)
    
    # Create downloader
    downloader = RealMedicalDataDownloader(args.output_dir)
    
    try:
        if args.dataset == 'all':
            # Download all available datasets
            results = downloader.download_all_available_datasets()
        else:
            # Download specific dataset
            if args.dataset == 'brats':
                results = {'brats': downloader.download_brats_sample_from_kaggle()}
            elif args.dataset == 'lung_ct':
                results = {'lung_ct': downloader.download_lung_ct_sample()}
            elif args.dataset == 'heart_mri':
                results = {'heart_mri': downloader.download_heart_mri_sample()}
            elif args.dataset == 'liver_tumor':
                results = {'liver_tumor': downloader.download_liver_tumor_sample()}
            elif args.dataset == 'medical_imagenet':
                results = {'medical_imagenet': downloader.download_medical_image_net_sample()}
            elif args.dataset == 'chest_xray':
                results = {'chest_xray': downloader.download_chest_xray_sample()}
        
        # Create summary
        summary = downloader.create_dataset_summary(results)
        
        # Print results
        print("=" * 60)
        print("REAL MEDICAL IMAGING DATASETS DOWNLOAD RESULTS")
        print("=" * 60)
        
        for dataset, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"{dataset:20} : {status}")
        
        print(f"\nTotal datasets: {summary['total_datasets']}")
        print(f"Successful: {summary['successful_count']}")
        print(f"Failed: {summary['total_datasets'] - summary['successful_count']}")
        
        if summary['successful_downloads']:
            print(f"\nSuccessfully downloaded datasets:")
            for dataset in summary['successful_downloads']:
                info = summary['datasets_info'].get(dataset, {})
                print(f"  - {dataset}: {info.get('file_count', 0)} files, {info.get('size_mb', 0):.1f} MB")
        
        # Save summary
        summary_file = Path(args.output_dir) / "download_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nDownload summary saved to: {summary_file}")
        print(f"Data directory: {args.output_dir}")
        
        if summary['successful_count'] > 0:
            print("\n🎉 Ready for training! Next steps:")
            print("1. Run preprocessing on the downloaded data")
            print("2. Start model training with real data")
            print("3. Test the API with trained models")
        else:
            print("\n⚠️  No datasets were successfully downloaded.")
            print("Please check your internet connection and Kaggle API setup.")
        
        return 0 if summary['successful_count'] > 0 else 1
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
