"""
Dataset downloader for medical imaging datasets.
"""

import os
import requests
import zipfile
import tarfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
import hashlib

logger = logging.getLogger(__name__)


class MedicalDatasetDownloader:
    """Downloader for medical imaging datasets."""
    
    def __init__(self, base_dir: str = "data/datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset configurations
        self.datasets = {
            "brats2021": {
                "name": "BRATS 2021",
                "description": "Brain Tumor Segmentation Challenge 2021",
                "url": "https://www.synapse.org/#!Synapse:syn27046444",
                "download_type": "manual",  # Requires registration
                "local_path": "brats2021",
                "expected_size_gb": 15.0,
                "files": [
                    "BraTS2021_Training_Data.tar",
                    "BraTS2021_ValidationData.tar"
                ]
            },
            "lidc_idri": {
                "name": "LIDC-IDRI",
                "description": "Lung Image Database Consortium and Image Database Resource Initiative",
                "url": "https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI",
                "download_type": "manual",  # Requires registration
                "local_path": "lidc_idri",
                "expected_size_gb": 120.0,
                "files": [
                    "LIDC-IDRI-0001.tar.gz",
                    "LIDC-IDRI-0002.tar.gz",
                    # ... more files
                ]
            },
            "medical_decathlon": {
                "name": "Medical Segmentation Decathlon",
                "description": "Multi-organ segmentation dataset",
                "url": "http://medicaldecathlon.com/",
                "download_type": "manual",  # Requires registration
                "local_path": "medical_decathlon",
                "expected_size_gb": 8.0,
                "files": [
                    "Task01_BrainTumour.tar",
                    "Task02_Heart.tar",
                    "Task03_Liver.tar",
                    "Task04_Hippocampus.tar",
                    "Task05_Prostate.tar",
                    "Task06_Lung.tar",
                    "Task07_Pancreas.tar",
                    "Task08_HepaticVessel.tar",
                    "Task09_Spleen.tar",
                    "Task10_Colon.tar"
                ]
            },
            "tcia": {
                "name": "TCIA",
                "description": "The Cancer Imaging Archive",
                "url": "https://www.cancerimagingarchive.net/",
                "download_type": "api",  # Can use TCIA API
                "local_path": "tcia",
                "expected_size_gb": 500.0,  # Varies by collection
                "collections": [
                    "TCGA-GBM",
                    "TCGA-LGG", 
                    "TCGA-LUAD",
                    "TCGA-LUSC",
                    "TCGA-BRCA"
                ]
            }
        }
    
    def download_dataset(self, dataset_name: str, force_download: bool = False) -> bool:
        """
        Download a medical imaging dataset.
        
        Args:
            dataset_name: Name of the dataset to download
            force_download: Force download even if already exists
            
        Returns:
            True if download successful
        """
        if dataset_name not in self.datasets:
            logger.error(f"Unknown dataset: {dataset_name}")
            return False
        
        dataset_config = self.datasets[dataset_name]
        dataset_path = self.base_dir / dataset_config["local_path"]
        
        # Check if already downloaded
        if dataset_path.exists() and not force_download:
            logger.info(f"Dataset {dataset_name} already exists at {dataset_path}")
            return True
        
        logger.info(f"Downloading {dataset_config['name']}...")
        
        if dataset_config["download_type"] == "manual":
            return self._download_manual_dataset(dataset_name, dataset_config, dataset_path)
        elif dataset_config["download_type"] == "api":
            return self._download_api_dataset(dataset_name, dataset_config, dataset_path)
        else:
            logger.error(f"Unsupported download type: {dataset_config['download_type']}")
            return False
    
    def _download_manual_dataset(self, dataset_name: str, config: Dict, dataset_path: Path) -> bool:
        """Handle manual download datasets (require registration)."""
        logger.info(f"Manual download required for {config['name']}")
        logger.info(f"Please visit: {config['url']}")
        logger.info(f"Download files to: {dataset_path}")
        logger.info(f"Expected files: {config['files']}")
        
        # Create directory structure
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Create download instructions
        instructions_file = dataset_path / "DOWNLOAD_INSTRUCTIONS.txt"
        with open(instructions_file, 'w') as f:
            f.write(f"Download Instructions for {config['name']}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Description: {config['description']}\n")
            f.write(f"URL: {config['url']}\n")
            f.write(f"Expected Size: {config['expected_size_gb']} GB\n\n")
            f.write("Steps:\n")
            f.write("1. Visit the URL above\n")
            f.write("2. Register/login if required\n")
            f.write("3. Download the following files:\n")
            for file in config['files']:
                f.write(f"   - {file}\n")
            f.write(f"4. Place downloaded files in: {dataset_path}\n")
            f.write("5. Run the extraction script\n\n")
            f.write("After downloading, run:\n")
            f.write(f"python -m src.data.dataset_downloader extract {dataset_name}\n")
        
        logger.info(f"Download instructions saved to: {instructions_file}")
        return True
    
    def _download_api_dataset(self, dataset_name: str, config: Dict, dataset_path: Path) -> bool:
        """Handle API-based downloads (like TCIA)."""
        if dataset_name == "tcia":
            return self._download_tcia_dataset(config, dataset_path)
        else:
            logger.error(f"API download not implemented for {dataset_name}")
            return False
    
    def _download_tcia_dataset(self, config: Dict, dataset_path: Path) -> bool:
        """Download TCIA dataset using their API."""
        try:
            from tcia_utils import nbia
            
            dataset_path.mkdir(parents=True, exist_ok=True)
            
            logger.info("Downloading TCIA datasets...")
            
            for collection in config['collections']:
                collection_path = dataset_path / collection
                collection_path.mkdir(exist_ok=True)
                
                logger.info(f"Downloading collection: {collection}")
                
                # Download collection metadata
                try:
                    nbia.getCollectionValues(collection)
                    logger.info(f"Successfully accessed collection: {collection}")
                except Exception as e:
                    logger.warning(f"Could not access collection {collection}: {e}")
                    continue
            
            logger.info("TCIA dataset setup completed")
            return True
            
        except ImportError:
            logger.error("tcia_utils not installed. Install with: pip install tcia-utils")
            return False
        except Exception as e:
            logger.error(f"Error downloading TCIA dataset: {e}")
            return False
    
    def extract_dataset(self, dataset_name: str) -> bool:
        """
        Extract downloaded dataset files.
        
        Args:
            dataset_name: Name of the dataset to extract
            
        Returns:
            True if extraction successful
        """
        if dataset_name not in self.datasets:
            logger.error(f"Unknown dataset: {dataset_name}")
            return False
        
        dataset_config = self.datasets[dataset_name]
        dataset_path = self.base_dir / dataset_config["local_path"]
        
        if not dataset_path.exists():
            logger.error(f"Dataset path does not exist: {dataset_path}")
            return False
        
        logger.info(f"Extracting {dataset_config['name']}...")
        
        extracted_files = []
        
        for file_name in dataset_config.get("files", []):
            file_path = dataset_path / file_name
            
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue
            
            logger.info(f"Extracting {file_name}...")
            
            try:
                if file_name.endswith('.tar'):
                    with tarfile.open(file_path, 'r') as tar:
                        tar.extractall(dataset_path)
                elif file_name.endswith('.tar.gz'):
                    with tarfile.open(file_path, 'r:gz') as tar:
                        tar.extractall(dataset_path)
                elif file_name.endswith('.zip'):
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(dataset_path)
                else:
                    logger.warning(f"Unknown file format: {file_name}")
                    continue
                
                extracted_files.append(file_name)
                logger.info(f"Successfully extracted {file_name}")
                
            except Exception as e:
                logger.error(f"Error extracting {file_name}: {e}")
                return False
        
        logger.info(f"Extraction completed. Extracted {len(extracted_files)} files.")
        return True
    
    def verify_dataset(self, dataset_name: str) -> Dict[str, any]:
        """
        Verify dataset integrity and structure.
        
        Args:
            dataset_name: Name of the dataset to verify
            
        Returns:
            Dictionary with verification results
        """
        if dataset_name not in self.datasets:
            return {"error": f"Unknown dataset: {dataset_name}"}
        
        dataset_config = self.datasets[dataset_name]
        dataset_path = self.base_dir / dataset_config["local_path"]
        
        if not dataset_path.exists():
            return {"error": f"Dataset path does not exist: {dataset_path}"}
        
        verification_results = {
            "dataset_name": dataset_name,
            "dataset_path": str(dataset_path),
            "exists": True,
            "files_found": [],
            "files_missing": [],
            "total_size_gb": 0.0,
            "structure_valid": False
        }
        
        # Check for expected files
        for file_name in dataset_config.get("files", []):
            file_path = dataset_path / file_name
            if file_path.exists():
                verification_results["files_found"].append(file_name)
                verification_results["total_size_gb"] += file_path.stat().st_size / (1024**3)
            else:
                verification_results["files_missing"].append(file_name)
        
        # Check for extracted content
        extracted_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
        if extracted_dirs:
            verification_results["structure_valid"] = True
            verification_results["extracted_directories"] = [d.name for d in extracted_dirs]
        
        return verification_results
    
    def list_available_datasets(self) -> List[Dict[str, str]]:
        """List all available datasets."""
        return [
            {
                "name": name,
                "display_name": config["name"],
                "description": config["description"],
                "url": config["url"],
                "download_type": config["download_type"],
                "expected_size_gb": config["expected_size_gb"]
            }
            for name, config in self.datasets.items()
        ]
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, any]:
        """Get detailed information about a dataset."""
        if dataset_name not in self.datasets:
            return {"error": f"Unknown dataset: {dataset_name}"}
        
        config = self.datasets[dataset_name]
        dataset_path = self.base_dir / config["local_path"]
        
        info = {
            "name": config["name"],
            "description": config["description"],
            "url": config["url"],
            "download_type": config["download_type"],
            "expected_size_gb": config["expected_size_gb"],
            "local_path": str(dataset_path),
            "exists": dataset_path.exists(),
            "verification": self.verify_dataset(dataset_name)
        }
        
        return info


def main():
    """Command line interface for dataset downloader."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Medical Dataset Downloader")
    parser.add_argument("action", choices=["list", "download", "extract", "verify", "info"],
                       help="Action to perform")
    parser.add_argument("dataset", nargs="?", help="Dataset name")
    parser.add_argument("--force", action="store_true", help="Force download")
    
    args = parser.parse_args()
    
    downloader = MedicalDatasetDownloader()
    
    if args.action == "list":
        datasets = downloader.list_available_datasets()
        print("Available Datasets:")
        print("=" * 50)
        for dataset in datasets:
            print(f"Name: {dataset['name']}")
            print(f"Display: {dataset['display_name']}")
            print(f"Description: {dataset['description']}")
            print(f"Size: {dataset['expected_size_gb']} GB")
            print(f"Type: {dataset['download_type']}")
            print("-" * 30)
    
    elif args.action == "download":
        if not args.dataset:
            print("Please specify a dataset name")
            return
        
        success = downloader.download_dataset(args.dataset, args.force)
        if success:
            print(f"Dataset {args.dataset} download initiated successfully")
        else:
            print(f"Failed to download dataset {args.dataset}")
    
    elif args.action == "extract":
        if not args.dataset:
            print("Please specify a dataset name")
            return
        
        success = downloader.extract_dataset(args.dataset)
        if success:
            print(f"Dataset {args.dataset} extracted successfully")
        else:
            print(f"Failed to extract dataset {args.dataset}")
    
    elif args.action == "verify":
        if not args.dataset:
            print("Please specify a dataset name")
            return
        
        results = downloader.verify_dataset(args.dataset)
        print(f"Verification results for {args.dataset}:")
        print(results)
    
    elif args.action == "info":
        if not args.dataset:
            print("Please specify a dataset name")
            return
        
        info = downloader.get_dataset_info(args.dataset)
        print(f"Dataset info for {args.dataset}:")
        print(info)


if __name__ == "__main__":
    main()
