#!/usr/bin/env python3
"""
Create synthetic medical imaging data for immediate testing and training.
"""

import argparse
import logging
import numpy as np
import os
from pathlib import Path
import pickle
import json

logger = logging.getLogger(__name__)


def create_synthetic_brats_data(num_samples: int = 100, output_dir: str = "data/datasets/brats_synthetic"):
    """Create synthetic BRATS brain tumor data."""
    logger.info(f"Creating {num_samples} synthetic BRATS samples...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    data_list = []
    
    for i in range(num_samples):
        # Create synthetic brain MRI image (64x64x64 for faster processing)
        brain_image = np.random.rand(64, 64, 64).astype(np.float32)
        brain_image = (brain_image * 1000) - 500  # CT-like intensities
        
        # Create brain-like structure
        center = np.array([32, 32, 32])
        for x in range(64):
            for y in range(64):
                for z in range(64):
                    dist = np.linalg.norm(np.array([x, y, z]) - center)
                    if dist < 25:  # Brain region
                        brain_image[x, y, z] -= 200  # Brain tissue
                    else:
                        brain_image[x, y, z] = -1000  # Background
        
        # Create synthetic segmentation mask
        segmentation = np.zeros_like(brain_image, dtype=np.uint8)
        
        # Add tumor regions
        if np.random.random() > 0.3:  # 70% chance of tumor
            # Whole tumor region
            tumor_center = center + np.random.randint(-10, 10, 3)
            tumor_size = np.random.randint(8, 15)
            
            for x in range(64):
                for y in range(64):
                    for z in range(64):
                        dist = np.linalg.norm(np.array([x, y, z]) - tumor_center)
                        if dist < tumor_size:
                            segmentation[x, y, z] = 1  # Whole tumor
            
            # Core region (smaller)
            if np.random.random() > 0.5:
                core_size = tumor_size // 2
                for x in range(64):
                    for y in range(64):
                        for z in range(64):
                            dist = np.linalg.norm(np.array([x, y, z]) - tumor_center)
                            if dist < core_size:
                                segmentation[x, y, z] = 2  # Core
            
            # Enhancing region (even smaller)
            if np.random.random() > 0.7:
                enh_size = core_size // 2
                for x in range(64):
                    for y in range(64):
                        for z in range(64):
                            dist = np.linalg.norm(np.array([x, y, z]) - tumor_center)
                            if dist < enh_size:
                                segmentation[x, y, z] = 3  # Enhancing
        
        # Save as numpy arrays
        case_id = f"BraTS2021_{i:05d}"
        case_dir = output_path / case_id
        case_dir.mkdir(exist_ok=True)
        
        np.save(case_dir / f"{case_id}_t1.npy", brain_image)
        np.save(case_dir / f"{case_id}_seg.npy", segmentation)
        
        # Add to data list
        data_list.append({
            'case_id': case_id,
            'image_path': str(case_dir / f"{case_id}_t1.npy"),
            'label_path': str(case_dir / f"{case_id}_seg.npy"),
            'has_tumor': np.any(segmentation > 0),
            'tumor_volume': np.sum(segmentation > 0)
        })
    
    # Save data list
    with open(output_path / "data_list.json", "w") as f:
        json.dump(data_list, f, indent=2)
    
    logger.info(f"Created {num_samples} synthetic BRATS samples in {output_path}")
    return data_list


def create_synthetic_lidc_data(num_samples: int = 100, output_dir: str = "data/datasets/lidc_synthetic"):
    """Create synthetic LIDC lung nodule data."""
    logger.info(f"Creating {num_samples} synthetic LIDC samples...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    data_list = []
    
    for i in range(num_samples):
        # Create synthetic lung CT image (128x128x32)
        lung_image = np.random.rand(128, 128, 32).astype(np.float32)
        lung_image = (lung_image * 1000) - 1000  # CT-like intensities
        
        # Create lung-like structure
        for z in range(32):
            for x in range(128):
                for y in range(128):
                    # Create lung regions (darker)
                    if 30 < x < 98 and 30 < y < 98:
                        lung_image[x, y, z] -= 300  # Lung tissue
                    else:
                        lung_image[x, y, z] = -1000  # Background
        
        # Create classification label (nodule present/absent)
        has_nodule = np.random.random() > 0.5  # 50% chance of nodule
        
        if has_nodule:
            # Add synthetic nodule
            nodule_center = np.array([
                np.random.randint(40, 88),
                np.random.randint(40, 88),
                np.random.randint(8, 24)
            ])
            nodule_size = np.random.randint(3, 8)
            
            for x in range(128):
                for y in range(128):
                    for z in range(32):
                        dist = np.linalg.norm(np.array([x, y, z]) - nodule_center)
                        if dist < nodule_size:
                            lung_image[x, y, z] += 200  # Nodule (brighter)
        
        # Save as numpy array
        case_id = f"LIDC-IDRI_{i:04d}"
        case_file = output_path / f"{case_id}.npy"
        np.save(case_file, lung_image)
        
        # Add to data list
        data_list.append({
            'case_id': case_id,
            'image_path': str(case_file),
            'label': 1 if has_nodule else 0,
            'has_nodule': has_nodule
        })
    
    # Save data list
    with open(output_path / "data_list.json", "w") as f:
        json.dump(data_list, f, indent=2)
    
    logger.info(f"Created {num_samples} synthetic LIDC samples in {output_path}")
    return data_list


def create_synthetic_decathlon_data(num_samples: int = 100, output_dir: str = "data/datasets/decathlon_synthetic"):
    """Create synthetic Medical Segmentation Decathlon data."""
    logger.info(f"Creating {num_samples} synthetic Decathlon samples...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create different organ tasks
    tasks = {
        'Task01_BrainTumour': {'organ': 'brain', 'classes': 4},
        'Task02_Heart': {'organ': 'heart', 'classes': 3},
        'Task03_Liver': {'organ': 'liver', 'classes': 3}
    }
    
    all_data = {}
    
    for task_name, task_info in tasks.items():
        task_dir = output_path / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        
        images_dir = task_dir / "imagesTr"
        labels_dir = task_dir / "labelsTr"
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        
        task_data = []
        
        for i in range(num_samples // len(tasks)):
            # Create synthetic organ image
            organ_image = np.random.rand(96, 96, 96).astype(np.float32)
            organ_image = (organ_image * 1000) - 500
            
            # Create organ-like structure
            center = np.array([48, 48, 48])
            for x in range(96):
                for y in range(96):
                    for z in range(96):
                        dist = np.linalg.norm(np.array([x, y, z]) - center)
                        if dist < 30:
                            organ_image[x, y, z] -= 200
            
            # Create segmentation mask
            segmentation = np.zeros_like(organ_image, dtype=np.uint8)
            
            # Add organ structures
            if np.random.random() > 0.3:
                structure_center = center + np.random.randint(-15, 15, 3)
                structure_size = np.random.randint(10, 20)
                
                for x in range(96):
                    for y in range(96):
                        for z in range(96):
                            dist = np.linalg.norm(np.array([x, y, z]) - structure_center)
                            if dist < structure_size:
                                segmentation[x, y, z] = np.random.randint(1, task_info['classes'])
            
            # Save files
            case_id = f"{task_name}_{i:03d}"
            image_file = images_dir / f"{case_id}_0000.nii.gz"
            label_file = labels_dir / f"{case_id}.nii.gz"
            
            # Save as numpy arrays (simplified)
            np.save(image_file.with_suffix('.npy'), organ_image)
            np.save(label_file.with_suffix('.npy'), segmentation)
            
            task_data.append({
                'case_id': case_id,
                'image_path': str(image_file.with_suffix('.npy')),
                'label_path': str(label_file.with_suffix('.npy')),
                'task': task_name,
                'organ': task_info['organ']
            })
        
        all_data[task_name] = task_data
        
        # Save task data list
        with open(task_dir / "data_list.json", "w") as f:
            json.dump(task_data, f, indent=2)
    
    # Save overall data list
    with open(output_path / "all_data.json", "w") as f:
        json.dump(all_data, f, indent=2)
    
    logger.info(f"Created synthetic Decathlon data in {output_path}")
    return all_data


def main():
    """Main script to create synthetic data."""
    parser = argparse.ArgumentParser(description="Create synthetic medical imaging data")
    parser.add_argument("--dataset", choices=['brats', 'lidc', 'decathlon', 'all'],
                       default='all', help="Dataset to create")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to create")
    parser.add_argument("--output_dir", type=str, default="data/datasets",
                       help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level)
    
    try:
        results = {}
        
        if args.dataset in ['brats', 'all']:
            results['brats'] = create_synthetic_brats_data(
                args.num_samples, 
                f"{args.output_dir}/brats_synthetic"
            )
        
        if args.dataset in ['lidc', 'all']:
            results['lidc'] = create_synthetic_lidc_data(
                args.num_samples,
                f"{args.output_dir}/lidc_synthetic"
            )
        
        if args.dataset in ['decathlon', 'all']:
            results['decathlon'] = create_synthetic_decathlon_data(
                args.num_samples,
                f"{args.output_dir}/decathlon_synthetic"
            )
        
        print("✅ Synthetic data creation completed!")
        print(f"Created datasets: {list(results.keys())}")
        print(f"Output directory: {args.output_dir}")
        print("\nNext steps:")
        print("1. Run training scripts with the synthetic data")
        print("2. Test the API with trained models")
        print("3. Download real datasets for production use")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to create synthetic data: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
