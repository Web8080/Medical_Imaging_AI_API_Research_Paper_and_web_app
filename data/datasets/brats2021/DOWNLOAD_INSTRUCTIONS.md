
# BRATS 2021 Dataset Download Instructions

## Official Download Steps:
1. Visit: https://www.synapse.org/#!Synapse:syn27046444
2. Register for a Synapse account
3. Request access to the BRATS 2021 dataset
4. Download the following files:
   - BraTS2021_Training_Data.tar (15GB)
   - BraTS2021_ValidationData.tar (3GB)

## Alternative Sources:
- Kaggle: https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation
- Direct download: https://www.kaggle.com/datasets/awsaf49/brats2021-training-data

## Expected Structure:
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

## Dataset Information:
- Total cases: 1,251
- Modality: MRI (T1, T1CE, T2, FLAIR)
- Task: Brain tumor segmentation
- Classes: Background, Necrotic core, Edema, Enhancing tumor
- Format: NIfTI (.nii.gz)
- Size: ~15GB
