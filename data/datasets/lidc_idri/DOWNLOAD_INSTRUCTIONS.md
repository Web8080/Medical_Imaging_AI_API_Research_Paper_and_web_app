
# LIDC-IDRI Dataset Download Instructions

## Official Download Steps:
1. Visit: https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI
2. Register for a TCIA account
3. Download the NBIA Data Retriever tool
4. Use the tool to download the LIDC-IDRI dataset

## Alternative Sources:
- Kaggle: https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images
- Direct download: https://www.kaggle.com/datasets/kmader/finding-lungs-in-ct-data

## Expected Structure:
```
lidc_idri/
├── LIDC-IDRI-0001/
│   ├── 1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.dcm
│   └── ...
└── ...
```

## Dataset Information:
- Total cases: 1,018
- Modality: CT
- Task: Lung nodule detection
- Format: DICOM
- Size: ~120GB
