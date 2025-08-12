# FOMO Fine-Tuning Data Documentation

This document provides a comprehensive overview of the FOMO fine-tuning dataset structure, including the three distinct tasks and their associated data formats.

## Overview

The FOMO fine-tuning dataset consists of three tasks:
1. **Task 1**: Infarct Detection - Binary classification of brain infarcts
2. **Task 2**: Meningioma Segmentation - Binary segmentation of brain meningiomas
3. **Task 3**: Brain Age Regression - Predicting patient age from MRI scans

## Task 1: Infarct Detection

### Data Description
- **Purpose**: Binary classification of brain infarcts
- **Data Type**: Skull-stripped brain MRI scans
- **Modalities**: SWI, FLAIR, DWI (b=1000), ADC
- **Format**: NIfTI files (.nii.gz)

### Directory Structure
```
fomo-fine-tuning/fomo-task1/
├── skull_stripped/
│   ├── sub_1/
│   │   └── ses_1/
│   │       ├── ss_adc.nii.gz
│   │       ├── ss_dwi_b1000.nii.gz
│   │       ├── ss_flair.nii.gz
│   │       └── ss_swi.nii.gz
│   └── sub_2/
│       └── ses_1/
│           ├── ss_adc.nii.gz
│           ├── ss_dwi_b1000.nii.gz
│           ├── ss_flair.nii.gz
│           └── ss_swi.nii.gz
└── labels_masked/
    ├── sub_1/
    │   └── ses_1/
    │       └── seg_masked.nii.gz
    └── sub_2/
        └── ses_1/
            └── seg_masked.nii.gz
```

### Data Characteristics
- Different resolutions across subjects
- Intensity ranges vary by modality:
  - SWI: Up to ~2500
  - FLAIR: Up to ~800
  - DWI: Up to ~2000
  - ADC: Can include negative values, up to ~4000

## Task 2: Meningioma Segmentation

### Data Description
- **Purpose**: Binary segmentation of brain meningiomas
- **Data Type**: Preprocessed brain MRI scans with segmentation masks
- **Format**: 
  - `.npy` files containing 4-channel 3D data
  - `.pkl` files containing metadata

### Directory Structure
```
fomo-fine-tuning/fomo-task2/
├── preprocessed_2/
│   ├── FOMO2_sub_1.npy
│   ├── FOMO2_sub_1.pkl
│   ├── FOMO2_sub_2.npy
│   └── FOMO2_sub_2.pkl
└── labels_masked/
    ├── sub_1/
    │   └── ses_1/
    │       └── seg_masked.nii.gz
    └── sub_2/
        └── ses_1/
            └── seg_masked.nii.gz
```

### Data Format Details
Each subject has:
- A `.npy` file containing a 4D array with shape `(4, height, width, depth)`
- A `.pkl` file containing metadata

### Channels
1. **Channel 0**: T1-weighted MRI
2. **Channel 1**: T2-weighted MRI
3. **Channel 2**: FLAIR MRI
4. **Channel 3**: Segmentation masks

### Data Characteristics
- All channels normalized to [0, 1] range
- Segmentation masks are binary (0 for background, 1 for meningioma)
- Different resolutions across subjects

## Task 3: Brain Age Regression

### Data Description
- **Purpose**: Predicting patient age from MRI scans
- **Data Type**: Skull-stripped brain MRI scans
- **Modalities**: T1-weighted, T2-weighted
- **Format**: NIfTI files (.nii.gz)

### Directory Structure
```
fomo-fine-tuning/fomo-task3/
├── preprocessed_2/
│   ├── sub_1/
│   │   └── ses_1/
│   │       ├── ss_t1.nii.gz
│   │       └── ss_t2.nii.gz
│   └── sub_2/
│       └── ses_1/
│           ├── ss_t1.nii.gz
│           └── ss_t2.nii.gz
└── labels/
    ├── sub_1/
    │   └── ses_1/
    │       └── label.txt
    └── sub_2/
        └── ses_1/
            └── label.txt
```

### Data Characteristics
- All data normalized to [0, 1] range
- Large dataset with 200 subjects
- Labels in `label.txt` files contain age values

## Visualization

We've created a visualization script (`scripts/visualize_fomo_data.py`) that generates axial, coronal, and sagittal views for all tasks:
1. **Task 1**: Views for each MRI modality (SWI, FLAIR, DWI, ADC)
2. **Task 2**: Views for each channel (T1, T2, FLAIR, and segmentation)
3. **Task 3**: Views for T1 and T2 scans

The visualizations are saved to `/tmp/fomo_visualizations/` organized by task:
- `/tmp/fomo_visualizations/task1/`: Task 1 visualizations
- `/tmp/fomo_visualizations/task2/`: Task 2 visualizations
- `/tmp/fomo_visualizations/task3/`: Task 3 visualizations

Each visualization shows three orthogonal views (axial, coronal, and sagittal) in a single image file.

## Data Overlays

Regarding your question about whether images are layered with masks:
- **Task 1**: The `labels_masked/` directory contains segmentation masks (`seg_masked.nii.gz`) that can be overlaid on the skull-stripped images from `skull_stripped/`
- **Task 2**: The 4th channel in the `.npy` files contains the segmentation masks, which can be overlaid on the other channels (T1, T2, FLAIR)
- **Task 3**: No segmentation masks are provided; this is a regression task based on raw MRI data

For overlay visualization, you would typically load both the base image and the mask, then display them together with different colormaps or transparency levels.

## Visualization Script Usage

To generate visualizations of the FOMO data:

1. Make sure you have the required Python packages installed:
   ```
   numpy
   matplotlib
   nibabel
   ```

2. Run the visualization script:
   ```
   python3 scripts/visualize_fomo_data.py
   ```

3. The script will automatically process all three tasks and save the visualizations to `/tmp/fomo_visualizations/`.

The script uses a fixed random seed for reproducibility, so it will always select the same subjects/files for visualization across runs.