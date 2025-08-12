# FOMO Fine-Tuning Data Analysis

This directory contains documentation and tools for analyzing the FOMO fine-tuning dataset.

## Contents

- `FOMO_DATA_DOCUMENTATION.md` - Comprehensive documentation of the FOMO dataset structure
- `scripts/visualize_fomo_data.py` - Python script for visualizing the dataset
- `fomo-fine-tuning/` - Main dataset directory (from original source)

## Dataset Documentation

See `FOMO_DATA_DOCUMENTATION.md` for detailed information about the structure of the three tasks:
1. Task 1: Infarct Detection
2. Task 2: Meningioma Segmentation
3. Task 3: Brain Age Regression

## Visualization Tool

The `scripts/visualize_fomo_data.py` script can be used to generate visualizations of the dataset:
```bash
python3 scripts/visualize_fomo_data.py
```

This script will create axial, coronal, and sagittal views for samples from all three tasks and save them to `/tmp/fomo_visualizations/`.