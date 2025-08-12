# FOMO Fine-Tuning Pipeline Implementation

This directory contains our implementation of a fine-tuning pipeline for the FOMO dataset, which builds upon the original source code.

## Contents

- `FOMO_DATA_DOCUMENTATION.md` - Comprehensive documentation of the FOMO dataset structure
- `FOMO_FINE_TUNING_PLAN.md` - Detailed plan for the fine-tuning pipeline
- `FOMO_ANALYSIS_README.md` - README for our data analysis work
- `fomo_finetuning_pipeline/` - Complete implementation of the fine-tuning pipeline
- `scripts/` - Utility scripts for data visualization

## Fine-Tuning Pipeline

The `fomo_finetuning_pipeline/` directory contains a complete implementation of a fine-tuning pipeline that:

- Fine-tunes a pretrained UNet-B model (`pretrained/epoch=12.ckpt`) on all three FOMO tasks
- Uses 5-fold cross-validation for robust evaluation given the small dataset sizes
- Supports all three task types:
  1. Task 1: Infarct Detection (Classification)
  2. Task 2: Meningioma Segmentation (Segmentation)
  3. Task 3: Brain Age Regression (Regression)
- Provides detailed metrics and result aggregation

## Key Components

### Data Handling
- Custom dataset classes for each task type
- Data loading and preprocessing specific to each task's data format
- Cross-validation splitting with appropriate stratification

### Model Architecture
- Fine-tuning wrapper for UNet-B that loads pretrained weights
- Task-specific heads for classification, segmentation, and regression
- Weight transfer mechanism that handles size mismatches

### Training Pipeline
- Cross-validation implementation with 5 folds
- Task-specific training configurations
- Checkpointing and early stopping

### Evaluation
- Comprehensive metrics for each task type
- Result aggregation across folds
- Detailed reporting of performance

## Usage

To run the fine-tuning pipeline:

```bash
cd fomo_finetuning_pipeline
python scripts/run_all_tasks.py
```

For detailed usage instructions, see `fomo_finetuning_pipeline/README.md`.