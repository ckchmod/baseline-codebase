# FOMO Fine-Tuning Pipeline Plan

This document outlines a comprehensive plan for fine-tuning a pretrained UNet-B model on the FOMO fine-tuning dataset for three separate tasks using cross-validation and evaluation pipelines.

## Overview

We will create a pipeline that fine-tunes a pretrained UNet-B model (`pretrained/epoch=12.ckpt`) on the FOMO fine-tuning dataset for three distinct medical imaging tasks:
1. Task 1: Infarct Detection (Classification)
2. Task 2: Meningioma Segmentation (Segmentation)
3. Task 3: Brain Age Regression (Regression)

## Project Structure

```
fomo-fine-tuning-pipeline/
├── configs/                 # Configuration files for each task
├── data/                    # Data loading and preprocessing
│   ├── datasets.py          # Custom dataset classes for each task
│   ├── datamodules.py       # PyTorch Lightning datamodules
│   └── preprocessing.py     # Task-specific preprocessing functions
├── models/                  # Model definitions and fine-tuning
│   ├── finetune_unet.py     # Fine-tuning wrapper for UNet-B
│   └── task_heads.py        # Task-specific output heads
├── training/                # Training and evaluation pipelines
│   ├── cross_validation.py  # Cross-validation implementation
│   ├── trainer.py           # Training loop
│   └── evaluator.py         # Evaluation pipeline
├── utils/                   # Utility functions
│   ├── metrics.py           # Task-specific metrics
│   └── visualization.py     # Visualization tools
├── scripts/                 # Execution scripts
│   ├── run_task1.py         # Script to run Task 1 fine-tuning
│   ├── run_task2.py         # Script to run Task 2 fine-tuning
│   └── run_task3.py         # Script to run Task 3 fine-tuning
└── results/                 # Output directory for results
```

## 1. Data Preparation

### 1.1 Data Loading
- Create custom dataset classes for each task that can handle the specific data formats:
  - Task 1: Load NIfTI files from `fomo-fine-tuning/fomo-task1/skull_stripped/`
  - Task 2: Load `.npy` files from `fomo-fine-tuning/fomo-task2/preprocessed_2/`
  - Task 3: Load NIfTI files from `fomo-fine-tuning/fomo-task3/preprocessed_2/`

### 1.2 Preprocessing
- Implement task-specific preprocessing pipelines:
  - Task 1: Normalize MRI modalities, combine multiple modalities as channels
  - Task 2: Extract channels from 4D arrays, handle segmentation masks
  - Task 3: Normalize T1/T2 scans, prepare age labels

### 1.3 Data Augmentation
- Implement appropriate augmentations for each task:
  - Task 1: Spatial transformations, intensity augmentations
  - Task 2: Spatial transformations that maintain alignment between image and mask
  - Task 3: Spatial transformations, intensity augmentations

## 2. Model Architecture

### 2.1 Base Model
- Load the pretrained UNet-B model from `pretrained/epoch=12.ckpt`
- The model uses `starting_filters=32` and `use_skip_connections=True`

### 2.2 Task-Specific Heads
- Task 1 (Classification): Replace decoder with classification head (global pooling + fully connected layer)
- Task 2 (Segmentation): Use existing segmentation decoder
- Task 3 (Regression): Replace decoder with regression head (global pooling + fully connected layer)

### 2.3 Fine-tuning Strategy
- Load pretrained weights with size matching verification
- Freeze early layers optionally
- Adjust learning rates for different parts of the network

## 3. Cross-Validation Pipeline

### 3.1 K-Fold Splitting
- Implement 5-fold cross-validation for all tasks
- Ensure stratified splitting where appropriate:
  - Task 1: Stratify by infarct presence
  - Task 2: Stratify by meningioma presence
  - Task 3: Stratify by age distribution

### 3.2 Training Loop
- For each fold:
  - Train model on training set
  - Validate on validation set
  - Save best model checkpoint
  - Log metrics and losses

### 3.3 Evaluation
- Test each fold's best model on the test set
- Aggregate results across folds
- Compute mean and standard deviation of metrics

## 4. Task-Specific Implementations

### 4.1 Task 1: Infarct Detection (Classification)
- **Input**: Multi-modal MRI scans (SWI, FLAIR, DWI, ADC)
- **Output**: Binary classification (infarct present/absent)
- **Loss**: Cross-entropy loss
- **Metrics**: Accuracy, Precision, Recall, F1-score, AUC-ROC

### 4.2 Task 2: Meningioma Segmentation (Segmentation)
- **Input**: Multi-modal MRI scans (T1, T2, FLAIR)
- **Output**: Binary segmentation mask
- **Loss**: Dice + Cross-entropy loss
- **Metrics**: Dice coefficient, Hausdorff distance, Surface distance

### 4.3 Task 3: Brain Age Regression (Regression)
- **Input**: T1/T2-weighted MRI scans
- **Output**: Predicted age (continuous value)
- **Loss**: Mean squared error
- **Metrics**: Mean absolute error, R-squared, Pearson correlation

## 5. Training Configuration

### 5.1 Hyperparameters
- Learning rate: 1e-4 with cosine annealing
- Batch size: 2-4 (depending on GPU memory)
- Epochs: 100-200 with early stopping
- Optimizer: AdamW
- Weight decay: 3e-5

### 5.2 Fine-tuning Strategies
- Full fine-tuning: Update all weights
- Partial fine-tuning: Freeze encoder, train decoder and task head
- Layer-wise learning rates: Lower LR for earlier layers

## 6. Evaluation Pipeline

### 6.1 Validation Metrics
- Track validation metrics during training
- Save best model based on validation performance
- Implement early stopping

### 6.2 Test Evaluation
- Evaluate best model from each fold on test set
- Compute confidence intervals for metrics
- Generate detailed performance reports

### 6.3 Visualization
- Generate sample predictions for qualitative assessment
- Create confusion matrices for classification tasks
- Visualize segmentation results overlaid on original images

## 7. Implementation Steps

### Phase 1: Data Infrastructure
1. Create dataset classes for each task
2. Implement data preprocessing pipelines
3. Set up data augmentation
4. Create PyTorch Lightning datamodules

### Phase 2: Model Infrastructure
1. Implement fine-tuning wrapper for UNet-B
2. Create task-specific heads
3. Set up model loading with weight transfer

### Phase 3: Training Pipeline
1. Implement cross-validation framework
2. Create training loop with logging
3. Set up checkpointing and early stopping

### Phase 4: Evaluation Pipeline
1. Implement task-specific metrics
2. Create evaluation functions
3. Set up result aggregation

### Phase 5: Execution and Testing
1. Create execution scripts for each task
2. Run preliminary experiments
3. Optimize hyperparameters

## 8. Expected Outputs

### 8.1 Model Checkpoints
- Best model from each fold for each task
- Final aggregated model

### 8.2 Results
- Cross-validation metrics with confidence intervals
- Test set performance
- Training curves and logs

### 8.3 Visualizations
- Sample predictions
- Performance plots
- Attention maps (if applicable)

## 9. Tools and Libraries

- PyTorch Lightning for training framework
- PyTorch for deep learning
- MONAI for medical imaging utilities
- scikit-learn for metrics and cross-validation utilities
- wandb for experiment tracking
- nibabel for NIfTI file handling
- numpy for numerical operations

## 10. Timeline

1. **Week 1**: Data infrastructure and preprocessing
2. **Week 2**: Model infrastructure and fine-tuning setup
3. **Week 3**: Training pipeline implementation
4. **Week 4**: Evaluation pipeline and testing
5. **Week 5**: Optimization and final experiments

This plan provides a comprehensive framework for fine-tuning the pretrained UNet-B model on the FOMO fine-tuning dataset with proper cross-validation and evaluation for all three tasks.