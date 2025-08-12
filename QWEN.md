# Qwen Context for FOMO25 Challenge Codebase

This document provides essential context about the FOMO25 Challenge codebase for Qwen Code, enabling effective interaction with the project.

## Overview

This repository contains the official baseline code for the FOMO25 Challenge, which investigates the few-shot generalization properties of foundation models for brain MRI data analysis. The challenge involves three distinct medical imaging tasks:

1. **Task 1**: Infarct Detection (Classification)
2. **Task 2**: Meningioma Segmentation (Segmentation)
3. **Task 3**: Brain Age Regression (Regression)

## Project Structure

The codebase is organized into several key directories:

- `fomo_finetuning_pipeline/`: The main fine-tuning pipeline implementation with cross-validation
- `fomo_finetuning_pipeline_gradual_unfreeze/`: A variant of the pipeline with gradual unfreezing
- `fomo-fine-tuning/`: Main dataset directory (external)
- `pretrained/`: Directory for pretrained model checkpoints
- `results/`: Output directory for training results
- `scripts/`: Utility scripts for data visualization
- `src/`: Core source code for models and training
- `test_results/`: Directory for test outputs

## Fine-Tuning Pipeline

The `fomo_finetuning_pipeline/` directory contains a complete implementation of a fine-tuning pipeline that:

- Fine-tunes a pretrained UNet-B model (`pretrained/epoch=12.ckpt`) on all three FOMO tasks
- Uses 5-fold cross-validation for robust evaluation given the small dataset sizes
- Supports all three task types:
  - Task 1: Classification (Infarct Detection)
  - Task 2: Segmentation (Meningioma Segmentation)
  - Task 3: Regression (Brain Age Regression)
- Provides detailed metrics and result aggregation

### Key Components

#### Data Handling
- Custom dataset classes for each task type
- Data loading and preprocessing specific to each task's data format
- Cross-validation splitting with appropriate stratification

#### Model Architecture
- Fine-tuning wrapper for UNet-B that loads pretrained weights
- Task-specific heads for classification, segmentation, and regression
- Weight transfer mechanism that handles size mismatches

#### Training Pipeline
- Cross-validation implementation with 5 folds
- Task-specific training configurations
- Checkpointing and early stopping

#### Evaluation
- Comprehensive metrics for each task type
- Result aggregation across folds
- Detailed reporting of performance

### Gradual Unfreezing Pipeline

The `fomo_finetuning_pipeline_gradual_unfreeze/` directory contains a variant of the pipeline with support for gradual unfreezing:

- Implements three unfreezing strategies:
  - `gradual`: Freeze encoder for first N epochs, then unfreeze
  - `head_only`: Only train the task-specific head
  - `full`: Train all parameters from the beginning (original behavior)
- Configurable unfreezing epoch for gradual strategy

## Usage

### Running Fine-Tuning Pipelines

To run the standard fine-tuning pipeline:

```bash
cd fomo_finetuning_pipeline
python scripts/run_task1.py  # Task 1
python scripts/run_task2.py  # Task 2
python scripts/run_task3.py  # Task 3
python scripts/run_all_tasks.py  # All tasks
```

To run the gradual unfreezing pipeline:

```bash
cd fomo_finetuning_pipeline_gradual_unfreeze
python scripts/run_task1_gradual_unfreeze.py  # Task 1
python scripts/run_task2_gradual_unfreeze.py  # Task 2
python scripts/run_task3_gradual_unfreeze.py  # Task 3
```

### Configuration

Each pipeline uses JSON configuration files located in the `configs/` directory. Key configuration parameters include:

- `patch_size`: Size of patches for training
- `batch_size`: Batch size for training
- `learning_rate`: Learning rate for optimization
- `max_epochs`: Maximum number of training epochs
- `pretrained_ckpt_path`: Path to the pretrained model checkpoint
- `unfreeze_strategy`: Unfreezing strategy (gradual unfreezing pipeline only)
- `unfreeze_epoch`: Epoch at which to unfreeze the encoder (gradual unfreezing pipeline only)

## Results

Results are saved in the `results/` directory, organized by task:

- `results/task1/`: Results for Task 1
- `results/task2/`: Results for Task 2
- `results/task3/`: Results for Task 3
- `results/task1_gradual_unfreeze/`: Results for Task 1 with gradual unfreezing
- `results/task2_gradual_unfreeze/`: Results for Task 2 with gradual unfreezing
- `results/task3_gradual_unfreeze/`: Results for Task 3 with gradual unfreezing

Each task's results directory contains:
- Fold-specific results CSV files
- Aggregated results CSV file
- Model checkpoints
- Training logs

## Dependencies

Key dependencies include:
- PyTorch for deep learning
- PyTorch Lightning for training framework
- MONAI for medical imaging utilities
- scikit-learn for metrics and cross-validation utilities
- wandb for experiment tracking
- nibabel for NIfTI file handling
- numpy for numerical operations

These can be installed with:
```bash
pip install -e .
```

## Hardware Requirements

The reference implementation was pretrained on 2xH100 GPUs with 80GB of memory. Depending on your hardware, you may need to adjust batch sizes and patch sizes accordingly.