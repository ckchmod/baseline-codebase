# FOMO Fine-Tuning Pipeline

This pipeline fine-tunes a pretrained UNet-B model on the FOMO fine-tuning dataset for three separate medical imaging tasks using cross-validation.

## Overview

The pipeline supports fine-tuning on three distinct medical imaging tasks:
1. **Task 1**: Infarct Detection (Classification)
2. **Task 2**: Meningioma Segmentation (Segmentation)
3. **Task 3**: Brain Age Regression (Regression)

Each task uses 5-fold cross-validation to ensure robust evaluation given the small dataset sizes.

## Directory Structure

```
fomo_finetuning_pipeline/
├── configs/                 # Configuration files for each task
├── data/                    # Data loading and preprocessing
│   ├── datasets.py          # Custom dataset classes for each task
│   ├── datamodules.py       # PyTorch Lightning datamodules
│   └── splits.py            # Data splitting utilities
├── models/                  # Model definitions and fine-tuning
│   └── finetune_unet.py     # Fine-tuning wrapper for UNet-B
├── training/                # Training and evaluation pipelines
│   └── cross_validation.py  # Cross-validation implementation
├── utils/                   # Utility functions
│   └── metrics.py           # Metrics aggregation utilities
├── scripts/                 # Execution scripts
│   ├── run_task1.py         # Script to run Task 1 fine-tuning
│   ├── run_task2.py         # Script to run Task 2 fine-tuning
│   ├── run_task3.py         # Script to run Task 3 fine-tuning
│   └── run_all_tasks.py     # Script to run all tasks
└── results/                 # Output directory for results
```

## Installation

1. Ensure you have the required dependencies installed:
   ```
   pip install torch torchvision lightning pytorch-metric-learning scikit-learn pandas nibabel
   ```

2. Make sure the FOMO dataset is organized as described in the main README.

## Usage

### Running Individual Tasks

To run a specific task, use the corresponding script:

```bash
# Task 1: Infarct Detection
python scripts/run_task1.py

# Task 2: Meningioma Segmentation
python scripts/run_task2.py

# Task 3: Brain Age Regression
python scripts/run_task3.py
```

### Running All Tasks

To run all tasks sequentially:

```bash
python scripts/run_all_tasks.py
```

To run a specific task using the unified script:

```bash
# Run Task 1
python scripts/run_all_tasks.py --task 1

# Run Task 2
python scripts/run_all_tasks.py --task 2

# Run Task 3
python scripts/run_all_tasks.py --task 3
```

## Configuration

Each task has its own configuration file in the `configs/` directory. You can modify parameters such as:

- `patch_size`: Size of patches for training
- `batch_size`: Batch size for training
- `learning_rate`: Learning rate for optimization
- `max_epochs`: Maximum number of training epochs
- `pretrained_ckpt_path`: Path to the pretrained model checkpoint

## Results

Results are saved in the `results/` directory, organized by task:

- `results/task1/`: Results for Task 1
- `results/task2/`: Results for Task 2
- `results/task3/`: Results for Task 3

Each task's results directory contains:
- Fold-specific results CSV files
- Aggregated results CSV file
- Model checkpoints
- Training logs

## Model Architecture

The pipeline uses a pretrained UNet-B model (`pretrained/epoch=12.ckpt`) as the backbone, which is fine-tuned for each specific task:

- **Task 1**: Replaces the decoder with a classification head
- **Task 2**: Uses the existing segmentation decoder
- **Task 3**: Replaces the decoder with a regression head

## Cross-Validation

All tasks use 5-fold cross-validation with appropriate stratification:

- **Task 1**: Stratified by infarct presence
- **Task 2**: Stratified by meningioma presence
- **Task 3**: Balanced splitting based on age distribution

## Output

The pipeline outputs:

1. Best model checkpoints for each fold
2. Detailed metrics for each fold
3. Aggregated results with mean ± standard deviation
4. Training logs and visualizations