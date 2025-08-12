# How to Run FOMO Fine-tuning Pipeline with Gradual Unfreezing

This document explains how to run the FOMO fine-tuning pipeline with gradual unfreezing from your local terminal.

## Direct Execution (Recommended)

Navigate to your project directory and run the script directly with the conda environment's Python:

```bash
# Navigate to your project directory
cd /Users/ckhome/Desktop/baseline-codebase

# Run Task 3 with gradual unfreezing
/Users/ckhome/miniconda3/envs/medsam/bin/python fomo_finetuning_pipeline_gradual_unfreeze/scripts/run_task3_gradual_unfreeze.py
```

## With Explicit PYTHONPATH

```bash
# Navigate to your project directory
cd /Users/ckhome/Desktop/baseline-codebase

# Run with explicit PYTHONPATH
PYTHONPATH=/Users/ckhome/Desktop/baseline-codebase:/Users/ckhome/Desktop/baseline-codebase/src /Users/ckhome/miniconda3/envs/medsam/bin/python fomo_finetuning_pipeline_gradual_unfreeze/scripts/run_task3_gradual_unfreeze.py
```

## For Other Tasks with Gradual Unfreezing

- **Task 1 (Infarct Detection)**: 
  ```bash
  /Users/ckhome/miniconda3/envs/medsam/bin/python fomo_finetuning_pipeline_gradual_unfreeze/scripts/run_task1_gradual_unfreeze.py
  ```

- **Task 2 (Meningioma Segmentation)**: 
  ```bash
  /Users/ckhome/miniconda3/envs/medsam/bin/python fomo_finetuning_pipeline_gradual_unfreeze/scripts/run_task2_gradual_unfreeze.py
  ```

## Configuration

The gradual unfreezing implementation uses two additional configuration parameters:

- `unfreeze_strategy`: One of "gradual", "head_only", or "full"
- `unfreeze_epoch`: Epoch at which to unfreeze the encoder (only used with "gradual" strategy)

## Key Features

1. **Gradual Unfreezing**: The encoder is initially frozen for the first N epochs, then unfrozen
2. **Flexible Strategies**: 
   - `gradual`: Freeze encoder for first N epochs, then unfreeze
   - `head_only`: Only train the task-specific head
   - `full`: Train all parameters from the beginning (original behavior)

## Verification

When running successfully, you should see output like:
- "Keeping encoder frozen at epoch 0" (when using gradual unfreezing)
- Reduced number of trainable parameters (only head parameters)
- Configuration parameters showing `unfreeze_strategy` and `unfreeze_epoch`

## Results

Results are saved in the `results/` directory, organized by task:
- `results/task1_gradual_unfreeze/`: Results for Task 1 with gradual unfreezing
- `results/task2_gradual_unfreeze/`: Results for Task 2 with gradual unfreezing
- `results/task3_gradual_unfreeze/`: Results for Task 3 with gradual unfreezing