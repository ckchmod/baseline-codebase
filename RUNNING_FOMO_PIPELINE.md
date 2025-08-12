# How to Run FOMO Fine-tuning Pipeline

This document explains how to run the FOMO fine-tuning pipeline from your local terminal.

## Direct Execution (Recommended)

Navigate to your project directory and run the script directly with the conda environment's Python:

```bash
# Navigate to your project directory
cd /Users/ckhome/Desktop/baseline-codebase

# Run Task 3 (Brain Age Regression)
/Users/ckhome/miniconda3/envs/medsam/bin/python fomo_finetuning_pipeline/scripts/run_task3.py
```

## With Explicit PYTHONPATH

```bash
# Navigate to your project directory
cd /Users/ckhome/Desktop/baseline-codebase

# Run with explicit PYTHONPATH
PYTHONPATH=/Users/ckhome/Desktop/baseline-codebase:/Users/ckhome/Desktop/baseline-codebase/src /Users/ckhome/miniconda3/envs/medsam/bin/python fomo_finetuning_pipeline/scripts/run_task3.py
```

## For Other Tasks

- **Task 1 (Infarct Detection)**: Replace `run_task3.py` with `run_task1.py`
- **Task 2 (Meningioma Segmentation)**: Replace `run_task3.py` with `run_task2.py`
- **All tasks**: Use `run_all_tasks.py` with optional flags:
  - `--task 1` for Task 1
  - `--task 2` for Task 2
  - `--task 3` for Task 3

## Using conda activate (If properly configured)

If you want to use `conda activate`, you need to initialize conda for your shell first:

```bash
# Initialize conda for your shell (run this once)
conda init bash

# Close and reopen your terminal, or run:
source ~/.bash_profile

# Then you can activate your environment and run:
conda activate medsam
python fomo_finetuning_pipeline/scripts/run_task3.py
```

## Key Points

1. The conda environment "medsam" contains all the required dependencies
2. The Python interpreter in the conda environment is located at `/Users/ckhome/miniconda3/envs/medsam/bin/python`
3. The pipeline requires proper PYTHONPATH to find the modules
4. Memory issues (signal 137) may occur with large 3D models - this is expected behavior on machines with limited memory