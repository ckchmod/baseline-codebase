# SLURM Scripts for FOMO Fine-tuning

This directory contains SLURM scripts for running the FOMO fine-tuning pipelines on a HPC cluster.

## Standard Fine-tuning Pipeline Scripts

1. `run_finetuning_task1.slurm` - Run Task 1 (Infarct Detection) using the standard pipeline
2. `run_finetuning_task2.slurm` - Run Task 2 (Meningioma Segmentation) using the standard pipeline
3. `run_finetuning_task3.slurm` - Run Task 3 (Brain Age Regression) using the standard pipeline
4. `run_finetuning_all_tasks.slurm` - Run all tasks using the standard pipeline

## Gradual Unfreezing Pipeline Scripts

1. `run_finetuning_gradual_task1.slurm` - Run Task 1 with gradual unfreezing
2. `run_finetuning_gradual_task2.slurm` - Run Task 2 with gradual unfreezing
3. `run_finetuning_gradual_task3.slurm` - Run Task 3 with gradual unfreezing
4. `run_finetuning_gradual_all.slurm` - Run all tasks with gradual unfreezing

## Usage

To submit a job to the SLURM scheduler, use the `sbatch` command:

```bash
# Submit a single task
sbatch slurm_scripts/run_finetuning_task1.slurm

# Submit all tasks with gradual unfreezing
sbatch slurm_scripts/run_finetuning_gradual_all.slurm
```

## Configuration

The scripts are configured with the following resources:
- 1 node with 16 CPUs
- 128GB memory
- 1 GPU
- 72-hour time limit

These settings can be modified by editing the `#SBATCH` directives at the top of each script.

## Output

Job output and error logs are saved to the `slurm_logs/` directory with filenames that include the job ID.

## Environment

The scripts assume:
1. You have a conda environment named `medsam` with all required dependencies
2. You have initialized conda with `source ~/software/init-conda`
3. Your project is located at `/Users/ckhome/Desktop/baseline-codebase`

If these paths are different in your environment, you will need to modify the scripts accordingly.