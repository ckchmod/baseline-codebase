#!/bin/bash
#SBATCH -N1
#SBATCH -n1
#SBATCH -c16
#SBATCH --mem=128GB
#SBATCH -p crc
#SBATCH --gres=gpu:1

#SBATCH -t 72:00:00
#SBATCH -J fomo_gradual_task1
#SBATCH -o slurm_logs/finetune_gradual_task1_%j.out
#SBATCH -e slurm_logs/finetune_gradual_task1_%j.err

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shinwoochris.kang@ucalgary.ca

set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job started on $(hostname)"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "----------------------------------------"

# Initialize conda
source ~/software/init-conda

# Activate conda environment
conda activate medsam

echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "----------------------------------------"

# Create slurm_logs directory if it doesn't exist
mkdir -p slurm_logs

echo "Running FOMO Task 1 with Gradual Unfreezing: Infarct Detection"
echo "=============================================================="

# Navigate to project directory
cd /Users/ckhome/Desktop/baseline-codebase

# Run Task 1 with gradual unfreezing and explicit PYTHONPATH
PYTHONPATH=/Users/ckhome/Desktop/baseline-codebase:/Users/ckhome/Desktop/baseline-codebase/src \
/Users/ckhome/miniconda3/envs/medsam/bin/python fomo_finetuning_pipeline_gradual_unfreeze/scripts/run_task1_gradual_unfreeze.py

echo "FOMO Task 1 with Gradual Unfreezing completed!"