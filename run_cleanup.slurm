#!/bin/bash
#SBATCH -N1
#SBATCH -n1
#SBATCH -c10
#SBATCH --mem=32GB
#SBATCH -p cpu2023
#SBATCH -t 4:00:00
#SBATCH -J fomo_cleanup
#SBATCH -o cleanup_%j.out
#SBATCH -e cleanup_%j.err

# Initialize conda
source ~/software/init-conda

# Activate conda environment
conda activate fomo

# Set the working directory
cd /work/forkert_lab/fomo/baseline-codebase

# Run the cleanup script
python clean_invalid_images_parallel.py /work/forkert_lab/fomo/preprocessed/FOMO60k --verbose

echo "Cleanup completed!" 