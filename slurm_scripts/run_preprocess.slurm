#!/bin/bash
#SBATCH -N1
#SBATCH -n1
#SBATCH -c10
#SBATCH --mem=32GB
#SBATCH -p cpu2023
#SBATCH -t 24:00:00
#SBATCH -J fomo_preprocess
#SBATCH -o slurm_logs/preprocess_%j.out
#SBATCH -e slurm_logs/preprocess_%j.err

# Initialize conda
source ~/software/init-conda

# Activate conda environment
conda activate fomo

# Set the working directory
cd /work/forkert_lab/fomo/baseline-codebase

# Run the preprocessing script
python src/data/fomo-60k/preprocess.py \
    --in_path=/work/forkert_lab/data/fomo/preprocessed_1 \
    --out_path=/work/forkert_lab/data/fomo/preprocessed_2 \
    
echo "Preprocessing completed!" 