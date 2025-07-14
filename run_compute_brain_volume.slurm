#!/bin/bash
#SBATCH -N1
#SBATCH -n1
#SBATCH -c15
#SBATCH --mem=64GB
#SBATCH -p cpu2023
#SBATCH -t 8:00:00
#SBATCH -J fomo_compute_brain_volume
#SBATCH -o compute_brain_volume_%j.out
#SBATCH -e compute_brain_volume_%j.err

# Initialize conda
source ~/software/init-conda

# Activate conda environment
conda activate fomo

# Set the working directory
cd /work/forkert_lab/fomo/baseline-codebase

# Run the preprocessing script
python compute_brain_volume.py \
    --data_dir /work/forkert_lab/kimberly/preprocessed \
    --output_csv /work/forkert_lab/fomo/brain_volumes_after_preprocessing.csv \
    
echo "Brain volume computation completed!" 