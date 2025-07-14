#!/bin/bash
#SBATCH -N1
#SBATCH -n1
#SBATCH -c10
#SBATCH --mem=16GB
#SBATCH -p cpu2019
#SBATCH -t 14:00:00
#SBATCH -J git_lfs_pull
#SBATCH -o git_lfs_pull_%j.out
#SBATCH -e git_lfs_pull_%j.err

# Initialize conda
source ~/software/init-conda

# Activate conda environment
conda activate fomo

# Set the working directory
cd /work/forkert_lab/fomo/baseline-codebase

# Run the parallel Git LFS pull script
python parallel_git_lfs_pull.py \
    --root_dir /work/forkert_lab/fomo/FOMO-MRI/fomo-60k \
    --num_workers 75 \
    --batch_size 50 \
    --verify

echo "Git LFS pull completed!" 