#!/bin/bash
#SBATCH -N1
#SBATCH -n1
#SBATCH -c8
#SBATCH --mem=64GB
#SBATCH -p crc
#SBATCH --gres=gpu:1
#SBATCH -t 72:00:00
#SBATCH -J fomo_pretrain
#SBATCH -o slurm_logs/pretrain_%j.out
#SBATCH -e slurm_logs/pretrain_%j.err

# Initialize conda
source ~/software/init-conda

# Activate conda environment
conda activate fomo

# Set the working directory
cd /work/forkert_lab/fomo/baseline-codebase

# Create slurm_logs directory if it doesn't exist
mkdir -p slurm_logs

# Run the pretraining script with invalid image handling
python src/pretrain.py \
    --save_dir=/local_scratch/crc/models \
    --pretrain_data_dir=/local_scratch/crc/data/fomo/preprocessed_2 \
    --model_name=unet_xl_lw_dec \
    --patch_size=96 \
    --batch_size=30 \
    --epochs=100 \
    --warmup_epochs=1 \
    --augmentation_preset=all \
    --checkpoint_every_n_epochs=1 \
    --experiment=fomo_pretrain_v2

echo "Pretraining completed!" 