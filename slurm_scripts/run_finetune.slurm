#!/bin/bash
#SBATCH -N1
#SBATCH -n1
#SBATCH -c8
#SBATCH --mem=64GB
#SBATCH -p crc
#SBATCH --gres=gpu:1
#SBATCH -t 72:00:00
#SBATCH -J fomo_finetune
#SBATCH -o slurm_logs/finetune_%j.out
#SBATCH -e slurm_logs/finetune_%j.err

# Initialize conda
source ~/software/init-conda

# Activate conda environment
conda activate fomo

# Set the working directory
cd /work/forkert_lab/fomo/baseline-codebase

# Create slurm_logs directory if it doesn't exist
mkdir -p slurm_logs

python src/finetune.py \
    --taskid=4 \
    --data_dir=/local_scratch/crc/data/pd_temp \
    --save_dir=/local_scratch/crc/models/finetuned \
    --pretrained_weights_path=/local_scratch/crc/models/models/preprocessed_2/unet_xl_lw_dec/versions/version_0/last.ckpt \
    --model_name=unet_xl \
    --patch_size=96 \
    --taskid=4 \
    --batch_size=16 \
    --epochs=100 \
    --train_batches_per_epoch=100 \
    --augmentation_preset=basic

echo "Finetuning completed!" 