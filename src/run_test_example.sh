#!/bin/bash

# Example script to run the test_model.py script for different tasks

# Set your checkpoint path here
CHECKPOINT_PATH="./data/models/Task001_FOMO1/unet_b/version_0/checkpoints/last.ckpt"

# Example 1: Test FOMO1 classification task (Task 1)
# echo "Testing FOMO1 classification task..."
# python src/test_model.py \
#     --checkpoint_path "$CHECKPOINT_PATH" \
#     --taskid 1 \
#     --data_dir "./data/preprocessed" \
#     --output_dir "./test_results" \
#     --batch_size 4 \
#     --num_workers 4 \
#     --augmentation_preset "none" \
#     --seed 42

# Example 2: Test FOMO2 classification task (Task 2)
# echo "Testing FOMO2 classification task..."
# python src/test_model.py \
#     --checkpoint_path "./data/models/Task002_FOMO2/unet_b/version_0/checkpoints/last.ckpt" \
#     --taskid 2 \
#     --data_dir "./data/preprocessed" \
#     --output_dir "./test_results" \
#     --batch_size 4 \
#     --num_workers 4 \
#     --augmentation_preset "none" \
#     --seed 42

# Example 3: Test FOMO3 regression task (Task 3)
# echo "Testing FOMO3 regression task..."
# python src/test_model.py \
#     --checkpoint_path "./data/models/Task003_FOMO3/unet_b/version_0/checkpoints/last.ckpt" \
#     --taskid 3 \
#     --data_dir "./data/preprocessed" \
#     --output_dir "./test_results" \
#     --batch_size 4 \
#     --num_workers 4 \
#     --augmentation_preset "none" \
#     --seed 42

# Example 4: Test PD classification task (Task 4)
echo "Testing PD classification task..."
python src/test_model.py \
    --checkpoint_path "/local_scratch/crc/models/finetuned/Task004_PD/unet_b/version_0/checkpoints/last.ckpt" \
    --taskid 4 \
    --data_dir "/local_scratch/crc/data/pd_temp/val/" \
    --output_dir "./test_results" \
    --batch_size 4 \
    --model_name "unet_b" \
    --patch_size 32 \
    --augmentation_preset "none" \
    --seed 42

echo "Testing completed!" 