#!/usr/bin/env python3
"""
Parallel script to clean preprocessed images by removing those with intensities outside the (0, 1) range.
This script scans through .npy files and removes any that don't meet the intensity requirements.
"""

import os
import sys
import numpy as np
import argparse
from pathlib import Path
from typing import List, Tuple
import logging
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clean_invalid_images_parallel.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def is_valid_image(data: np.ndarray) -> bool:
    """
    Check if image intensities are within expected range (0, 1).
    
    Args:
        data: numpy array containing image data
        
    Returns:
        bool: True if image is valid, False otherwise
    """
    if data is None:
        return False
    
    min_val = data.min()
    max_val = data.max()
    
    return 0 <= min_val and max_val <= 1


def check_single_file(file_path_str: str) -> Tuple[str, bool, float, float]:
    """
    Check if a single image file has valid intensities.
    This function is designed to be used with multiprocessing.
    
    Args:
        file_path_str: String path to the .npy file
        
    Returns:
        Tuple of (file_path, is_valid, min_val, max_val)
    """
    file_path = Path(file_path_str)
    try:
        data = np.load(file_path)
        min_val = data.min()
        max_val = data.max()
        is_valid = is_valid_image(data)
        return str(file_path), is_valid, min_val, max_val
    except Exception as e:
        return str(file_path), False, float('nan'), float('nan')


def remove_invalid_images_parallel(data_dir: str, dry_run: bool = False, num_workers: int = None) -> None:
    """
    Scan through preprocessed images and remove invalid ones using parallel processing.
    
    Args:
        data_dir: Directory containing preprocessed images
        dry_run: If True, only report what would be removed without actually removing
        num_workers: Number of parallel workers (default: CPU count - 1)
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.error(f"Data directory {data_dir} does not exist")
        return
    
    # Find all .npy files
    npy_files = list(data_path.glob("*.npy"))
    logger.info(f"Found {len(npy_files)} .npy files to check")
    
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    logger.info(f"Using {num_workers} parallel workers")
    
    # Process files in parallel
    start_time = time.time()
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(check_single_file, [str(f) for f in npy_files]),
            total=len(npy_files),
            desc="Checking images"
        ))
    
    processing_time = time.time() - start_time
    logger.info(f"Processing completed in {processing_time:.2f} seconds")
    
    # Analyze results
    invalid_files = []
    valid_count = 0
    
    for file_path, is_valid, min_val, max_val in results:
        if is_valid:
            valid_count += 1
        else:
            invalid_files.append((file_path, min_val, max_val))
            logger.warning(f"Invalid image: {Path(file_path).name} - min={min_val:.4f}, max={max_val:.4f}")
    
    logger.info(f"Found {len(invalid_files)} invalid images out of {len(npy_files)} total")
    logger.info(f"Valid images: {valid_count}")
    
    if dry_run:
        logger.info("DRY RUN - No files will be removed")
        return
    
    # Remove invalid files and their corresponding .pkl files
    logger.info("Starting file removal...")
    removed_count = 0
    
    for file_path, min_val, max_val in tqdm(invalid_files, desc="Removing invalid files"):
        try:
            npy_file = Path(file_path)
            # Remove .npy file
            npy_file.unlink()
            
            # Remove corresponding .pkl file if it exists
            pkl_file = npy_file.with_suffix('.pkl')
            if pkl_file.exists():
                pkl_file.unlink()
            
            removed_count += 1
            
        except Exception as e:
            logger.error(f"Error removing {file_path}: {e}")
    
    logger.info(f"Successfully removed {removed_count} invalid image files")


def main():
    parser = argparse.ArgumentParser(
        description="Clean preprocessed images by removing those with invalid intensities (parallel version)"
    )
    parser.add_argument(
        "--data_dir",
        help="Directory containing preprocessed images (.npy files)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be removed without actually removing files"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count - 1)"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Starting parallel image cleaning process")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Workers: {args.workers or 'auto'}")
    
    remove_invalid_images_parallel(args.data_dir, args.dry_run, args.workers)
    
    logger.info("Image cleaning process completed")


if __name__ == "__main__":
    main() 