import numpy as np
import argparse
import os
import multiprocessing as mp
from functools import partial
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    save_pickle,
    maybe_mkdir_p as ensure_dir_exists,
)
from yucca.functional.preprocessing import preprocess_case_for_training_without_label
from yucca.functional.utils.loading import read_file_to_nifti_or_np
from utils.utils import parallel_process


def search_files_in_subdir(subdir):
    """
    Search for .nii.gz files in a single subdirectory.
    
    Args:
        subdir: Path to subdirectory to search
        
    Returns:
        list: List of (subject_name, session_name, scan_file, scan_path) tuples
    """
    scan_infos = []
    skip_dirs = {'.git', '__pycache__', '.DS_Store'}
    
    # Extract subject name from the subdir path
    subject_name = os.path.basename(subdir)
    
    try:
        for root, dirs, files in os.walk(subdir):
            # Skip unwanted directories
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            
            # Only process files that end with .nii.gz AND contain "skull_stripped" in their name
            scan_files = [f for f in files if f.endswith(".nii.gz") and "skull_stripped" in f]
            
            for scan_file in scan_files:
                scan_path = os.path.join(root, scan_file)
                
                # Extract session name from the path
                # Path format: subdir/session_name/scan_file
                rel_path = os.path.relpath(scan_path, subdir)
                path_parts = rel_path.split(os.sep)
                
                if len(path_parts) >= 2:  # Ensure we have session/file structure
                    session_name = path_parts[0]
                    scan_infos.append((subject_name, session_name, scan_file, scan_path))
    except Exception as e:
        print(f"Error searching {subdir}: {e}")
    
    return scan_infos


def process_single_scan(scan_info, preprocess_config, target_dir, skip_existing=True):
    """
    Process a single scan for pretraining data.

    Args:
        scan_info: A tuple containing (subject_name, session_name, scan_file, scan_path)
        preprocess_config: Preprocessing configuration dictionary
        target_dir: Target directory for preprocessed data
        skip_existing: If True, skip processing if files already exist

    Returns:
        Success message or error message
    """
    subject_name, session_name, scan_file, scan_path = scan_info

    # Extract filename without extension to use as identifier
    scan_name = os.path.splitext(os.path.splitext(scan_file)[0])[0]

    # Create a unique filename for the preprocessed data
    filename = f"{subject_name}_{session_name}_{scan_name}"
    save_path = join(target_dir, filename)
    
    # Check if preprocessed files already exist
    if skip_existing and os.path.exists(save_path + ".npy") and os.path.exists(save_path + ".pkl"):
        print(f"Skipped {subject_name}/{session_name}/{scan_file} (already exists)")
        return

    # Check if file is a Git LFS pointer before processing
    try:
        with open(scan_path, 'rb') as f:
            header = f.read(100)  # Read first 100 bytes
            if header.startswith(b'version https://git-lfs.github.com/spec/v1'):
                print(f"Skipped {subject_name}/{session_name}/{scan_file} (Git LFS pointer - not downloaded)")
                return
    except Exception as e:
        print(f"Error checking file type for {subject_name}/{session_name}/{scan_file}: {str(e)}")
        return

    try:
        images, image_props = preprocess_case_for_training_without_label(
            images=[read_file_to_nifti_or_np(scan_path)], **preprocess_config
        )
        image = images[0]

        np.save(save_path + ".npy", image)
        save_pickle(image_props, save_path + ".pkl")

        print(f"Processed {subject_name}/{session_name}/{scan_file}")
    except Exception as e:
        print(f"Error processing {subject_name}/{session_name}/{scan_file}: {str(e)}")


def preprocess_pretrain_data(in_path: str, out_path: str, num_workers: int = None, skip_existing: bool = True):
    """
    Preprocess all pretraining data in parallel.

    Args:
        in_path: Path to the source data directory
        out_path: Path to store preprocessed data
        num_workers: Number of parallel workers (default: CPU count - 1)
        skip_existing: If True, skip processing if files already exist
    """
    target_dir = join(out_path, "FOMO60k")
    ensure_dir_exists(target_dir)

    preprocess_config = {
        "normalization_operation": ["volume_wise_znorm"],
        "crop_to_nonzero": True,
        "target_orientation": "RAS",
        "target_spacing": [1.0, 1.0, 1.0],
        "keep_aspect_ratio_when_using_target_size": False,
        "transpose": [0, 1, 2],
    }

    print(f"Preprocessing configuration: {preprocess_config}")
    print(f"Target directory: {target_dir}")
    print(f"Skip existing: {skip_existing}")

    # Get all immediate subdirectories (subjects) for parallel search
    subdirs = [os.path.join(in_path, d) for d in os.listdir(in_path) 
               if os.path.isdir(os.path.join(in_path, d))]
    
    print(f"Found {len(subdirs)} subdirectories to search")
    print(f"Using {num_workers} parallel workers for file discovery")
    
    # Search for files in parallel
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(search_files_in_subdir, subdirs)
    
    # Combine results from all workers
    scan_infos = []
    for result in results:
        scan_infos.extend(result)
    
    print(f"Found {len(scan_infos)} scans to process")
    
    # Sort for consistent processing order
    scan_infos.sort()

    # Create partial function with fixed arguments
    process_func = partial(
        process_single_scan, preprocess_config=preprocess_config, target_dir=target_dir, skip_existing=skip_existing
    )

    # Process all scans in parallel using the shared utility function
    print(f"Processing {len(scan_infos)} scans in parallel using {num_workers} workers")
    parallel_process(process_func, scan_infos, num_workers, desc="Preprocessing scans")
    print(f"Preprocessing completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_path", type=str, required=True, help="Path to pretrain data"
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Path to put preprocessed pretrain data",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers to use. Default is CPU count - 1",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="If set, reprocess files even if they already exist (default: skip existing files)",
    )
    args = parser.parse_args()
    preprocess_pretrain_data(
        in_path=args.in_path, out_path=args.out_path, num_workers=args.num_workers, skip_existing=not args.no_skip_existing
    )