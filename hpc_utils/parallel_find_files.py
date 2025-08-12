#!/usr/bin/env python3
import os
import multiprocessing as mp
from pathlib import Path
from functools import partial
import time


def find_nifti_files_in_subdir(subdir):
    """
    Find all .nii.gz files in a single subdirectory.
    
    Args:
        subdir: Path to subdirectory to search
        
    Returns:
        list: List of .nii.gz file paths found in this subdirectory
    """
    nifti_files = []
    try:
        for file_path in Path(subdir).rglob("*.nii.gz"):
            nifti_files.append(str(file_path))
    except Exception as e:
        print(f"Error searching {subdir}: {e}")
    
    return nifti_files


def find_nifti_files_parallel(root_dir, num_workers=None):
    """
    Find all .nii.gz files in parallel using multiprocessing.
    
    Args:
        root_dir: Root directory to search
        num_workers: Number of parallel workers (default: CPU count - 1)
        
    Returns:
        list: List of all .nii.gz file paths found
    """
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    
    root_path = Path(root_dir)
    
    # Get all immediate subdirectories (subjects)
    subdirs = [str(d) for d in root_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(subdirs)} subdirectories to search")
    print(f"Using {num_workers} parallel workers")
    
    # Use multiprocessing to search subdirectories in parallel
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(find_nifti_files_in_subdir, subdirs)
    
    # Flatten results
    all_files = []
    for file_list in results:
        all_files.extend(file_list)
    
    return all_files


def check_single_file(file_path):
    """
    Check if a single file is a real NIfTI file or Git LFS pointer.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        tuple: (file_type, file_path) where file_type is 'real', 'lfs', 'unknown', or 'error'
    """
    try:
        with open(file_path, 'rb') as f:
            # Read first few bytes to check file type
            header = f.read(10)
            # Check for gzip magic number
            if header.startswith(b'\x1f\x8b'):
                return ('real', file_path)
            else:
                # Check if it's a Git LFS pointer
                f.seek(0)
                first_line = f.readline().decode('utf-8', errors='ignore').strip()
                if first_line.startswith('version https://git-lfs.github.com/spec/v1'):
                    return ('lfs', file_path)
                else:
                    return ('unknown', file_path)
    except Exception as e:
        return ('error', file_path)


def check_file_types_parallel(file_paths, num_workers=None):
    """
    Check file types in parallel to distinguish between real NIfTI files and Git LFS pointers.
    
    Args:
        file_paths: List of file paths to check
        num_workers: Number of parallel workers (default: CPU count - 1)
        
    Returns:
        tuple: (real_files, lfs_pointers, unknown_files, error_files)
    """
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    
    print(f"Checking file types using {num_workers} workers...")
    
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(check_single_file, file_paths)
    
    real_files = [path for file_type, path in results if file_type == 'real']
    lfs_pointers = [path for file_type, path in results if file_type == 'lfs']
    unknown_files = [path for file_type, path in results if file_type == 'unknown']
    error_files = [path for file_type, path in results if file_type == 'error']
    
    return real_files, lfs_pointers, unknown_files, error_files


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Find NIfTI files in parallel")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory to search")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--check_types", action="store_true", help="Also check file types")
    
    args = parser.parse_args()
    
    print(f"Searching for .nii.gz files in: {args.root_dir}")
    start_time = time.time()
    
    # Find all NIfTI files
    nifti_files = find_nifti_files_parallel(args.root_dir, args.num_workers)
    
    find_time = time.time() - start_time
    print(f"Found {len(nifti_files)} .nii.gz files in {find_time:.2f} seconds")
    
    if args.check_types:
        print("\nChecking file types...")
        type_start_time = time.time()
        
        real_files, lfs_pointers, unknown_files, error_files = check_file_types_parallel(
            nifti_files, args.num_workers
        )
        
        type_time = time.time() - type_start_time
        
        print(f"\nFile type analysis completed in {type_time:.2f} seconds:")
        print(f"  Real NIfTI files (gzip compressed): {len(real_files)}")
        print(f"  Git LFS pointer files: {len(lfs_pointers)}")
        print(f"  Unknown file types: {len(unknown_files)}")
        print(f"  Error reading files: {len(error_files)}")
        
        if lfs_pointers:
            print(f"\nSample Git LFS pointer files:")
            for i, path in enumerate(lfs_pointers[:5]):
                print(f"  {i+1}. {path}")
            if len(lfs_pointers) > 5:
                print(f"  ... and {len(lfs_pointers) - 5} more")
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds") 