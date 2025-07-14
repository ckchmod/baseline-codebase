#!/usr/bin/env python3
import os
import glob
from pathlib import Path

def check_file_type(file_path):
    """Check if a file is a real gzip file or Git LFS pointer"""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(100)
            if header.startswith(b'version https://git-lfs.github.com/spec/v1'):
                return 'lfs_pointer'
            else:
                return 'real_gzip'
    except Exception as e:
        return f'error: {str(e)}'

def main():
    data_dir = "/work/forkert_lab/fomo/FOMO-MRI"
    
    print(f"Scanning directory: {data_dir}")
    
    # Find all .nii.gz files
    pattern = os.path.join(data_dir, "**/*.nii.gz")
    files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(files)} .nii.gz files")
    
    # Sample first few files to check types
    print("\nChecking first 10 files:")
    real_gzip_count = 0
    lfs_pointer_count = 0
    error_count = 0
    
    for i, file_path in enumerate(files[:10]):
        file_type = check_file_type(file_path)
        rel_path = os.path.relpath(file_path, data_dir)
        print(f"  {i+1:2d}. {rel_path} -> {file_type}")
        
        if file_type == 'real_gzip':
            real_gzip_count += 1
        elif file_type == 'lfs_pointer':
            lfs_pointer_count += 1
        else:
            error_count += 1
    
    # Now check all files efficiently
    print(f"\nChecking all {len(files)} files...")
    real_gzip_total = 0
    lfs_pointer_total = 0
    error_total = 0
    
    for i, file_path in enumerate(files):
        if i % 1000 == 0:
            print(f"  Progress: {i}/{len(files)} files checked")
        
        file_type = check_file_type(file_path)
        if file_type == 'real_gzip':
            real_gzip_total += 1
        elif file_type == 'lfs_pointer':
            lfs_pointer_total += 1
        else:
            error_total += 1
    
    print(f"\nFinal Results:")
    print(f"  Total .nii.gz files: {len(files)}")
    print(f"  Real gzip files: {real_gzip_total}")
    print(f"  Git LFS pointers: {lfs_pointer_total}")
    print(f"  Errors: {error_total}")
    print(f"  Percentage real files: {real_gzip_total/len(files)*100:.1f}%")

if __name__ == "__main__":
    main() 