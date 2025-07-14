#!/usr/bin/env python3
"""
Parallel File Copy Script

This script copies files from a source directory to a destination directory using
parallel processing for improved performance.

Usage:
    python parallel_copy.py <source_path> <destination_path> [options]

Options:
    --workers N     Number of worker processes (default: 4)
    --recursive     Copy directories recursively
    --preserve      Preserve file attributes (timestamps, permissions)
    --verbose       Show detailed progress
"""

import os
import sys
import shutil
import argparse
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time


def copy_file(args):
    """Copy a single file with error handling."""
    src, dst, preserve = args
    try:
        # Create destination directory if it doesn't exist
        dst_dir = os.path.dirname(dst)
        if dst_dir and not os.path.exists(dst_dir):
            os.makedirs(dst_dir, exist_ok=True)
        
        # Copy file
        if preserve:
            shutil.copy2(src, dst)
        else:
            shutil.copy(src, dst)
        
        return (src, dst, True, None)
    except Exception as e:
        return (src, dst, False, str(e))


def scan_directory_parallel(src_path, dst_path, recursive=False, workers=4):
    """Scan directory in parallel to get list of files to copy."""
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    
    if src_path.is_file():
        # Single file copy
        return [(str(src_path), str(dst_path))]
    
    if not recursive:
        # Non-recursive copy (only files in root directory)
        files_to_copy = []
        for item in src_path.iterdir():
            if item.is_file():
                dst_file = dst_path / item.name
                files_to_copy.append((str(item), str(dst_file)))
        return files_to_copy
    
    # Recursive copy with parallel scanning
    def scan_subdirectory(subdir):
        """Scan a single subdirectory and return files found."""
        files_in_subdir = []
        try:
            for root, dirs, files in os.walk(subdir):
                for file in files:
                    src_file = Path(root) / file
                    # Calculate relative path from source
                    rel_path = src_file.relative_to(src_path)
                    dst_file = dst_path / rel_path
                    files_in_subdir.append((str(src_file), str(dst_file)))
        except Exception as e:
            print(f"Warning: Error scanning {subdir}: {e}")
        return files_in_subdir
    
    # Get immediate subdirectories for parallel processing
    subdirs = [d for d in src_path.iterdir() if d.is_dir()]
    
    if not subdirs:
        # No subdirectories, scan root directory
        return scan_subdirectory(src_path)
    
    # Add files from root directory
    files_to_copy = []
    for item in src_path.iterdir():
        if item.is_file():
            dst_file = dst_path / item.name
            files_to_copy.append((str(item), str(dst_file)))
    
    # Scan subdirectories in parallel
    print(f"Scanning {len(subdirs)} subdirectories in parallel...")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(scan_subdirectory, subdir) for subdir in subdirs]
        
        for future in as_completed(futures):
            try:
                files_to_copy.extend(future.result())
            except Exception as e:
                print(f"Warning: Error in parallel scanning: {e}")
    
    return files_to_copy


def get_files_to_copy(src_path, dst_path, recursive=False, workers=4):
    """Get list of files to copy with their destination paths."""
    return scan_directory_parallel(src_path, dst_path, recursive, workers)


def main():
    parser = argparse.ArgumentParser(description='Parallel file copy utility')
    parser.add_argument('source', help='Source path (file or directory)')
    parser.add_argument('destination', help='Destination path')
    parser.add_argument('--workers', type=int, default=mp.cpu_count()-1,
                       help='Number of worker processes (default: number of cpus - 1)')
    parser.add_argument('--recursive', action='store_true',
                       help='Copy directories recursively')
    parser.add_argument('--preserve', action='store_true',
                       help='Preserve file attributes (timestamps, permissions)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed progress')
    
    args = parser.parse_args()
    
    # Validate source path
    if not os.path.exists(args.source):
        print(f"Error: Source path '{args.source}' does not exist")
        sys.exit(1)
    
    # Create destination directory if it doesn't exist
    dst_path = Path(args.destination)
    if not dst_path.exists():
        if args.source.endswith('/') or os.path.isdir(args.source):
            dst_path.mkdir(parents=True, exist_ok=True)
        else:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get list of files to copy
    print(f"Scanning source path: {args.source}")
    files_to_copy = get_files_to_copy(args.source, args.destination, args.recursive)
    
    if not files_to_copy:
        print("No files found to copy")
        return
    
    print(f"Found {len(files_to_copy)} files to copy")
    print(f"Using {args.workers} worker processes")
    
    # Prepare copy arguments
    copy_args = [(src, dst, args.preserve) for src, dst in files_to_copy]
    
    # Start parallel copy
    start_time = time.time()
    successful_copies = 0
    failed_copies = 0
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all copy tasks
        future_to_file = {executor.submit(copy_file, args): args[0] 
                         for args in copy_args}
        
        # Process completed tasks
        for future in as_completed(future_to_file):
            src, dst, success, error = future.result()
            
            if success:
                successful_copies += 1
                if args.verbose:
                    print(f"✓ Copied: {src} -> {dst}")
            else:
                failed_copies += 1
                print(f"✗ Failed: {src} -> {dst} (Error: {error})")
            
            # Progress update
            total = successful_copies + failed_copies
            if total % 10 == 0 or total == len(files_to_copy):
                print(f"Progress: {total}/{len(files_to_copy)} files processed")
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Summary
    print(f"\nCopy completed in {duration:.2f} seconds")
    print(f"Successful copies: {successful_copies}")
    print(f"Failed copies: {failed_copies}")
    
    if failed_copies > 0:
        sys.exit(1)


if __name__ == '__main__':
    main() 