#!/usr/bin/env python3
import os
import subprocess
import multiprocessing as mp
from pathlib import Path
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed


def download_lfs_file(file_path):
    """
    Download a single Git LFS file using git lfs pull.
    
    Args:
        file_path: Path to the Git LFS pointer file
        
    Returns:
        tuple: (success, file_path, error_message)
    """
    try:
        # Change to the directory containing the file
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        
        # Run git lfs pull for this specific file
        result = subprocess.run(
            ['git', 'lfs', 'pull', '--include', file_name],
            cwd=file_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            return (True, file_path, None)
        else:
            return (False, file_path, result.stderr)
            
    except subprocess.TimeoutExpired:
        return (False, file_path, "Timeout")
    except Exception as e:
        return (False, file_path, str(e))


def batch_download_lfs_files(file_paths, num_workers=None, batch_size=100):
    """
    Download Git LFS files in parallel batches.
    
    Args:
        file_paths: List of Git LFS pointer file paths
        num_workers: Number of parallel workers
        batch_size: Number of files to process in each batch
        
    Returns:
        tuple: (successful_downloads, failed_downloads)
    """
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    
    print(f"Downloading {len(file_paths)} Git LFS files using {num_workers} workers")
    print(f"Processing in batches of {batch_size}")
    
    successful_downloads = []
    failed_downloads = []
    
    # Process files in batches
    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(file_paths) + batch_size - 1) // batch_size
        
        print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch)} files)")
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all files in this batch
            future_to_file = {executor.submit(download_lfs_file, file_path): file_path 
                            for file_path in batch}
            
            # Process completed downloads
            for future in as_completed(future_to_file):
                success, file_path, error = future.result()
                if success:
                    successful_downloads.append(file_path)
                    print(f"✓ Downloaded: {os.path.basename(file_path)}")
                else:
                    failed_downloads.append((file_path, error))
                    print(f"✗ Failed: {os.path.basename(file_path)} - {error}")
    
    return successful_downloads, failed_downloads


def get_lfs_pointer_files(root_dir):
    """
    Get all Git LFS pointer files in the directory tree, ignoring already downloaded files.
    
    Args:
        root_dir: Root directory to search
        
    Returns:
        list: List of Git LFS pointer file paths (excluding already downloaded files)
    """
    lfs_files = []
    
    for file_path in Path(root_dir).rglob("*.nii.gz"):
        try:
            # First check if it's already a real gzip file
            with open(file_path, 'rb') as f:
                header = f.read(10)
                if header.startswith(b'\x1f\x8b'):
                    # This is already a real gzip file, skip it
                    continue
            
            # If not gzip, check if it's a Git LFS pointer
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                if first_line.startswith('version https://git-lfs.github.com/spec/v1'):
                    lfs_files.append(str(file_path))
        except:
            continue
    
    return lfs_files


def verify_downloads(file_paths):
    """
    Verify that files have been successfully downloaded.
    
    Args:
        file_paths: List of file paths to verify
        
    Returns:
        tuple: (verified_files, still_pointers)
    """
    verified_files = []
    still_pointers = []
    
    for file_path in file_paths:
        try:
            with open(file_path, 'rb') as f:
                header = f.read(10)
                if header.startswith(b'\x1f\x8b'):
                    verified_files.append(file_path)
                else:
                    still_pointers.append(file_path)
        except:
            still_pointers.append(file_path)
    
    return verified_files, still_pointers


def count_files_in_subdir(subdir):
    """Count files in a single subdirectory."""
    real_files = 0
    lfs_pointers = 0
    unknown_files = 0
    
    try:
        for file_path in Path(subdir).rglob("*.nii.gz"):
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(10)
                    if header.startswith(b'\x1f\x8b'):
                        real_files += 1
                    else:
                        # Check if it's a Git LFS pointer
                        f.seek(0)
                        first_line = f.readline().decode('utf-8', errors='ignore').strip()
                        if first_line.startswith('version https://git-lfs.github.com/spec/v1'):
                            lfs_pointers += 1
                        else:
                            unknown_files += 1
            except:
                unknown_files += 1
    except Exception as e:
        print(f"Error processing subdirectory {subdir}: {e}")
    
    return real_files, lfs_pointers, unknown_files


def count_file_types(root_dir, num_workers=None):
    """
    Count different types of files in the directory tree using parallel processing.
    
    Args:
        root_dir: Root directory to search
        num_workers: Number of parallel workers (default: CPU count - 1)
        
    Returns:
        dict: Counts of different file types
    """
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    
    root_path = Path(root_dir)
    
    # Get all immediate subdirectories (subjects)
    subdirs = [str(d) for d in root_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(subdirs)} subdirectories to analyze")
    print(f"Using {num_workers} parallel workers")
    
    # Process subdirectories in parallel
    with mp.Pool(processes=num_workers) as pool:
        results = list(pool.map(count_files_in_subdir, subdirs))
    
    # Aggregate results
    total_real_files = sum(r[0] for r in results)
    total_lfs_pointers = sum(r[1] for r in results)
    total_unknown_files = sum(r[2] for r in results)
    
    return {
        'real_files': total_real_files,
        'lfs_pointers': total_lfs_pointers,
        'unknown_files': total_unknown_files,
        'total': total_real_files + total_lfs_pointers + total_unknown_files
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Git LFS files in parallel")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing Git LFS files")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for processing")
    parser.add_argument("--verify", action="store_true", help="Verify downloads after completion")
    parser.add_argument("--dry_run", action="store_true", help="Only count files, don't download")
    
    args = parser.parse_args()
    
    print(f"Analyzing files in: {args.root_dir}")
    start_time = time.time()
    
    # Count file types first
    if args.num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    else:
        num_workers = args.num_workers
    
    file_counts = count_file_types(args.root_dir, num_workers)
    
    print(f"\nFile Analysis:")
    print(f"  Total .nii.gz files: {file_counts['total']}")
    print(f"  Already downloaded (real gzip): {file_counts['real_files']}")
    print(f"  Git LFS pointers (need download): {file_counts['lfs_pointers']}")
    print(f"  Unknown files: {file_counts['unknown_files']}")
    
    if args.dry_run:
        print("\nDry run completed. No files downloaded.")
        exit(0)
    
    if file_counts['lfs_pointers'] == 0:
        print("\nNo Git LFS pointer files found to download!")
        exit(0)
    
    # Use parallel processing to find LFS pointer files
    # Get all subdirectories to search in parallel
    root_path = Path(args.root_dir)
    subdirs = [str(d) for d in root_path.rglob("*") if d.is_dir()]
    
    print(f"\nSearching {len(subdirs)} directories using {num_workers} workers...")
    
    # Process subdirectories in parallel to find LFS files
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(get_lfs_pointer_files, subdirs)
    
    # Combine results from all workers
    lfs_files = []
    for result in results:
        lfs_files.extend(result)
    
    scan_time = time.time() - start_time
    print(f"Found {len(lfs_files)} Git LFS pointer files to download in {scan_time:.2f} seconds")
    
    if not lfs_files:
        print("No Git LFS pointer files found!")
        exit(0)
    
    # Download files in parallel
    download_start_time = time.time()
    successful_downloads, failed_downloads = batch_download_lfs_files(
        lfs_files, args.num_workers, args.batch_size
    )
    download_time = time.time() - download_start_time
    
    print(f"\nDownload Summary:")
    print(f"  Successfully downloaded: {len(successful_downloads)}")
    print(f"  Failed downloads: {len(failed_downloads)}")
    print(f"  Download time: {download_time:.2f} seconds")
    
    if failed_downloads:
        print(f"\nFailed downloads (first 10):")
        for file_path, error in failed_downloads[:10]:
            print(f"  {os.path.basename(file_path)}: {error}")
        if len(failed_downloads) > 10:
            print(f"  ... and {len(failed_downloads) - 10} more")
    
    # Verify downloads if requested
    if args.verify:
        print(f"\nVerifying downloads...")
        verify_start_time = time.time()
        verified_files, still_pointers = verify_downloads(successful_downloads)
        verify_time = time.time() - verify_start_time
        
        print(f"Verification Summary:")
        print(f"  Verified as real files: {len(verified_files)}")
        print(f"  Still pointers: {len(still_pointers)}")
        print(f"  Verification time: {verify_time:.2f} seconds")
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds") 