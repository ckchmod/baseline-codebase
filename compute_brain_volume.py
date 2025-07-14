#! /usr/bin/env python3

import os
import glob
import numpy as np
import nibabel as nib
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import argparse
from pathlib import Path


def calculate_brain_volume(image_data, voxel_size=None):
    """
    Calculate brain volume from MRI image data.
    
    Parameters:
    image_data: numpy array of the image data
    voxel_size: tuple of voxel dimensions in mm (optional)
    
    Returns:
    dict: Dictionary containing volume information
    """
    
    # Count non-zero voxels (brain tissue)
    non_zero_voxels = np.count_nonzero(image_data)
    total_voxels = image_data.size
    
    # Calculate volume in voxels
    volume_voxels = non_zero_voxels
    
    # Calculate volume in cubic mm if voxel size is provided
    volume_mm3 = None
    if voxel_size is not None:
        if len(voxel_size) == 3:
            voxel_volume = voxel_size[0] * voxel_size[1] * voxel_size[2]
            volume_mm3 = volume_voxels * voxel_volume
        else:
            # If voxel size is isotropic, use the first value
            voxel_volume = voxel_size[0] ** 3
            volume_mm3 = volume_voxels * voxel_volume
    
    # Calculate volume in cubic cm
    volume_cm3 = volume_mm3 / 1000 if volume_mm3 is not None else None
    
    return {
        'volume_voxels': volume_voxels,
        'volume_mm3': volume_mm3,
        'volume_cm3': volume_cm3,
        'total_voxels': total_voxels,
        'brain_ratio': volume_voxels / total_voxels if total_voxels > 0 else 0
    }


def process_single_file(file_path):
    """
    Process a single nifti file and return its volume information.
    
    Parameters:
    file_path: Path to the nifti file
    
    Returns:
    dict: Dictionary containing file info and volume data
    """
    try:
        # Load the image
        img = nib.load(file_path)
        img_data = img.get_fdata()
        
        # Get voxel size from header
        header = img.header
        voxel_size = None
        try:
            # Try to get voxel size from header
            if hasattr(header, 'get_zooms'):
                voxel_size = header.get_zooms()[:3]  # Get first 3 dimensions
            elif hasattr(header, 'pixdim'):
                voxel_size = header['pixdim'][1:4]  # Get pixdim 1-3
        except:
            # If we can't get voxel size, we'll calculate volume in voxels only
            pass
        
        # Calculate brain volume
        volume_info = calculate_brain_volume(img_data, voxel_size)
        
        # Get file information
        file_name = os.path.basename(file_path)
        file_dir = os.path.dirname(file_path)
        
        # Extract subject and session info from path
        path_parts = Path(file_path).parts
        subject = None
        session = None
        
        for i, part in enumerate(path_parts):
            if part.startswith('sub_'):
                subject = part
            elif part.startswith('ses_'):
                session = part
        
        # Get image dimensions
        dimensions = img_data.shape
        
        return {
            'file_path': file_path,
            'file_name': file_name,
            'subject': subject,
            'session': session,
            'dimensions': f"{dimensions[0]}x{dimensions[1]}x{dimensions[2]}",
            'volume_voxels': volume_info['volume_voxels'],
            'volume_mm3': volume_info['volume_mm3'],
            'volume_cm3': volume_info['volume_cm3'],
            'total_voxels': volume_info['total_voxels'],
            'brain_ratio': volume_info['brain_ratio'],
            'voxel_size_mm': str(voxel_size) if voxel_size is not None else None,
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'subject': None,
            'session': None,
            'dimensions': None,
            'volume_voxels': None,
            'volume_mm3': None,
            'volume_cm3': None,
            'total_voxels': None,
            'brain_ratio': None,
            'voxel_size_mm': None,
            'status': f'error: {str(e)}'
        }


def main():
    parser = argparse.ArgumentParser(description='Process nifti files and calculate brain volumes')
    parser.add_argument('--data_dir', type=str, 
                       default='/data/Data/FOMO-MRI/fomo-60k',
                       help='Directory containing nifti files')
    parser.add_argument('--output_csv', type=str, 
                       default='brain_volumes.csv',
                       help='Output CSV file path')
    parser.add_argument('--n_workers', type=int, 
                       default=None,
                       help='Number of worker processes (default: number of CPU cores)')
    parser.add_argument('--pattern', type=str, 
                       default='**/*.nii.gz',
                       help='File pattern to search for')
    
    args = parser.parse_args()
    
    # Find all nifti files
    search_pattern = os.path.join(args.data_dir, args.pattern)
    nifti_files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(nifti_files)} nifti files in {args.data_dir}")
    
    if len(nifti_files) == 0:
        print("No files found. Exiting.")
        return
    
    # Set number of workers
    if args.n_workers is None:
        args.n_workers = cpu_count()
    
    print(f"Using {args.n_workers} worker processes")
    
    # Process files in parallel with progress bar
    results = []
    with Pool(processes=args.n_workers) as pool:
        # Use tqdm to show progress
        for result in tqdm(pool.imap(process_single_file, nifti_files), 
                         total=len(nifti_files), 
                         desc="Processing files"):
            results.append(result)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Add some statistics
    successful_results = df[df['status'] == 'success']
    error_results = df[df['status'] != 'success']
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {len(successful_results)} files")
    print(f"Errors: {len(error_results)} files")
    
    if len(successful_results) > 0:
        print(f"\nVolume statistics (successful files):")
        
        # Volume in voxels
        if 'volume_voxels' in successful_results.columns:
            vol_voxels = successful_results['volume_voxels'].dropna()
            if len(vol_voxels) > 0:
                print(f"  Volume (voxels):")
                print(f"    Mean: {vol_voxels.mean():.0f}")
                print(f"    Std: {vol_voxels.std():.0f}")
                print(f"    Min: {vol_voxels.min():.0f}")
                print(f"    Max: {vol_voxels.max():.0f}")
        
        # Volume in mm³
        if 'volume_mm3' in successful_results.columns:
            vol_mm3 = successful_results['volume_mm3'].dropna()
            if len(vol_mm3) > 0:
                print(f"  Volume (mm³):")
                print(f"    Mean: {vol_mm3.mean():.0f}")
                print(f"    Std: {vol_mm3.std():.0f}")
                print(f"    Min: {vol_mm3.min():.0f}")
                print(f"    Max: {vol_mm3.max():.0f}")
        
        # Volume in cm³
        if 'volume_cm3' in successful_results.columns:
            vol_cm3 = successful_results['volume_cm3'].dropna()
            if len(vol_cm3) > 0:
                print(f"  Volume (cm³):")
                print(f"    Mean: {vol_cm3.mean():.2f}")
                print(f"    Std: {vol_cm3.std():.2f}")
                print(f"    Min: {vol_cm3.min():.2f}")
                print(f"    Max: {vol_cm3.max():.2f}")
        
        # Brain ratio
        if 'brain_ratio' in successful_results.columns:
            brain_ratio = successful_results['brain_ratio'].dropna()
            if len(brain_ratio) > 0:
                print(f"  Brain ratio (brain voxels / total voxels):")
                print(f"    Mean: {brain_ratio.mean():.4f}")
                print(f"    Std: {brain_ratio.std():.4f}")
                print(f"    Min: {brain_ratio.min():.4f}")
                print(f"    Max: {brain_ratio.max():.4f}")
    
    # Save to CSV
    df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to: {args.output_csv}")
    
    # Show sample of results
    print(f"\nSample of results:")
    display_columns = ['file_name', 'subject', 'session', 'volume_voxels', 'volume_cm3', 'brain_ratio']
    available_columns = [col for col in display_columns if col in df.columns]
    print(df[available_columns].head(10).to_string(index=False))
    
    # Show error summary if any
    if len(error_results) > 0:
        print(f"\nError summary:")
        error_counts = error_results['status'].value_counts()
        for error_type, count in error_counts.items():
            print(f"  {error_type}: {count}")


if __name__ == "__main__":
    main() 