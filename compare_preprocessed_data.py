#!/usr/bin/env python3
"""
Script to compare preprocessed data from different directories.
Randomly selects one sample and compares the corresponding images from:
1. /work/forkert_lab/data/fomo/preprocessed_2/FOMO60k/
2. /work/forkert_lab/data/fomo/preprocessed_1
3. /work/forkert_lab/data/fomo/FOMO-MRI/fomo-60k/ (original data)

Only processes files that contain "skull_stripped" in their name.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pickle
import nibabel as nib
from pathlib import Path
import glob

def load_npy_data(file_path):
    """Load numpy array data."""
    data = np.load(file_path)
    
    # Load properties if available
    pkl_path = file_path.replace('.npy', '.pkl')
    properties = None
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            properties = pickle.load(f)
    
    return data, properties

def load_nii_data(file_path):
    """Load NIfTI data."""
    img = nib.load(file_path)
    data = img.get_fdata()
    return data, img.header

def find_corresponding_files(base_filename, dir1, dir2, dir3):
    """Find corresponding files in the three directories."""
    # Extract subject and session info from filename
    # Example: sub_10069_ses_1_t1_skull_stripped.npy
    parts = base_filename.replace('.npy', '').split('_')
    if len(parts) >= 4:
        subject = f"sub_{parts[1]}"
        session = f"ses_{parts[3]}"
        modality = parts[4] if len(parts) > 4 else 't1'
        
        print(f"Looking for: Subject={subject}, Session={session}, Modality={modality}")
        
        # Find file in dir1 (preprocessed_2)
        file1 = os.path.join(dir1, base_filename)
        
        # Find file in dir2 (kimberly/preprocessed)
        # Look for .nii.gz files with skull_stripped
        pattern2 = os.path.join(dir2, subject, session, f"{modality}_skull_stripped.nii.gz")
        file2 = pattern2 if os.path.exists(pattern2) else None
        
        # Find file in dir3 (original data)
        # This might be more complex, let's try different patterns
        possible_patterns3 = [
            os.path.join(dir3, subject, session, f"{modality}_skull_stripped.nii.gz"),
            os.path.join(dir3, subject, session, f"{modality}.nii.gz"),
            os.path.join(dir3, subject, session, f"{modality}_brain.nii.gz"),
            os.path.join(dir3, subject, session, f"{modality}_orig.nii.gz"),
        ]
        file3 = None
        for pattern in possible_patterns3:
            if os.path.exists(pattern):
                file3 = pattern
                break
        
        return file1, file2, file3
    
    return None, None, None

def display_comparison(data1, data2, data3, titles, save_path=None):
    """Display comparison of three datasets."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # Use middle slices for each view
    slice_indices = {
        'axial': data1.shape[0] // 2 if data1 is not None else 0,
        'coronal': data1.shape[1] // 2 if data1 is not None else 0,
        'sagittal': data1.shape[2] // 2 if data1 is not None else 0
    }
    
    datasets = [data1, data2, data3]
    
    for i, (data, title) in enumerate(zip(datasets, titles)):
        if data is None:
            axes[0, i].text(0.5, 0.5, 'File not found', ha='center', va='center', transform=axes[0, i].transAxes)
            axes[1, i].text(0.5, 0.5, 'File not found', ha='center', va='center', transform=axes[1, i].transAxes)
            axes[2, i].text(0.5, 0.5, 'File not found', ha='center', va='center', transform=axes[2, i].transAxes)
            axes[0, i].set_title(f'{title}\n(Not found)')
            axes[1, i].set_title(f'{title}\n(Not found)')
            axes[2, i].set_title(f'{title}\n(Not found)')
            continue
            
        # Axial view
        axes[0, i].imshow(data[slice_indices['axial'], :, :], cmap='gray')
        axes[0, i].set_title(f'{title}\nAxial (Slice {slice_indices["axial"]})')
        axes[0, i].axis('off')
        
        # Coronal view
        axes[1, i].imshow(data[:, slice_indices['coronal'], :], cmap='gray')
        axes[1, i].set_title(f'{title}\nCoronal (Slice {slice_indices["coronal"]})')
        axes[1, i].axis('off')
        
        # Sagittal view
        axes[2, i].imshow(data[:, :, slice_indices['sagittal']], cmap='gray')
        axes[2, i].set_title(f'{title}\nSagittal (Slice {slice_indices["sagittal"]})')
        axes[2, i].axis('off')
    
    plt.suptitle('Comparison of Preprocessed Data from Different Sources', fontsize=16, y=0.95)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    
    plt.show()
    
    return slice_indices

def print_data_info(data, title, properties=None):
    """Print information about the data."""
    if data is None:
        print(f"{title}: File not found")
        return
    
    print(f"\n{title}:")
    print(f"  Shape: {data.shape}")
    print(f"  Data type: {data.dtype}")
    print(f"  Range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"  Mean: {data.mean():.3f}")
    print(f"  Std: {data.std():.3f}")
    
    if properties:
        print(f"  Properties keys: {list(properties.keys())}")
        if 'spacing' in properties:
            print(f"  Spacing: {properties['spacing']}")

def main():
    # Set random seed for reproducibility
    # random.seed(42)
    # np.random.seed(42)
    
    # Configure matplotlib
    plt.rcParams['figure.figsize'] = (15, 10)
    plt.rcParams['image.cmap'] = 'gray'
    plt.rcParams['image.interpolation'] = 'nearest'
    
    # Define directories
    dir1 = "/work/forkert_lab/data/fomo/preprocessed_2/FOMO60k"
    dir2 = "/work/forkert_lab/data/fomo/preprocessed_1"
    dir3 = "/work/forkert_lab/data/fomo/FOMO-MRI/fomo-60k"
    
    # Get all .npy files from dir1 that contain "skull_stripped"
    npy_files = [f for f in os.listdir(dir1) if f.endswith('.npy') and 'skull_stripped' in f]
    print(f"Total skull_stripped files in preprocessed_2: {len(npy_files)}")
    
    if len(npy_files) == 0:
        print("No skull_stripped files found in the directory!")
        return
    
    # Randomly select one file
    selected_file = random.choice(npy_files)
    print(f"\nSelected file: {selected_file}")
    
    # Find corresponding files in other directories
    file1, file2, file3 = find_corresponding_files(selected_file, dir1, dir2, dir3)
    
    print(f"\nFile paths:")
    print(f"Dir1 (preprocessed_2): {file1}")
    print(f"Dir2 (preprocessed_1): {file2}")
    print(f"Dir3 (original): {file3}")
    
    # Load data
    data1, properties1 = load_npy_data(file1) if file1 else (None, None)
    data2, header2 = load_nii_data(file2) if file2 else (None, None)
    data3, header3 = load_nii_data(file3) if file3 else (None, None)
    
    # Print information about each dataset
    print_data_info(data1, "Preprocessed_2", properties1)
    print_data_info(data2, "preprocessed_1")
    print_data_info(data3, "Original Data")
    
    # Create output directory
    output_dir = "comparison_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Display comparison
    titles = ["Preprocessed_2", "Preprocessed_1", "Original"]
    save_path = os.path.join(output_dir, f"{selected_file.replace('.npy', '')}_comparison.png")
    
    slice_indices = display_comparison(data1, data2, data3, titles, save_path)
    
    # Additional analysis: compute differences if all datasets are available
    if data1 is not None and data2 is not None:
        print(f"\nComparing preprocessed_2 vs preprocessed_1:")
        if data1.shape == data2.shape:
            diff = data1 - data2
            print(f"  Mean absolute difference: {np.mean(np.abs(diff)):.6f}")
            print(f"  Max absolute difference: {np.max(np.abs(diff)):.6f}")
            print(f"  Correlation: {np.corrcoef(data1.flatten(), data2.flatten())[0,1]:.6f}")
        else:
            print(f"  Shapes don't match: {data1.shape} vs {data2.shape}")
    
    if data1 is not None and data3 is not None:
        print(f"\nComparing preprocessed_2 vs original:")
        if data1.shape == data3.shape:
            diff = data1 - data3
            print(f"  Mean absolute difference: {np.mean(np.abs(diff)):.6f}")
            print(f"  Max absolute difference: {np.max(np.abs(diff)):.6f}")
            print(f"  Correlation: {np.corrcoef(data1.flatten(), data3.flatten())[0,1]:.6f}")
        else:
            print(f"  Shapes don't match: {data1.shape} vs {data3.shape}")
    
    print(f"\nComparison saved to: {output_dir}")

if __name__ == "__main__":
    main() 