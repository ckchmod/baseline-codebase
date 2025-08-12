#!/usr/bin/env python3
"""
Script to visualize FOMO fine-tuning data.
Displays axial, coronal, and sagittal views of preprocessed brain imaging data.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import random
import nibabel as nib
import pickle
from pathlib import Path

def load_nifti_data(file_path):
    """Load NIfTI data."""
    img = nib.load(file_path)
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    return data, affine, header

def display_orthogonal_views(data, title, save_path=None):
    """Display axial, coronal, and sagittal views of 3D data."""
    # Use middle slices for each view
    slice_indices = {
        'axial': data.shape[2] // 2,
        'coronal': data.shape[1] // 2,
        'sagittal': data.shape[0] // 2
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Axial view (xy plane)
    axes[0].imshow(data[:, :, slice_indices['axial']].T, cmap='gray', origin='lower')
    axes[0].set_title(f'Axial View (Slice {slice_indices["axial"]})')
    axes[0].axis('off')
    
    # Coronal view (xz plane)
    axes[1].imshow(data[:, slice_indices['coronal'], :].T, cmap='gray', origin='lower')
    axes[1].set_title(f'Coronal View (Slice {slice_indices["coronal"]})')
    axes[1].axis('off')
    
    # Sagittal view (yz plane)
    axes[2].imshow(data[slice_indices['sagittal'], :, :].T, cmap='gray', origin='lower')
    axes[2].set_title(f'Sagittal View (Slice {slice_indices["sagittal"]})')
    axes[2].axis('off')
    
    plt.suptitle(title, fontsize=16, y=0.95)
    plt.tight_layout()
    
    if save_path:
        try:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        except Exception as e:
            print(f"Error saving visualization to {save_path}: {str(e)}")
    
    plt.show()
    
    return slice_indices

def display_task2_data(data, title_prefix, save_path_prefix):
    """Display Task 2 data with multiple channels."""
    # The data has shape (4, height, width, depth)
    # We'll visualize each channel separately
    channel_names = ['T1-weighted', 'T2-weighted', 'FLAIR', 'Segmentation']
    
    # Convert object array to regular array
    if data.dtype == object:
        # Create a new array with the same shape but float dtype
        converted_data = np.zeros(data.shape, dtype=np.float32)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    for l in range(data.shape[3]):
                        converted_data[i, j, k, l] = data[i, j, k, l]
        data = converted_data
    
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    
    for i in range(min(4, data.shape[0])):
        channel_data = data[i]
        title = f"{title_prefix} - {channel_names[i] if i < len(channel_names) else f'Channel {i}'}"
        save_path = f"{save_path_prefix}_channel_{i}.png"
        
        print(f"\nChannel {i} ({channel_names[i] if i < len(channel_names) else f'Channel {i}'}):")
        print(f"  Shape: {channel_data.shape}")
        print(f"  Range: [{channel_data.min():.3f}, {channel_data.max():.3f}]")
        print(f"  Mean: {channel_data.mean():.3f}")
        print(f"  Std: {channel_data.std():.3f}")
        
        # Display orthogonal views
        slice_indices = display_orthogonal_views(channel_data, title, save_path)

def visualize_task1_data(task_dir, num_examples=3):
    """Visualize examples from FOMO task 1."""
    print(f"\n{'='*80}")
    print(f"Processing FOMO Task 1: {task_dir}")
    print(f"{'='*80}")
    
    # Visualize skull-stripped data
    skull_stripped_dir = os.path.join(task_dir, "skull_stripped")
    
    if not os.path.exists(skull_stripped_dir):
        print(f"No skull_stripped directory found in {task_dir}")
        return
    
    # Get all subject directories
    subject_dirs = [d for d in os.listdir(skull_stripped_dir) 
                   if os.path.isdir(os.path.join(skull_stripped_dir, d))]
    
    print(f"Total subjects found: {len(subject_dirs)}")
    
    if len(subject_dirs) == 0:
        print("No subject directories found.")
        return
    
    # Select random subjects to visualize
    selected_subjects = random.sample(subject_dirs, min(num_examples, len(subject_dirs)))
    print(f"\nSelected subjects for visualization:")
    for i, subject in enumerate(selected_subjects, 1):
        print(f"{i}. {subject}")
    
    # Create output directory for saved images
    output_dir = f"/tmp/fomo_visualizations/task1"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process and display each selected subject
    for i, subject in enumerate(selected_subjects, 1):
        print(f"\n{'-'*60}")
        print(f"Subject {i}: {subject}")
        print(f"{'-'*60}")
        
        subject_dir = os.path.join(skull_stripped_dir, subject)
        session_dir = os.path.join(subject_dir, "ses_1")
        
        # Look for NIfTI files (skull-stripped images)
        nifti_files = [f for f in os.listdir(session_dir) if f.startswith('ss_') and f.endswith('.nii.gz')]
        
        if len(nifti_files) == 0:
            print(f"No skull-stripped NIfTI files found in {session_dir}")
            continue
            
        for nifti_file in nifti_files:
            nifti_path = os.path.join(session_dir, nifti_file)
            
            try:
                # Load data
                data, affine, header = load_nifti_data(nifti_path)
                
                print(f"File: {nifti_file}")
                print(f"Data shape: {data.shape}")
                print(f"Data type: {data.dtype}")
                print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
                print(f"Data mean: {data.mean():.3f}")
                print(f"Data std: {data.std():.3f}")
                
                # Display orthogonal views
                save_path = os.path.join(output_dir, f"{subject}_{nifti_file.replace('.nii.gz', '')}_views.png")
                slice_indices = display_orthogonal_views(data, f"Task 1 - {subject} - {nifti_file}", save_path)
                
            except Exception as e:
                print(f"Error processing {nifti_file}: {str(e)}")
                continue

def visualize_task2_data(task_dir, num_examples=3):
    """Visualize examples from FOMO task 2."""
    print(f"\n{'='*80}")
    print(f"Processing FOMO Task 2: {task_dir}")
    print(f"{'='*80}")
    
    preprocessed_dir = os.path.join(task_dir, "preprocessed_2")
    
    if not os.path.exists(preprocessed_dir):
        print(f"No preprocessed_2 directory found in {task_dir}")
        return
    
    # Get all .npy files
    npy_files = [f for f in os.listdir(preprocessed_dir) if f.endswith('.npy')]
    print(f"Total preprocessed files found: {len(npy_files)}")
    
    if len(npy_files) == 0:
        print("No .npy files found in this directory.")
        return
        
    # Select random files to visualize
    selected_files = random.sample(npy_files, min(num_examples, len(npy_files)))
    print(f"\nSelected files for visualization:")
    for i, file in enumerate(selected_files, 1):
        print(f"{i}. {file}")
    
    # Create output directory for saved images
    output_dir = f"/tmp/fomo_visualizations/task2"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process and display each selected file
    for i, filename in enumerate(selected_files, 1):
        print(f"\n{'-'*60}")
        print(f"File {i}: {filename}")
        print(f"{'-'*60}")
        
        file_path = os.path.join(preprocessed_dir, filename)
        
        try:
            # Load data with allow_pickle=True
            data = np.load(file_path, allow_pickle=True)
            
            # Display data for each channel
            save_path_prefix = os.path.join(output_dir, f"{filename.replace('.npy', '')}")
            display_task2_data(data, f"Task 2 - {filename}", save_path_prefix)
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

def visualize_task3_data(task_dir, num_examples=3):
    """Visualize examples from FOMO task 3."""
    print(f"\n{'='*80}")
    print(f"Processing FOMO Task 3: {task_dir}")
    print(f"{'='*80}")
    
    preprocessed_dir = os.path.join(task_dir, "preprocessed_2")
    
    if not os.path.exists(preprocessed_dir):
        print(f"No preprocessed_2 directory found in {task_dir}")
        return
    
    # Get all subject directories
    subject_dirs = [d for d in os.listdir(preprocessed_dir) 
                   if os.path.isdir(os.path.join(preprocessed_dir, d))]
    
    print(f"Total subjects found: {len(subject_dirs)}")
    
    if len(subject_dirs) == 0:
        print("No subject directories found.")
        return
    
    # Select random subjects to visualize
    selected_subjects = random.sample(subject_dirs, min(num_examples, len(subject_dirs)))
    print(f"\nSelected subjects for visualization:")
    for i, subject in enumerate(selected_subjects, 1):
        print(f"{i}. {subject}")
    
    # Create output directory for saved images
    output_dir = f"/tmp/fomo_visualizations/task3"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process and display each selected subject
    for i, subject in enumerate(selected_subjects, 1):
        print(f"\n{'-'*60}")
        print(f"Subject {i}: {subject}")
        print(f"{'-'*60}")
        
        subject_dir = os.path.join(preprocessed_dir, subject)
        session_dir = os.path.join(subject_dir, "ses_1")
        
        # Look for NIfTI files
        nifti_files = [f for f in os.listdir(session_dir) if f.endswith('.nii.gz')]
        
        if len(nifti_files) == 0:
            print(f"No NIfTI files found in {session_dir}")
            continue
            
        for nifti_file in nifti_files:
            nifti_path = os.path.join(session_dir, nifti_file)
            
            try:
                # Load data
                data, affine, header = load_nifti_data(nifti_path)
                
                print(f"File: {nifti_file}")
                print(f"Data shape: {data.shape}")
                print(f"Data type: {data.dtype}")
                print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
                print(f"Data mean: {data.mean():.3f}")
                print(f"Data std: {data.std():.3f}")
                
                # Display orthogonal views
                save_path = os.path.join(output_dir, f"{subject}_{nifti_file.replace('.nii.gz', '')}_views.png")
                slice_indices = display_orthogonal_views(data, f"Task 3 - {subject} - {nifti_file}", save_path)
                
            except Exception as e:
                print(f"Error processing {nifti_file}: {str(e)}")
                continue

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Configure matplotlib for better visualization
    plt.rcParams['figure.figsize'] = (15, 10)
    plt.rcParams['image.cmap'] = 'gray'
    plt.rcParams['image.interpolation'] = 'nearest'
    
    # Define paths to FOMO task directories
    base_dir = "fomo-fine-tuning"
    task_dirs = {
        "task1": os.path.join(base_dir, "fomo-task1"),
        "task2": os.path.join(base_dir, "fomo-task2"),
        "task3": os.path.join(base_dir, "fomo-task3")
    }
    
    print("FOMO Data Visualization Script")
    print("="*50)
    
    # Check which task directories exist
    existing_task_dirs = {task: path for task, path in task_dirs.items() if os.path.exists(path)}
    
    if not existing_task_dirs:
        print(f"No FOMO task directories found in {base_dir}")
        return
    
    print(f"Found {len(existing_task_dirs)} FOMO task directories:")
    for task, task_dir in existing_task_dirs.items():
        print(f"{task}: {task_dir}")
    
    # Visualize all tasks
    print("\nVisualizing all tasks...")
    for task, task_dir in existing_task_dirs.items():
        if task == "task1":
            visualize_task1_data(task_dir)
        elif task == "task2":
            visualize_task2_data(task_dir)
        elif task == "task3":
            visualize_task3_data(task_dir)

if __name__ == "__main__":
    main()