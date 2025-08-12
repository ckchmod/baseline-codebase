#!/usr/bin/env python3
"""
Simple script to visualize preprocessed FOMO data.
Randomly selects 5 files and displays their axial, coronal, and sagittal views.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pickle
from pathlib import Path

def load_preprocessed_data(file_path):
    """Load preprocessed data and its properties."""
    # Load the numpy array
    data = np.load(file_path)
    
    # Load properties if available
    pkl_path = file_path.replace('.npy', '.pkl')
    properties = None
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            properties = pickle.load(f)
    
    return data, properties

def display_orthogonal_views(data, title, save_path=None):
    """Display axial, coronal, and sagittal views of 3D data."""
    # Use middle slices for each view
    slice_indices = {
        'axial': data.shape[0] // 2,
        'coronal': data.shape[1] // 2,
        'sagittal': data.shape[2] // 2
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Axial view (xy plane)
    axes[0].imshow(data[slice_indices['axial'], :, :], cmap='gray')
    axes[0].set_title(f'Axial View (Slice {slice_indices["axial"]})')
    axes[0].axis('off')
    
    # Coronal view (xz plane)
    axes[1].imshow(data[:, slice_indices['coronal'], :], cmap='gray')
    axes[1].set_title(f'Coronal View (Slice {slice_indices["coronal"]})')
    axes[1].axis('off')
    
    # Sagittal view (yz plane)
    axes[2].imshow(data[:, :, slice_indices['sagittal']], cmap='gray')
    axes[2].set_title(f'Sagittal View (Slice {slice_indices["sagittal"]})')
    axes[2].axis('off')
    
    plt.suptitle(title, fontsize=16, y=0.95)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()
    
    return slice_indices

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Configure matplotlib for better visualization
    plt.rcParams['figure.figsize'] = (15, 10)
    plt.rcParams['image.cmap'] = 'gray'
    plt.rcParams['image.interpolation'] = 'nearest'
    
    # Define paths
    preprocessed_dir = "/work/forkert_lab/fomo/preprocessed/FOMO60k"
    
    # Get all .npy files (excluding splits.pkl)
    npy_files = [f for f in os.listdir(preprocessed_dir) if f.endswith('.npy')]
    print(f"Total preprocessed files found: {len(npy_files)}")
    
    # Randomly select 5 files
    selected_files = random.sample(npy_files, 5)
    print(f"\nSelected files:")
    for i, file in enumerate(selected_files, 1):
        print(f"{i}. {file}")
    
    # Create output directory for saved images
    output_dir = "preprocessed_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process and display each selected file
    all_shapes = []
    all_ranges = []
    all_means = []
    all_stds = []
    
    for i, filename in enumerate(selected_files, 1):
        print(f"\n{'='*60}")
        print(f"File {i}: {filename}")
        print(f"{'='*60}")
        
        file_path = os.path.join(preprocessed_dir, filename)
        
        try:
            # Load data
            data, properties = load_preprocessed_data(file_path)
            
            print(f"Data shape: {data.shape}")
            print(f"Data type: {data.dtype}")
            print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
            print(f"Data mean: {data.mean():.3f}")
            print(f"Data std: {data.std():.3f}")
            
            if properties:
                print(f"\nProperties keys: {list(properties.keys())}")
                if 'spacing' in properties:
                    print(f"Spacing: {properties['spacing']}")
                if 'shape_before_cropping' in properties:
                    print(f"Original shape: {properties['shape_before_cropping']}")
            
            # Display orthogonal views
            save_path = os.path.join(output_dir, f"{filename.replace('.npy', '')}_views.png")
            slice_indices = display_orthogonal_views(data, f"{filename}", save_path)
            
            # Display additional slices for better visualization
            if data.shape[0] > 10 and data.shape[1] > 10 and data.shape[2] > 10:
                # Show slices at 25%, 50%, and 75% of each dimension
                additional_slices = {
                    'axial': [data.shape[0] // 4, data.shape[0] // 2, 3 * data.shape[0] // 4],
                    'coronal': [data.shape[1] // 4, data.shape[1] // 2, 3 * data.shape[1] // 4],
                    'sagittal': [data.shape[2] // 4, data.shape[2] // 2, 3 * data.shape[2] // 4]
                }
                
                # Display multiple axial slices
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                for j, slice_idx in enumerate(additional_slices['axial']):
                    axes[j].imshow(data[slice_idx, :, :], cmap='gray')
                    axes[j].set_title(f'Axial Slice {slice_idx}')
                    axes[j].axis('off')
                plt.suptitle(f'{filename} - Multiple Axial Slices', fontsize=16)
                plt.tight_layout()
                
                # Save additional slices
                additional_save_path = os.path.join(output_dir, f"{filename.replace('.npy', '')}_axial_slices.png")
                plt.savefig(additional_save_path, dpi=150, bbox_inches='tight')
                print(f"Saved axial slices to {additional_save_path}")
                plt.show()
            
            # Collect statistics
            all_shapes.append(data.shape)
            all_ranges.append((data.min(), data.max()))
            all_means.append(data.mean())
            all_stds.append(data.std())
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    if all_shapes:
        print(f"\nShape statistics:")
        print(f"Min shape: {min(all_shapes)}")
        print(f"Max shape: {max(all_shapes)}")
        
        print(f"\nIntensity statistics:")
        print(f"Min range: [{min([r[0] for r in all_ranges]):.3f}, {min([r[1] for r in all_ranges]):.3f}]")
        print(f"Max range: [{max([r[0] for r in all_ranges]):.3f}, {max([r[1] for r in all_ranges]):.3f}]")
        print(f"Mean of means: {np.mean(all_means):.3f} ± {np.std(all_means):.3f}")
        print(f"Mean of stds: {np.mean(all_stds):.3f} ± {np.std(all_stds):.3f}")
    
    print(f"\nVisualizations saved to: {output_dir}")

if __name__ == "__main__":
    main() 