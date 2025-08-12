import os
import numpy as np
import nibabel as nib
import pickle
from torch.utils.data import Dataset
from typing import Tuple, Optional, Literal
from batchgenerators.utilities.file_and_folder_operations import load_pickle


class FOMOFinetuneDataset(Dataset):
    """
    Base dataset class for FOMO fine-tuning tasks.
    """

    def __init__(
        self,
        samples: list,
        patch_size: Tuple[int, int, int],
        task_type: Literal["classification", "segmentation", "regression"],
        composed_transforms: Optional = None,
    ):
        super().__init__()
        self.samples = samples
        self.patch_size = patch_size
        self.task_type = task_type
        self.composed_transforms = composed_transforms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses must implement __getitem__ method")


class FOMOFinetuneTask1Dataset(FOMOFinetuneDataset):
    """
    Dataset class for Task 1: Infarct Detection (Classification)
    """

    def __init__(
        self,
        samples: list,
        patch_size: Tuple[int, int, int],
        task_type: str = "classification",
        composed_transforms: Optional = None,
    ):
        super().__init__(samples, patch_size, task_type, composed_transforms)

    def __getitem__(self, idx):
        subject_id = self.samples[idx]
        
        # Load all modalities for this subject
        subject_dir = f"fomo-fine-tuning/fomo-task1/skull_stripped/{subject_id}/ses_1"
        
        # Load the four modalities
        # Handle both ss_swi.nii.gz and ss_t2s.nii.gz
        swi_path = os.path.join(subject_dir, "ss_swi.nii.gz")
        t2s_path = os.path.join(subject_dir, "ss_t2s.nii.gz")
        
        # Use ss_swi if it exists, otherwise use ss_t2s
        if os.path.exists(swi_path):
            first_modality_path = swi_path
        elif os.path.exists(t2s_path):
            first_modality_path = t2s_path
        else:
            raise FileNotFoundError(f"Neither ss_swi.nii.gz nor ss_t2s.nii.gz found for subject {subject_id}")
        
        flair_path = os.path.join(subject_dir, "ss_flair.nii.gz")
        dwi_path = os.path.join(subject_dir, "ss_dwi_b1000.nii.gz")
        adc_path = os.path.join(subject_dir, "ss_adc.nii.gz")
        
        # Load images
        first_modality_img = nib.load(first_modality_path)
        flair_img = nib.load(flair_path)
        dwi_img = nib.load(dwi_path)
        adc_img = nib.load(adc_path)
        
        # Get data arrays and convert to float32
        first_modality_data = first_modality_img.get_fdata().astype(np.float32)
        flair_data = flair_img.get_fdata().astype(np.float32)
        dwi_data = dwi_img.get_fdata().astype(np.float32)
        adc_data = adc_img.get_fdata().astype(np.float32)
        
        # Stack modalities as channels (shape: [4, H, W, D])
        image = np.stack([first_modality_data, flair_data, dwi_data, adc_data], axis=0)
        
        # Resize or pad image to patch_size
        target_shape = self.patch_size  # [128, 128, 128]
        current_shape = image.shape[1:]  # [H, W, D]
        
        # Pad if current shape is smaller than target shape
        pad_width = []
        for i in range(3):
            diff = max(0, target_shape[i] - current_shape[i])
            pad_width.append((0, diff))
        
        if any(pad[1] > 0 for pad in pad_width):
            image = np.pad(image, [(0, 0)] + pad_width, mode='constant', constant_values=0)
        
        # Crop if current shape is larger than target shape
        if any(current_shape[i] > target_shape[i] for i in range(3)):
            image = image[:, :target_shape[0], :target_shape[1], :target_shape[2]]
        
        # Load label from labels_masked directory
        # For Task 1, we derive the label from the segmentation mask
        # If there's any segmentation present, it's a positive case (label 1)
        # Otherwise it's a negative case (label 0)
        seg_mask_path = f"fomo-fine-tuning/fomo-task1/labels_masked/{subject_id}/ses_1/seg_masked.nii.gz"
        seg_mask = nib.load(seg_mask_path).get_fdata()
        label = 1 if np.any(seg_mask > 0) else 0
        
        # Apply transforms if provided
        if self.composed_transforms:
            # Convert to dictionary format expected by transforms
            data_dict = {
                "image": image,
                "label": label,
                "file_path": subject_id
            }
            data_dict = self.composed_transforms(data_dict)
            image = data_dict["image"]
            label = data_dict["label"]
        
        return {
            "image": image,
            "label": label,
            "file_path": subject_id
        }


class FOMOFinetuneTask2Dataset(FOMOFinetuneDataset):
    """
    Dataset class for Task 2: Meningioma Segmentation (Segmentation)
    """

    def __init__(
        self,
        samples: list,
        patch_size: Tuple[int, int, int],
        task_type: str = "segmentation",
        composed_transforms: Optional = None,
    ):
        super().__init__(samples, patch_size, task_type, composed_transforms)

    def __getitem__(self, idx):
        file_name = self.samples[idx]
        
        # Load the .npy file containing the 4D array
        npy_path = f"fomo-fine-tuning/fomo-task2/preprocessed_2/{file_name}.npy"
        data = np.load(npy_path, allow_pickle=True)
        
        # If it's an object array, convert to regular array
        if data.dtype == object:
            # Create a new array with the same shape but float dtype
            converted_data = np.zeros(data.shape, dtype=np.float32)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    for k in range(data.shape[2]):
                        for l in range(data.shape[3]):
                            converted_data[i, j, k, l] = data[i, j, k, l]
            data = converted_data
        else:
            # Ensure data is float32
            data = data.astype(np.float32)
        
        # Separate image channels (first 3) and segmentation mask (4th channel)
        image = data[:3]  # T1, T2, FLAIR
        label = data[3]   # Segmentation mask
        
        # Add channel dimension to label to make it [1, H, W, D]
        label = label[np.newaxis, ...]
        
        # Resize or pad image to patch_size
        target_shape = self.patch_size  # [128, 128, 128]
        current_shape = image.shape[1:]  # [H, W, D]
        
        # Pad if current shape is smaller than target shape
        pad_width = []
        for i in range(3):
            diff = max(0, target_shape[i] - current_shape[i])
            pad_width.append((0, diff))
        
        if any(pad[1] > 0 for pad in pad_width):
            image = np.pad(image, [(0, 0)] + pad_width, mode='constant', constant_values=0)
            label = np.pad(label, [(0, 0)] + pad_width, mode='constant', constant_values=0)
        
        # Crop if current shape is larger than target shape
        if any(current_shape[i] > target_shape[i] for i in range(3)):
            image = image[:, :target_shape[0], :target_shape[1], :target_shape[2]]
            label = label[:, :target_shape[0], :target_shape[1], :target_shape[2]]
        
        # Apply transforms if provided
        if self.composed_transforms:
            # Convert to dictionary format expected by transforms
            data_dict = {
                "image": image,
                "label": label,
                "file_path": file_name
            }
            data_dict = self.composed_transforms(data_dict)
            image = data_dict["image"]
            label = data_dict["label"]
        
        return {
            "image": image,
            "label": label,
            "file_path": file_name
        }


class FOMOFinetuneTask3Dataset(FOMOFinetuneDataset):
    """
    Dataset class for Task 3: Brain Age Regression (Regression)
    """

    def __init__(
        self,
        samples: list,
        patch_size: Tuple[int, int, int],
        task_type: str = "regression",
        composed_transforms: Optional = None,
    ):
        super().__init__(samples, patch_size, task_type, composed_transforms)

    def __getitem__(self, idx):
        subject_id = self.samples[idx]
        
        # Load T1 and T2 scans
        subject_dir = f"fomo-fine-tuning/fomo-task3/preprocessed_2/{subject_id}/ses_1"
        
        t1_path = os.path.join(subject_dir, "ss_t1.nii.gz")
        t2_path = os.path.join(subject_dir, "ss_t2.nii.gz")
        
        # Load images
        t1_img = nib.load(t1_path)
        t2_img = nib.load(t2_path)
        
        # Get data arrays and convert to float32
        t1_data = t1_img.get_fdata().astype(np.float32)
        t2_data = t2_img.get_fdata().astype(np.float32)
        
        # Stack modalities as channels (shape: [2, H, W, D])
        # First ensure both images have the same shape by padding/cropping
        t1_shape = t1_data.shape
        t2_shape = t2_data.shape
        
        # If shapes are different, we need to handle this
        if t1_shape != t2_shape:
            # Find the maximum shape
            max_shape = [max(t1_shape[i], t2_shape[i]) for i in range(3)]
            
            # Pad t1 if needed
            if t1_shape != tuple(max_shape):
                pad_width = []
                for i in range(3):
                    diff = max_shape[i] - t1_shape[i]
                    pad_width.append((0, diff))
                t1_data = np.pad(t1_data, pad_width, mode='constant', constant_values=0)
            
            # Pad t2 if needed
            if t2_shape != tuple(max_shape):
                pad_width = []
                for i in range(3):
                    diff = max_shape[i] - t2_shape[i]
                    pad_width.append((0, diff))
                t2_data = np.pad(t2_data, pad_width, mode='constant', constant_values=0)
        
        image = np.stack([t1_data, t2_data], axis=0)
        
        # Resize or pad image to patch_size
        target_shape = self.patch_size  # [128, 128, 128]
        current_shape = image.shape[1:]  # [H, W, D]
        
        # Pad if current shape is smaller than target shape
        pad_width = []
        for i in range(3):
            diff = max(0, target_shape[i] - current_shape[i])
            pad_width.append((0, diff))
        
        if any(pad[1] > 0 for pad in pad_width):
            image = np.pad(image, [(0, 0)] + pad_width, mode='constant', constant_values=0)
        
        # Crop if current shape is larger than target shape
        if any(current_shape[i] > target_shape[i] for i in range(3)):
            image = image[:, :target_shape[0], :target_shape[1], :target_shape[2]]
        
        # Load label (age) from labels directory
        label_path = f"fomo-fine-tuning/fomo-task3/labels/{subject_id}/ses_1/label.txt"
        with open(label_path, 'r') as f:
            label = float(f.read().strip())
        
        # Apply transforms if provided
        if self.composed_transforms:
            # Convert to dictionary format expected by transforms
            data_dict = {
                "image": image,
                "label": np.array([label]),  # Make it an array for consistency
                "file_path": subject_id
            }
            data_dict = self.composed_transforms(data_dict)
            image = data_dict["image"]
            label = data_dict["label"]
        
        return {
            "image": image,
            "label": label,
            "file_path": subject_id
        }