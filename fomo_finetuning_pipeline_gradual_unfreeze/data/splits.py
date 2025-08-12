import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
import os


def get_task1_samples():
    """
    Get list of subject IDs for Task 1
    """
    skull_stripped_dir = "fomo-fine-tuning/fomo-task1/skull_stripped"
    labels_masked_dir = "fomo-fine-tuning/fomo-task1/labels_masked"
    
    # Get subject IDs that exist in both directories
    skull_stripped_subjects = [d for d in os.listdir(skull_stripped_dir) 
                              if os.path.isdir(os.path.join(skull_stripped_dir, d))]
    labels_masked_subjects = [d for d in os.listdir(labels_masked_dir) 
                             if os.path.isdir(os.path.join(labels_masked_dir, d))]
    
    # Return intersection of both lists
    subject_ids = list(set(skull_stripped_subjects) & set(labels_masked_subjects))
    return subject_ids


def get_task1_labels(subject_ids):
    """
    Get labels for Task 1 subjects
    """
    import nibabel as nib
    import numpy as np
    
    labels = []
    for subject_id in subject_ids:
        # For Task 1, we derive the label from the segmentation mask
        # If there's any segmentation present, it's a positive case (label 1)
        # Otherwise it's a negative case (label 0)
        seg_mask_path = f"fomo-fine-tuning/fomo-task1/labels_masked/{subject_id}/ses_1/seg_masked.nii.gz"
        seg_mask = nib.load(seg_mask_path).get_fdata()
        label = 1 if np.any(seg_mask > 0) else 0
        labels.append(label)
    return labels


def get_task2_samples():
    """
    Get list of file names for Task 2
    """
    preprocessed_dir = "fomo-fine-tuning/fomo-task2/preprocessed_2"
    npy_files = [f.replace('.npy', '') for f in os.listdir(preprocessed_dir) 
                 if f.endswith('.npy')]
    return npy_files


def get_task2_labels(file_names):
    """
    Get labels for Task 2 samples (presence of meningioma)
    """
    labels = []
    for file_name in file_names:
        # For Task 2, we'll use the maximum value in the segmentation mask as the label
        # 0 = no meningioma, 1 = meningioma present
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
        
        # The segmentation mask is the 4th channel
        seg_mask = data[3]
        label = 1 if np.any(seg_mask > 0) else 0
        labels.append(label)
    return labels


def get_task3_samples():
    """
    Get list of subject IDs for Task 3
    """
    preprocessed_dir = "fomo-fine-tuning/fomo-task3/preprocessed_2"
    subject_ids = [d for d in os.listdir(preprocessed_dir) 
                   if os.path.isdir(os.path.join(preprocessed_dir, d))]
    return subject_ids


def get_task3_labels(subject_ids):
    """
    Get labels (ages) for Task 3 subjects
    """
    labels = []
    for subject_id in subject_ids:
        label_path = f"fomo-fine-tuning/fomo-task3/labels/{subject_id}/ses_1/label.txt"
        with open(label_path, 'r') as f:
            label = float(f.read().strip())
        labels.append(label)
    return labels


def create_kfold_splits_task1(n_splits=5, random_state=42):
    """
    Create stratified k-fold splits for Task 1
    """
    subject_ids = get_task1_samples()
    labels = get_task1_labels(subject_ids)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []
    
    for train_idx, test_idx in skf.split(subject_ids, labels):
        train_subjects = [subject_ids[i] for i in train_idx]
        test_subjects = [subject_ids[i] for i in test_idx]
        
        # Further split train into train and validation
        # Use 80% for training and 20% for validation
        val_size = len(train_subjects) // 5
        val_subjects = train_subjects[:val_size]
        train_subjects = train_subjects[val_size:]
        
        splits.append({
            'train': train_subjects,
            'val': val_subjects,
            'test': test_subjects
        })
    
    return splits


def create_kfold_splits_task2(n_splits=5, random_state=42):
    """
    Create stratified k-fold splits for Task 2
    """
    file_names = get_task2_samples()
    labels = get_task2_labels(file_names)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []
    
    for train_idx, test_idx in skf.split(file_names, labels):
        train_files = [file_names[i] for i in train_idx]
        test_files = [file_names[i] for i in test_idx]
        
        # Further split train into train and validation
        # Use 80% for training and 20% for validation
        val_size = len(train_files) // 5
        val_files = train_files[:val_size]
        train_files = train_files[val_size:]
        
        splits.append({
            'train': train_files,
            'val': val_files,
            'test': test_files
        })
    
    return splits


def create_kfold_splits_task3(n_splits=5, random_state=42):
    """
    Create k-fold splits for Task 3 (regression task, no stratification)
    """
    subject_ids = get_task3_samples()
    labels = get_task3_labels(subject_ids)
    
    # For regression, we'll try to balance the age distribution across folds
    # by sorting subjects by age and then distributing them across folds
    sorted_indices = np.argsort(labels)
    subject_ids = [subject_ids[i] for i in sorted_indices]
    
    # Create approximately balanced folds
    kf = KFold(n_splits=n_splits, shuffle=False)  # We'll shuffle manually
    splits = []
    
    for train_idx, test_idx in kf.split(subject_ids):
        train_subjects = [subject_ids[i] for i in train_idx]
        test_subjects = [subject_ids[i] for i in test_idx]
        
        # Shuffle the train subjects
        np.random.seed(random_state)
        np.random.shuffle(train_subjects)
        
        # Further split train into train and validation
        # Use 80% for training and 20% for validation
        val_size = len(train_subjects) // 5
        val_subjects = train_subjects[:val_size]
        train_subjects = train_subjects[val_size:]
        
        splits.append({
            'train': train_subjects,
            'val': val_subjects,
            'test': test_subjects
        })
    
    return splits