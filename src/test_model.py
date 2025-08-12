#!/usr/bin/env python

import argparse
import os
import logging
import torch
import numpy as np
import json
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import lightning as L
from torch.utils.data import DataLoader

from models.supervised_base import BaseSupervisedModel
from data.dataset import FOMODataset
from data.task_configs import task1_config, task2_config, task3_config, task4_config
from utils.utils import SimplePathConfig, setup_seed
from yucca.pipeline.configuration.split_data import get_split_config
from yucca.modules.data.augmentation.YuccaAugmentationComposer import YuccaAugmentationComposer
from augmentations.finetune_augmentation_presets import get_finetune_augmentation_params


def get_task_config(taskid):
    """Get task configuration based on task ID."""
    if taskid == 1:
        task_cfg = task1_config
    elif taskid == 2:
        task_cfg = task2_config
    elif taskid == 3:
        task_cfg = task3_config
    elif taskid == 4:
        task_cfg = task4_config
    else:
        raise ValueError(f"Unknown taskid: {taskid}. Supported IDs are 1, 2, 3, and 4")
    return task_cfg


def load_model_from_checkpoint(checkpoint_path, task_type, config):
    """Load model from checkpoint."""
    # Create model instance
    model = BaseSupervisedModel.create(
        task_type=task_type,
        config=config,
        learning_rate=1e-4,  # Not used for inference
        do_compile=False,
        compile_mode="default",
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Load state dict
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model


def create_test_dataset(data_dir, task_name, patch_size, task_type, augmentation_preset):
    """Create test dataset using all available data."""
    # Set up path configuration
    train_data_dir = data_dir
    path_config = SimplePathConfig(train_data_dir=train_data_dir)
    
    # Get all samples from the data directory
    from batchgenerators.utilities.file_and_folder_operations import subfiles
    all_files = subfiles(train_data_dir, suffix=".npy", join=False)
    # Remove .npy extension to get base filenames and create full paths
    test_samples = [os.path.join(train_data_dir, f[:-4]) for f in all_files]
    
    # Configure augmentations (validation transforms only)
    aug_params = get_finetune_augmentation_params(augmentation_preset)
    tt_preset = "classification" if task_type == "regression" else task_type
    augmenter = YuccaAugmentationComposer(
        patch_size=(patch_size,) * 3,
        task_type_preset=tt_preset,
        parameter_dict=aug_params,
        deep_supervision=False,
    )
    
    # Create dataset
    if task_type == "segmentation":
        from yucca.modules.data.datasets.YuccaDataset import YuccaTrainDataset
        dataset = YuccaTrainDataset(
            samples=test_samples,
            patch_size=(patch_size,) * 3,
            composed_transforms=augmenter.val_transforms,
            task_type=task_type,
        )
    else:
        dataset = FOMODataset(
            samples=test_samples,
            patch_size=(patch_size,) * 3,
            composed_transforms=augmenter.val_transforms,
            task_type=task_type,
        )
    
    return dataset, test_samples


def evaluate_classification(model, dataloader, device, num_classes):
    """Evaluate classification model and compute metrics including AUROC."""
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_probabilities = []
    all_targets = []
    all_file_paths = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Get inputs and targets
            inputs = batch["image"]
            targets = batch["label"]
            file_paths = batch["file_path"]
            
            # Forward pass
            outputs = model(inputs)
            
            
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            # For AUROC, we need probabilities for each class
            prob_positive = probabilities
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(prob_positive.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_file_paths.extend(file_paths)
            
            # Debug first batch
            if len(all_targets) == len(predictions):
                print(f"First batch - targets shape: {targets.shape}, predictions shape: {predictions.shape}")
                print(f"First batch - targets: {targets.cpu().numpy()}")
                print(f"First batch - predictions: {predictions.cpu().numpy()}")
    
    # Convert to numpy arrays and ensure proper shapes
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_targets = np.array(all_targets)
    
    # Ensure targets are 1D
    if all_targets.ndim > 1:
        all_targets = all_targets.squeeze()
    
    # Ensure predictions are 1D
    if all_predictions.ndim > 1:
        all_predictions = all_predictions.squeeze()
    
    # Ensure probabilities have correct shape
    if all_probabilities.ndim == 1:
        # Binary case - probabilities should be 1D
        pass
    else:
        # Multi-class case - probabilities should be 2D (samples, classes)
        if all_probabilities.ndim > 2:
            all_probabilities = all_probabilities.squeeze()
    
    print(f"Targets shape: {all_targets.shape}, dtype: {all_targets.dtype}")
    print(f"Predictions shape: {all_predictions.shape}, dtype: {all_predictions.dtype}")
    print(f"Probabilities shape: {all_probabilities.shape}, dtype: {all_probabilities.dtype}")
    print(f"Target values: {np.unique(all_targets)}")
    print(f"Prediction values: {np.unique(all_predictions)}")
    print(f"Target sample: {all_targets[:5]}")
    print(f"Prediction sample: {all_predictions[:5]}")
    print(f"Target min/max: {all_targets.min()}/{all_targets.max()}")
    print(f"Prediction min/max: {all_predictions.min()}/{all_predictions.max()}")
    
    # Compute metrics
    metrics = {}
    
    # Ensure both targets and predictions are 1D arrays with same dtype
    # Force flatten and convert to int
    y_true = all_targets.flatten().astype(int)
    y_pred = all_predictions.flatten().astype(int)
    
    # Additional safety check - ensure they are truly 1D
    if y_true.ndim != 1:
        y_true = y_true.ravel()
    if y_pred.ndim != 1:
        y_pred = y_pred.ravel()
    
    print(f"After conversion - y_true shape: {y_true.shape}, dtype: {y_true.dtype}")
    print(f"After conversion - y_pred shape: {y_pred.shape}, dtype: {y_pred.dtype}")
    print(f"After conversion - y_true sample: {y_true[:5]}")
    print(f"After conversion - y_pred sample: {y_pred[:5]}")
    print(f"After conversion - y_true unique: {np.unique(y_true)}")
    print(f"After conversion - y_pred unique: {np.unique(y_pred)}")
    print(f"After conversion - y_true is 1D: {y_true.ndim == 1}")
    print(f"After conversion - y_pred is 1D: {y_pred.ndim == 1}")
    
    # Final safety check - ensure they are exactly the same format
    assert y_true.ndim == 1, f"y_true is not 1D: shape {y_true.shape}"
    assert y_pred.ndim == 1, f"y_pred is not 1D: shape {y_pred.shape}"
    assert y_true.dtype == y_pred.dtype, f"dtype mismatch: {y_true.dtype} vs {y_pred.dtype}"
    
    # Basic classification metrics with individual error handling
    try:
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        print("Accuracy computed successfully")
    except Exception as e:
        print(f"Error computing accuracy: {e}")
        metrics['accuracy'] = None
    
    try:
        metrics['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        print("Precision computed successfully")
    except Exception as e:
        print(f"Error computing precision: {e}")
        metrics['precision'] = None
    
    try:
        metrics['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        print("Recall computed successfully")
    except Exception as e:
        print(f"Error computing recall: {e}")
        metrics['recall'] = None
    
    try:
        metrics['f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        print("F1 computed successfully")
    except Exception as e:
        print(f"Error computing F1: {e}")
        metrics['f1'] = None
    
    # AUROC
    if num_classes == 2:
        # Binary classification
        try:
            # Ensure we have binary targets (0 and 1) and proper probability format
            y_true_binary = y_true
            y_score_binary = all_probabilities.ravel().astype(float)
            
            # Check if we have both classes
            unique_classes = np.unique(y_true_binary)
            if len(unique_classes) < 2:
                print(f"Warning: Only one class found in targets: {unique_classes}")
                metrics['auroc'] = None
            else:
                metrics['auroc'] = roc_auc_score(y_true_binary, y_score_binary)
        except ValueError as e:
            print(f"Warning: Could not compute AUROC: {e}")
            metrics['auroc'] = None
    else:
        # Multi-class classification - compute AUROC for each class vs rest
        auroc_scores = []
        for i in range(num_classes):
            try:
                # One-vs-rest AUROC
                y_true_binary = (y_true == i).astype(int)
                y_score_binary = all_probabilities[:, i].astype(float)
                
                # Check if we have both classes for this class
                unique_classes = np.unique(y_true_binary)
                if len(unique_classes) < 2:
                    print(f"Warning: Only one class found for class {i}: {unique_classes}")
                    auroc_scores.append(None)
                else:
                    auroc_class = roc_auc_score(y_true_binary, y_score_binary)
                    auroc_scores.append(auroc_class)
            except ValueError as e:
                print(f"Warning: Could not compute AUROC for class {i}: {e}")
                auroc_scores.append(None)
        
        metrics['auroc_per_class'] = auroc_scores
        valid_scores = [score for score in auroc_scores if score is not None]
        metrics['auroc_mean'] = np.mean(valid_scores) if valid_scores else None
    
    return metrics, {
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'targets': all_targets,
        'file_paths': all_file_paths
    }


def evaluate_regression(model, dataloader, device):
    """Evaluate regression model and compute metrics."""
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_file_paths = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Get inputs and targets
            inputs = batch["image"]
            targets = batch["label"]
            file_paths = batch["file_path"]
            
            # Forward pass
            outputs = model(inputs)
            predictions = outputs.squeeze()
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_file_paths.extend(file_paths)
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Compute regression metrics
    mse = np.mean((all_predictions - all_targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_predictions - all_targets))
    
    # R-squared
    ss_res = np.sum((all_targets - all_predictions) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    return metrics, {
        'predictions': all_predictions,
        'targets': all_targets,
        'file_paths': all_file_paths
    }


def main():
    logging.getLogger().setLevel(logging.INFO)
    
    parser = argparse.ArgumentParser(description="Test finetuned model and compute AUROC")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint (.ckpt file)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/preprocessed",
        help="Path to data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_results",
        help="Directory to save test results"
    )
    parser.add_argument(
        "--taskid",
        type=int,
        required=True,
        help="Task ID (1: FOMO1 classification, 2: FOMO2 classification, 3: FOMO3 regression, 4: PD classification)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for testing"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading"
    )

    parser.add_argument(
        "--augmentation_preset",
        type=str,
        choices=["all", "basic", "none"],
        default="none",
        help="Augmentation preset for testing"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="unet_b",
        help="Model name (unet_b, unet_xl, etc.)"
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=32,
        help="Patch size for the model"
    )
    
    args = parser.parse_args()
    
    # Set seed
    setup_seed(args.seed)
    
    # Get task configuration
    task_cfg = get_task_config(args.taskid)
    task_type = task_cfg["task_type"]
    task_name = task_cfg["task_name"]
    num_classes = task_cfg["num_classes"]
    modalities = len(task_cfg["modalities"])
    
    print(f"Testing model for task {args.taskid}: {task_name}")
    print(f"Task type: {task_type}")
    print(f"Number of classes: {num_classes}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create test dataset
    print("Creating test dataset...")
    dataset, test_samples = create_test_dataset(
        data_dir=args.data_dir,
        task_name=task_name,
        patch_size=args.patch_size,
        task_type=task_type,
        augmentation_preset=args.augmentation_preset
    )
    
    print(f"Test dataset size: {len(dataset)}")
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False
    )
    
    # Load model
    print("Loading model from checkpoint...")
    config = {
        # Task information
        "task": task_name,
        "task_id": args.taskid,
        "task_type": task_type,
        "experiment": f"test_{task_name}",
        "model_name": args.model_name,
        "model_dimensions": "3D",
        "run_type": "test",
        # Directories
        "save_dir": args.output_dir,
        "train_data_dir": args.data_dir,
        "version_dir": args.output_dir,
        "version": 0,
        # Reproducibility
        "seed": args.seed,
        # Dataset properties
        "num_classes": num_classes,
        "num_modalities": modalities,
        "image_extension": ".npy",
        "allow_missing_modalities": False,
        "labels": task_cfg["labels"],
        # Training parameters (not used for testing but needed for model creation)
        "batch_size": args.batch_size,
        "learning_rate": 1e-4,  # Not used for testing
        "patch_size": (args.patch_size,) * 3,
        "precision": "bf16-mixed",
        "augmentation_preset": args.augmentation_preset,
        "epochs": 1,  # Not used for testing
        "train_batches_per_epoch": 1,  # Not used for testing
        "effective_batch_size": args.batch_size,
        # Dataset metrics
        "train_dataset_size": len(test_samples),
        "val_dataset_size": len(test_samples),
        "max_iterations": 1,  # Not used for testing
        # Hardware settings
        "num_devices": 1,
        "num_workers": args.num_workers,
        # Model compilation
        "compile": False,
        "compile_mode": "default",
        # Trainer specific params
        "fast_dev_run": False,
    }
    
    model = load_model_from_checkpoint(args.checkpoint_path, task_type, config)
    
    # Evaluate model
    print("Evaluating model...")
    if task_type == "regression":
        metrics, results = evaluate_regression(model, dataloader, device)
    else:
        metrics, results = evaluate_classification(model, dataloader, device, num_classes)
    
    # Print results
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    for metric_name, metric_value in metrics.items():
        if metric_value is not None:
            print(f"{metric_name}: {metric_value:.4f}")
        else:
            print(f"{metric_name}: N/A")
    
    # Save results
    results_file = os.path.join(args.output_dir, f"test_results_task{args.taskid}.json")
    results_data = {
        "task_id": args.taskid,
        "task_name": task_name,
        "task_type": task_type,
        "metrics": metrics,
        "test_samples": test_samples,
        "config": {
            "checkpoint_path": args.checkpoint_path,
            "batch_size": args.batch_size,
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    
    # Save detailed predictions
    predictions_file = os.path.join(args.output_dir, f"predictions_task{args.taskid}.npz")
    np.savez(
        predictions_file,
        predictions=results['predictions'],
        targets=results['targets'],
        file_paths=results['file_paths'],
        probabilities=results.get('probabilities', None)
    )
    
    print(f"Detailed predictions saved to: {predictions_file}")


if __name__ == "__main__":
    main() 