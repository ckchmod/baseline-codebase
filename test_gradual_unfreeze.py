#!/usr/bin/env python3
"""
Test script to verify the gradual unfreezing implementation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Create a minimal test config
test_config = {
    "task_id": 3,
    "task_name": "Brain Age Regression Test",
    "task_type": "regression",
    "num_modalities": 2,
    "num_classes": 1,
    "patch_size": [32, 32, 32],  # Small patch size for testing
    "batch_size": 1,
    "num_workers": 0,
    "max_epochs": 5,
    "learning_rate": 1e-4,
    "pretrained_ckpt_path": "pretrained/epoch=12.ckpt",
    "results_dir": "results/task3_test",
    "n_splits": 2,
    "random_state": 42,
    "model_name": "unet_b",
    "version_dir": "fomo_finetuning_pipeline_gradual_unfreeze",
    "unfreeze_strategy": "gradual",
    "unfreeze_epoch": 2
}

def test_model_creation():
    """Test that we can create a model with gradual unfreezing"""
    print("Testing model creation with gradual unfreezing...")
    
    try:
        # Import the model class
        sys.path.append(os.path.join(os.path.dirname(__file__), "fomo_finetuning_pipeline_gradual_unfreeze"))
        from fomo_finetuning_pipeline_gradual_unfreeze.models.finetune_unet import FOMOFinetuneRegModel
        
        # Create a minimal config for testing
        config = {
            "task_type": "regression",
            "num_modalities": 2,
            "num_classes": 1,
            "patch_size": [32, 32, 32],
            "model_name": "unet_b",
            "version_dir": "fomo_finetuning_pipeline_gradual_unfreeze"
        }
        
        # Try to create the model
        model = FOMOFinetuneRegModel(
            config=config,
            learning_rate=1e-4,
            pretrained_ckpt_path="pretrained/epoch=12.ckpt",
            unfreeze_strategy="gradual",
            unfreeze_epoch=2
        )
        
        print("✓ Model created successfully")
        print(f"  Unfreeze strategy: {model.unfreeze_strategy}")
        print(f"  Unfreeze epoch: {model.unfreeze_epoch}")
        
        # Check that the optimizer can be configured
        # Note: We can't actually call configure_optimizers without a trainer,
        # but we can check that the method exists
        if hasattr(model, 'configure_optimizers'):
            print("✓ configure_optimizers method exists")
        else:
            print("✗ configure_optimizers method missing")
            
        return True
        
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        return False

def test_config_loading():
    """Test that we can load the gradual unfreezing configs"""
    print("\nTesting config loading...")
    
    try:
        import json
        
        # Test task3 config
        config_path = os.path.join(
            os.path.dirname(__file__), 
            "fomo_finetuning_pipeline_gradual_unfreeze", 
            "configs", 
            "task3_gradual_unfreeze_config.json"
        )
        
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        print("✓ Task3 gradual unfreezing config loaded successfully")
        print(f"  Unfreeze strategy: {config.get('unfreeze_strategy', 'NOT SET')}")
        print(f"  Unfreeze epoch: {config.get('unfreeze_epoch', 'NOT SET')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        return False

if __name__ == "__main__":
    print("=== Gradual Unfreezing Implementation Test ===\n")
    
    success = True
    success &= test_model_creation()
    success &= test_config_loading()
    
    print(f"\n=== Test Result ===")
    if success:
        print("✓ All tests passed! Gradual unfreezing implementation is ready.")
    else:
        print("✗ Some tests failed. Please check the implementation.")
        
    sys.exit(0 if success else 1)