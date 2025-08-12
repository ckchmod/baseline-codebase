import os
import sys
import torch
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Try to import wandb logger, but don't fail if it's not available
try:
    from lightning.pytorch.loggers import WandbLogger
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WandbLogger not available, using CSVLogger only")

from data.splits import create_kfold_splits_task1, create_kfold_splits_task2, create_kfold_splits_task3
from data.datamodules import FOMOFinetuneTask1DataModule, FOMOFinetuneTask2DataModule, FOMOFinetuneTask3DataModule
from fomo_finetuning_pipeline_gradual_unfreeze.models.finetune_unet import FOMOFinetuneClsModel, FOMOFinetuneSegModel, FOMOFinetuneRegModel
from fomo_finetuning_pipeline_gradual_unfreeze.utils.metrics import aggregate_metrics


class FOMOCrossValidationPipeline:
    """
    Cross-validation pipeline for FOMO fine-tuning tasks
    """
    
    def __init__(self, task_id, config):
        self.task_id = task_id
        self.config = config
        self.results_dir = config.get("results_dir", "results")
        self.n_splits = config.get("n_splits", 5)
        self.random_state = config.get("random_state", 42)
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
    def run_task1_cv(self):
        """
        Run cross-validation for Task 1: Infarct Detection
        """
        print("Running cross-validation for Task 1: Infarct Detection")
        
        # Create data splits
        splits = create_kfold_splits_task1(n_splits=self.n_splits, random_state=self.random_state)
        
        # Store results for each fold
        fold_results = []
        
        for fold_idx, split in enumerate(splits):
            print(f"\n=== Fold {fold_idx + 1}/{self.n_splits} ===")
            
            # Create data module
            data_module = FOMOFinetuneTask1DataModule(
                task_type="classification",
                patch_size=self.config["patch_size"],
                batch_size=self.config["batch_size"],
                num_workers=self.config["num_workers"],
                train_samples=split["train"],
                val_samples=split["val"],
                test_samples=split["test"],
            )
            
            # Create model with gradual unfreezing parameters
            model_kwargs = {
                "config": self.config,
                "learning_rate": self.config["learning_rate"],
                "pretrained_ckpt_path": self.config["pretrained_ckpt_path"],
            }
            
            # Add unfreezing parameters if they exist in config
            if "unfreeze_strategy" in self.config:
                model_kwargs["unfreeze_strategy"] = self.config["unfreeze_strategy"]
            if "unfreeze_epoch" in self.config:
                model_kwargs["unfreeze_epoch"] = self.config["unfreeze_epoch"]
                
            model = FOMOFinetuneClsModel(**model_kwargs)
            
            # Create trainer
            loggers = [CSVLogger(
                self.results_dir, 
                name=f"task1_fold_{fold_idx + 1}"
            )]
            
            # Add WandbLogger if available and user is logged in
            if WANDB_AVAILABLE:
                try:
                    # Check if user is logged in to wandb
                    wandb.Api()
                    wandb_logger = WandbLogger(
                        name=f"task1_fold_{fold_idx + 1}",
                        project="fomo-finetuning",
                        group="task1"
                    )
                    loggers.append(wandb_logger)
                    print("Using WandbLogger for logging")
                except Exception as e:
                    print(f"WandbLogger available but not configured: {e}")
                    print("Using CSVLogger only")
            
            checkpoint_callback = ModelCheckpoint(
                dirpath=os.path.join(self.results_dir, f"task1_fold_{fold_idx + 1}"),
                filename="best_model",
                monitor="val/accuracy",
                mode="max",
                save_top_k=1,
            )
            
            early_stop_callback = EarlyStopping(
                monitor="val/loss",
                min_delta=0.001,
                patience=10,
                verbose=True,
                mode="min",
            )
            
            trainer = L.Trainer(
                max_epochs=self.config["max_epochs"],
                logger=loggers,
                callbacks=[checkpoint_callback, early_stop_callback],
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
            )
            
            # Train model
            trainer.fit(model, data_module)
            
            # Test model
            test_results = trainer.test(model, data_module)
            fold_results.append(test_results[0])
            
            # Save fold results
            fold_df = pd.DataFrame([test_results[0]])
            fold_df.to_csv(
                os.path.join(self.results_dir, f"task1_fold_{fold_idx + 1}_results.csv"),
                index=False
            )
        
        # Aggregate results across folds
        aggregated_results = aggregate_metrics(fold_results)
        
        # Save aggregated results
        agg_df = pd.DataFrame([aggregated_results])
        agg_df.to_csv(
            os.path.join(self.results_dir, "task1_aggregated_results.csv"),
            index=False
        )
        
        print("\n=== Task 1 CV Complete ===")
        print(f"Aggregated Results: {aggregated_results}")
        
        return aggregated_results
    
    def run_task2_cv(self):
        """
        Run cross-validation for Task 2: Meningioma Segmentation
        """
        print("Running cross-validation for Task 2: Meningioma Segmentation")
        
        # Create data splits
        splits = create_kfold_splits_task2(n_splits=self.n_splits, random_state=self.random_state)
        
        # Store results for each fold
        fold_results = []
        
        for fold_idx, split in enumerate(splits):
            print(f"\n=== Fold {fold_idx + 1}/{self.n_splits} ===")
            
            # Create data module
            data_module = FOMOFinetuneTask2DataModule(
                task_type="segmentation",
                patch_size=self.config["patch_size"],
                batch_size=self.config["batch_size"],
                num_workers=self.config["num_workers"],
                train_samples=split["train"],
                val_samples=split["val"],
                test_samples=split["test"],
            )
            
            # Create model with gradual unfreezing parameters
            model_kwargs = {
                "config": self.config,
                "learning_rate": self.config["learning_rate"],
                "pretrained_ckpt_path": self.config["pretrained_ckpt_path"],
            }
            
            # Add unfreezing parameters if they exist in config
            if "unfreeze_strategy" in self.config:
                model_kwargs["unfreeze_strategy"] = self.config["unfreeze_strategy"]
            if "unfreeze_epoch" in self.config:
                model_kwargs["unfreeze_epoch"] = self.config["unfreeze_epoch"]
                
            model = FOMOFinetuneSegModel(**model_kwargs)
            
            # Create trainer
            loggers = [CSVLogger(
                self.results_dir, 
                name=f"task2_fold_{fold_idx + 1}"
            )]
            
            # Add WandbLogger if available and user is logged in
            if WANDB_AVAILABLE:
                try:
                    # Check if user is logged in to wandb
                    wandb.Api()
                    wandb_logger = WandbLogger(
                        name=f"task2_fold_{fold_idx + 1}",
                        project="fomo-finetuning",
                        group="task2"
                    )
                    loggers.append(wandb_logger)
                    print("Using WandbLogger for logging")
                except Exception as e:
                    print(f"WandbLogger available but not configured: {e}")
                    print("Using CSVLogger only")
            
            checkpoint_callback = ModelCheckpoint(
                dirpath=os.path.join(self.results_dir, f"task2_fold_{fold_idx + 1}"),
                filename="best_model",
                monitor="val/dice",
                mode="max",
                save_top_k=1,
            )
            
            early_stop_callback = EarlyStopping(
                monitor="val/loss",
                min_delta=0.001,
                patience=10,
                verbose=True,
                mode="min",
            )
            
            trainer = L.Trainer(
                max_epochs=self.config["max_epochs"],
                logger=loggers,
                callbacks=[checkpoint_callback, early_stop_callback],
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
            )
            
            # Train model
            trainer.fit(model, data_module)
            
            # Test model
            test_results = trainer.test(model, data_module)
            fold_results.append(test_results[0])
            
            # Save fold results
            fold_df = pd.DataFrame([test_results[0]])
            fold_df.to_csv(
                os.path.join(self.results_dir, f"task2_fold_{fold_idx + 1}_results.csv"),
                index=False
            )
        
        # Aggregate results across folds
        aggregated_results = aggregate_metrics(fold_results)
        
        # Save aggregated results
        agg_df = pd.DataFrame([aggregated_results])
        agg_df.to_csv(
            os.path.join(self.results_dir, "task2_aggregated_results.csv"),
            index=False
        )
        
        print("\n=== Task 2 CV Complete ===")
        print(f"Aggregated Results: {aggregated_results}")
        
        return aggregated_results
    
    def run_task3_cv(self):
        """
        Run cross-validation for Task 3: Brain Age Regression
        """
        print("Running cross-validation for Task 3: Brain Age Regression")
        
        # Create data splits
        splits = create_kfold_splits_task3(n_splits=self.n_splits, random_state=self.random_state)
        
        # Store results for each fold
        fold_results = []
        
        for fold_idx, split in enumerate(splits):
            print(f"\n=== Fold {fold_idx + 1}/{self.n_splits} ===")
            
            # Create data module
            data_module = FOMOFinetuneTask3DataModule(
                task_type="regression",
                patch_size=self.config["patch_size"],
                batch_size=self.config["batch_size"],
                num_workers=self.config["num_workers"],
                train_samples=split["train"],
                val_samples=split["val"],
                test_samples=split["test"],
            )
            
            # Create model with gradual unfreezing parameters
            model_kwargs = {
                "config": self.config,
                "learning_rate": self.config["learning_rate"],
                "pretrained_ckpt_path": self.config["pretrained_ckpt_path"],
            }
            
            # Add unfreezing parameters if they exist in config
            if "unfreeze_strategy" in self.config:
                model_kwargs["unfreeze_strategy"] = self.config["unfreeze_strategy"]
            if "unfreeze_epoch" in self.config:
                model_kwargs["unfreeze_epoch"] = self.config["unfreeze_epoch"]
                
            model = FOMOFinetuneRegModel(**model_kwargs)
            
            # Create trainer
            loggers = [CSVLogger(
                self.results_dir, 
                name=f"task3_fold_{fold_idx + 1}"
            )]
            
            # Add WandbLogger if available and user is logged in
            if WANDB_AVAILABLE:
                try:
                    # Check if user is logged in to wandb
                    wandb.Api()
                    wandb_logger = WandbLogger(
                        name=f"task3_fold_{fold_idx + 1}",
                        project="fomo-finetuning",
                        group="task3"
                    )
                    loggers.append(wandb_logger)
                    print("Using WandbLogger for logging")
                except Exception as e:
                    print(f"WandbLogger available but not configured: {e}")
                    print("Using CSVLogger only")
            
            checkpoint_callback = ModelCheckpoint(
                dirpath=os.path.join(self.results_dir, f"task3_fold_{fold_idx + 1}"),
                filename="best_model",
                monitor="val/mae",
                mode="min",
                save_top_k=1,
            )
            
            early_stop_callback = EarlyStopping(
                monitor="val/loss",
                min_delta=0.001,
                patience=10,
                verbose=True,
                mode="min",
            )
            
            trainer = L.Trainer(
                max_epochs=self.config["max_epochs"],
                logger=loggers,
                callbacks=[checkpoint_callback, early_stop_callback],
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
            )
            
            # Train model
            trainer.fit(model, data_module)
            
            # Test model
            test_results = trainer.test(model, data_module)
            fold_results.append(test_results[0])
            
            # Save fold results
            fold_df = pd.DataFrame([test_results[0]])
            fold_df.to_csv(
                os.path.join(self.results_dir, f"task3_fold_{fold_idx + 1}_results.csv"),
                index=False
            )
        
        # Aggregate results across folds
        aggregated_results = aggregate_metrics(fold_results)
        
        # Save aggregated results
        agg_df = pd.DataFrame([aggregated_results])
        agg_df.to_csv(
            os.path.join(self.results_dir, "task3_aggregated_results.csv"),
            index=False
        )
        
        print("\n=== Task 3 CV Complete ===")
        print(f"Aggregated Results: {aggregated_results}")
        
        return aggregated_results
    
    def run_cross_validation(self):
        """
        Run cross-validation for the specified task
        """
        if self.task_id == 1:
            return self.run_task1_cv()
        elif self.task_id == 2:
            return self.run_task2_cv()
        elif self.task_id == 3:
            return self.run_task3_cv()
        else:
            raise ValueError(f"Unsupported task ID: {self.task_id}")