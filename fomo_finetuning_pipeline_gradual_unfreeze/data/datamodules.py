import lightning as pl
from torchvision.transforms import Compose
import logging
import torch
from typing import Literal, Optional, Tuple
from torch.utils.data import DataLoader, Sampler
from data.datasets import FOMOFinetuneTask1Dataset, FOMOFinetuneTask2Dataset, FOMOFinetuneTask3Dataset


class FOMOFinetuneDataModule(pl.LightningDataModule):
    """
    Base data module class for FOMO fine-tuning tasks.
    """

    def __init__(
        self,
        task_type: str,
        patch_size: Tuple[int, int, int],
        batch_size: int,
        num_workers: int,
        train_samples: list,
        val_samples: list,
        test_samples: list,
        composed_train_transforms: Optional[Compose] = None,
        composed_val_transforms: Optional[Compose] = None,
        composed_test_transforms: Optional[Compose] = None,
    ):
        super().__init__()
        
        self.task_type = task_type
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        
        self.composed_train_transforms = composed_train_transforms
        self.composed_val_transforms = composed_val_transforms
        self.composed_test_transforms = composed_test_transforms
        
        logging.info(f"Using {self.num_workers} workers")

    def setup(self, stage: Literal["fit", "test", "predict"]):
        raise NotImplementedError("Subclasses must implement setup method")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=False,
        )


class FOMOFinetuneTask1DataModule(FOMOFinetuneDataModule):
    """
    Data module for Task 1: Infarct Detection (Classification)
    """

    def setup(self, stage: Literal["fit", "test", "predict"]):
        if stage == "fit" or stage == "test":
            self.train_dataset = FOMOFinetuneTask1Dataset(
                samples=self.train_samples,
                patch_size=self.patch_size,
                composed_transforms=self.composed_train_transforms,
            )
            
            self.val_dataset = FOMOFinetuneTask1Dataset(
                samples=self.val_samples,
                patch_size=self.patch_size,
                composed_transforms=self.composed_val_transforms,
            )
            
        if stage == "test":
            self.test_dataset = FOMOFinetuneTask1Dataset(
                samples=self.test_samples,
                patch_size=self.patch_size,
                composed_transforms=self.composed_test_transforms,
            )


class FOMOFinetuneTask2DataModule(FOMOFinetuneDataModule):
    """
    Data module for Task 2: Meningioma Segmentation (Segmentation)
    """

    def setup(self, stage: Literal["fit", "test", "predict"]):
        if stage == "fit" or stage == "test":
            self.train_dataset = FOMOFinetuneTask2Dataset(
                samples=self.train_samples,
                patch_size=self.patch_size,
                composed_transforms=self.composed_train_transforms,
            )
            
            self.val_dataset = FOMOFinetuneTask2Dataset(
                samples=self.val_samples,
                patch_size=self.patch_size,
                composed_transforms=self.composed_val_transforms,
            )
            
        if stage == "test":
            self.test_dataset = FOMOFinetuneTask2Dataset(
                samples=self.test_samples,
                patch_size=self.patch_size,
                composed_transforms=self.composed_test_transforms,
            )


class FOMOFinetuneTask3DataModule(FOMOFinetuneDataModule):
    """
    Data module for Task 3: Brain Age Regression (Regression)
    """

    def setup(self, stage: Literal["fit", "test", "predict"]):
        if stage == "fit" or stage == "test":
            self.train_dataset = FOMOFinetuneTask3Dataset(
                samples=self.train_samples,
                patch_size=self.patch_size,
                composed_transforms=self.composed_train_transforms,
            )
            
            self.val_dataset = FOMOFinetuneTask3Dataset(
                samples=self.val_samples,
                patch_size=self.patch_size,
                composed_transforms=self.composed_val_transforms,
            )
            
        if stage == "test":
            self.test_dataset = FOMOFinetuneTask3Dataset(
                samples=self.test_samples,
                patch_size=self.patch_size,
                composed_transforms=self.composed_test_transforms,
            )