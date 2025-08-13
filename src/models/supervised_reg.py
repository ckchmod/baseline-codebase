from typing import Optional
import torch
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, R2Score

from models.supervised_base import BaseSupervisedModel


class SupervisedRegModel(BaseSupervisedModel):
    """
    Supervised model for regression tasks.
    Inherits from BaseSupervisedModel and implements regression-specific functionality.
    """

    def __init__(
        self,
        config: dict = {},
        learning_rate: float = 1e-3,
        do_compile: Optional[bool] = False,
        compile_mode: Optional[str] = "default",
        weight_decay: float = 3e-5,
        amsgrad: bool = False,
        eps: float = 1e-8,
        betas: tuple = (0.9, 0.999),
    ):
        super().__init__(
            config=config,
            learning_rate=learning_rate,
            do_compile=do_compile,
            compile_mode=compile_mode,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            eps=eps,
            betas=betas,
            deep_supervision=False,  # Regression doesn't use deep supervision
        )

    def _configure_metrics(self, prefix: str):
        """
        Configure regression-specific metrics

        Args:
            prefix: Prefix for metric names (train or val)

        Returns:
            MetricCollection: Collection of regression metrics
        """
        return MetricCollection(
            {
                f"{prefix}/mse": MeanSquaredError(),
                f"{prefix}/mae": MeanAbsoluteError(),
                f"{prefix}/r2": R2Score(),
            }
        )

    def _configure_losses(self):
        """
        Configure regression-specific loss functions

        Returns:
            tuple: (train_loss_fn, val_loss_fn)
        """
        # For regression, we typically use MSE loss
        loss_fn = torch.nn.MSELoss()
        return loss_fn, loss_fn

    def _process_batch(self, batch):
        """
        Process regression batch data

        Args:
            batch: Input batch

        Returns:
            tuple: (inputs, target, file_path)
        """
        inputs, target, file_path = batch["image"], batch["label"], batch["file_path"]
        # Keep target as float for regression tasks
        target = target.float()
        return inputs, target, file_path

    def compute_metrics(self, metrics, output, target, ignore_index=None):
        """
        Compute regression metrics

        Args:
            metrics: Metrics collection
            output: Model output
            target: Ground truth
            ignore_index: Index to ignore in metrics (not used in regression)

        Returns:
            dict: Dictionary of computed metrics
        """
        # Ensure both output and target are float tensors
        output = output.float()
        target = target.float()
        
        # Simplified shape handling - ensure both tensors are 1D
        # If either is 2D with last dimension of 1, squeeze it
        if len(output.shape) == 2 and output.shape[1] == 1:
            output = output.squeeze(-1)
        if len(target.shape) == 2 and target.shape[1] == 1:
            target = target.squeeze(-1)
            
        # Ensure both are 1D tensors
        if len(output.shape) > 1:
            output = output.view(-1)
        if len(target.shape) > 1:
            target = target.view(-1)
        
        # Additional safety check - ensure same length
        if output.shape[0] != target.shape[0]:
            raise ValueError(f"Output and target have different lengths: {output.shape[0]} vs {target.shape[0]}")
        
        return metrics(output, target)