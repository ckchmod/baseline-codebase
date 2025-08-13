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
        class FloatMSELoss(torch.nn.MSELoss):
            def forward(self, input, target):
                # Ensure both input and target are float tensors
                input = input.float()
                target = target.float()
                return super().forward(input, target)
        
        loss_fn = FloatMSELoss()
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
        # Ensure inputs and targets are float tensors for regression tasks
        inputs = inputs.float()
        target = target.float()
        # Ensure target has the correct shape for regression
        # If target is [batch_size, 1], squeeze it to [batch_size]
        if len(target.shape) > 1 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        return inputs, target, file_path

    def training_step(self, batch, _batch_idx):
        """Training step with float tensor handling for regression tasks"""
        input_channels = batch["image"].shape[1]
        assert input_channels == self.num_modalities, (
            f"Expected {self.num_modalities} input channels, but got {input_channels}. "
            "This often happens when the task config has changed _after_ data has been preprocessed. "
            "Check your data preprocessing."
        )

        inputs, target, _ = self._process_batch(batch)

        output = self(inputs)
        # Ensure output and target are float for loss computation
        output = output.float()
        target = target.float()
        loss = self.loss_fn_train(output, target)

        if self.deep_supervision and hasattr(output, "__iter__"):
            # If deep_supervision is enabled, output and target will be a list of (downsampled) tensors.
            # We only need the original ground truth and its corresponding prediction which is always the first entry in each list.
            output = output[0]
            target = target[0]

        metrics = self.compute_metrics(self.train_metrics, output, target)
        self.log_dict(
            {"train/loss": loss} | metrics,
            prog_bar=self.progress_bar,
            logger=True,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
        )

        return loss

    def validation_step(self, batch, _batch_idx):
        """Validation step with float tensor handling for regression tasks"""
        input_channels = batch["image"].shape[1]
        assert input_channels == self.num_modalities, (
            f"Expected {self.num_modalities} input channels, but got {input_channels}. "
            "This often happens when the task config has changed _after_ data has been preprocessed. "
            "Check your data preprocessing."
        )

        inputs, target, _ = self._process_batch(batch)

        output = self(inputs)
        # Ensure output and target are float for loss computation
        output = output.float()
        target = target.float()
        loss = self.loss_fn_val(output, target)
        metrics = self.compute_metrics(self.val_metrics, output, target)
        self.log_dict(
            {"val/loss": loss} | metrics,
            prog_bar=self.progress_bar,
            logger=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )

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
        
        # For regression tasks, ensure output and target have the same shape
        # Model outputs shape [batch_size, 1] but targets might be [batch_size]
        if output.shape != target.shape:
            # If output has an extra dimension, squeeze it
            if len(output.shape) > len(target.shape) and output.shape[-1] == 1:
                output = output.squeeze(-1)
            # If target has an extra dimension, squeeze it
            elif len(target.shape) > len(output.shape) and target.shape[-1] == 1:
                target = target.squeeze(-1)
            # If both are the same length but output has 1 in the last dimension
            elif len(output.shape) == len(target.shape) and output.shape[-1] == 1:
                output = output.squeeze(-1)
        
        return metrics(output, target)
