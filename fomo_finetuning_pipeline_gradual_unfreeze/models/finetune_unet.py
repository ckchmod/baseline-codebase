import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from models.supervised_base import BaseSupervisedModel
from models.supervised_cls import SupervisedClsModel
from models.supervised_seg import SupervisedSegModel
from models.supervised_reg import SupervisedRegModel


class FOMOFinetuneModel(BaseSupervisedModel):
    """
    Fine-tuning model for FOMO tasks that loads a pretrained UNet-B model
    and adapts it for specific downstream tasks.
    """

    def __init__(
        self,
        config: dict = {},
        learning_rate: float = 1e-4,
        do_compile: bool = False,
        compile_mode: str = "default",
        weight_decay: float = 3e-5,
        amsgrad: bool = False,
        eps: float = 1e-8,
        betas: tuple = (0.9, 0.999),
        deep_supervision: bool = False,
        pretrained_ckpt_path: str = "pretrained/epoch=12.ckpt",
        task_type: str = "segmentation",
        unfreeze_strategy: str = "gradual",  # Options: "gradual", "full", "head_only"
        unfreeze_epoch: int = 10,  # Epoch to unfreeze encoder when using gradual unfreezing
    ):
        # Set task_type before calling super().__init__
        self.task_type = task_type
        self.pretrained_ckpt_path = pretrained_ckpt_path
        self.unfreeze_strategy = unfreeze_strategy
        self.unfreeze_epoch = unfreeze_epoch
        
        super().__init__(
            config=config,
            learning_rate=learning_rate,
            do_compile=do_compile,
            compile_mode=compile_mode,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            eps=eps,
            betas=betas,
            deep_supervision=deep_supervision,
        )

    def load_model(self):
        """Load the pretrained UNet-B model and adapt it for the specific task"""
        print(f"Loading pretrained model from: {self.pretrained_ckpt_path}")
        
        # Load the pretrained model state dict
        checkpoint = torch.load(self.pretrained_ckpt_path, map_location="cpu")
        pretrained_state_dict = checkpoint["state_dict"]
        
        # Create the base UNet-B model
        print(f"Loading Model: 3D {self.model_name}")
        from models import networks
        model_class = getattr(networks, self.model_name)
        
        conv_op = torch.nn.Conv3d
        norm_op = torch.nn.InstanceNorm3d
        
        # Pass task_type directly to UNet without mapping
        model_kwargs = {
            # Applies to all models
            "input_channels": self.num_modalities,
            "num_classes": self.num_classes,
            "output_channels": self.num_classes,
            "deep_supervision": self.deep_supervision,
            # Applies to most CNN-based architectures
            "conv_op": conv_op,
            # Applies to most CNN-based architectures (exceptions: UXNet)
            "norm_op": norm_op,
            # MedNeXt
            "checkpoint_style": None,
            # ensure not pretraining
            "mode": self.task_type,  # Pass task_type directly
        }
        
        # Filter kwargs for the specific model class
        from yucca.functional.utils.kwargs import filter_kwargs
        model_kwargs = filter_kwargs(model_class, model_kwargs)
        self.model = model_class(**model_kwargs)
        
        # Load pretrained weights
        self.load_pretrained_weights(pretrained_state_dict)
        
        # Initialize unfreezing strategy
        self.setup_unfreezing_strategy()

    def load_pretrained_weights(self, pretrained_state_dict):
        """Load pretrained weights, handling potential size mismatches"""
        # Filter out layers that have changed in size
        old_params = self.model.state_dict()
        filtered_state_dict = {
            k: v
            for k, v in pretrained_state_dict.items()
            if (k in old_params) and (old_params[k].shape == v.shape)
        }
        
        rejected_keys_new = [k for k in pretrained_state_dict.keys() if k not in old_params]
        rejected_keys_shape = [
            k for k in pretrained_state_dict.keys() 
            if k in old_params and old_params[k].shape != pretrained_state_dict[k].shape
        ]
        
        print(f"Rejected the following keys:")
        print(f"Not in old dict: {rejected_keys_new}")
        print(f"Wrong shape: {rejected_keys_shape}")
        
        # Load the filtered state dict
        self.model.load_state_dict(filtered_state_dict, strict=False)
        
        print(f"Successfully loaded pretrained weights")

    def setup_unfreezing_strategy(self):
        """Set up the unfreezing strategy based on configuration"""
        if self.unfreeze_strategy == "head_only":
            # Freeze encoder, only train head
            for param in self.model.encoder.parameters():
                param.requires_grad = False
        elif self.unfreeze_strategy == "gradual":
            # Initially freeze encoder for gradual unfreezing
            for param in self.model.encoder.parameters():
                param.requires_grad = False
        # For "full" strategy, all parameters remain trainable (default behavior)

    def configure_optimizers(self):
        """Configure optimizers with gradual unfreezing support"""
        # Set up task-specific loss functions
        self.loss_fn_train, self.loss_fn_val = self._configure_losses()

        # Determine which parameters to train based on unfreezing strategy
        if self.unfreeze_strategy == "head_only":
            # Only train the task head
            trainable_params = list(self.model.decoder.parameters())
            print("Training only the task head (decoder)")
        elif self.unfreeze_strategy == "gradual":
            # Check if we should unfreeze the encoder based on current epoch
            if hasattr(self, 'current_epoch') and self.current_epoch >= self.unfreeze_epoch:
                # Unfreeze encoder after specified epoch
                for param in self.model.encoder.parameters():
                    param.requires_grad = True
                trainable_params = self.model.parameters()
                print(f"Unfreezing encoder at epoch {self.current_epoch}")
            else:
                # Keep encoder frozen
                trainable_params = list(self.model.decoder.parameters())
                print(f"Keeping encoder frozen at epoch {self.current_epoch if hasattr(self, 'current_epoch') else 0}")
        else:
            # Full fine-tuning (default)
            trainable_params = self.model.parameters()
            print("Full fine-tuning (all parameters trainable)")

        self.optim = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
            eps=self.eps,
            betas=self.betas,
        )

        # Scheduler with early cut-off factor of 1.15
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, T_max=int(self.trainer.max_epochs * 1.15), eta_min=1e-9
        )

        # Return the optimizer and scheduler - the loss is not returned
        return {"optimizer": self.optim, "lr_scheduler": self.lr_scheduler}


class FOMOFinetuneClsModel(FOMOFinetuneModel):
    """
    Fine-tuning model for classification tasks (Task 1: Infarct Detection)
    """
    
    def __init__(self, **kwargs):
        super().__init__(task_type="classification", **kwargs)
        
    def _configure_metrics(self, prefix: str):
        return SupervisedClsModel._configure_metrics(self, prefix)
        
    def _configure_losses(self):
        return SupervisedClsModel._configure_losses(self)
        
    def compute_metrics(self, metrics, output, target, ignore_index=None):
        return SupervisedClsModel.compute_metrics(self, metrics, output, target, ignore_index)


class FOMOFinetuneSegModel(FOMOFinetuneModel):
    """
    Fine-tuning model for segmentation tasks (Task 2: Meningioma Segmentation)
    """
    
    def __init__(self, **kwargs):
        super().__init__(task_type="segmentation", **kwargs)
        
    def _configure_metrics(self, prefix: str):
        return SupervisedSegModel._configure_metrics(self, prefix)
        
    def _configure_losses(self):
        return SupervisedSegModel._configure_losses(self)
        
    def compute_metrics(self, metrics, output, target, ignore_index=0):
        return SupervisedSegModel.compute_metrics(self, metrics, output, target, ignore_index)


class FOMOFinetuneRegModel(FOMOFinetuneModel):
    """
    Fine-tuning model for regression tasks (Task 3: Brain Age Regression)
    """
    
    def __init__(self, **kwargs):
        super().__init__(task_type="regression", **kwargs)
        
    def _configure_metrics(self, prefix: str):
        return SupervisedRegModel._configure_metrics(self, prefix)
        
    def _configure_losses(self):
        return SupervisedRegModel._configure_losses(self)
        
    def compute_metrics(self, metrics, output, target, ignore_index=None):
        return SupervisedRegModel.compute_metrics(self, metrics, output, target, ignore_index)