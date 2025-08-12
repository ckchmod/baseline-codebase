# FOMO Fine-tuning Pipeline with Gradual Unfreezing

This is a modified version of the FOMO fine-tuning pipeline that implements gradual unfreezing for better fine-tuning performance.

## Features

1. **Gradual Unfreezing**: The encoder is initially frozen for a specified number of epochs, then unfrozen for the remainder of training.
2. **Flexible Unfreezing Strategies**: 
   - `gradual`: Freeze encoder for first N epochs, then unfreeze (default)
   - `head_only`: Only train the task-specific head
   - `full`: Train all parameters from the beginning (original behavior)

## Configuration

The new configuration parameters are:

- `unfreeze_strategy`: One of "gradual", "head_only", or "full"
- `unfreeze_epoch`: Epoch at which to unfreeze the encoder (only used with "gradual" strategy)

## Usage

To run the tasks with gradual unfreezing:

```bash
# Task 1 (Infarct Detection)
python fomo_finetuning_pipeline_gradual_unfreeze/scripts/run_task1_gradual_unfreeze.py

# Task 2 (Meningioma Segmentation)
python fomo_finetuning_pipeline_gradual_unfreeze/scripts/run_task2_gradual_unfreeze.py

# Task 3 (Brain Age Regression)
python fomo_finetuning_pipeline_gradual_unfreeze/scripts/run_task3_gradual_unfreeze.py
```

## Implementation Details

The implementation modifies the `FOMOFinetuneModel` class to support different unfreezing strategies:

1. **Initialization**: The model can be configured with different unfreezing strategies
2. **Setup**: The `setup_unfreezing_strategy()` method initializes the freezing based on the selected strategy
3. **Optimizer Configuration**: The `configure_optimizers()` method dynamically adjusts which parameters are trained based on the current epoch and strategy

## Benefits

1. **Stable Initial Training**: Starting with a frozen encoder allows the task-specific head to adapt without disrupting the pretrained features
2. **Better Convergence**: Gradual unfreezing can lead to better convergence and performance
3. **Flexibility**: Different strategies can be tested to find what works best for each task