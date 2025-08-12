import sys
import os

# Add the parent directory to the path so we can import from other modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from data.datasets import FOMOFinetuneDataset
    print("Successfully imported FOMOFinetuneDataset")
except Exception as e:
    print(f"Error importing FOMOFinetuneDataset: {e}")

try:
    from data.datamodules import FOMOFinetuneDataModule
    print("Successfully imported FOMOFinetuneDataModule")
except Exception as e:
    print(f"Error importing FOMOFinetuneDataModule: {e}")

try:
    from models.finetune_unet import FOMOFinetuneModel
    print("Successfully imported FOMOFinetuneModel")
except Exception as e:
    print(f"Error importing FOMOFinetuneModel: {e}")

print("Basic import test completed")