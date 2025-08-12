import os
import json
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from training.cross_validation import FOMOCrossValidationPipeline


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def main():
    # Load configuration
    import os
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "task2_config.json")
    config = load_config(config_path)
    
    # Create results directory
    os.makedirs(config["results_dir"], exist_ok=True)
    
    # Create and run cross-validation pipeline
    pipeline = FOMOCrossValidationPipeline(task_id=2, config=config)
    results = pipeline.run_cross_validation()
    
    print("\n=== Final Results ===")
    for key, value in results.items():
        if not key.endswith(("_mean", "_std", "_values")):
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()