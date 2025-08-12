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


def run_task(task_id):
    """Run a specific task"""
    # Determine config file based on task_id
    if task_id == 1:
        config_path = "configs/task1_config.json"
        task_name = "Infarct Detection"
    elif task_id == 2:
        config_path = "configs/task2_config.json"
        task_name = "Meningioma Segmentation"
    elif task_id == 3:
        config_path = "configs/task3_config.json"
        task_name = "Brain Age Regression"
    else:
        raise ValueError(f"Unsupported task ID: {task_id}")
    
    print(f"Running Task {task_id}: {task_name}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Create results directory
    os.makedirs(config["results_dir"], exist_ok=True)
    
    # Create and run cross-validation pipeline
    pipeline = FOMOCrossValidationPipeline(task_id=task_id, config=config)
    results = pipeline.run_cross_validation()
    
    print(f"\n=== Final Results for Task {task_id}: {task_name} ===")
    for key, value in results.items():
        if not key.endswith(("_mean", "_std", "_values")):
            print(f"{key}: {value}")
    
    return results


def main():
    """Main function to run all tasks or a specific task"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run FOMO fine-tuning tasks")
    parser.add_argument(
        "--task", 
        type=int, 
        choices=[1, 2, 3, 0], 
        default=0,
        help="Task to run (1: Infarct Detection, 2: Meningioma Segmentation, 3: Brain Age Regression, 0: All tasks)"
    )
    
    args = parser.parse_args()
    
    if args.task == 0:
        # Run all tasks
        print("Running all FOMO fine-tuning tasks...")
        results = {}
        for task_id in [1, 2, 3]:
            results[task_id] = run_task(task_id)
    else:
        # Run specific task
        run_task(args.task)


if __name__ == "__main__":
    main()