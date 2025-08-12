import numpy as np
import pandas as pd


def aggregate_metrics(fold_results):
    """
    Aggregate metrics across folds, computing mean and standard deviation
    
    Args:
        fold_results: List of dictionaries containing metrics for each fold
        
    Returns:
        Dictionary with aggregated metrics (mean ± std)
    """
    if not fold_results:
        return {}
    
    # Get all metric keys from the first fold
    metric_keys = fold_results[0].keys()
    
    # Aggregate metrics
    aggregated = {}
    for key in metric_keys:
        values = [fold[key] for fold in fold_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        aggregated[key] = f"{mean_val:.4f} ± {std_val:.4f}"
        
        # Also store raw values for further analysis
        aggregated[f"{key}_mean"] = mean_val
        aggregated[f"{key}_std"] = std_val
        aggregated[f"{key}_values"] = values
    
    return aggregated


def save_detailed_results(fold_results, output_path):
    """
    Save detailed results to a CSV file
    
    Args:
        fold_results: List of dictionaries containing metrics for each fold
        output_path: Path to save the CSV file
    """
    # Convert to DataFrame
    df = pd.DataFrame(fold_results)
    
    # Add fold column
    df.insert(0, 'fold', range(1, len(fold_results) + 1))
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    return df


def load_fold_results(results_dir, task_id, n_folds=5):
    """
    Load results from all folds for a given task
    
    Args:
        results_dir: Directory containing results
        task_id: Task ID (1, 2, or 3)
        n_folds: Number of folds
        
    Returns:
        List of dictionaries containing metrics for each fold
    """
    fold_results = []
    
    for fold_idx in range(1, n_folds + 1):
        csv_path = f"{results_dir}/task{task_id}_fold_{fold_idx}_results.csv"
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                fold_results.append(df.iloc[0].to_dict())
        except FileNotFoundError:
            print(f"Results file not found for Task {task_id}, Fold {fold_idx}")
            
    return fold_results