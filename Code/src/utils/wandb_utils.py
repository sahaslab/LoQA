"""Utilities for logging to Weights & Biases (wandb)."""

import os
import wandb
from typing import Dict, Any, Optional, List

def log_json_predictions(
    rows: List[Dict[str, Any]],
    file_path: str,
    dataset_name: str,
    split_name: str,
    qg_model_name: str,
    qg_prompt_version: str,
    pd_model_name: str,
    pd_prompt_version: str,
    num_samples: int,
    project: str = "loqa-predictions",
    config: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
) -> None:
    """Log predictions to wandb with interactive table visualization.
    
    Args:
        rows: List of prediction dictionaries
        file_path: Path to the predictions JSON file
        dataset_name: Name of the dataset
        model_name: Name of the model used
        model_origin: Origin of the model
        max_samples: Number of samples processed
        project: Wandb project name
        config: Additional config parameters
        api_key: Wandb API key (uses WANDB_API_KEY env var if None)
    """
    # Login with API key
    api_key = api_key or os.getenv("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
    
    # Initialize wandb run
    run_name = f"{qg_model_name}-{qg_prompt_version}-{dataset_name}-{split_name}-{pd_model_name}-{pd_prompt_version}-{num_samples}"
    init_config = {
        "dataset_name": dataset_name,
        "split_name": split_name,
        "qg_model_name": qg_model_name,
        "qg_prompt_version": qg_prompt_version,
        "pd_model_name": pd_model_name,
        "pd_prompt_version": pd_prompt_version,
        "num_samples": num_samples,
    }
    if config:
        init_config.update(config)
    
    wandb.init(
        project=project,
        name=run_name,
        config=init_config,
        tags=[qg_model_name, qg_prompt_version, dataset_name, split_name, pd_model_name, pd_prompt_version],
        settings=wandb.Settings(code_dir=None),
    )
    
    # Create wandb Table for interactive visualization
    # Get all unique column names from rows
    if rows:
        columns = list(rows[0].keys())
        
        # Create table data
        table_data = []
        for row in rows:
            table_data.append([row.get(col, "") for col in columns])
        
        # Create and log the table
        predictions_table = wandb.Table(columns=columns, data=table_data)
        wandb.log({"predictions_table": predictions_table})
        
        print(f"Created interactive table with {len(rows)} rows and {len(columns)} columns")
    
    # Log artifact (file download)
    artifact = wandb.Artifact(
        name=f"predictions-{run_name}",
        type="model predictions",
        description=f"Predictions for {dataset_name} dataset -{split_name} split with -{pd_model_name} model with -{pd_prompt_version} prompt",
        metadata={
            "total_rows": len(rows),
            "columns": list(rows[0].keys()) if rows else [],
        }
    )
    artifact.add_file(file_path)
    wandb.log_artifact(artifact)
    
    # Log summary metrics
    wandb.log({
        "samples_processed": num_samples,
    })
    
    print(f"Logged to wandb: {wandb.run.url}")
    wandb.finish()

def log_scores(
    results: Dict[str, Any],
    file_path: Optional[str],
    pred_key: str,
    dataset_name: str,
    split_name: str,
    qg_model_name: str,
    qg_prompt_version: str,
    pd_model_name: str,
    pd_prompt_version: str,
    project: str = "loqa-scores",
    config: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
) -> None:
    """Log scores to wandb."""

    # Login with API key
    api_key = api_key or os.getenv("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
    
    # Initialize wandb run
    run_name = f"sc-{pred_key}-{qg_model_name}-{qg_prompt_version}-{dataset_name}-{split_name}-{pd_model_name}-{pd_prompt_version}"
    init_config = {
        "pred_key": pred_key,
        "dataset_name": dataset_name,
        "split_name": split_name,
        "qg_model_name": qg_model_name,
        "qg_prompt_version": qg_prompt_version,
        "pd_model_name": pd_model_name,
        "pd_prompt_version": pd_prompt_version,
    }
    if config:
        init_config.update(config)
    
    wandb.init(
        project=project,
        name=run_name,
        config=init_config,
        tags=[pred_key, qg_model_name, qg_prompt_version, dataset_name, split_name, pd_model_name, pd_prompt_version],
        settings=wandb.Settings(code_dir=None),
    )
    
    # Flatten nested results for better wandb visualization
    # Convert {'overall': {...}} to {'overall/exact-match-precision': ..., ...}
    flattened_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            for metric_name, metric_value in value.items():
                flattened_key = f"{key}/{metric_name}"
                flattened_results[flattened_key] = metric_value
        else:
            flattened_results[key] = value
    
    # Log scores as metrics (creates automatic charts)
    wandb.log(flattened_results)
    
    # Create and log summary table for overall scores
    if 'overall' in results:
        overall_metrics = results['overall']
        overall_table = wandb.Table(
            columns=["Pred Key", "Metric", "Precision", "Recall", "F1", "Count"],
            data=[
                [pred_key, "Exact Match", 
                 overall_metrics.get('exact-match-precision', 0.0),
                 overall_metrics.get('exact-match-recall', 0.0),
                 overall_metrics.get('exact-match-f1', 0.0),
                 overall_metrics.get('exact-match-count', 0)],
                [pred_key, "Relaxed Match",
                 overall_metrics.get('relaxed-match-precision', 0.0),
                 overall_metrics.get('relaxed-match-recall', 0.0),
                 overall_metrics.get('relaxed-match-f1', 0.0),
                 overall_metrics.get('relaxed-match-count', 0)],
            ]
        )
        wandb.log({"overall_scores_table": overall_table})
        
        # Create summary table with counts
        counts_table = wandb.Table(
            columns=["Pred Key", "Type", "Count"],
            data=[
                [pred_key, "Ground Truth", overall_metrics.get('ground-truth-count', 0)],
                [pred_key, "Predictions", overall_metrics.get('prediction-count', 0)],
                [pred_key, "Exact Matches", overall_metrics.get('exact-match-count', 0)],
                [pred_key, "Relaxed Matches", overall_metrics.get('relaxed-match-count', 0)],
            ]
        )
        wandb.log({"counts_table": counts_table})
    
    # Log artifact (file download) if file exists
    if file_path and os.path.exists(file_path) and os.path.isfile(file_path):
        artifact = wandb.Artifact(
            name=f"{run_name}",
            type="evaluation scores",
            description=f"Scores for {dataset_name} dataset - {split_name} split with {pd_model_name} model ({pred_key} predictions)",
            metadata={
                "pred_key": pred_key,
                "dataset_name": dataset_name,
                "split_name": split_name,
            }
        )
        artifact.add_file(file_path)
        wandb.log_artifact(artifact)
        print(f"Logged scores artifact: {file_path}")
    
    print(f"Logged to wandb: {wandb.run.url}")
    wandb.finish()