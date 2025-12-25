"""
Evaluation Utilities
====================

Functions for model evaluation and metrics computation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from pathlib import Path


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    return_predictions: bool = False
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: Data loader
        device: Device to evaluate on
        return_predictions: Whether to return predictions and targets
    
    Returns:
        Dictionary of evaluation metrics (and optionally predictions)
    """
    model.eval()
    predictions_list = []
    targets_list = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            if len(batch) == 3:  # GNN data (x, y, adj)
                x, y, adj = batch
                x = x.to(device)
                y = y.to(device)
                adj = adj.to(device)
                
                # Forward pass
                predictions = model(x, adj)
                
                # Reshape predictions to match targets
                if len(predictions.shape) > 2:
                    predictions = predictions.squeeze(-1)
                if len(y.shape) > 2:
                    y = y.squeeze(-1)
            else:  # Image data (x, y)
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                
                # Forward pass
                predictions = model(x)
            
            predictions_list.append(predictions.cpu().numpy())
            targets_list.append(y.cpu().numpy())
    
    # Concatenate all predictions and targets
    predictions = np.concatenate(predictions_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    
    # Compute metrics
    metrics = compute_metrics(predictions, targets)
    
    if return_predictions:
        metrics['predictions'] = predictions
        metrics['targets'] = targets
    
    return metrics


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
    
    Returns:
        Dictionary of metrics
    """
    # Flatten arrays if needed
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Mean Absolute Error
    mae = np.mean(np.abs(predictions - targets))
    
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((targets - predictions) / (np.abs(targets) + 1e-6))) * 100
    
    # R-squared
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-6))
    
    # Correlation coefficient
    correlation = np.corrcoef(predictions, targets)[0, 1]
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape),
        'r2': float(r2),
        'correlation': float(correlation)
    }


def plot_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: str = None,
    title: str = 'Visibility Predictions vs Ground Truth'
) -> None:
    """
    Plot predictions against ground truth.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
        save_path: Path to save plot (if None, display instead)
        title: Plot title
    """
    plt.figure(figsize=(12, 5))
    
    # Flatten arrays
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Time series plot
    plt.subplot(1, 2, 1)
    indices = np.arange(min(len(predictions), 200))  # Plot first 200 points
    plt.plot(indices, targets[indices], label='Ground Truth', alpha=0.7)
    plt.plot(indices, predictions[indices], label='Predictions', alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Visibility (km)')
    plt.title('Time Series Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Scatter plot
    plt.subplot(1, 2, 2)
    plt.scatter(targets, predictions, alpha=0.5, s=10)
    
    # Add diagonal line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.xlabel('Ground Truth Visibility (km)')
    plt.ylabel('Predicted Visibility (km)')
    plt.title('Prediction Scatter Plot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def evaluate_per_station(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_stations: int
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate metrics for each station separately.
    
    Args:
        predictions: Predicted values [batch_size, num_stations]
        targets: Ground truth values [batch_size, num_stations]
        num_stations: Number of stations
    
    Returns:
        Dictionary mapping station index to metrics
    """
    station_metrics = {}
    
    for station_idx in range(num_stations):
        station_preds = predictions[:, station_idx]
        station_targets = targets[:, station_idx]
        
        metrics = compute_metrics(station_preds, station_targets)
        station_metrics[station_idx] = metrics
    
    return station_metrics


def print_metrics(
    metrics: Dict[str, float],
    prefix: str = ''
) -> None:
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
        prefix: Optional prefix for print statements
    """
    print(f"\n{prefix}Evaluation Metrics:")
    print("=" * 50)
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{key.upper():20s}: {value:.4f}")
    print("=" * 50)
