"""
Data Utility Functions
======================

Helper functions for data processing and preparation.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional


def prepare_graph_data(
    data: np.ndarray,
    num_stations: int,
    adjacency_type: str = 'distance'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare graph structure data for GNN models.
    
    Args:
        data: Input data array
        num_stations: Number of stations
        adjacency_type: Type of adjacency matrix ('distance', 'correlation', 'fully_connected')
    
    Returns:
        Processed data and adjacency matrix
    """
    if adjacency_type == 'fully_connected':
        adj_matrix = np.ones((num_stations, num_stations))
        np.fill_diagonal(adj_matrix, 0)
    elif adjacency_type == 'correlation':
        # Compute correlation-based adjacency
        if len(data.shape) == 3:  # [time, stations, features]
            corr_matrix = np.corrcoef(data[:, :, 0].T)  # Use first feature
        else:
            corr_matrix = np.corrcoef(data.T)
        adj_matrix = np.abs(corr_matrix)
        np.fill_diagonal(adj_matrix, 0)
    else:  # distance-based (placeholder)
        adj_matrix = np.ones((num_stations, num_stations))
        np.fill_diagonal(adj_matrix, 0)
    
    # Normalize adjacency matrix
    row_sum = adj_matrix.sum(axis=1, keepdims=True)
    adj_matrix = adj_matrix / (row_sum + 1e-6)
    
    return data, adj_matrix


def normalize_features(
    data: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    feature_wise: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize features using z-score normalization.
    
    Args:
        data: Input data [time, stations, features] or [time, features]
        mean: Pre-computed mean (if None, compute from data)
        std: Pre-computed std (if None, compute from data)
        feature_wise: If True, normalize each feature independently
    
    Returns:
        Normalized data, mean, and std
    """
    if mean is None:
        if feature_wise:
            mean = np.mean(data, axis=tuple(range(len(data.shape) - 1)), keepdims=True)
        else:
            mean = np.mean(data)
    
    if std is None:
        if feature_wise:
            std = np.std(data, axis=tuple(range(len(data.shape) - 1)), keepdims=True)
        else:
            std = np.std(data)
        std = np.where(std == 0, 1.0, std)  # Avoid division by zero
    
    normalized_data = (data - mean) / std
    
    return normalized_data, mean, std


def denormalize_features(
    data: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray
) -> np.ndarray:
    """
    Denormalize features back to original scale.
    
    Args:
        data: Normalized data
        mean: Mean used for normalization
        std: Std used for normalization
    
    Returns:
        Denormalized data
    """
    return data * std + mean


def create_time_features(
    timestamps: np.ndarray,
    freq: str = 'H'
) -> np.ndarray:
    """
    Create temporal features from timestamps.
    
    Args:
        timestamps: Array of timestamps
        freq: Frequency of data ('H' for hourly, 'D' for daily, etc.)
    
    Returns:
        Time features array
    """
    import pandas as pd
    
    if not isinstance(timestamps, pd.DatetimeIndex):
        timestamps = pd.DatetimeIndex(timestamps)
    
    # Extract time features
    hour = timestamps.hour / 24.0
    day = timestamps.day / 31.0
    weekday = timestamps.weekday / 7.0
    month = timestamps.month / 12.0
    
    time_features = np.stack([
        np.sin(2 * np.pi * hour),
        np.cos(2 * np.pi * hour),
        np.sin(2 * np.pi * day),
        np.cos(2 * np.pi * day),
        np.sin(2 * np.pi * weekday),
        np.cos(2 * np.pi * weekday),
        np.sin(2 * np.pi * month),
        np.cos(2 * np.pi * month),
    ], axis=-1)
    
    return time_features


def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Dict[str, float]:
    """
    Calculate evaluation metrics for visibility prediction.
    
    Args:
        predictions: Predicted visibility values
        targets: Ground truth visibility values
    
    Returns:
        Dictionary of metrics
    """
    # Mean Absolute Error
    mae = np.mean(np.abs(predictions - targets))
    
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((targets - predictions) / (targets + 1e-6))) * 100
    
    # R-squared
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-6))
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }


def split_data(
    data: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        data: Input data
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
    
    Returns:
        Train, validation, and test splits
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), \
        "Ratios must sum to 1.0"
    
    n_samples = len(data)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    return train_data, val_data, test_data
