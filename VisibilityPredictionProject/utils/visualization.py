"""
Visualization Utilities
========================

Functions for visualizing model outputs and training progress.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional
import torch


def visualize_adjacency(
    adj_matrix: np.ndarray,
    station_names: Optional[List[str]] = None,
    save_path: str = None,
    title: str = 'Station Adjacency Matrix'
) -> None:
    """
    Visualize adjacency matrix as a heatmap.
    
    Args:
        adj_matrix: Adjacency matrix [num_stations, num_stations]
        station_names: Optional list of station names
        save_path: Path to save plot
        title: Plot title
    """
    plt.figure(figsize=(10, 8))
    
    if station_names is None:
        station_names = [f'S{i}' for i in range(len(adj_matrix))]
    
    sns.heatmap(
        adj_matrix,
        xticklabels=station_names,
        yticklabels=station_names,
        cmap='YlOrRd',
        annot=False,
        fmt='.2f',
        cbar_kws={'label': 'Connection Strength'}
    )
    
    plt.title(title)
    plt.xlabel('Target Station')
    plt.ylabel('Source Station')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved adjacency visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: Dict[str, List[float]] = None,
    val_metrics: Dict[str, List[float]] = None,
    save_path: str = None
) -> None:
    """
    Plot training and validation curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_metrics: Optional dictionary of training metrics
        val_metrics: Optional dictionary of validation metrics
        save_path: Path to save plot
    """
    n_metrics = 1
    if train_metrics:
        n_metrics += len(train_metrics)
    
    fig, axes = plt.subplots(1, min(n_metrics, 3), figsize=(5 * min(n_metrics, 3), 4))
    if n_metrics == 1:
        axes = [axes]
    
    # Plot loss
    ax = axes[0]
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Train Loss', marker='o')
    ax.plot(epochs, val_losses, 'r-', label='Val Loss', marker='s')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot additional metrics
    if train_metrics and val_metrics:
        metric_idx = 1
        for metric_name in list(train_metrics.keys())[:2]:  # Plot up to 2 additional metrics
            if metric_idx >= len(axes):
                break
            
            ax = axes[metric_idx]
            ax.plot(epochs, train_metrics[metric_name], 'b-', label=f'Train {metric_name.upper()}', marker='o')
            ax.plot(epochs, val_metrics[metric_name], 'r-', label=f'Val {metric_name.upper()}', marker='s')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name.upper())
            ax.set_title(f'{metric_name.upper()} Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
            metric_idx += 1
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_attention(
    attention_weights: torch.Tensor,
    save_path: str = None,
    title: str = 'Attention Weights'
) -> None:
    """
    Visualize attention weights.
    
    Args:
        attention_weights: Attention weights [seq_len, seq_len] or [batch, heads, seq_len, seq_len]
        save_path: Path to save plot
        title: Plot title
    """
    # Handle different attention weight shapes
    if len(attention_weights.shape) == 4:
        # Average over batch and heads
        attention_weights = attention_weights.mean(dim=(0, 1))
    elif len(attention_weights.shape) == 3:
        # Average over batch
        attention_weights = attention_weights.mean(dim=0)
    
    # Convert to numpy
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_weights,
        cmap='viridis',
        cbar_kws={'label': 'Attention Weight'}
    )
    
    plt.title(title)
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_station_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    station_idx: int,
    save_path: str = None
) -> None:
    """
    Plot predictions for a specific station.
    
    Args:
        predictions: Predicted values [num_samples, num_stations]
        targets: Ground truth values [num_samples, num_stations]
        station_idx: Index of station to plot
        save_path: Path to save plot
    """
    plt.figure(figsize=(12, 4))
    
    station_preds = predictions[:, station_idx]
    station_targets = targets[:, station_idx]
    
    indices = np.arange(min(len(station_preds), 200))
    
    plt.plot(indices, station_targets[indices], label='Ground Truth', alpha=0.7, linewidth=2)
    plt.plot(indices, station_preds[indices], label='Predictions', alpha=0.7, linewidth=2)
    
    plt.xlabel('Time Step')
    plt.ylabel('Visibility (km)')
    plt.title(f'Station {station_idx} Visibility Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved station plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_spatial_temporal_features(
    features: np.ndarray,
    save_path: str = None
) -> None:
    """
    Visualize spatial-temporal features.
    
    Args:
        features: Features array [time, stations, features]
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, min(features.shape[2], 3), figsize=(15, 4))
    if features.shape[2] == 1:
        axes = [axes]
    
    for i in range(min(features.shape[2], 3)):
        ax = axes[i]
        im = ax.imshow(features[:, :, i].T, aspect='auto', cmap='viridis', interpolation='nearest')
        ax.set_xlabel('Time')
        ax.set_ylabel('Station')
        ax.set_title(f'Feature {i}')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved spatial-temporal visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()
