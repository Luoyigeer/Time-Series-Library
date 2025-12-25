"""
Training Utilities
==================

Helper functions for model training and checkpointing.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
from tqdm import tqdm


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    max_grad_norm: float = 1.0
) -> Dict[str, float]:
    """
    Train model for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        epoch: Current epoch number
        max_grad_norm: Maximum gradient norm for clipping
    
    Returns:
        Dictionary of training metrics
    """
    model.train()
    total_loss = 0.0
    predictions_list = []
    targets_list = []
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, batch in enumerate(progress_bar):
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
        
        # Compute loss
        loss = criterion(predictions, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        predictions_list.append(predictions.detach().cpu().numpy())
        targets_list.append(y.detach().cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
    
    # Compute epoch metrics
    avg_loss = total_loss / len(dataloader)
    predictions = np.concatenate(predictions_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    return {
        'loss': avg_loss,
        'mae': mae,
        'rmse': rmse
    }


def validate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """
    Validate model for one epoch.
    
    Args:
        model: Model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
    
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    total_loss = 0.0
    predictions_list = []
    targets_list = []
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
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
            
            # Compute loss
            loss = criterion(predictions, y)
            
            # Track metrics
            total_loss += loss.item()
            predictions_list.append(predictions.cpu().numpy())
            targets_list.append(y.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
    
    # Compute epoch metrics
    avg_loss = total_loss / len(dataloader)
    predictions = np.concatenate(predictions_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    # Compute R-squared
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-6))
    
    return {
        'loss': avg_loss,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_path: str,
    is_best: bool = False
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch
        metrics: Dictionary of metrics
        save_path: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = save_path.parent / 'best_model.pth'
        torch.save(checkpoint, best_path)
        print(f"Saved best model to {best_path}")


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device('cpu')
) -> Tuple[nn.Module, Optional[torch.optim.Optimizer], int, Dict]:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        optimizer: Optional optimizer to load state into
        device: Device to load model on
    
    Returns:
        Loaded model, optimizer, epoch, and metrics
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    print(f"Loaded checkpoint from epoch {epoch}")
    
    return model, optimizer, epoch, metrics
