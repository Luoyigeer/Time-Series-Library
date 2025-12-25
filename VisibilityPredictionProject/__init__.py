"""
VisibilityPredictionProject
===========================

A comprehensive framework for visibility prediction using GNNs and transfer learning.

This project provides:
- Multi-station visibility prediction using Graph Neural Networks
- Single-image visibility detection using vision models  
- Transfer learning capabilities from GNN to vision models
- NOAA data processing pipeline
- Complete training and evaluation utilities

Main Components:
- models: Neural network architectures (GNN, Vision, Transfer)
- data_loader: Data loading utilities for NOAA and image data
- utils: Training, evaluation, and visualization tools
- scripts: Ready-to-use training scripts

Quick Start:
    >>> from VisibilityPredictionProject.models import STGNNVisibility
    >>> model = STGNNVisibility(num_stations=10, hidden_dim=128)
    >>> # Train the model...
    
For detailed documentation, see README.md
"""

__version__ = '1.0.0'
__author__ = 'Time-Series-Library Contributors'

# Import main components for easy access
from .models import (
    GraphVisibilityNet,
    STGNNVisibility,
    VisibilityVisionNet,
    TransferLearningAdapter
)

from .data_loader import (
    NOAAVisibilityLoader,
    VisibilityImageLoader,
    create_noaa_dataloader,
    create_image_dataloader
)

__all__ = [
    # Models
    'GraphVisibilityNet',
    'STGNNVisibility',
    'VisibilityVisionNet',
    'TransferLearningAdapter',
    
    # Data loaders
    'NOAAVisibilityLoader',
    'VisibilityImageLoader',
    'create_noaa_dataloader',
    'create_image_dataloader',
]
