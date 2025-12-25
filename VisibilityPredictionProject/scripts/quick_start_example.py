"""
Simple Example: Quick Start with VisibilityPredictionProject
============================================================

This script demonstrates basic usage of the VisibilityPredictionProject.
"""

import torch
import sys
from pathlib import Path

# Add project to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from VisibilityPredictionProject.models import GraphVisibilityNet, STGNNVisibility
from VisibilityPredictionProject.data_loader import NOAAVisibilityLoader, create_noaa_dataloader


def example_1_basic_model():
    """Example 1: Create and use a basic GNN model."""
    print("\n" + "="*60)
    print("Example 1: Basic GNN Model")
    print("="*60)
    
    # Create model
    model = GraphVisibilityNet(
        num_stations=10,
        input_dim=7,
        hidden_dim=64,
        output_dim=1
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create dummy input
    batch_size = 4
    seq_len = 24
    num_stations = 10
    input_dim = 7
    
    x = torch.randn(batch_size, seq_len, num_stations, input_dim)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        predictions = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Sample predictions:\n{predictions[0]}")


def example_2_data_loading():
    """Example 2: Load and process NOAA data."""
    print("\n" + "="*60)
    print("Example 2: Data Loading")
    print("="*60)
    
    # Create dataset (will use synthetic data if path doesn't exist)
    dataset = NOAAVisibilityLoader(
        data_path='./data/noaa_visibility',
        seq_len=24,
        pred_len=1,
        split='train'
    )
    
    print(f"Dataset contains {len(dataset)} samples")
    
    # Get a sample
    x, y, adj_matrix = dataset[0]
    
    print(f"Input features shape: {x.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Adjacency matrix shape: {adj_matrix.shape}")


def example_3_training():
    """Example 3: Simple training loop."""
    print("\n" + "="*60)
    print("Example 3: Training Loop")
    print("="*60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = STGNNVisibility(
        num_stations=5,  # Small model for demo
        input_dim=7,
        hidden_dim=32,
        output_dim=1
    ).to(device)
    
    # Create data loader
    train_loader = create_noaa_dataloader(
        './data/noaa_visibility',
        batch_size=8,
        split='train',
        num_stations=5
    )
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    # Train for a few steps
    model.train()
    print("\nTraining for 5 batches...")
    
    for i, (x, y, adj) in enumerate(train_loader):
        if i >= 5:  # Only 5 batches for demo
            break
        
        # Move to device
        x = x.to(device)
        y = y.to(device)
        adj = adj.to(device)
        
        # Forward pass
        predictions = model(x, adj)
        
        # Reshape if needed
        if len(predictions.shape) > 2:
            predictions = predictions.squeeze(-1)
        if len(y.shape) > 2:
            y = y.squeeze(-1)
        
        # Compute loss
        loss = criterion(predictions, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Batch {i+1}/5 - Loss: {loss.item():.4f}")
    
    print("Training demo complete!")


def example_4_visualization():
    """Example 4: Visualize model outputs."""
    print("\n" + "="*60)
    print("Example 4: Visualization")
    print("="*60)
    
    try:
        from VisibilityPredictionProject.utils import visualize_adjacency
        import numpy as np
        
        # Create model and get adjacency matrix
        model = GraphVisibilityNet(num_stations=5)
        adj_matrix = model.get_adjacency_matrix().numpy()
        
        print("Learned adjacency matrix:")
        print(adj_matrix)
        
        # Visualize (saves to file)
        visualize_adjacency(
            adj_matrix,
            station_names=[f'Station {i}' for i in range(5)],
            save_path='./adjacency_example.png'
        )
        print("Adjacency matrix visualization saved to adjacency_example.png")
        
    except Exception as e:
        print(f"Visualization skipped: {e}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print(" VisibilityPredictionProject - Quick Start Examples")
    print("="*70)
    
    try:
        example_1_basic_model()
    except Exception as e:
        print(f"Example 1 failed: {e}")
    
    try:
        example_2_data_loading()
    except Exception as e:
        print(f"Example 2 failed: {e}")
    
    try:
        example_3_training()
    except Exception as e:
        print(f"Example 3 failed: {e}")
    
    try:
        example_4_visualization()
    except Exception as e:
        print(f"Example 4 failed: {e}")
    
    print("\n" + "="*70)
    print(" All examples completed!")
    print("="*70)
    print("\nNext steps:")
    print("1. See README.md for full documentation")
    print("2. Check docs/NOAA_pipeline_guide.md for data preparation")
    print("3. Run scripts/train_multi_station.py for full training")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
