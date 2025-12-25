"""
Training Script for Multi-Station Visibility Prediction
=======================================================

This script trains GNN models for multi-station visibility prediction using NOAA data.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from VisibilityPredictionProject.models import GraphVisibilityNet, STGNNVisibility
from VisibilityPredictionProject.data_loader import create_noaa_dataloader
from VisibilityPredictionProject.utils import (
    train_epoch, validate_epoch, save_checkpoint,
    load_checkpoint, plot_training_curves, print_metrics
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train GNN models for multi-station visibility prediction'
    )
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='./data/noaa_visibility',
                       help='Path to NOAA visibility data')
    parser.add_argument('--seq_len', type=int, default=24,
                       help='Input sequence length (hours)')
    parser.add_argument('--pred_len', type=int, default=1,
                       help='Prediction horizon (hours)')
    parser.add_argument('--num_stations', type=int, default=10,
                       help='Number of weather stations')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='GraphVisibilityNet',
                       choices=['GraphVisibilityNet', 'STGNNVisibility'],
                       help='Model architecture')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/visibility',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    with open(save_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Create data loaders
    print("\nLoading data...")
    train_loader = create_noaa_dataloader(
        args.data_path,
        batch_size=args.batch_size,
        split='train',
        seq_len=args.seq_len,
        pred_len=args.pred_len
    )
    
    val_loader = create_noaa_dataloader(
        args.data_path,
        batch_size=args.batch_size,
        split='val',
        seq_len=args.seq_len,
        pred_len=args.pred_len
    )
    
    # Create model
    print(f"\nCreating {args.model} model...")
    if args.model == 'GraphVisibilityNet':
        model = GraphVisibilityNet(
            num_stations=args.num_stations,
            input_dim=7,  # Default number of features
            hidden_dim=args.hidden_dim,
            output_dim=args.pred_len,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    else:  # STGNNVisibility
        model = STGNNVisibility(
            num_stations=args.num_stations,
            input_dim=7,
            hidden_dim=args.hidden_dim,
            output_dim=args.pred_len,
            dropout=args.dropout
        )
    
    model = model.to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer and loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        model, optimizer, start_epoch, metrics = load_checkpoint(
            model, args.resume, optimizer, device
        )
        best_val_loss = metrics.get('loss', float('inf'))
    
    # Training loop
    print("\nStarting training...")
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    patience_counter = 0
    
    for epoch in range(start_epoch + 1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        train_losses.append(train_metrics['loss'])
        train_maes.append(train_metrics['mae'])
        
        # Validate
        val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch
        )
        val_losses.append(val_metrics['loss'])
        val_maes.append(val_metrics['mae'])
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Print epoch results
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"Train - Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.4f}, RMSE: {train_metrics['rmse']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.4f}, RMSE: {val_metrics['rmse']:.4f}, R2: {val_metrics['r2']:.4f}")
        
        # Save checkpoint
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
        else:
            patience_counter += 1
        
        save_checkpoint(
            model, optimizer, epoch, val_metrics,
            save_dir / f'checkpoint_epoch_{epoch}.pth',
            is_best=is_best
        )
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    # Plot training curves
    print("\nPlotting training curves...")
    plot_training_curves(
        train_losses,
        val_losses,
        train_metrics={'mae': train_maes},
        val_metrics={'mae': val_maes},
        save_path=save_dir / 'training_curves.png'
    )
    
    # Load best model and evaluate
    print("\nLoading best model for final evaluation...")
    model, _, _, _ = load_checkpoint(
        model, save_dir / 'best_model.pth', device=device
    )
    
    final_metrics = validate_epoch(
        model, val_loader, criterion, device, epoch='Final'
    )
    print_metrics(final_metrics, prefix='Final ')
    
    print(f"\nTraining complete! Best model saved to {save_dir / 'best_model.pth'}")


if __name__ == '__main__':
    main()
