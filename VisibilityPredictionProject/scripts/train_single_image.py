"""
Training Script for Single-Image Visibility Detection with Transfer Learning
============================================================================

This script trains vision models for image-based visibility detection,
optionally using transfer learning from multi-station GNN models.
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

from VisibilityPredictionProject.models import (
    VisibilityVisionNet, GraphVisibilityNet,
    STGNNVisibility, TransferLearningAdapter
)
from VisibilityPredictionProject.data_loader import create_image_dataloader
from VisibilityPredictionProject.utils import (
    train_epoch, validate_epoch, save_checkpoint,
    load_checkpoint, plot_training_curves, print_metrics
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train vision models for single-image visibility detection'
    )
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='./data/visibility_images',
                       help='Path to visibility image data')
    parser.add_argument('--labels_file', type=str, default=None,
                       help='Path to CSV file with image labels')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Image size (height and width)')
    
    # Model arguments
    parser.add_argument('--num_classes', type=int, default=1,
                       help='Number of output classes (1 for regression)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained weights')
    
    # Transfer learning arguments
    parser.add_argument('--use_transfer', action='store_true',
                       help='Use transfer learning from GNN model')
    parser.add_argument('--gnn_checkpoint', type=str, default=None,
                       help='Path to pretrained GNN model checkpoint')
    parser.add_argument('--gnn_model_type', type=str, default='GraphVisibilityNet',
                       choices=['GraphVisibilityNet', 'STGNNVisibility'],
                       help='Type of GNN model to load')
    parser.add_argument('--freeze_gnn', action='store_true',
                       help='Freeze GNN weights during transfer learning')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/vision',
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
    train_loader = create_image_dataloader(
        args.data_path,
        batch_size=args.batch_size,
        split='train',
        labels_file=args.labels_file,
        image_size=(args.image_size, args.image_size)
    )
    
    val_loader = create_image_dataloader(
        args.data_path,
        batch_size=args.batch_size,
        split='val',
        labels_file=args.labels_file,
        image_size=(args.image_size, args.image_size)
    )
    
    # Create model
    print("\nCreating vision model...")
    vision_model = VisibilityVisionNet(
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        dropout=args.dropout
    )
    
    # Setup transfer learning if requested
    if args.use_transfer and args.gnn_checkpoint:
        print("\nSetting up transfer learning...")
        
        # Load GNN model
        if args.gnn_model_type == 'GraphVisibilityNet':
            gnn_model = GraphVisibilityNet()
        else:
            gnn_model = STGNNVisibility()
        
        # Load GNN checkpoint
        checkpoint = torch.load(args.gnn_checkpoint, map_location=device)
        gnn_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded GNN checkpoint from {args.gnn_checkpoint}")
        
        # Create transfer learning adapter
        model = TransferLearningAdapter(
            gnn_model=gnn_model,
            vision_model=vision_model,
            freeze_gnn=args.freeze_gnn
        )
    else:
        model = vision_model
    
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
