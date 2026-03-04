"""
Phase 2 Training Script: Audio-to-Articulatory Parameter Model

Trains a Bi-LSTM model to predict articulatory parameters from audio features.

Usage:
    python scripts/train_phase2.py --data_dir data/processed/phase2 --output_dir models/phase2
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import time
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.modeling.models.bilstm import BiLSTMArticulationPredictor, print_model_summary
from src.modeling.dataset import create_dataloaders, load_dataset_statistics
from src.modeling.losses import ArticulatoryLoss
from src.modeling.evaluate import evaluate_model, print_evaluation_results, save_evaluation_results
from src.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Phase 2 model")

    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing processed Phase 2 data')
    parser.add_argument('--stats_file', type=str, default=None,
                       help='Path to normalization statistics file')
    parser.add_argument('--audio_feature_type', type=str, default='mel',
                       choices=['mel', 'mfcc', 'both'],
                       help='Type of audio features')

    # Model arguments
    parser.add_argument('--input_dim', type=int, default=80,
                       help='Input feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='LSTM hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout probability')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay for optimizer')
    parser.add_argument('--smoothness_weight', type=float, default=0.1,
                       help='Weight for temporal smoothness loss')

    # Optimization arguments
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'adamw'],
                       help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='plateau',
                       choices=['plateau', 'cosine', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')

    # System arguments
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Output arguments
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for models and logs')
    parser.add_argument('--save_frequency', type=int, default=10,
                       help='Save checkpoint every N epochs')

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    """Get torch device"""
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        return torch.device(device_arg)


def train_epoch(model: nn.Module,
               dataloader: DataLoader,
               criterion: nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               epoch: int) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")

    for batch in progress_bar:
        audio = batch['audio'].to(device)
        params = batch['params'].to(device)
        lengths = batch['lengths'].to(device)
        mask = batch['mask'].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(audio, lengths)

        # Compute loss
        loss = criterion(predictions, params, mask)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()

        # Track loss
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / num_batches
    return avg_loss


def validate_epoch(model: nn.Module,
                  dataloader: DataLoader,
                  criterion: nn.Module,
                  device: torch.device,
                  epoch: int) -> float:
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]")

    with torch.no_grad():
        for batch in progress_bar:
            audio = batch['audio'].to(device)
            params = batch['params'].to(device)
            lengths = batch['lengths'].to(device)
            mask = batch['mask'].to(device)

            # Forward pass
            predictions = model(audio, lengths)

            # Compute loss
            loss = criterion(predictions, params, mask)

            # Track loss
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    """Main training loop"""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = setup_logger(output_dir / 'training.log')
    logger.info("=" * 80)
    logger.info("Phase 2 Training: Audio-to-Articulatory Parameter Prediction")
    logger.info("=" * 80)

    # Log arguments
    logger.info("\nArguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    # Set random seed
    set_seed(args.seed)
    logger.info(f"\nSet random seed: {args.seed}")

    # Get device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    # Create dataloaders
    logger.info(f"\nLoading data from {args.data_dir}...")
    dataloaders = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        audio_feature_type=args.audio_feature_type
    )

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']

    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")

    # Load normalization statistics
    stats = None
    if args.stats_file and Path(args.stats_file).exists():
        logger.info(f"\nLoading normalization statistics from {args.stats_file}")
        stats = load_dataset_statistics(args.stats_file)

    # Create model
    logger.info("\nCreating model...")
    model = BiLSTMArticulationPredictor(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        output_dim=10,
        dropout=args.dropout
    )
    model = model.to(device)

    print_model_summary(model)

    # Create loss function
    criterion = ArticulatoryLoss(smoothness_weight=args.smoothness_weight)
    logger.info(f"\nLoss function: ArticulatoryLoss (smoothness_weight={args.smoothness_weight})")

    # Create optimizer
    if args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=args.learning_rate,
                        weight_decay=args.weight_decay)
    else:
        optimizer = AdamW(model.parameters(), lr=args.learning_rate,
                         weight_decay=args.weight_decay)

    logger.info(f"Optimizer: {args.optimizer.upper()}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Weight decay: {args.weight_decay}")

    # Create scheduler
    scheduler = None
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5,
                                     factor=0.5, verbose=True)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    if scheduler:
        logger.info(f"Scheduler: {args.scheduler}")

    # Training loop
    logger.info("\n" + "=" * 80)
    logger.info("Starting Training")
    logger.info("=" * 80)

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    training_history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }

    start_time = time.time()

    for epoch in range(args.num_epochs):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device, epoch)

        # Update scheduler
        if scheduler:
            if args.scheduler == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Save history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['learning_rate'].append(current_lr)

        # Log epoch results
        epoch_time = time.time() - epoch_start
        logger.info(f"\nEpoch {epoch+1}/{args.num_epochs} ({epoch_time:.1f}s)")
        logger.info(f"  Train Loss: {train_loss:.6f}")
        logger.info(f"  Val Loss:   {val_loss:.6f}")
        logger.info(f"  LR:         {current_lr:.6e}")

        # Save checkpoint
        if (epoch + 1) % args.save_frequency == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1:03d}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            logger.info(f"  Saved checkpoint: {checkpoint_path}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            best_model_path = output_dir / 'best_model.pth'
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"  ✓ New best model! Val Loss: {val_loss:.6f}")

        else:
            patience_counter += 1
            logger.info(f"  No improvement for {patience_counter} epochs")

        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    # Training complete
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 80)
    logger.info("Training Complete")
    logger.info("=" * 80)
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Best epoch: {best_epoch+1}")
    logger.info(f"Best val loss: {best_val_loss:.6f}")

    # Save training history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    logger.info(f"\nSaved training history to {history_path}")

    # Final evaluation on test set
    logger.info("\n" + "=" * 80)
    logger.info("Final Evaluation on Test Set")
    logger.info("=" * 80)

    # Load best model
    model.load_state_dict(torch.load(output_dir / 'best_model.pth'))

    # Denormalization function
    denormalize_fn = None
    if stats is not None:
        def denormalize_fn(x):
            return x * stats['param_std'] + stats['param_mean']

    # Evaluate
    test_results = evaluate_model(model, test_loader, device, denormalize_fn)
    print_evaluation_results(test_results, split_name="Test")

    # Save evaluation results
    results_path = output_dir / 'test_results.json'
    save_evaluation_results(test_results, results_path)

    logger.info(f"\n✓ Training complete! Best model saved to {output_dir / 'best_model.pth'}")


if __name__ == '__main__':
    main()
