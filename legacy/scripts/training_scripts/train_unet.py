#!/usr/bin/env python3
"""
Train U-Net for Vocal Tract Segmentation
"""

import sys
import os
from pathlib import Path
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from src.segmentation.unet_lightning import UNetLightning
from src.segmentation.pseudo_label_dataset import (
    PseudoLabelDataset,
    get_train_transform,
    get_val_transform
)


def train_unet(args):
    """Train U-Net model"""
    
    print("=" * 70)
    print("🚀 U-Net Training")
    print("=" * 70)
    
    # Set random seed for reproducibility
    pl.seed_everything(42, workers=True)
    
    # 1. Create datasets
    print("\n📦 Loading datasets...")
    train_dataset = PseudoLabelDataset(
        args.data_dir,
        split='train',
        transform=get_train_transform()
    )
    val_dataset = PseudoLabelDataset(
        args.data_dir,
        split='val',
        transform=get_val_transform()
    )
    test_dataset = PseudoLabelDataset(
        args.data_dir,
        split='test',
        transform=get_val_transform()
    )
    
    # 2. Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"✅ Train batches: {len(train_loader)}")
    print(f"✅ Val batches: {len(val_loader)}")
    print(f"✅ Test batches: {len(test_loader)}")
    
    # 3. Create model
    print(f"\n🏗️ Creating U-Net model...")
    model = UNetLightning(
        n_channels=1,
        n_classes=1,
        bilinear=True,
        lr=args.lr,
        bce_weight=0.5,
        dice_weight=0.5
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Parameters: {num_params:,}")
    
    # 4. Setup callbacks
    checkpoint_dir = Path(args.output_dir) / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='unet-{epoch:02d}-{val_dice:.4f}',
        monitor='val_dice',
        mode='max',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_dice',
        patience=15,
        mode='max',
        verbose=True
    )
    
    # 5. Setup logger
    log_dir = Path(args.output_dir) / 'logs'
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name='unet_training'
    )
    
    # 6. Create trainer
    print(f"\n⚙️ Training configuration:")
    print(f"   - Max epochs: {args.max_epochs}")
    print(f"   - Batch size: {args.batch_size}")
    print(f"   - Learning rate: {args.lr}")
    print(f"   - Device: {'GPU' if torch.cuda.is_available() and not args.cpu else 'CPU'}")
    
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='cpu' if args.cpu else 'auto',
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        deterministic=True,
        enable_progress_bar=True
    )
    
    # 7. Train
    print(f"\n{'='*70}")
    print("🎯 Starting training...")
    if args.resume:
        print(f"📂 Resuming from checkpoint: {args.resume}")
    print(f"{'='*70}\n")

    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume if args.resume else None)
    
    # 8. Test
    print(f"\n{'='*70}")
    print("📊 Testing best model...")
    print(f"{'='*70}\n")
    
    test_results = trainer.test(model, test_loader, ckpt_path='best')
    
    # 9. Save final model
    final_model_path = Path(args.output_dir) / 'unet_best.pth'
    torch.save(model.state_dict(), final_model_path)
    
    print(f"\n{'='*70}")
    print("📋 Training Summary")
    print(f"{'='*70}")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best val Dice: {checkpoint_callback.best_model_score:.4f}")
    print(f"Test results: {test_results}")
    print(f"Final model saved: {final_model_path}")
    print(f"{'='*70}\n")
    
    return model, test_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train U-Net for vocal tract segmentation')
    
    # Data
    parser.add_argument('--data-dir', type=str, default='data/pseudo_labels',
                       help='Path to pseudo-labels directory')
    parser.add_argument('--output-dir', type=str, default='models/unet_scratch',
                       help='Output directory for models and logs')
    
    # Training
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--max-epochs', type=int, default=100,
                       help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    
    # System
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of dataloader workers')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU training')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    args = parser.parse_args()
    
    # Train
    model, results = train_unet(args)
    
    print("✅ Training complete!")
