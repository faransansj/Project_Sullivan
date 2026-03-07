#!/usr/bin/env python3
"""
Train U-Net for Vocal Tract Segmentation

Trains a binary U-Net (airway vs background) on pseudo-labeled MRI frames.
Saves the best checkpoint to models/unet_scratch/unet_best.pth, which is
the path expected by segment_mp4_dataset.py.

Usage:
    python scripts/train_unet.py
    python scripts/train_unet.py --epochs 50 --batch-size 16
    python scripts/train_unet.py --device cuda --epochs 100
"""

import argparse
import sys
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.segmentation.unet_lightning import UNetLightning
from src.segmentation.pseudo_label_dataset import (
    PseudoLabelDataset,
    get_train_transform,
    get_val_transform,
)
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Train U-Net for vocal tract segmentation")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/pseudo_labels",
        help="Directory containing pseudo-labels (output of generate_pseudo_labels_from_hdf5.py)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/unet_scratch",
        help="Directory to save checkpoints and logs",
    )
    parser.add_argument("--epochs",     type=int,   default=50,    help="Max training epochs")
    parser.add_argument("--batch-size", type=int,   default=16,    help="Batch size")
    parser.add_argument("--lr",         type=float, default=3e-4,  help="Learning rate")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device (auto = use GPU if available)",
    )
    parser.add_argument("--workers",    type=int,   default=4,     help="DataLoader workers")
    parser.add_argument("--patience",   type=int,   default=10,    help="Early stopping patience")

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_dir     = project_root / args.data_dir
    output_dir   = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("TrainUNet", log_file=str(project_root / "logs" / "train_unet.log"))

    # Resolve device
    if args.device == "auto":
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    else:
        accelerator = args.device

    logger.info("=" * 60)
    logger.info("U-NET TRAINING")
    logger.info("=" * 60)
    logger.info(f"Data dir    : {data_dir}")
    logger.info(f"Output dir  : {output_dir}")
    logger.info(f"Epochs      : {args.epochs}")
    logger.info(f"Batch size  : {args.batch_size}")
    logger.info(f"LR          : {args.lr}")
    logger.info(f"Device      : {accelerator}")
    logger.info("=" * 60)

    # Datasets
    train_ds = PseudoLabelDataset(str(data_dir), split="train", transform=get_train_transform())
    val_ds   = PseudoLabelDataset(str(data_dir), split="val",   transform=get_val_transform())

    logger.info(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(accelerator == "gpu"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(accelerator == "gpu"),
    )

    # Model
    model = UNetLightning(
        n_channels=1,
        n_classes=1,
        bilinear=True,
        lr=args.lr,
        bce_weight=0.5,
        dice_weight=0.5,
    )
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="unet_best",
        monitor="val_dice",
        mode="max",
        save_top_k=1,
        verbose=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_dice",
        mode="max",
        patience=args.patience,
        verbose=True,
    )

    tb_logger = TensorBoardLogger(
        save_dir=str(output_dir / "logs"),
        name="unet_training",
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=tb_logger,
        log_every_n_steps=5,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_loader, val_loader)

    # Save best weights as plain state_dict (unet_best.pth)
    # segment_mp4_dataset.py can load both Lightning .ckpt and plain .pth
    best_ckpt = output_dir / "unet_best.ckpt"
    pth_path  = output_dir / "unet_best.pth"

    if best_ckpt.exists():
        ckpt = torch.load(str(best_ckpt), map_location="cpu")
        torch.save(ckpt["state_dict"], str(pth_path))
        logger.info(f"Saved plain state_dict → {pth_path}")

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best checkpoint : {best_ckpt}")
    logger.info(f"Plain weights   : {pth_path}")
    logger.info(f"Best val_dice   : {checkpoint_cb.best_model_score:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
