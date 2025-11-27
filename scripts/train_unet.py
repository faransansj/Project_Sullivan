#!/usr/bin/env python3
"""
Train U-Net from scratch on pseudo-labeled USC-TIMIT data.

This script trains a U-Net segmentation model using pseudo-labels generated
from traditional computer vision methods.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.segmentation.dataset import (
    SegmentationDatasetSplit,
    create_train_val_test_splits,
)
from src.segmentation.unet import UNet
from src.utils.logger import get_logger, setup_logger

# Setup logging
setup_logger(level="INFO", console=True)
logger = get_logger(__name__)


class SegmentationMetrics:
    """Compute segmentation metrics."""

    @staticmethod
    def dice_score(
        pred: torch.Tensor, target: torch.Tensor, num_classes: int, epsilon: float = 1e-7
    ) -> torch.Tensor:
        """
        Compute Dice score for each class.

        Args:
            pred: (B, H, W) predicted class indices
            target: (B, H, W) ground truth class indices
            num_classes: Number of classes
            epsilon: Small value to avoid division by zero

        Returns:
            (num_classes,) tensor of Dice scores per class
        """
        dice_scores = []

        for c in range(num_classes):
            pred_c = (pred == c).float()
            target_c = (target == c).float()

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            dice = (2.0 * intersection + epsilon) / (union + epsilon)
            dice_scores.append(dice)

        return torch.stack(dice_scores)

    @staticmethod
    def iou_score(
        pred: torch.Tensor, target: torch.Tensor, num_classes: int, epsilon: float = 1e-7
    ) -> torch.Tensor:
        """
        Compute IoU (Jaccard) score for each class.

        Args:
            pred: (B, H, W) predicted class indices
            target: (B, H, W) ground truth class indices
            num_classes: Number of classes
            epsilon: Small value to avoid division by zero

        Returns:
            (num_classes,) tensor of IoU scores per class
        """
        iou_scores = []

        for c in range(num_classes):
            pred_c = (pred == c).float()
            target_c = (target == c).float()

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum() - intersection

            iou = (intersection + epsilon) / (union + epsilon)
            iou_scores.append(iou)

        return torch.stack(iou_scores)


class UNetLightning(L.LightningModule):
    """PyTorch Lightning module for U-Net training."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-5,
    ):
        """
        Initialize U-Net Lightning module.

        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            learning_rate: Learning rate
            weight_decay: Weight decay for AdamW
        """
        super().__init__()
        self.save_hyperparameters()

        # Model (UNet expects n_classes parameter)
        self.model = UNet(n_classes=num_classes)

        # Loss function (CrossEntropyLoss with class weights)
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.metrics = SegmentationMetrics()

        # Store outputs for epoch-level metrics
        self.validation_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        """Training step."""
        images, masks, metadata = batch

        # Forward pass
        logits = self(images)

        # Compute loss
        loss = self.criterion(logits, masks)

        # Compute metrics
        pred_masks = torch.argmax(logits, dim=1)
        dice_scores = self.metrics.dice_score(
            pred_masks, masks, num_classes=self.hparams.num_classes
        )

        # Log metrics
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/dice_mean", dice_scores.mean(), prog_bar=True)

        # Log per-class Dice scores
        class_names = ["background", "tongue", "jaw", "lips"]
        for i, name in enumerate(class_names[: self.hparams.num_classes]):
            self.log(f"train/dice_{name}", dice_scores[i])

        return loss

    def validation_step(self, batch: Tuple, batch_idx: int) -> Dict:
        """Validation step."""
        images, masks, metadata = batch

        # Forward pass
        logits = self(images)

        # Compute loss
        loss = self.criterion(logits, masks)

        # Compute metrics
        pred_masks = torch.argmax(logits, dim=1)
        dice_scores = self.metrics.dice_score(
            pred_masks, masks, num_classes=self.hparams.num_classes
        )
        iou_scores = self.metrics.iou_score(
            pred_masks, masks, num_classes=self.hparams.num_classes
        )

        # Store outputs
        output = {
            "loss": loss,
            "dice_scores": dice_scores,
            "iou_scores": iou_scores,
        }
        self.validation_step_outputs.append(output)

        return output

    def on_validation_epoch_end(self):
        """Compute and log validation metrics at epoch end."""
        outputs = self.validation_step_outputs

        # Average loss
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        # Average Dice and IoU scores
        avg_dice = torch.stack([x["dice_scores"] for x in outputs]).mean(dim=0)
        avg_iou = torch.stack([x["iou_scores"] for x in outputs]).mean(dim=0)

        # Log metrics
        self.log("val/loss", avg_loss, prog_bar=True)
        self.log("val/dice_mean", avg_dice.mean(), prog_bar=True)
        self.log("val/iou_mean", avg_iou.mean(), prog_bar=True)

        # Log per-class metrics
        class_names = ["background", "tongue", "jaw", "lips"]
        for i, name in enumerate(class_names[: self.hparams.num_classes]):
            self.log(f"val/dice_{name}", avg_dice[i])
            self.log(f"val/iou_{name}", avg_iou[i])

        # Clear stored outputs
        self.validation_step_outputs.clear()

    def test_step(self, batch: Tuple, batch_idx: int) -> Dict:
        """Test step (same as validation step)."""
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        # Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


def main():
    """Main training function."""
    logger.info("=" * 70)
    logger.info("U-NET TRAINING")
    logger.info("=" * 70)

    # Configuration
    pseudo_labels_dir = Path("data/processed/pseudo_labels")
    output_dir = Path("models/unet_scratch")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Hyperparameters
    batch_size = 8
    num_epochs = 100
    learning_rate = 3e-4
    num_workers = 4
    num_classes = 4  # background, tongue, jaw, lips

    # Data augmentation
    augment_train = True

    # Create train/val/test splits
    logger.info("\nCreating train/val/test splits...")
    train_paths, val_paths, test_paths = create_train_val_test_splits(
        pseudo_labels_dir, val_ratio=0.15, test_ratio=0.15, random_seed=42
    )

    # Create datasets
    logger.info("\nCreating datasets...")
    train_dataset = SegmentationDatasetSplit(train_paths, augment=augment_train)
    val_dataset = SegmentationDatasetSplit(val_paths, augment=False)
    test_dataset = SegmentationDatasetSplit(test_paths, augment=False)

    logger.info(f"  Train: {len(train_dataset)} samples")
    logger.info(f"  Val: {len(val_dataset)} samples")
    logger.info(f"  Test: {len(test_dataset)} samples")

    # Create dataloaders
    logger.info("\nCreating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True,
    )

    # Initialize model
    logger.info("\nInitializing U-Net model...")
    model = UNetLightning(
        in_channels=1, num_classes=num_classes, learning_rate=learning_rate
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")

    # Setup callbacks
    logger.info("\nSetting up callbacks...")
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="unet-{epoch:03d}-{val/dice_mean:.4f}",
            monitor="val/dice_mean",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/dice_mean",
            patience=20,
            mode="max",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Setup logger
    csv_logger = CSVLogger(
        save_dir=output_dir / "logs", name="unet_training"
    )

    # Initialize trainer
    logger.info("\nInitializing trainer...")
    trainer = L.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        logger=csv_logger,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        gradient_clip_val=1.0,
        deterministic=False,
    )

    # Train model
    logger.info("\n" + "=" * 70)
    logger.info("STARTING TRAINING")
    logger.info("=" * 70 + "\n")

    trainer.fit(model, train_loader, val_loader)

    # Test model
    logger.info("\n" + "=" * 70)
    logger.info("TESTING BEST MODEL")
    logger.info("=" * 70 + "\n")

    trainer.test(model, test_loader, ckpt_path="best")

    # Save final model
    logger.info("\nSaving final model...")
    final_model_path = output_dir / "unet_final.pth"
    torch.save(model.model.state_dict(), final_model_path)
    logger.info(f"  Saved to: {final_model_path}")

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nModel checkpoints: {output_dir / 'checkpoints'}")
    logger.info(f"Training logs: {output_dir / 'logs'}")
    logger.info(f"Final model: {final_model_path}")


if __name__ == "__main__":
    main()
