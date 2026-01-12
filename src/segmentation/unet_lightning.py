"""
PyTorch Lightning Module for U-Net Training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any
import numpy as np

from .unet_simple import UNet


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation"""
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) - logits
            target: (B, 1, H, W) - binary mask (0 or 1)
        """
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined BCE + Dice loss"""
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def dice_coefficient(pred, target, threshold=0.5, smooth=1.0):
    """Calculate Dice coefficient for evaluation"""
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()


class UNetLightning(pl.LightningModule):
    """
    PyTorch Lightning module for U-Net training.
    
    Args:
        n_channels: Number of input channels (1 for grayscale)
        n_classes: Number of output classes (1 for binary)
        bilinear: Use bilinear upsampling
        lr: Learning rate
        bce_weight: Weight for BCE loss
        dice_weight: Weight for Dice loss
    """
    
    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 1,
        bilinear: bool = True,
        lr: float = 1e-4,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model
        self.model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
        
        # Loss
        self.criterion = CombinedLoss(bce_weight=bce_weight, dice_weight=dice_weight)
        
        # Metrics storage
        self.validation_step_outputs = []
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, masks = batch
        
        # Forward
        logits = self(images)
        loss = self.criterion(logits, masks)
        
        # Calculate Dice for monitoring
        dice = dice_coefficient(logits, masks)
        
        # Log
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_dice', dice, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        
        # Forward
        logits = self(images)
        loss = self.criterion(logits, masks)
        
        # Calculate Dice
        dice = dice_coefficient(logits, masks)
        
        # Store for epoch-level aggregation
        self.validation_step_outputs.append({
            'val_loss': loss,
            'val_dice': dice
        })
        
        # Log
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_dice', dice, on_step=False, on_epoch=True, prog_bar=True)
        
        return {'val_loss': loss, 'val_dice': dice}
    
    def on_validation_epoch_end(self):
        # Calculate average metrics
        if len(self.validation_step_outputs) > 0:
            avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
            avg_dice = np.mean([x['val_dice'] for x in self.validation_step_outputs])
            
            self.log('val_loss_epoch', avg_loss)
            self.log('val_dice_epoch', avg_dice)
            
            # Clear for next epoch
            self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        images, masks = batch
        
        # Forward
        logits = self(images)
        loss = self.criterion(logits, masks)
        
        # Calculate Dice
        dice = dice_coefficient(logits, masks)
        
        # Log
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_dice', dice, on_step=False, on_epoch=True)
        
        return {'test_loss': loss, 'test_dice': dice}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        
        # Learning rate scheduler (reduce on plateau)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',  # Maximize Dice
            factor=0.5,
            patience=5
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_dice',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def predict_step(self, batch, batch_idx):
        """For inference"""
        if isinstance(batch, tuple):
            images, _ = batch
        else:
            images = batch
        
        logits = self(images)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
        return preds


if __name__ == "__main__":
    # Test
    model = UNetLightning(n_channels=1, n_classes=1, lr=1e-4)
    
    # Dummy batch
    images = torch.randn(4, 1, 59, 59)
    masks = torch.randint(0, 2, (4, 1, 59, 59)).float()
    
    batch = (images, masks)
    
    # Test training step
    loss = model.training_step(batch, 0)
    print(f"Training loss: {loss:.4f}")
    
    # Test validation step
    val_output = model.validation_step(batch, 0)
    print(f"Validation Dice: {val_output['val_dice']:.4f}")
    
    print("✅ Lightning module test passed!")
