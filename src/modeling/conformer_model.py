"""
Conformer Model for High-Performance Acoustic-to-Articulatory Inversion

Implements a Conformer architecture which combines Transformer's global attention
with CNN's local feature extraction, optimized for speech tasks.

Based on "Conformer: Convolution-augmented Transformer for Speech Recognition"
(Gulati et al., 2020)

Phase 4 — Accuracy Improvement Pipeline
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchaudio.models import Conformer
from typing import Dict, Optional, Tuple, List
import numpy as np

from .model_utils import (
    create_padding_mask,
    count_parameters,
    format_parameter_count
)


class ConformerInversionModel(pl.LightningModule):
    """
    Conformer-based model for predicting articulatory parameters.

    Architecture:
    - Linear projection to d_model
    - N x Conformer Blocks (Feed Forward -> MHSA -> Convolution -> Feed Forward)
    - Linear output projection

    Parameters
    ----------
    input_dim : int
        Input feature dimension (80 for Mel, 1024 for HuBERT-Large)
    d_model : int
        Conformer hidden dimension
    num_layers : int
        Number of Conformer blocks
    num_heads : int
        Number of attention heads
    ffn_dim : int
        Feed-forward network dimension
    depthwise_conv_kernel_size : int
        Kernel size for depthwise convolution in Conformer blocks
    output_dim : int
        Output parameter dimension (14 geometric + 10 PCA = 24)
    dropout : float
        Dropout probability
    learning_rate : float
        Learning rate for AdamW optimizer
    weight_decay : float
        Weight decay for regularization
    mse_weight : float
        Weight for MSE (position) loss
    pcc_weight : float
        Weight for PCC (correlation) loss
    velocity_weight : float
        Weight for velocity (first-order temporal) loss
    acceleration_weight : float
        Weight for acceleration (second-order temporal) loss
    """

    def __init__(
        self,
        input_dim: int = 1024,
        d_model: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        depthwise_conv_kernel_size: int = 31,
        output_dim: int = 24,
        dropout: float = 0.1,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.01,
        mse_weight: float = 1.0,
        pcc_weight: float = 2.0,
        velocity_weight: float = 1.0,
        acceleration_weight: float = 0.5
    ):
        super().__init__()
        self.save_hyperparameters()

        self.output_dim = output_dim

        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # Conformer Encoder (torchaudio >= 0.9.0)
        self.conformer = Conformer(
            input_dim=d_model,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim),
        )

        # Loss function
        self.criterion = nn.MSELoss(reduction='none')

        # Print model info
        param_count = sum(p.numel() for p in self.parameters())
        print(f"\n🚀 Conformer Inversion Model Initialized:")
        print(f"  Parameters: {format_parameter_count(param_count)} ({param_count:,})")
        print(f"  Layers: {num_layers}, Heads: {num_heads}, d_model: {d_model}")
        print(f"  Input: {input_dim}, Output: {output_dim}")
        print(f"  Conv kernel: {depthwise_conv_kernel_size}\n")

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape (batch, seq_len, input_dim)
        lengths : torch.Tensor, optional
            Actual sequence lengths, shape (batch,)

        Returns
        -------
        output : torch.Tensor
            Predicted parameters, shape (batch, seq_len, output_dim)
        """
        # Project input: (B, T, D_in) -> (B, T, D_model)
        x = self.input_projection(x)

        # Conformer expects (B, T, D) and lengths
        if lengths is not None:
            x, _ = self.conformer(x, lengths)
        else:
            seq_len = x.shape[1]
            dummy_lengths = torch.full(
                (x.shape[0],), seq_len, dtype=torch.long, device=x.device
            )
            x, _ = self.conformer(x, dummy_lengths)

        # Output: (B, T, D_model) -> (B, T, D_out)
        output = self.output_projection(x)

        return output

    # =========================================================================
    # Training / Validation / Test Steps — Full Hybrid Loss Parity
    # =========================================================================

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        audio, params, lengths, _ = batch
        pred = self(audio, lengths)

        # Create mask for valid frames
        mask = self._create_mask(lengths, pred.shape[1]).to(pred.device)
        mask = mask.unsqueeze(-1)  # (batch, seq_len, 1)

        # Position loss (MSE)
        position_loss = (self.criterion(pred, params) * mask).sum() / mask.sum()

        # Temporal losses
        temporal = self._compute_temporal_loss(pred, params, mask)
        velocity_loss = temporal['velocity_loss']
        acceleration_loss = temporal['acceleration_loss']

        # PCC loss
        pcc_loss = self._compute_pcc_loss(pred, params, mask)

        # Split logging for 24-dim output (Geometric vs PCA)
        if self.output_dim == 24:
            geo_error = self.criterion(pred[:, :, :14], params[:, :, :14])
            self.log('train_mse_geo', (geo_error * mask).sum() / mask.sum(),
                     on_step=False, on_epoch=True)
            pca_error = self.criterion(pred[:, :, 14:], params[:, :, 14:])
            self.log('train_mse_pca', (pca_error * mask).sum() / mask.sum(),
                     on_step=False, on_epoch=True)

        # Combined hybrid loss
        loss = (
            self.hparams.mse_weight * position_loss
            + self.hparams.pcc_weight * pcc_loss
            + self.hparams.velocity_weight * velocity_loss
            + self.hparams.acceleration_weight * acceleration_loss
        )

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mse', position_loss, on_step=False, on_epoch=True)
        self.log('train_pcc_loss', pcc_loss, on_step=False, on_epoch=True)
        self.log('train_velocity_loss', velocity_loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: Tuple, batch_idx: int) -> Dict:
        audio, params, lengths, _ = batch
        pred = self(audio, lengths)

        mask = self._create_mask(lengths, pred.shape[1]).to(pred.device)
        mask = mask.unsqueeze(-1)

        position_loss = (self.criterion(pred, params) * mask).sum() / mask.sum()
        temporal = self._compute_temporal_loss(pred, params, mask)
        pcc_loss = self._compute_pcc_loss(pred, params, mask)

        if self.output_dim == 24:
            geo_error = self.criterion(pred[:, :, :14], params[:, :, :14])
            self.log('val_mse_geo', (geo_error * mask).sum() / mask.sum(),
                     on_step=False, on_epoch=True)
            pca_error = self.criterion(pred[:, :, 14:], params[:, :, 14:])
            self.log('val_mse_pca', (pca_error * mask).sum() / mask.sum(),
                     on_step=False, on_epoch=True)

        loss = (
            self.hparams.mse_weight * position_loss
            + self.hparams.pcc_weight * pcc_loss
            + self.hparams.velocity_weight * temporal['velocity_loss']
            + self.hparams.acceleration_weight * temporal['acceleration_loss']
        )

        metrics = self._compute_metrics(pred, params, mask)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mse', position_loss, on_step=False, on_epoch=True)
        self.log('val_pcc_loss', pcc_loss, on_step=False, on_epoch=True)
        self.log('val_rmse', metrics['rmse'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_pearson', metrics['pearson'], on_step=False, on_epoch=True, prog_bar=True)

        return {'val_loss': loss, **metrics}

    def test_step(self, batch: Tuple, batch_idx: int) -> Dict:
        audio, params, lengths, _ = batch
        pred = self(audio, lengths)

        mask = self._create_mask(lengths, pred.shape[1]).to(pred.device)
        mask = mask.unsqueeze(-1)

        position_loss = (self.criterion(pred, params) * mask).sum() / mask.sum()
        temporal = self._compute_temporal_loss(pred, params, mask)
        metrics = self._compute_metrics(pred, params, mask)

        loss = position_loss + (
            self.hparams.velocity_weight * temporal['velocity_loss']
            + self.hparams.acceleration_weight * temporal['acceleration_loss']
        )

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_rmse', metrics['rmse'], on_step=False, on_epoch=True)
        self.log('test_mae', metrics['mae'], on_step=False, on_epoch=True)
        self.log('test_pearson', metrics['pearson'], on_step=False, on_epoch=True)

        return {'test_loss': loss, **metrics}

    def predict_step(self, batch: Tuple, batch_idx: int) -> Dict:
        audio, params, lengths, utterance_names = batch
        pred = self(audio, lengths)
        return {
            'predictions': pred,
            'targets': params,
            'lengths': lengths,
            'utterance_names': utterance_names,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.98),
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            anneal_strategy='cos',
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
        }

    # =========================================================================
    # Shared Utilities (matching TransformerModel interface)
    # =========================================================================

    def _create_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """Create mask: 1 for valid frames, 0 for padding."""
        batch_size = lengths.shape[0]
        mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len)
        return (mask < lengths.unsqueeze(1)).float()

    def _compute_pcc_loss(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Pearson Correlation loss: 1 - PCC (per-sequence)."""
        batch_size = pred.shape[0]
        total_pcc_loss = 0.0
        valid_sequences = 0
        mask_bool = mask.squeeze(-1) > 0

        for b in range(batch_size):
            valid = mask_bool[b]
            if valid.sum() < 2:
                continue

            p = pred[b][valid]
            t = target[b][valid]

            p_mean = torch.mean(p, dim=0)
            t_mean = torch.mean(t, dim=0)
            p_std = torch.std(p, dim=0) + 1e-8
            t_std = torch.std(t, dim=0) + 1e-8

            p_norm = (p - p_mean) / p_std
            t_norm = (t - t_mean) / t_std

            pcc = torch.mean(p_norm * t_norm, dim=0)
            total_pcc_loss += torch.mean(1.0 - pcc)
            valid_sequences += 1

        if valid_sequences > 0:
            return total_pcc_loss / valid_sequences
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    def _compute_temporal_loss(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Velocity + Acceleration matching losses."""
        pred_vel = pred[:, 1:, :] - pred[:, :-1, :]
        target_vel = target[:, 1:, :] - target[:, :-1, :]
        vel_mask = mask[:, :-1, :] * mask[:, 1:, :]

        vel_loss = (self.criterion(pred_vel, target_vel) * vel_mask).sum() / vel_mask.sum()

        pred_acc = pred_vel[:, 1:, :] - pred_vel[:, :-1, :]
        target_acc = target_vel[:, 1:, :] - target_vel[:, :-1, :]
        acc_mask = vel_mask[:, :-1, :] * vel_mask[:, 1:, :]

        acc_loss = (self.criterion(pred_acc, target_acc) * acc_mask).sum() / acc_mask.sum()

        return {'velocity_loss': vel_loss, 'acceleration_loss': acc_loss}

    def _compute_metrics(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> Dict[str, float]:
        """Compute RMSE, MAE, and Pearson correlation."""
        pred_flat = pred * mask
        target_flat = target * mask

        pred_np = pred_flat.detach().cpu().numpy()
        target_np = target_flat.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy().squeeze(-1)

        valid = mask_np > 0
        pred_valid = pred_np[valid]
        target_valid = target_np[valid]

        rmse = float(np.sqrt(np.mean((pred_valid - target_valid) ** 2)))
        mae = float(np.mean(np.abs(pred_valid - target_valid)))

        correlations = []
        for i in range(pred_np.shape[-1]):
            p = pred_np[:, :, i][valid]
            t = target_np[:, :, i][valid]
            if len(p) > 1:
                corr = np.corrcoef(p, t)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        pearson = float(np.mean(correlations)) if correlations else 0.0

        return {'rmse': rmse, 'mae': mae, 'pearson': pearson}
