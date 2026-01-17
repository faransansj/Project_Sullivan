"""
Transformer Model for Acoustic-to-Articulatory Inversion

Implements a Transformer encoder architecture for predicting articulatory
parameters from audio features.

Based on "Attention Is All You Need" (Vaswani et al., 2017)

Author: Project Sullivan
Date: 2025-11-30
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Optional, Tuple
import numpy as np

from .positional_encoding import create_positional_encoding
from .model_utils import (
    create_padding_mask,
    create_loss_mask,
    count_parameters,
    format_parameter_count,
    get_activation_function
)


class TransformerModel(pl.LightningModule):
    """
    Transformer encoder for articulatory parameter prediction.

    Architecture:
    - Input projection: maps audio features to d_model
    - Positional encoding: adds position information
    - Transformer encoder: N layers of multi-head self-attention
    - Output projection: maps d_model to parameter dimension

    Parameters
    ----------
    input_dim : int
        Dimension of input audio features (80 for mel, 13 for MFCC)
    d_model : int, default=256
        Dimension of transformer model
    num_layers : int, default=4
        Number of transformer encoder layers
    num_heads : int, default=8
        Number of attention heads
    d_ff : int, default=1024
        Dimension of feed-forward network
    output_dim : int
        Dimension of output parameters (14 for geometric, 10 for PCA)
    dropout : float, default=0.1
        Dropout probability
    pos_encoding : str, default='learnable'
        Type of positional encoding ('sinusoidal' or 'learnable')
    activation : str, default='gelu'
        Activation function for feed-forward network
    learning_rate : float, default=5e-4
        Learning rate for optimizer
    weight_decay : float, default=0.01
        Weight decay for AdamW optimizer
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 1024,
        output_dim: int = 14,
        dropout: float = 0.1,
        pos_encoding: str = 'learnable',
        activation: str = 'gelu',
        learning_rate: float = 5e-4,
        weight_decay: float = 0.01,
        max_seq_len: int = 5000,
        velocity_weight: float = 1.0,
        acceleration_weight: float = 0.5,
        pcc_weight: float = 1.0,
        mse_weight: float = 1.0
    ):
        super().__init__()
        self.save_hyperparameters()

        # Store hyperparameters
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.output_dim = output_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.velocity_weight = velocity_weight
        self.acceleration_weight = acceleration_weight
        self.pcc_weight = pcc_weight
        self.mse_weight = mse_weight

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoding = create_positional_encoding(
            encoding_type=pos_encoding,
            d_model=d_model,
            max_len=max_seq_len,
            dropout=dropout
        )

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,  # (batch, seq, feature)
            norm_first=True  # Pre-norm for better training stability
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)  # Final layer norm
        )

        # Output projection
        self.output_projection = nn.Linear(d_model, output_dim)

        # Dropout for output
        self.dropout_layer = nn.Dropout(dropout)

        # Loss function - use 'none' reduction for proper masking
        self.criterion = nn.MSELoss(reduction='none')

        # Print model info
        param_count = count_parameters(self)
        print(f"\nTransformer Model Initialized:")
        print(f"  Parameters: {format_parameter_count(param_count)} ({param_count:,})")
        print(f"  d_model: {d_model}, Layers: {num_layers}, Heads: {num_heads}")
        print(f"  Input dim: {input_dim}, Output dim: {output_dim}\n")

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input audio features, shape (batch, seq_len, input_dim)
        lengths : torch.Tensor, optional
            Actual sequence lengths (before padding), shape (batch,)

        Returns
        -------
        output : torch.Tensor
            Predicted parameters, shape (batch, seq_len, output_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Input projection: (batch, seq_len, input_dim) -> (batch, seq_len, d_model)
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Create padding mask for attention
        # PyTorch Transformer uses True for positions to MASK (padding)
        src_key_padding_mask = None
        if lengths is not None:
            src_key_padding_mask = create_padding_mask(lengths, max_len=seq_len)
            # src_key_padding_mask: (batch, seq_len), True = padding

        # Transformer encoder
        # Shape: (batch, seq_len, d_model)
        x = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask
        )

        # Dropout
        x = self.dropout_layer(x)

        # Output projection: (batch, seq_len, d_model) -> (batch, seq_len, output_dim)
        output = self.output_projection(x)

        return output

    def _compute_pcc_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute Pearson Correlation Coefficient loss: 1 - PCC
        Per-Sequence implementation to capture local dynamic trends.
        """
        batch_size = pred.shape[0]
        total_pcc_loss = 0.0
        valid_sequences = 0

        # mask is (batch, seq_len, 1)
        mask_bool = mask.squeeze(-1) > 0  # (batch, seq_len)

        for b in range(batch_size):
            # Extract valid frames for this utterance
            valid_indices = mask_bool[b]

            # Skip if sequence is too short for correlation
            if valid_indices.sum() < 2:
                continue

            p = pred[b][valid_indices]  # (T, num_features)
            t = target[b][valid_indices]  # (T, num_features)

            # Compute stats per feature for this specific utterance
            p_mean = torch.mean(p, dim=0)
            t_mean = torch.mean(t, dim=0)
            
            p_std = torch.std(p, dim=0) + 1e-8
            t_std = torch.std(t, dim=0) + 1e-8
            
            # Standardize (Z-score normalization per sequence)
            p_norm = (p - p_mean) / p_std
            t_norm = (t - t_mean) / t_std
            
            # PCC is now just the mean of the element-wise product of standardized vectors
            # (equivalent to Cov / (std_p * std_t))
            # p_norm * t_norm gives correlation contribution per frame
            pcc = torch.mean(p_norm * t_norm, dim=0)
            
            # Loss = 1 - PCC (average across features)
            total_pcc_loss += torch.mean(1.0 - pcc)
            valid_sequences += 1

        if valid_sequences > 0:
            return total_pcc_loss / valid_sequences
        else:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        """
        Training step with Hybrid Loss (MSE + PCC + Temporal).
        """
        audio, params, lengths, _ = batch

        # Forward pass
        pred_params = self(audio, lengths)

        # Create mask for valid frames
        mask = self._create_mask(lengths, pred_params.shape[1]).to(pred_params.device)
        mask = mask.unsqueeze(-1)  # (batch, seq_len, 1)

        # Compute position loss (per-frame MSE)
        squared_error = self.criterion(pred_params, params)
        masked_error = squared_error * mask
        position_loss = masked_error.sum() / mask.sum()

        # Compute temporal losses (velocity + acceleration)
        temporal_losses = self._compute_temporal_loss(pred_params, params, mask)
        velocity_loss = temporal_losses['velocity_loss']
        acceleration_loss = temporal_losses['acceleration_loss']

        # Compute PCC loss
        pcc_loss = self._compute_pcc_loss(pred_params, params, mask)

        # Combined loss
        loss = (self.mse_weight * position_loss) + \
               (self.pcc_weight * pcc_loss) + \
               (self.velocity_weight * velocity_loss) + \
               (self.acceleration_weight * acceleration_loss)

        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mse', position_loss, on_step=False, on_epoch=True)
        self.log('train_pcc_loss', pcc_loss, on_step=False, on_epoch=True)
        self.log('train_velocity_loss', velocity_loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: Tuple, batch_idx: int) -> Dict:
        """
        Validation step with Hybrid Loss.
        """
        audio, params, lengths, _ = batch

        # Forward pass
        pred_params = self(audio, lengths)

        # Create mask for valid frames
        mask = self._create_mask(lengths, pred_params.shape[1]).to(pred_params.device)
        mask = mask.unsqueeze(-1)

        # Compute position loss
        squared_error = self.criterion(pred_params, params)
        masked_error = squared_error * mask
        position_loss = masked_error.sum() / mask.sum()

        # Compute temporal losses
        temporal_losses = self._compute_temporal_loss(pred_params, params, mask)
        velocity_loss = temporal_losses['velocity_loss']
        acceleration_loss = temporal_losses['acceleration_loss']

        # Compute PCC loss
        pcc_loss = self._compute_pcc_loss(pred_params, params, mask)

        # Combined loss
        loss = (self.mse_weight * position_loss) + \
               (self.pcc_weight * pcc_loss) + \
               (self.velocity_weight * velocity_loss) + \
               (self.acceleration_weight * acceleration_loss)

        # Compute metrics
        metrics = self._compute_metrics(pred_params, params, mask)

        # Logging
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mse', position_loss, on_step=False, on_epoch=True)
        self.log('val_pcc_loss', pcc_loss, on_step=False, on_epoch=True)
        self.log('val_rmse', metrics['rmse'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_pearson', metrics['pearson'], on_step=False, on_epoch=True)

        return {'val_loss': loss, **metrics}

    def test_step(self, batch: Tuple, batch_idx: int) -> Dict:
        """
        Test step with temporal loss.

        Parameters
        ----------
        batch : tuple
            (audio, params, lengths, utterance_names)
        batch_idx : int
            Batch index

        Returns
        -------
        metrics : dict
            Test metrics
        """
        audio, params, lengths, _ = batch

        # Forward pass
        pred_params = self(audio, lengths)

        # Create mask for valid frames
        mask = self._create_mask(lengths, pred_params.shape[1]).to(pred_params.device)
        mask = mask.unsqueeze(-1)

        # Compute position loss
        squared_error = self.criterion(pred_params, params)
        masked_error = squared_error * mask
        position_loss = masked_error.sum() / mask.sum()

        # Compute temporal losses
        temporal_losses = self._compute_temporal_loss(pred_params, params, mask)
        velocity_loss = temporal_losses['velocity_loss']
        acceleration_loss = temporal_losses['acceleration_loss']

        # Combined loss
        loss = position_loss + (self.velocity_weight * velocity_loss) + (self.acceleration_weight * acceleration_loss)

        # Compute metrics
        metrics = self._compute_metrics(pred_params, params, mask)

        # Logging
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_position_loss', position_loss, on_step=False, on_epoch=True)
        self.log('test_velocity_loss', velocity_loss, on_step=False, on_epoch=True)
        self.log('test_acceleration_loss', acceleration_loss, on_step=False, on_epoch=True)
        self.log('test_rmse', metrics['rmse'], on_step=False, on_epoch=True)
        self.log('test_mae', metrics['mae'], on_step=False, on_epoch=True)
        self.log('test_pearson', metrics['pearson'], on_step=False, on_epoch=True)

        return {'test_loss': loss, **metrics}

    def configure_optimizers(self):
        """
        Configure optimizer and scheduler.

        Uses AdamW with CosineAnnealingWarmRestarts for Transformer training.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.98),  # Transformer-specific betas
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double period after each restart
            eta_min=1e-6  # Minimum learning rate
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def _create_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """
        Create mask for padded sequences (copied from baseline_lstm.py).

        Parameters
        ----------
        lengths : torch.Tensor
            Actual sequence lengths, shape (batch,)
        max_len : int
            Maximum sequence length in batch

        Returns
        -------
        mask : torch.Tensor
            Mask tensor, shape (batch, max_len), 1 for valid frames, 0 for padding
        """
        batch_size = lengths.shape[0]
        mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len)
        mask = (mask < lengths.unsqueeze(1)).float()
        return mask

    def _compute_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics (copied from baseline_lstm.py).

        Parameters
        ----------
        pred : torch.Tensor
            Predicted parameters, shape (batch, seq_len, param_dim)
        target : torch.Tensor
            Target parameters, shape (batch, seq_len, param_dim)
        mask : torch.Tensor
            Mask for valid frames, shape (batch, seq_len, 1)

        Returns
        -------
        metrics : dict
            Dictionary of metrics (rmse, mae, pearson)
        """
        # Flatten and apply mask
        pred_flat = pred * mask
        target_flat = target * mask

        # Move to CPU for numpy operations
        pred_np = pred_flat.detach().cpu().numpy()
        target_np = target_flat.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy().squeeze(-1)  # (batch, seq_len)

        # Filter out padded values
        valid_mask = mask_np > 0
        pred_valid = pred_np[valid_mask]
        target_valid = target_np[valid_mask]

        # RMSE
        rmse = np.sqrt(np.mean((pred_valid - target_valid) ** 2))

        # MAE
        mae = np.mean(np.abs(pred_valid - target_valid))

        # Pearson correlation (average across parameters)
        correlations = []
        for i in range(pred_np.shape[-1]):
            pred_param = pred_np[:, :, i][valid_mask]
            target_param = target_np[:, :, i][valid_mask]

            if len(pred_param) > 1:
                corr = np.corrcoef(pred_param, target_param)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        pearson = np.mean(correlations) if correlations else 0.0

        return {
            'rmse': rmse,
            'mae': mae,
            'pearson': pearson
        }

    def _compute_temporal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute temporal loss components (velocity and acceleration matching).

        Parameters
        ----------
        pred : torch.Tensor
            Predicted parameters, shape (batch, seq_len, output_dim)
        target : torch.Tensor
            Target parameters, shape (batch, seq_len, output_dim)
        mask : torch.Tensor
            Mask for valid frames, shape (batch, seq_len, 1)

        Returns
        -------
        losses : dict
            Dictionary with 'velocity_loss' and 'acceleration_loss'
        """
        batch_size, seq_len, output_dim = pred.shape

        # Compute velocity (first-order difference)
        # velocity[t] = position[t+1] - position[t]
        pred_velocity = pred[:, 1:, :] - pred[:, :-1, :]  # (batch, seq_len-1, output_dim)
        target_velocity = target[:, 1:, :] - target[:, :-1, :]

        # Mask for velocity (both frames must be valid)
        velocity_mask = mask[:, :-1, :] * mask[:, 1:, :]  # (batch, seq_len-1, 1)

        # Velocity loss
        velocity_error = self.criterion(pred_velocity, target_velocity)
        velocity_loss = (velocity_error * velocity_mask).sum() / velocity_mask.sum()

        # Compute acceleration (second-order difference)
        # acceleration[t] = velocity[t+1] - velocity[t] = position[t+2] - 2*position[t+1] + position[t]
        pred_acceleration = pred_velocity[:, 1:, :] - pred_velocity[:, :-1, :]  # (batch, seq_len-2, output_dim)
        target_acceleration = target_velocity[:, 1:, :] - target_velocity[:, :-1, :]

        # Mask for acceleration (three frames must be valid)
        acceleration_mask = velocity_mask[:, :-1, :] * velocity_mask[:, 1:, :]  # (batch, seq_len-2, 1)

        # Acceleration loss
        acceleration_error = self.criterion(pred_acceleration, target_acceleration)
        acceleration_loss = (acceleration_error * acceleration_mask).sum() / acceleration_mask.sum()

        return {
            'velocity_loss': velocity_loss,
            'acceleration_loss': acceleration_loss
        }

    def predict_step(self, batch: Tuple, batch_idx: int) -> Dict:
        """
        Prediction step (copied from baseline_lstm.py).

        Parameters
        ----------
        batch : tuple
            (audio, params, lengths, utterance_names)
        batch_idx : int
            Batch index

        Returns
        -------
        predictions : dict
            Predictions and metadata
        """
        audio, params, lengths, utterance_names = batch

        # Forward pass
        pred_params = self(audio, lengths)

        return {
            'predictions': pred_params,
            'targets': params,
            'lengths': lengths,
            'utterance_names': utterance_names
        }