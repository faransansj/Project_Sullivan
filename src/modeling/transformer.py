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
        max_seq_len: int = 5000
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

        # Loss function
        self.criterion = nn.MSELoss()

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

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        """
        Training step (copied from baseline_lstm.py).

        Parameters
        ----------
        batch : tuple
            (audio, params, lengths, utterance_names)
        batch_idx : int
            Batch index

        Returns
        -------
        loss : torch.Tensor
            Training loss
        """
        audio, params, lengths, _ = batch

        # Forward pass
        pred_params = self(audio, lengths)

        # Compute loss only on non-padded frames
        mask = self._create_mask(lengths, pred_params.shape[1]).to(pred_params.device)
        mask = mask.unsqueeze(-1)  # (batch, seq_len, 1)

        # Masked loss
        loss = self.criterion(pred_params * mask, params * mask)

        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Tuple, batch_idx: int) -> Dict:
        """
        Validation step (copied from baseline_lstm.py).

        Parameters
        ----------
        batch : tuple
            (audio, params, lengths, utterance_names)
        batch_idx : int
            Batch index

        Returns
        -------
        metrics : dict
            Validation metrics
        """
        audio, params, lengths, _ = batch

        # Forward pass
        pred_params = self(audio, lengths)

        # Compute loss
        mask = self._create_mask(lengths, pred_params.shape[1]).to(pred_params.device)
        mask = mask.unsqueeze(-1)

        loss = self.criterion(pred_params * mask, params * mask)

        # Compute metrics
        metrics = self._compute_metrics(pred_params, params, mask)

        # Logging
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_rmse', metrics['rmse'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mae', metrics['mae'], on_step=False, on_epoch=True)
        self.log('val_pearson', metrics['pearson'], on_step=False, on_epoch=True)

        return {'val_loss': loss, **metrics}

    def test_step(self, batch: Tuple, batch_idx: int) -> Dict:
        """
        Test step (copied from baseline_lstm.py).

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

        # Compute loss
        mask = self._create_mask(lengths, pred_params.shape[1]).to(pred_params.device)
        mask = mask.unsqueeze(-1)

        loss = self.criterion(pred_params * mask, params * mask)

        # Compute metrics
        metrics = self._compute_metrics(pred_params, params, mask)

        # Logging
        self.log('test_loss', loss, on_step=False, on_epoch=True)
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
