"""
Baseline Bi-LSTM Model for Acoustic-to-Articulatory Inversion

This module implements a simple bidirectional LSTM baseline model
for predicting articulatory parameters from audio features.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Optional, Tuple
import numpy as np


class BaselineLSTM(pl.LightningModule):
    """
    Baseline Bi-LSTM model for articulatory parameter prediction.

    Architecture:
    - Input: Audio features (mel-spectrogram or MFCC)
    - Bi-LSTM layers
    - Fully connected output layer
    - Output: Articulatory parameters

    Parameters
    ----------
    input_dim : int
        Dimension of input audio features (80 for mel, 13 for MFCC)
    hidden_dim : int, default=128
        Hidden dimension of LSTM
    num_layers : int, default=2
        Number of LSTM layers
    output_dim : int
        Dimension of output parameters (14 for geometric, 10 for PCA)
    dropout : float, default=0.3
        Dropout probability
    learning_rate : float, default=1e-3
        Learning rate for optimizer
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 14,
        dropout: float = 0.3,
        learning_rate: float = 1e-3
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout
        self.learning_rate = learning_rate

        # Bi-LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Output layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

        # Loss function
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
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

        # Pack padded sequence if lengths provided
        if lengths is not None:
            # Sort by length (required by pack_padded_sequence)
            lengths_cpu = lengths.cpu()
            sorted_lengths, sorted_indices = torch.sort(lengths_cpu, descending=True)
            _, unsorted_indices = torch.sort(sorted_indices)

            x = x[sorted_indices]

            # Pack
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, sorted_lengths, batch_first=True, enforce_sorted=True
            )

            # LSTM
            packed_output, _ = self.lstm(packed_x)

            # Unpack
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True, total_length=seq_len
            )

            # Unsort
            lstm_out = lstm_out[unsorted_indices]
        else:
            # No packing
            lstm_out, _ = self.lstm(x)

        # Dropout
        lstm_out = self.dropout_layer(lstm_out)

        # Output layer
        output = self.fc(lstm_out)

        return output

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        """Training step."""
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
        """Validation step."""
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
        """Test step."""
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
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def _create_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """
        Create mask for padded sequences.

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
        Compute evaluation metrics.

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
            Dictionary of metrics
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
        """Prediction step."""
        audio, params, lengths, utterance_names = batch

        # Forward pass
        pred_params = self(audio, lengths)

        return {
            'predictions': pred_params,
            'targets': params,
            'lengths': lengths,
            'utterance_names': utterance_names
        }
