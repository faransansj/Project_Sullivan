"""
Bi-LSTM Model for Articulatory Parameter Prediction

This module implements a Bidirectional LSTM baseline model
for predicting articulatory parameters from audio features.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class BiLSTMArticulationPredictor(nn.Module):
    """
    Bidirectional LSTM for articulatory parameter prediction

    Architecture:
        Audio Features → Bi-LSTM layers → Fully Connected → Parameters
    """

    def __init__(self,
                 input_dim: int = 80,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 output_dim: int = 10,
                 dropout: float = 0.3,
                 bidirectional: bool = True):
        """
        Args:
            input_dim: Input feature dimension (e.g., 80 for mel-spectrogram)
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            output_dim: Output dimension (10 for articulatory parameters)
            dropout: Dropout probability
            bidirectional: Use bidirectional LSTM
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.bidirectional = bidirectional

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Output projection
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, output_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: (batch, time, input_dim) input audio features
            lengths: (batch,) sequence lengths for pack_padded_sequence

        Returns:
            (batch, time, output_dim) predicted parameters
        """
        batch_size, seq_len, _ = x.shape

        # Pack padded sequences if lengths provided
        if lengths is not None:
            # Sort by length (required for pack_padded_sequence)
            lengths_cpu = lengths.cpu()
            sorted_lengths, sorted_idx = torch.sort(lengths_cpu, descending=True)
            _, unsort_idx = torch.sort(sorted_idx)

            x = x[sorted_idx]

            # Pack sequences
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, sorted_lengths, batch_first=True, enforce_sorted=True
            )

            # LSTM forward
            packed_output, (hidden, cell) = self.lstm(packed_x)

            # Unpack sequences
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True, total_length=seq_len
            )

            # Unsort to original order
            lstm_out = lstm_out[unsort_idx]

        else:
            # Standard forward pass
            lstm_out, (hidden, cell) = self.lstm(x)

        # Apply dropout
        lstm_out = self.dropout(lstm_out)

        # Project to output dimension
        output = self.fc(lstm_out)

        return output

    def get_num_params(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BiLSTMWithAttention(nn.Module):
    """
    Bi-LSTM with attention mechanism for articulatory parameter prediction

    Extension of basic Bi-LSTM with self-attention to better capture
    long-range dependencies in speech.
    """

    def __init__(self,
                 input_dim: int = 80,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 output_dim: int = 10,
                 dropout: float = 0.3,
                 attention_heads: int = 4):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            output_dim: Output dimension
            dropout: Dropout probability
            attention_heads: Number of attention heads
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # Bi-LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Multi-head self-attention
        lstm_output_dim = hidden_dim * 2
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(lstm_output_dim)

        # Output projection
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self,
                x: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with attention

        Args:
            x: (batch, time, input_dim) input audio features
            lengths: (batch,) sequence lengths

        Returns:
            (batch, time, output_dim) predicted parameters
        """
        batch_size, seq_len, _ = x.shape

        # Bi-LSTM encoding
        if lengths is not None:
            # Pack padded sequences
            lengths_cpu = lengths.cpu()
            sorted_lengths, sorted_idx = torch.sort(lengths_cpu, descending=True)
            _, unsort_idx = torch.sort(sorted_idx)

            x = x[sorted_idx]

            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, sorted_lengths, batch_first=True, enforce_sorted=True
            )

            packed_output, _ = self.lstm(packed_x)

            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True, total_length=seq_len
            )

            lstm_out = lstm_out[unsort_idx]

        else:
            lstm_out, _ = self.lstm(x)

        # Self-attention
        # Create attention mask if lengths provided
        if lengths is not None:
            # Create mask: True for padding positions
            mask = torch.arange(seq_len, device=x.device)[None, :] >= lengths[:, None]
        else:
            mask = None

        attn_out, _ = self.attention(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=mask
        )

        # Residual connection + layer norm
        lstm_out = self.layer_norm(lstm_out + attn_out)

        # Output projection
        output = self.fc(lstm_out)

        return output

    def get_num_params(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in model

    Args:
        model: PyTorch model

    Returns:
        (total_params, trainable_params) tuple
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_summary(model: nn.Module):
    """Print model architecture summary"""
    print("=" * 80)
    print("Model Architecture Summary")
    print("=" * 80)
    print(model)
    print("=" * 80)

    total, trainable = count_parameters(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Model size: {total * 4 / 1024 / 1024:.2f} MB (FP32)")
    print("=" * 80)
