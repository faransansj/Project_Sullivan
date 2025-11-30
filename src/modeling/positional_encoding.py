"""
Positional Encoding Modules for Transformer and Conformer

Provides various positional encoding strategies:
- Sinusoidal: Standard Transformer positional encoding (Vaswani et al., 2017)
- Learnable: Trainable position embeddings
- Relative: Relative positional encoding for Conformer (Shaw et al., 2018)

Author: Project Sullivan
Date: 2025-11-30
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from "Attention Is All You Need".

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        d_model (int): Dimension of the model
        max_len (int): Maximum sequence length
        dropout (float): Dropout rate
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute the div_term for scaling
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cos to odd indices
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            # Handle odd d_model case
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        # Add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added, shape (batch, seq_len, d_model)
        """
        # x: (batch, seq_len, d_model)
        # self.pe: (1, max_len, d_model)

        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional embeddings.

    Unlike sinusoidal encoding, these are trainable parameters.
    Often works better for shorter sequences or domain-specific tasks.

    Args:
        d_model (int): Dimension of the model
        max_len (int): Maximum sequence length
        dropout (float): Dropout rate
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Learnable position embeddings
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learnable positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added, shape (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding for Conformer.

    Instead of absolute positions, encodes relative distances between positions.
    Based on "Self-Attention with Relative Position Representations" (Shaw et al., 2018).

    This is used in Conformer's multi-head attention mechanism.

    Args:
        d_model (int): Dimension of the model
        max_len (int): Maximum relative distance to consider
        num_heads (int): Number of attention heads
    """

    def __init__(self, d_model: int, max_len: int = 512, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Relative position embeddings
        # Shape: (2 * max_len - 1, d_k)
        # Covers relative positions from -max_len+1 to max_len-1
        self.rel_embeddings = nn.Parameter(
            torch.randn(2 * max_len - 1, self.d_k) * 0.02
        )

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Generate relative positional encoding matrix.

        Args:
            seq_len: Sequence length

        Returns:
            Relative position embeddings of shape (seq_len, seq_len, d_k)
        """
        # Clamp sequence length to max_len
        seq_len = min(seq_len, self.max_len)

        # Create relative position matrix
        # rel_pos[i, j] = j - i (relative position from i to j)
        positions = torch.arange(seq_len, device=self.rel_embeddings.device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)

        # Shift to make all indices non-negative
        # Range: [0, 2*max_len-2]
        relative_positions = relative_positions + self.max_len - 1

        # Clamp to valid range
        relative_positions = torch.clamp(
            relative_positions, 0, 2 * self.max_len - 2
        )

        # Index into embeddings
        # Shape: (seq_len, seq_len, d_k)
        rel_pos_embeddings = self.rel_embeddings[relative_positions]

        return rel_pos_embeddings

    def get_relative_bias(self, seq_len: int) -> torch.Tensor:
        """
        Get relative position bias for attention scores.

        This is added to attention logits before softmax.

        Args:
            seq_len: Sequence length

        Returns:
            Relative position bias of shape (num_heads, seq_len, seq_len)
        """
        # Get relative embeddings
        rel_embeddings = self.forward(seq_len)  # (seq_len, seq_len, d_k)

        # Expand for all heads
        # Shape: (1, seq_len, seq_len, d_k) -> (num_heads, seq_len, seq_len, d_k)
        rel_embeddings = rel_embeddings.unsqueeze(0).expand(
            self.num_heads, -1, -1, -1
        )

        # Sum across d_k dimension to get bias scores
        # Shape: (num_heads, seq_len, seq_len)
        bias = rel_embeddings.sum(dim=-1)

        return bias


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) from "RoFormer" (Su et al., 2021).

    More recent alternative to sinusoidal encoding, used in modern models like LLaMA.
    Rotates key and query representations based on their positions.

    Args:
        d_model (int): Dimension of the model
        max_len (int): Maximum sequence length
        base (int): Base for frequency calculation (default: 10000)
    """

    def __init__(self, d_model: int, max_len: int = 5000, base: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.base = base

        # Precompute frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)

        # Cache for rotary embeddings
        self._cache = None
        self._cache_len = 0

    def _compute_rotary_emb(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute rotary embeddings for given sequence length."""
        if self._cache is not None and seq_len <= self._cache_len:
            return self._cache[:seq_len]

        # Compute position encodings
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, d_model//2)

        # Create rotation matrix
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, d_model)

        # Cache for future use
        self._cache = emb
        self._cache_len = seq_len

        return emb

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dimensions."""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor, seq_dim: int = 1) -> torch.Tensor:
        """
        Apply rotary position embeddings.

        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            seq_dim: Dimension index for sequence length

        Returns:
            Tensor with rotary embeddings applied
        """
        seq_len = x.shape[seq_dim]
        rotary_emb = self._compute_rotary_emb(seq_len, x.device)

        # Expand to match input dimensions
        # rotary_emb: (seq_len, d_model)
        # Need to broadcast to x's shape
        emb_cos = rotary_emb.cos()
        emb_sin = rotary_emb.sin()

        # Apply rotation
        x_rotated = (x * emb_cos) + (self.rotate_half(x) * emb_sin)

        return x_rotated


# Factory function for easy instantiation
def create_positional_encoding(
    encoding_type: str,
    d_model: int,
    max_len: int = 5000,
    dropout: float = 0.1,
    **kwargs
) -> nn.Module:
    """
    Factory function to create positional encoding modules.

    Args:
        encoding_type: Type of encoding ('sinusoidal', 'learnable', 'relative', 'rotary')
        d_model: Model dimension
        max_len: Maximum sequence length
        dropout: Dropout rate
        **kwargs: Additional arguments for specific encoding types

    Returns:
        Positional encoding module

    Example:
        >>> pos_enc = create_positional_encoding('sinusoidal', d_model=256, max_len=1000)
        >>> x = torch.randn(8, 100, 256)  # (batch, seq_len, d_model)
        >>> x_with_pos = pos_enc(x)
    """
    encoding_type = encoding_type.lower()

    if encoding_type == 'sinusoidal':
        return SinusoidalPositionalEncoding(d_model, max_len, dropout)
    elif encoding_type == 'learnable':
        return LearnablePositionalEncoding(d_model, max_len, dropout)
    elif encoding_type == 'relative':
        num_heads = kwargs.get('num_heads', 8)
        return RelativePositionalEncoding(d_model, max_len, num_heads)
    elif encoding_type == 'rotary':
        base = kwargs.get('base', 10000)
        return RotaryPositionalEmbedding(d_model, max_len, base)
    else:
        raise ValueError(
            f"Unknown encoding type: {encoding_type}. "
            f"Choose from: 'sinusoidal', 'learnable', 'relative', 'rotary'"
        )
