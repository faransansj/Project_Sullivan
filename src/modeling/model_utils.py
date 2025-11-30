"""
Model Utilities for Transformer and Conformer

Provides shared utilities for sequence masking, padding, and other common operations.

Author: Project Sullivan
Date: 2025-11-30
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


def create_padding_mask(
    lengths: torch.Tensor,
    max_len: Optional[int] = None,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Create padding mask from sequence lengths.

    For PyTorch's MultiheadAttention and Transformer modules, the mask should be:
    - True for positions that should be MASKED (padding)
    - False for positions that should be ATTENDED

    Args:
        lengths: Tensor of shape (batch,) containing actual sequence lengths
        max_len: Maximum sequence length. If None, uses lengths.max()
        device: Device to create mask on. If None, uses lengths.device

    Returns:
        Boolean mask of shape (batch, max_len) where True = padding position

    Example:
        >>> lengths = torch.tensor([10, 8, 5])
        >>> mask = create_padding_mask(lengths, max_len=10)
        >>> mask.shape
        torch.Size([3, 10])
        >>> mask[0].sum()  # First sequence has no padding
        tensor(0)
        >>> mask[2].sum()  # Third sequence has 5 padding positions
        tensor(5)
    """
    if device is None:
        device = lengths.device

    if max_len is None:
        max_len = lengths.max().item()

    batch_size = lengths.size(0)

    # Create range tensor: (max_len,)
    positions = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)

    # Expand lengths: (batch, 1)
    lengths_expanded = lengths.unsqueeze(1)  # (batch, 1)

    # Create mask: positions >= lengths -> True (padding)
    padding_mask = positions >= lengths_expanded  # (batch, max_len)

    return padding_mask


def create_loss_mask(
    lengths: torch.Tensor,
    max_len: Optional[int] = None,
    output_dim: int = 1,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Create mask for computing loss on variable-length sequences.

    This mask is used to zero out loss contributions from padding positions.

    Args:
        lengths: Tensor of shape (batch,) containing actual sequence lengths
        max_len: Maximum sequence length. If None, uses lengths.max()
        output_dim: Output dimension to broadcast mask to (default: 1)
        device: Device to create mask on. If None, uses lengths.device

    Returns:
        Float mask of shape (batch, max_len, output_dim) where 1.0 = valid, 0.0 = padding

    Example:
        >>> lengths = torch.tensor([10, 8, 5])
        >>> mask = create_loss_mask(lengths, max_len=10, output_dim=14)
        >>> mask.shape
        torch.Size([3, 10, 14])
        >>> pred = torch.randn(3, 10, 14)
        >>> target = torch.randn(3, 10, 14)
        >>> loss = ((pred - target) ** 2 * mask).sum() / mask.sum()
    """
    if device is None:
        device = lengths.device

    if max_len is None:
        max_len = lengths.max().item()

    # Create padding mask (True = padding)
    padding_mask = create_padding_mask(lengths, max_len, device)

    # Invert to get valid positions (False -> 1.0, True -> 0.0)
    loss_mask = (~padding_mask).float()  # (batch, max_len)

    # Add output dimension
    loss_mask = loss_mask.unsqueeze(-1)  # (batch, max_len, 1)

    # Expand to output_dim if needed
    if output_dim > 1:
        loss_mask = loss_mask.expand(-1, -1, output_dim)  # (batch, max_len, output_dim)

    return loss_mask


def create_causal_mask(
    seq_len: int,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Create causal (triangular) attention mask for autoregressive models.

    Prevents attending to future positions.

    Args:
        seq_len: Sequence length
        device: Device to create mask on

    Returns:
        Boolean mask of shape (seq_len, seq_len) where True = masked position

    Example:
        >>> mask = create_causal_mask(5)
        >>> mask
        tensor([[False,  True,  True,  True,  True],
                [False, False,  True,  True,  True],
                [False, False, False,  True,  True],
                [False, False, False, False,  True],
                [False, False, False, False, False]])
    """
    if device is None:
        device = torch.device('cpu')

    # Create upper triangular matrix (excluding diagonal)
    # triu(..., diagonal=1) creates upper triangle with diagonal offset
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    return mask


def combine_masks(
    padding_mask: Optional[torch.Tensor] = None,
    causal_mask: Optional[torch.Tensor] = None,
    custom_mask: Optional[torch.Tensor] = None
) -> Optional[torch.Tensor]:
    """
    Combine multiple attention masks using logical OR.

    A position is masked if it's masked in ANY of the input masks.

    Args:
        padding_mask: Padding mask of shape (batch, seq_len) or (batch, 1, seq_len)
        causal_mask: Causal mask of shape (seq_len, seq_len)
        custom_mask: Custom mask of shape (batch, seq_len, seq_len) or (seq_len, seq_len)

    Returns:
        Combined mask or None if all inputs are None

    Example:
        >>> lengths = torch.tensor([5, 3])
        >>> padding_mask = create_padding_mask(lengths, max_len=5)
        >>> causal_mask = create_causal_mask(5)
        >>> combined = combine_masks(padding_mask, causal_mask)
        >>> combined.shape
        torch.Size([2, 5, 5])
    """
    result_mask = None

    if padding_mask is not None:
        # Ensure padding_mask has shape (batch, 1, seq_len) for broadcasting
        if padding_mask.dim() == 2:
            padding_mask = padding_mask.unsqueeze(1)  # (batch, 1, seq_len)
        result_mask = padding_mask

    if causal_mask is not None:
        if result_mask is None:
            result_mask = causal_mask
        else:
            # Broadcast and combine
            result_mask = result_mask | causal_mask.unsqueeze(0)

    if custom_mask is not None:
        if result_mask is None:
            result_mask = custom_mask
        else:
            result_mask = result_mask | custom_mask

    return result_mask


def get_activation_function(activation: str) -> nn.Module:
    """
    Get activation function by name.

    Args:
        activation: Name of activation function ('relu', 'gelu', 'swish', 'silu', 'tanh')

    Returns:
        Activation module

    Example:
        >>> act = get_activation_function('gelu')
        >>> x = torch.randn(10)
        >>> y = act(x)
    """
    activation = activation.lower()

    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation in ['swish', 'silu']:
        return nn.SiLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'elu':
        return nn.ELU()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU()
    else:
        raise ValueError(
            f"Unknown activation: {activation}. "
            f"Choose from: 'relu', 'gelu', 'swish', 'silu', 'tanh', 'elu', 'leaky_relu'"
        )


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int = 1,
    keepdim: bool = False
) -> torch.Tensor:
    """
    Compute mean of tensor along dimension, excluding masked positions.

    Args:
        tensor: Input tensor
        mask: Boolean mask (True = masked/invalid)
        dim: Dimension to reduce
        keepdim: Whether to keep reduced dimension

    Returns:
        Masked mean

    Example:
        >>> x = torch.tensor([[1., 2., 3., 0., 0.],
        ...                   [4., 5., 0., 0., 0.]])
        >>> lengths = torch.tensor([3, 2])
        >>> mask = create_padding_mask(lengths, max_len=5)
        >>> masked_mean(x, mask, dim=1)
        tensor([2., 4.5])  # [mean(1,2,3), mean(4,5)]
    """
    # Invert mask: True -> 0.0, False -> 1.0
    valid_mask = (~mask).float()

    # Expand mask to match tensor dimensions if needed
    while valid_mask.dim() < tensor.dim():
        valid_mask = valid_mask.unsqueeze(-1)

    # Compute masked sum
    masked_sum = (tensor * valid_mask).sum(dim=dim, keepdim=keepdim)

    # Compute count of valid elements
    valid_count = valid_mask.sum(dim=dim, keepdim=keepdim)

    # Avoid division by zero
    valid_count = valid_count.clamp(min=1e-9)

    return masked_sum / valid_count


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count number of parameters in model.

    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters

    Returns:
        Number of parameters

    Example:
        >>> model = nn.Linear(100, 50)
        >>> count_parameters(model)
        5050  # 100*50 + 50 (bias)
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def format_parameter_count(count: int) -> str:
    """
    Format parameter count in human-readable form.

    Args:
        count: Number of parameters

    Returns:
        Formatted string

    Example:
        >>> format_parameter_count(1234567)
        '1.23M'
        >>> format_parameter_count(5432)
        '5.43K'
    """
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.2f}K"
    else:
        return str(count)


class LayerScale(nn.Module):
    """
    Layer Scale module for improving training stability.

    Introduced in "Going deeper with Image Transformers" (Touvron et al., 2021).
    Scales residual branches with learnable parameters initialized close to 0.

    Args:
        dim: Dimension of the layer
        init_value: Initial value for scale parameters (default: 1e-4)

    Example:
        >>> layer_scale = LayerScale(dim=256, init_value=1e-4)
        >>> x = torch.randn(8, 100, 256)
        >>> x_scaled = layer_scale(x)
    """

    def __init__(self, dim: int, init_value: float = 1e-4):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim) * init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


def init_weights(module: nn.Module, std: float = 0.02):
    """
    Initialize module weights.

    Args:
        module: Module to initialize
        std: Standard deviation for normal initialization

    Example:
        >>> model = nn.TransformerEncoder(...)
        >>> model.apply(lambda m: init_weights(m, std=0.02))
    """
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            module.bias.data.zero_()
