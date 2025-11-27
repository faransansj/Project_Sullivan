"""
Loss Functions for Articulatory Parameter Prediction

Implements custom loss functions including temporal smoothness regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ArticulatoryLoss(nn.Module):
    """
    Combined loss for articulatory parameter prediction

    Loss = MSE Loss + α * Smoothness Loss
    """

    def __init__(self,
                 smoothness_weight: float = 0.1,
                 reduction: str = 'mean'):
        """
        Args:
            smoothness_weight: Weight for temporal smoothness term (alpha)
            reduction: Loss reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.reduction = reduction

        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute combined loss

        Args:
            predictions: (batch, time, 10) predicted parameters
            targets: (batch, time, 10) ground truth parameters
            mask: (batch, time) boolean mask (True for valid positions)

        Returns:
            Scalar loss value
        """
        # MSE loss
        mse = self.compute_mse_loss(predictions, targets, mask)

        # Smoothness loss
        smoothness = self.compute_smoothness_loss(predictions, mask)

        # Total loss
        total_loss = mse + self.smoothness_weight * smoothness

        return total_loss

    def compute_mse_loss(self,
                        predictions: torch.Tensor,
                        targets: torch.Tensor,
                        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute MSE loss with optional masking

        Args:
            predictions: (batch, time, 10) predictions
            targets: (batch, time, 10) targets
            mask: (batch, time) mask

        Returns:
            MSE loss
        """
        if mask is None:
            return self.mse_loss(predictions, targets)

        # Apply mask
        mask = mask.unsqueeze(-1)  # (batch, time, 1)
        diff = (predictions - targets) ** 2
        masked_diff = diff * mask

        if self.reduction == 'mean':
            return masked_diff.sum() / mask.sum()
        elif self.reduction == 'sum':
            return masked_diff.sum()
        else:
            return masked_diff

    def compute_smoothness_loss(self,
                               predictions: torch.Tensor,
                               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute temporal smoothness loss

        Penalizes large changes between consecutive frames

        Args:
            predictions: (batch, time, 10) predictions
            mask: (batch, time) mask

        Returns:
            Smoothness loss
        """
        if predictions.size(1) <= 1:
            return torch.tensor(0.0, device=predictions.device)

        # Compute differences between consecutive frames
        diff = predictions[:, 1:, :] - predictions[:, :-1, :]  # (batch, time-1, 10)
        smoothness = (diff ** 2).mean(dim=-1)  # (batch, time-1)

        if mask is not None:
            # Mask for differences (both frames must be valid)
            diff_mask = mask[:, 1:] & mask[:, :-1]  # (batch, time-1)
            smoothness = smoothness * diff_mask

            if self.reduction == 'mean':
                return smoothness.sum() / diff_mask.sum()
            elif self.reduction == 'sum':
                return smoothness.sum()
        else:
            if self.reduction == 'mean':
                return smoothness.mean()
            elif self.reduction == 'sum':
                return smoothness.sum()

        return smoothness


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE loss for different parameter dimensions

    Allows different weights for different articulatory parameters
    (e.g., tongue parameters may be more important than lip parameters)
    """

    def __init__(self,
                 weights: Optional[torch.Tensor] = None,
                 reduction: str = 'mean'):
        """
        Args:
            weights: (10,) weight for each parameter dimension
            reduction: Loss reduction method
        """
        super().__init__()

        if weights is None:
            # Default: equal weights
            weights = torch.ones(10)

        self.register_buffer('weights', weights)
        self.reduction = reduction

    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute weighted MSE loss

        Args:
            predictions: (batch, time, 10) predictions
            targets: (batch, time, 10) targets
            mask: (batch, time) mask

        Returns:
            Weighted MSE loss
        """
        # Compute per-dimension squared error
        squared_error = (predictions - targets) ** 2  # (batch, time, 10)

        # Apply weights
        weighted_error = squared_error * self.weights  # (batch, time, 10)

        # Sum across parameter dimensions
        loss = weighted_error.sum(dim=-1)  # (batch, time)

        # Apply mask if provided
        if mask is not None:
            loss = loss * mask

            if self.reduction == 'mean':
                return loss.sum() / mask.sum()
            elif self.reduction == 'sum':
                return loss.sum()
        else:
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()

        return loss


class HuberLoss(nn.Module):
    """
    Huber loss for robust parameter prediction

    Less sensitive to outliers than MSE loss
    """

    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        """
        Args:
            delta: Threshold for switching between MSE and MAE
            reduction: Loss reduction method
        """
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute Huber loss

        Args:
            predictions: (batch, time, 10) predictions
            targets: (batch, time, 10) targets
            mask: (batch, time) mask

        Returns:
            Huber loss
        """
        error = predictions - targets
        abs_error = torch.abs(error)

        # Huber loss formula
        quadratic = torch.min(abs_error, torch.tensor(self.delta, device=error.device))
        linear = abs_error - quadratic

        loss = 0.5 * quadratic ** 2 + self.delta * linear
        loss = loss.mean(dim=-1)  # (batch, time)

        # Apply mask if provided
        if mask is not None:
            loss = loss * mask

            if self.reduction == 'mean':
                return loss.sum() / mask.sum()
            elif self.reduction == 'sum':
                return loss.sum()
        else:
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()

        return loss


def get_loss_function(loss_type: str = "articulatory",
                     **kwargs) -> nn.Module:
    """
    Factory function for loss functions

    Args:
        loss_type: Type of loss ("mse", "articulatory", "weighted", "huber")
        **kwargs: Additional arguments for loss function

    Returns:
        Loss function module
    """
    if loss_type == "mse":
        return nn.MSELoss(**kwargs)

    elif loss_type == "articulatory":
        return ArticulatoryLoss(**kwargs)

    elif loss_type == "weighted":
        return WeightedMSELoss(**kwargs)

    elif loss_type == "huber":
        return HuberLoss(**kwargs)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
