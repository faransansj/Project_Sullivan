"""
Modeling Module for Acoustic-to-Articulatory Inference

This module contains neural network models and dataset utilities for training
and inference of articulatory parameters from audio.
"""

from .dataset import ArticulatoryDataset, create_dataloaders
from .baseline_lstm import BaselineLSTM

__version__ = "0.2.0"

__all__ = [
    'ArticulatoryDataset',
    'create_dataloaders',
    'BaselineLSTM',
]