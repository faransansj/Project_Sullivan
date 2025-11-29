"""
Audio Feature Extraction Module

This module provides tools for extracting audio features synchronized with MRI frames.

Available extractors:
- MelSpectrogramExtractor: Mel-spectrogram features
- MFCCExtractor: MFCC features
"""

from .mel_spectrogram import MelSpectrogramExtractor
from .mfcc import MFCCExtractor

__all__ = [
    'MelSpectrogramExtractor',
    'MFCCExtractor',
]
