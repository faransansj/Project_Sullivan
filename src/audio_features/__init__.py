"""
Audio Feature Extraction Module

This module provides tools for extracting audio features synchronized with MRI frames.

Available extractors:
- MelSpectrogramExtractor: Mel-spectrogram features
- MFCCExtractor: MFCC features
- HuBERTExtractor: HuBERT self-supervised features
"""

from .mel_spectrogram import MelSpectrogramExtractor
from .mfcc import MFCCExtractor
from .hubert_extractor import HuBERTExtractor

__all__ = [
    'MelSpectrogramExtractor',
    'MFCCExtractor',
    'HuBERTExtractor',
]

