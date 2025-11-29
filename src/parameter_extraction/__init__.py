"""
Articulatory Parameter Extraction Module

This module provides tools for extracting low-dimensional articulatory parameters
from vocal tract segmentation masks.

Available extractors:
- GeometricFeatureExtractor: Hand-crafted geometric features
- PCAFeatureExtractor: PCA-based dimensionality reduction
"""

from .geometric_features import GeometricFeatureExtractor
from .pca_features import PCAFeatureExtractor

__all__ = [
    'GeometricFeatureExtractor',
    'PCAFeatureExtractor',
]
