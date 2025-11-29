"""
Geometric Feature Extraction for Articulatory Parameters

This module extracts interpretable geometric features from vocal tract
segmentation masks, including tongue position, jaw opening, lip aperture, etc.
"""

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist
from typing import Dict, Tuple, Optional
import warnings


class GeometricFeatureExtractor:
    """
    Extract geometric articulatory features from segmentation masks.

    The extractor computes interpretable features for each articulator:
    - Tongue: area, centroid, tip position, dorsum height, curvature
    - Jaw: opening degree, vertical position
    - Lips: aperture, area, vertical position
    - Vocal tract: constriction degree and location

    Parameters
    ----------
    image_height : int, default=84
        Height of the segmentation mask
    image_width : int, default=84
        Width of the segmentation mask
    class_mapping : dict, optional
        Mapping of class names to class indices
        Default: {0: 'background', 1: 'tongue', 2: 'jaw', 3: 'lips'}
    normalize : bool, default=True
        Whether to normalize features to [0, 1] range
    """

    def __init__(
        self,
        image_height: int = 84,
        image_width: int = 84,
        class_mapping: Optional[Dict[int, str]] = None,
        normalize: bool = True
    ):
        self.image_height = image_height
        self.image_width = image_width
        self.normalize = normalize

        if class_mapping is None:
            self.class_mapping = {
                0: 'background',
                1: 'tongue',
                2: 'jaw',
                3: 'lips'
            }
        else:
            self.class_mapping = class_mapping

        # Feature names (computed dynamically based on what's extracted)
        self._feature_names = None

    def extract(self, segmentation: np.ndarray) -> Dict[str, float]:
        """
        Extract geometric features from a single segmentation mask.

        Parameters
        ----------
        segmentation : np.ndarray
            Segmentation mask of shape (height, width) with class indices

        Returns
        -------
        features : dict
            Dictionary of feature names to values
        """
        assert segmentation.shape == (self.image_height, self.image_width), \
            f"Expected shape ({self.image_height}, {self.image_width}), got {segmentation.shape}"

        features = {}

        # Extract features for each articulator
        features.update(self._extract_tongue_features(segmentation))
        features.update(self._extract_jaw_features(segmentation))
        features.update(self._extract_lip_features(segmentation))
        features.update(self._extract_constriction_features(segmentation))

        # Normalize if requested
        if self.normalize:
            features = self._normalize_features(features)

        return features

    def extract_batch(self, segmentations: np.ndarray) -> np.ndarray:
        """
        Extract features from a batch of segmentation masks.

        Parameters
        ----------
        segmentations : np.ndarray
            Batch of segmentation masks, shape (num_frames, height, width)

        Returns
        -------
        features : np.ndarray
            Feature array of shape (num_frames, num_features)
        """
        num_frames = segmentations.shape[0]
        feature_list = []

        for i in range(num_frames):
            frame_features = self.extract(segmentations[i])
            feature_list.append(list(frame_features.values()))

        # Cache feature names from first extraction
        if self._feature_names is None:
            first_features = self.extract(segmentations[0])
            self._feature_names = list(first_features.keys())

        return np.array(feature_list, dtype=np.float32)

    def _extract_tongue_features(self, segmentation: np.ndarray) -> Dict[str, float]:
        """Extract tongue-related features."""
        tongue_mask = (segmentation == 1).astype(np.uint8)
        features = {}

        # Area (normalized by image area)
        tongue_area = np.sum(tongue_mask)
        features['tongue_area'] = tongue_area / (self.image_height * self.image_width)

        if tongue_area == 0:
            # No tongue detected, return zeros
            features['tongue_centroid_x'] = 0.0
            features['tongue_centroid_y'] = 0.0
            features['tongue_tip_y'] = 0.0
            features['tongue_dorsum_height'] = 0.0
            features['tongue_width'] = 0.0
            return features

        # Centroid
        y_coords, x_coords = np.where(tongue_mask > 0)
        centroid_x = np.mean(x_coords) / self.image_width
        centroid_y = np.mean(y_coords) / self.image_height
        features['tongue_centroid_x'] = centroid_x
        features['tongue_centroid_y'] = centroid_y

        # Tongue tip (highest point, minimum y)
        tongue_tip_y = np.min(y_coords) / self.image_height
        features['tongue_tip_y'] = tongue_tip_y

        # Tongue dorsum height (average y of top 20% of tongue)
        top_20_threshold = np.percentile(y_coords, 20)
        dorsum_points = y_coords[y_coords <= top_20_threshold]
        if len(dorsum_points) > 0:
            dorsum_height = np.mean(dorsum_points) / self.image_height
        else:
            dorsum_height = tongue_tip_y
        features['tongue_dorsum_height'] = dorsum_height

        # Tongue width (horizontal extent)
        tongue_width = (np.max(x_coords) - np.min(x_coords)) / self.image_width
        features['tongue_width'] = tongue_width

        return features

    def _extract_jaw_features(self, segmentation: np.ndarray) -> Dict[str, float]:
        """Extract jaw-related features."""
        jaw_mask = (segmentation == 2).astype(np.uint8)
        features = {}

        jaw_area = np.sum(jaw_mask)
        features['jaw_area'] = jaw_area / (self.image_height * self.image_width)

        if jaw_area == 0:
            features['jaw_centroid_y'] = 0.0
            features['jaw_opening'] = 0.0
            return features

        # Centroid
        y_coords, x_coords = np.where(jaw_mask > 0)
        centroid_y = np.mean(y_coords) / self.image_height
        features['jaw_centroid_y'] = centroid_y

        # Jaw opening (vertical extent of jaw)
        jaw_opening = (np.max(y_coords) - np.min(y_coords)) / self.image_height
        features['jaw_opening'] = jaw_opening

        return features

    def _extract_lip_features(self, segmentation: np.ndarray) -> Dict[str, float]:
        """Extract lip-related features."""
        lip_mask = (segmentation == 3).astype(np.uint8)
        features = {}

        lip_area = np.sum(lip_mask)
        features['lip_area'] = lip_area / (self.image_height * self.image_width)

        if lip_area == 0:
            features['lip_centroid_y'] = 0.0
            features['lip_aperture'] = 0.0
            return features

        # Centroid
        y_coords, x_coords = np.where(lip_mask > 0)
        centroid_y = np.mean(y_coords) / self.image_height
        features['lip_centroid_y'] = centroid_y

        # Lip aperture (vertical extent - approximates lip opening)
        lip_aperture = (np.max(y_coords) - np.min(y_coords)) / self.image_height
        features['lip_aperture'] = lip_aperture

        return features

    def _extract_constriction_features(self, segmentation: np.ndarray) -> Dict[str, float]:
        """
        Extract vocal tract constriction features.

        Constriction is the narrowest point in the vocal tract,
        important for phoneme production.
        """
        features = {}

        # Create vocal tract mask (all non-background pixels)
        vocal_tract_mask = (segmentation > 0).astype(np.uint8)

        # For each row, compute the width of vocal tract
        row_widths = []
        row_positions = []

        for y in range(self.image_height):
            row = vocal_tract_mask[y, :]
            if np.any(row):
                width = np.sum(row)
                row_widths.append(width)
                row_positions.append(y)

        if len(row_widths) == 0:
            features['constriction_degree'] = 0.0
            features['constriction_location_y'] = 0.0
            return features

        # Constriction degree (minimum width, normalized)
        min_width = np.min(row_widths)
        constriction_degree = min_width / self.image_width
        features['constriction_degree'] = constriction_degree

        # Constriction location (y position of minimum width)
        min_idx = np.argmin(row_widths)
        constriction_y = row_positions[min_idx] / self.image_height
        features['constriction_location_y'] = constriction_y

        return features

    def _normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize features to [0, 1] range.

        Most features are already normalized by image dimensions,
        but this ensures all values are in valid range.
        """
        normalized = {}
        for key, value in features.items():
            # Clip to [0, 1] range
            normalized[key] = np.clip(value, 0.0, 1.0)
        return normalized

    @property
    def feature_names(self) -> list:
        """Get list of feature names in extraction order."""
        if self._feature_names is None:
            # Extract from a dummy segmentation to get names
            dummy_seg = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
            dummy_features = self.extract(dummy_seg)
            self._feature_names = list(dummy_features.keys())
        return self._feature_names

    @property
    def num_features(self) -> int:
        """Get total number of features."""
        return len(self.feature_names)

    def __repr__(self) -> str:
        return (f"GeometricFeatureExtractor(image_size=({self.image_height}, {self.image_width}), "
                f"num_features={self.num_features}, normalize={self.normalize})")
