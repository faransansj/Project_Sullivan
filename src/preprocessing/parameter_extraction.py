"""
Articulatory Parameter Extraction from Segmentation Masks

This module extracts low-dimensional articulatory parameters from
vocal tract segmentation masks for use in Phase 2 audio-to-parameter modeling.

Classes:
    0: Background/Air
    1: Tongue
    2: Jaw/Palate
    3: Lips
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import ndimage
from scipy.interpolate import interp1d
from skimage import measure
import cv2


@dataclass
class ArticulatoryParameters:
    """Container for articulatory parameters extracted from a single frame"""

    # Tongue parameters (5 dimensions)
    tongue_area: float  # Normalized tongue area
    tongue_centroid_x: float  # Horizontal position (normalized)
    tongue_centroid_y: float  # Vertical position (normalized)
    tongue_tip_y: float  # Tongue tip vertical position
    tongue_curvature: float  # Mean curvature of tongue surface

    # Jaw parameters (2 dimensions)
    jaw_height: float  # Vertical jaw opening
    jaw_angle: float  # Jaw angle relative to palate

    # Lip parameters (2 dimensions)
    lip_aperture: float  # Vertical lip opening
    lip_protrusion: float  # Horizontal lip extension

    # Constriction parameters (1 dimension)
    constriction_degree: float  # Narrowest point in vocal tract

    # Metadata
    frame_index: int
    utterance_name: str

    def to_vector(self) -> np.ndarray:
        """Convert to 10-dimensional vector for modeling"""
        return np.array([
            self.tongue_area,
            self.tongue_centroid_x,
            self.tongue_centroid_y,
            self.tongue_tip_y,
            self.tongue_curvature,
            self.jaw_height,
            self.jaw_angle,
            self.lip_aperture,
            self.lip_protrusion,
            self.constriction_degree
        ], dtype=np.float32)

    @classmethod
    def from_vector(cls, vec: np.ndarray, frame_index: int = 0,
                    utterance_name: str = "") -> 'ArticulatoryParameters':
        """Create from 10-dimensional vector"""
        assert len(vec) == 10, f"Expected 10-dim vector, got {len(vec)}"
        return cls(
            tongue_area=vec[0],
            tongue_centroid_x=vec[1],
            tongue_centroid_y=vec[2],
            tongue_tip_y=vec[3],
            tongue_curvature=vec[4],
            jaw_height=vec[5],
            jaw_angle=vec[6],
            lip_aperture=vec[7],
            lip_protrusion=vec[8],
            constriction_degree=vec[9],
            frame_index=frame_index,
            utterance_name=utterance_name
        )


class ParameterExtractor:
    """Extract articulatory parameters from segmentation masks"""

    def __init__(self,
                 image_height: int = 84,
                 image_width: int = 84,
                 normalize: bool = True):
        """
        Args:
            image_height: Height of segmentation mask
            image_width: Width of segmentation mask
            normalize: Whether to normalize parameters to [0, 1]
        """
        self.H = image_height
        self.W = image_width
        self.normalize = normalize

        # Normalization statistics (will be updated during fitting)
        self.param_mean = None
        self.param_std = None

    def extract_from_mask(self,
                         mask: np.ndarray,
                         frame_index: int = 0,
                         utterance_name: str = "") -> ArticulatoryParameters:
        """
        Extract parameters from a single segmentation mask

        Args:
            mask: (H, W) segmentation mask with class labels
            frame_index: Frame number in sequence
            utterance_name: Name of utterance

        Returns:
            ArticulatoryParameters object
        """
        assert mask.shape == (self.H, self.W), \
            f"Expected mask shape ({self.H}, {self.W}), got {mask.shape}"

        # Extract individual class masks
        tongue_mask = (mask == 1).astype(np.uint8)
        jaw_mask = (mask == 2).astype(np.uint8)
        lip_mask = (mask == 3).astype(np.uint8)
        air_mask = (mask == 0).astype(np.uint8)

        # Extract tongue parameters
        tongue_params = self._extract_tongue_parameters(tongue_mask)

        # Extract jaw parameters
        jaw_params = self._extract_jaw_parameters(jaw_mask, tongue_mask)

        # Extract lip parameters
        lip_params = self._extract_lip_parameters(lip_mask)

        # Extract constriction parameters
        constriction = self._extract_constriction(air_mask, tongue_mask)

        return ArticulatoryParameters(
            tongue_area=tongue_params['area'],
            tongue_centroid_x=tongue_params['centroid_x'],
            tongue_centroid_y=tongue_params['centroid_y'],
            tongue_tip_y=tongue_params['tip_y'],
            tongue_curvature=tongue_params['curvature'],
            jaw_height=jaw_params['height'],
            jaw_angle=jaw_params['angle'],
            lip_aperture=lip_params['aperture'],
            lip_protrusion=lip_params['protrusion'],
            constriction_degree=constriction,
            frame_index=frame_index,
            utterance_name=utterance_name
        )

    def _extract_tongue_parameters(self, tongue_mask: np.ndarray) -> Dict[str, float]:
        """Extract tongue-related parameters"""
        # Area (normalized by image size)
        area = np.sum(tongue_mask) / (self.H * self.W)

        # Centroid
        if np.sum(tongue_mask) > 0:
            moments = cv2.moments(tongue_mask)
            if moments['m00'] > 0:
                cx = moments['m10'] / moments['m00'] / self.W  # Normalize
                cy = moments['m01'] / moments['m00'] / self.H
            else:
                cx, cy = 0.5, 0.5
        else:
            cx, cy = 0.5, 0.5

        # Tongue tip (uppermost point of tongue)
        if np.sum(tongue_mask) > 0:
            tongue_points = np.argwhere(tongue_mask > 0)
            tip_y = np.min(tongue_points[:, 0]) / self.H  # Normalized
        else:
            tip_y = 0.5

        # Curvature (measure of tongue body curvature)
        curvature = self._compute_curvature(tongue_mask)

        return {
            'area': area,
            'centroid_x': cx,
            'centroid_y': cy,
            'tip_y': tip_y,
            'curvature': curvature
        }

    def _extract_jaw_parameters(self,
                                jaw_mask: np.ndarray,
                                tongue_mask: np.ndarray) -> Dict[str, float]:
        """Extract jaw-related parameters"""
        # Jaw height (vertical distance between jaw and tongue)
        if np.sum(jaw_mask) > 0 and np.sum(tongue_mask) > 0:
            jaw_points = np.argwhere(jaw_mask > 0)
            tongue_points = np.argwhere(tongue_mask > 0)

            # Find lowest jaw point and highest tongue point
            jaw_lowest = np.max(jaw_points[:, 0])
            tongue_highest = np.min(tongue_points[:, 0])

            height = (jaw_lowest - tongue_highest) / self.H
        else:
            height = 0.5

        # Jaw angle (approximation using jaw contour slope)
        angle = self._compute_jaw_angle(jaw_mask)

        return {
            'height': max(0.0, height),  # Ensure non-negative
            'angle': angle
        }

    def _extract_lip_parameters(self, lip_mask: np.ndarray) -> Dict[str, float]:
        """Extract lip-related parameters"""
        if np.sum(lip_mask) > 0:
            lip_points = np.argwhere(lip_mask > 0)

            # Lip aperture (vertical extent)
            aperture = (np.max(lip_points[:, 0]) - np.min(lip_points[:, 0])) / self.H

            # Lip protrusion (horizontal extent from right edge)
            # Assuming lips are on the left side of the image
            protrusion = (self.W - np.min(lip_points[:, 1])) / self.W
        else:
            aperture = 0.0
            protrusion = 0.0

        return {
            'aperture': aperture,
            'protrusion': protrusion
        }

    def _extract_constriction(self,
                             air_mask: np.ndarray,
                             tongue_mask: np.ndarray) -> float:
        """
        Extract constriction degree (narrowest point in vocal tract)

        Returns:
            Normalized constriction degree (0 = fully closed, 1 = fully open)
        """
        if np.sum(air_mask) == 0:
            return 0.0

        # Find narrowest horizontal slice in the vocal tract
        min_width = self.W
        for row in range(self.H):
            air_in_row = np.sum(air_mask[row, :])
            if air_in_row > 0:
                min_width = min(min_width, air_in_row)

        constriction = min_width / self.W
        return constriction

    def _compute_curvature(self, mask: np.ndarray) -> float:
        """
        Compute mean curvature of mask contour

        Returns:
            Mean absolute curvature (normalized)
        """
        if np.sum(mask) == 0:
            return 0.0

        # Find contours
        contours = measure.find_contours(mask, 0.5)
        if len(contours) == 0:
            return 0.0

        # Use largest contour
        contour = max(contours, key=len)

        if len(contour) < 5:
            return 0.0

        # Compute curvature using finite differences
        # k = |dx*d2y - dy*d2x| / (dx^2 + dy^2)^(3/2)
        dx = np.gradient(contour[:, 1])
        dy = np.gradient(contour[:, 0])
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)

        curvature = np.abs(dx * d2y - dy * d2x) / (dx**2 + dy**2)**(3/2)
        curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)

        return np.mean(curvature)

    def _compute_jaw_angle(self, jaw_mask: np.ndarray) -> float:
        """
        Compute jaw angle relative to horizontal

        Returns:
            Angle in radians, normalized to [0, 1]
        """
        if np.sum(jaw_mask) == 0:
            return 0.5

        # Find jaw contour
        contours = measure.find_contours(jaw_mask, 0.5)
        if len(contours) == 0:
            return 0.5

        contour = max(contours, key=len)

        if len(contour) < 10:
            return 0.5

        # Fit a line to the lower jaw contour (lower half of points)
        sorted_points = contour[contour[:, 0].argsort()]
        lower_half = sorted_points[len(sorted_points)//2:]

        if len(lower_half) < 2:
            return 0.5

        # Linear regression
        x = lower_half[:, 1]
        y = lower_half[:, 0]

        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            angle = np.arctan(slope)
            # Normalize to [0, 1]
            normalized_angle = (angle + np.pi/2) / np.pi
            return np.clip(normalized_angle, 0.0, 1.0)
        else:
            return 0.5

    def extract_from_sequence(self,
                             masks: np.ndarray,
                             utterance_name: str = "") -> np.ndarray:
        """
        Extract parameters from a sequence of masks

        Args:
            masks: (num_frames, H, W) array of segmentation masks
            utterance_name: Name of utterance

        Returns:
            (num_frames, 10) array of articulatory parameters
        """
        num_frames = len(masks)
        parameters = np.zeros((num_frames, 10), dtype=np.float32)

        for i, mask in enumerate(masks):
            params = self.extract_from_mask(mask, frame_index=i,
                                           utterance_name=utterance_name)
            parameters[i] = params.to_vector()

        return parameters

    def fit_normalization(self, all_parameters: np.ndarray):
        """
        Fit normalization statistics from all training data

        Args:
            all_parameters: (num_samples, 10) array of parameters from all frames
        """
        self.param_mean = np.mean(all_parameters, axis=0)
        self.param_std = np.std(all_parameters, axis=0)

        # Prevent division by zero
        self.param_std = np.where(self.param_std < 1e-6, 1.0, self.param_std)

    def normalize_parameters(self, parameters: np.ndarray) -> np.ndarray:
        """
        Normalize parameters using fitted statistics

        Args:
            parameters: (num_frames, 10) array of parameters

        Returns:
            Normalized parameters
        """
        if self.param_mean is None or self.param_std is None:
            raise ValueError("Must call fit_normalization() first")

        return (parameters - self.param_mean) / self.param_std

    def denormalize_parameters(self, normalized_parameters: np.ndarray) -> np.ndarray:
        """
        Denormalize parameters back to original scale

        Args:
            normalized_parameters: (num_frames, 10) normalized parameters

        Returns:
            Original scale parameters
        """
        if self.param_mean is None or self.param_std is None:
            raise ValueError("Must call fit_normalization() first")

        return normalized_parameters * self.param_std + self.param_mean


def validate_parameters(parameters: np.ndarray,
                       expected_range: Tuple[float, float] = (0.0, 1.0)) -> Dict[str, any]:
    """
    Validate extracted parameters for quality assurance

    Args:
        parameters: (num_frames, 10) array of parameters
        expected_range: Expected value range for unnormalized parameters

    Returns:
        Dictionary with validation results
    """
    results = {
        'valid': True,
        'warnings': [],
        'statistics': {}
    }

    # Check for NaN or Inf
    if np.any(np.isnan(parameters)) or np.any(np.isinf(parameters)):
        results['valid'] = False
        results['warnings'].append("Contains NaN or Inf values")

    # Check value ranges (for unnormalized parameters)
    param_min = np.min(parameters, axis=0)
    param_max = np.max(parameters, axis=0)

    if np.any(param_min < expected_range[0] - 0.1) or \
       np.any(param_max > expected_range[1] + 0.1):
        results['warnings'].append(
            f"Some parameters outside expected range {expected_range}"
        )

    # Check temporal smoothness
    if len(parameters) > 1:
        diffs = np.diff(parameters, axis=0)
        mean_change = np.mean(np.abs(diffs), axis=0)

        # Flag if average change per frame is too large
        if np.any(mean_change > 0.3):
            results['warnings'].append(
                "High temporal variability detected (possible segmentation errors)"
            )

        results['statistics']['mean_temporal_change'] = mean_change

    # Store basic statistics
    results['statistics']['mean'] = np.mean(parameters, axis=0)
    results['statistics']['std'] = np.std(parameters, axis=0)
    results['statistics']['min'] = param_min
    results['statistics']['max'] = param_max

    return results
