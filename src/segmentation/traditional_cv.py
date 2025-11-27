"""
Traditional computer vision methods for vocal tract segmentation.

This module implements classical CV techniques to generate pseudo-labels
for training a U-Net from scratch on USC-TIMIT data.

Methods:
- Otsu thresholding: Separate tissue from air
- GrabCut: Interactive foreground/background separation
- Active Contours (Snakes): Refine boundaries
- Watershed: Segment connected regions
"""

import warnings
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage
from skimage import filters, morphology, segmentation
from skimage.segmentation import active_contour

from ..utils.logger import get_logger

logger = get_logger(__name__)


class VocalTractSegmenter:
    """
    Traditional CV-based segmentation for rtMRI vocal tract images.

    This class provides multiple segmentation algorithms that can be
    combined to generate pseudo-labels for U-Net training.
    """

    def __init__(self,
                 tissue_air_threshold: Optional[float] = None,
                 min_tissue_area: int = 50,
                 gaussian_sigma: float = 1.0):
        """
        Initialize vocal tract segmenter.

        Args:
            tissue_air_threshold: Threshold for tissue/air separation (auto if None)
            min_tissue_area: Minimum area for tissue regions (pixels)
            gaussian_sigma: Sigma for Gaussian smoothing
        """
        self.tissue_air_threshold = tissue_air_threshold
        self.min_tissue_area = min_tissue_area
        self.gaussian_sigma = gaussian_sigma

    def segment_tissue_air(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Segment tissue from air using Otsu's thresholding.

        Args:
            frame: MRI frame (H, W), normalized [0, 1]

        Returns:
            Tuple of (binary_mask, threshold_value)
            - binary_mask: (H, W) bool array, True=tissue, False=air
            - threshold_value: Computed Otsu threshold
        """
        # Ensure input is in [0, 1] range
        if frame.max() > 1.0:
            frame = frame / frame.max()

        # Apply Gaussian smoothing to reduce noise
        if self.gaussian_sigma > 0:
            frame_smooth = ndimage.gaussian_filter(frame, sigma=self.gaussian_sigma)
        else:
            frame_smooth = frame

        # Compute Otsu threshold
        if self.tissue_air_threshold is None:
            try:
                threshold = filters.threshold_otsu(frame_smooth)
            except ValueError:
                # Fallback if image has uniform intensity
                threshold = 0.5
                logger.warning("Otsu thresholding failed, using default threshold 0.5")
        else:
            threshold = self.tissue_air_threshold

        # Create binary mask
        tissue_mask = frame_smooth > threshold

        # Remove small regions
        if self.min_tissue_area > 0:
            tissue_mask = morphology.remove_small_objects(
                tissue_mask,
                min_size=self.min_tissue_area
            )

        return tissue_mask, threshold

    def segment_multilevel_otsu(self, frame: np.ndarray,
                                n_classes: int = 3) -> np.ndarray:
        """
        Multi-level Otsu thresholding for multiple tissue types.

        Args:
            frame: MRI frame (H, W), normalized [0, 1]
            n_classes: Number of classes to segment (2-5)

        Returns:
            Segmentation map (H, W) with values [0, n_classes-1]
        """
        # Ensure input is in [0, 1] range
        if frame.max() > 1.0:
            frame = frame / frame.max()

        # Apply Gaussian smoothing
        if self.gaussian_sigma > 0:
            frame_smooth = ndimage.gaussian_filter(frame, sigma=self.gaussian_sigma)
        else:
            frame_smooth = frame

        # Compute multi-level thresholds
        try:
            thresholds = filters.threshold_multiotsu(frame_smooth, classes=n_classes)
        except ValueError:
            # Fallback to uniform bins
            logger.warning(f"Multi-Otsu failed, using uniform bins for {n_classes} classes")
            thresholds = np.linspace(frame_smooth.min(), frame_smooth.max(), n_classes + 1)[1:-1]

        # Create segmentation map
        seg_map = np.digitize(frame_smooth, bins=thresholds)

        return seg_map

    def segment_grabcut(self,
                       frame: np.ndarray,
                       rect: Optional[Tuple[int, int, int, int]] = None,
                       mask: Optional[np.ndarray] = None,
                       n_iter: int = 5) -> np.ndarray:
        """
        GrabCut segmentation for foreground/background separation.

        Args:
            frame: MRI frame (H, W), normalized [0, 1]
            rect: Initial rectangle (x, y, w, h) for foreground region
                  If None, uses central 60% of image
            mask: Initial mask (H, W) with values:
                  0=background, 1=foreground, 2=probable_bg, 3=probable_fg
            n_iter: Number of GrabCut iterations

        Returns:
            Binary mask (H, W) where True=foreground, False=background
        """
        # Convert to uint8 for OpenCV
        if frame.max() <= 1.0:
            frame_uint8 = (frame * 255).astype(np.uint8)
        else:
            frame_uint8 = frame.astype(np.uint8)

        # Convert grayscale to BGR for GrabCut
        frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2BGR)

        # Initialize mask
        h, w = frame.shape
        if mask is None:
            mask = np.zeros((h, w), dtype=np.uint8)

            # If no rect provided, use central 60% of image
            if rect is None:
                margin_h = int(h * 0.2)
                margin_w = int(w * 0.2)
                rect = (margin_w, margin_h, w - 2*margin_w, h - 2*margin_h)

        # GrabCut models
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        # Run GrabCut
        try:
            if rect is not None:
                cv2.grabCut(frame_bgr, mask, rect, bgd_model, fgd_model,
                           n_iter, cv2.GC_INIT_WITH_RECT)
            else:
                cv2.grabCut(frame_bgr, mask, None, bgd_model, fgd_model,
                           n_iter, cv2.GC_INIT_WITH_MASK)
        except cv2.error as e:
            logger.error(f"GrabCut failed: {e}")
            # Return central region as fallback
            fallback_mask = np.zeros((h, w), dtype=bool)
            margin_h = int(h * 0.2)
            margin_w = int(w * 0.2)
            fallback_mask[margin_h:h-margin_h, margin_w:w-margin_w] = True
            return fallback_mask

        # Convert mask to binary (0,2=background, 1,3=foreground)
        binary_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                               True, False)

        return binary_mask

    def segment_active_contour(self,
                              frame: np.ndarray,
                              init_contour: np.ndarray,
                              alpha: float = 0.015,
                              beta: float = 10.0,
                              gamma: float = 0.001,
                              n_iter: int = 100) -> np.ndarray:
        """
        Active contour (snake) segmentation.

        Args:
            frame: MRI frame (H, W), normalized [0, 1]
            init_contour: Initial contour points (N, 2) with (x, y) coordinates
            alpha: Snake length shape parameter (continuity)
            beta: Snake smoothness shape parameter (curvature)
            gamma: Time step (explicit numerical scheme)
            n_iter: Number of iterations

        Returns:
            Refined contour points (N, 2)
        """
        # Ensure input is in [0, 1] range
        if frame.max() > 1.0:
            frame = frame / frame.max()

        # Apply Gaussian smoothing
        if self.gaussian_sigma > 0:
            frame_smooth = ndimage.gaussian_filter(frame, sigma=self.gaussian_sigma)
        else:
            frame_smooth = frame

        # Run active contour
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            try:
                refined_contour = active_contour(
                    frame_smooth,
                    init_contour,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    max_iterations=n_iter,
                    coordinates='rc'  # row-column format
                )
            except Exception as e:
                logger.warning(f"Active contour failed: {e}, returning initial contour")
                refined_contour = init_contour

        return refined_contour

    def segment_watershed(self,
                         frame: np.ndarray,
                         markers: Optional[np.ndarray] = None,
                         compactness: float = 0.001) -> np.ndarray:
        """
        Watershed segmentation for separating touching regions.

        Args:
            frame: MRI frame (H, W), normalized [0, 1]
            markers: Initial markers (H, W) with labeled regions
                     If None, uses distance transform of tissue mask
            compactness: Compactness parameter for watershed

        Returns:
            Labeled segmentation (H, W) with integer region labels
        """
        # Ensure input is in [0, 1] range
        if frame.max() > 1.0:
            frame = frame / frame.max()

        # Apply Gaussian smoothing
        if self.gaussian_sigma > 0:
            frame_smooth = ndimage.gaussian_filter(frame, sigma=self.gaussian_sigma)
        else:
            frame_smooth = frame

        # If no markers provided, create from tissue/air segmentation
        if markers is None:
            tissue_mask, _ = self.segment_tissue_air(frame)

            # Distance transform
            distance = ndimage.distance_transform_edt(tissue_mask)

            # Find local maxima as markers
            from scipy.ndimage import maximum_filter
            local_max = (distance == maximum_filter(distance, size=10))
            markers = ndimage.label(local_max)[0]

        # Compute gradient for watershed
        gradient = filters.sobel(frame_smooth)

        # Run watershed
        labels = segmentation.watershed(
            gradient,
            markers=markers,
            compactness=compactness
        )

        return labels

    def create_vocal_tract_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Create a mask of the vocal tract region (excludes external background).

        This uses morphological operations to identify the air-filled
        vocal tract region and surrounding tissue.

        Args:
            frame: MRI frame (H, W), normalized [0, 1]

        Returns:
            Binary mask (H, W) where True=vocal tract region, False=external
        """
        # Get tissue mask
        tissue_mask, _ = self.segment_tissue_air(frame)

        # Fill holes to include air spaces within tissue
        filled = ndimage.binary_fill_holes(tissue_mask)

        # Morphological closing to smooth boundaries
        filled = morphology.binary_closing(filled, morphology.disk(3))

        # Keep only largest connected component (main vocal tract region)
        labeled = morphology.label(filled)
        if labeled.max() == 0:
            # No regions found, return full image
            return np.ones_like(frame, dtype=bool)

        # Find largest region
        region_sizes = np.bincount(labeled.ravel())
        region_sizes[0] = 0  # Exclude background
        largest_region = region_sizes.argmax()

        vocal_tract_mask = (labeled == largest_region)

        return vocal_tract_mask


def create_initial_contour(center: Tuple[int, int],
                          radius: int,
                          n_points: int = 100) -> np.ndarray:
    """
    Create circular initial contour for active contour segmentation.

    Args:
        center: (row, col) center of circle
        radius: Radius of circle in pixels
        n_points: Number of points in contour

    Returns:
        Contour points (n_points, 2) in (row, col) format
    """
    theta = np.linspace(0, 2 * np.pi, n_points)
    r, c = center

    contour = np.array([
        r + radius * np.cos(theta),
        c + radius * np.sin(theta)
    ]).T

    return contour


def estimate_tongue_region(frame: np.ndarray) -> Tuple[int, int, int]:
    """
    Estimate tongue region location for initial segmentation.

    For midsagittal rtMRI, the tongue is typically in the lower-center
    portion of the image.

    Args:
        frame: MRI frame (H, W)

    Returns:
        Tuple of (center_row, center_col, radius)
    """
    h, w = frame.shape

    # Tongue is typically in lower-center region
    # Approximate location: 60% down, centered horizontally
    center_row = int(h * 0.6)
    center_col = w // 2

    # Radius approximately 1/4 of image height
    radius = int(h * 0.25)

    return center_row, center_col, radius
