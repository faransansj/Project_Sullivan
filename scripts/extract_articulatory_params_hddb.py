#!/usr/bin/env python3
"""
HDDB Articulatory Parameter Extraction
========================================

Extract articulatory parameters from binary segmentation masks.

Features:
- 14 geometric features (airway area, centroid, bounding box, etc.)
- 10 PCA components from flattened masks

Author: Claude & Research Assistant
Date: 2026-01-11
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime
from sklearn.decomposition import PCA
from scipy import ndimage
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger


class ArtParamExtractor:
    """Extract articulatory parameters from binary segmentation masks"""

    def __init__(self, output_dir: str):
        """
        Initialize extractor

        Args:
            output_dir: Output directory for parameters
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger("ArtParamExtractor")

        # PCA model (will be fitted on all training data)
        self.pca = None

    def extract_geometric_features(self, mask: np.ndarray) -> np.ndarray:
        """
        Extract 14 geometric features from binary mask

        Args:
            mask: (H, W) binary mask (0=background, 1=airway)

        Returns:
            features: (14,) array of geometric features
        """
        h, w = mask.shape

        # Find airway contour
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            # No airway detected - return zeros
            return np.zeros(14, dtype=np.float32)

        # Use largest contour (main airway)
        contour = max(contours, key=cv2.contourArea)

        # 1. Airway area (normalized by image size)
        area = cv2.contourArea(contour) / (h * w)

        # 2-3. Centroid (normalized)
        M = cv2.moments(contour)
        if M['m00'] > 0:
            cx = M['m10'] / M['m00'] / w
            cy = M['m01'] / M['m00'] / h
        else:
            cx, cy = 0.5, 0.5

        # 4-7. Bounding box (normalized)
        x, y, bbox_w, bbox_h = cv2.boundingRect(contour)
        bbox_top = y / h
        bbox_bottom = (y + bbox_h) / h
        bbox_left = x / w
        bbox_right = (x + bbox_w) / w

        # 8. Aspect ratio
        aspect_ratio = bbox_w / (bbox_h + 1e-8)

        # 9. Solidity (area / convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area * (h * w) / (hull_area + 1e-8)

        # 10. Extent (area / bounding box area)
        bbox_area = bbox_w * bbox_h
        extent = area * (h * w) / (bbox_area + 1e-8)

        # 11. Perimeter (normalized)
        perimeter = cv2.arcLength(contour, True) / (2 * (h + w))

        # 12. Circularity (4π * area / perimeter²)
        circularity = 4 * np.pi * area * (h * w) / (perimeter**2 * (2 * (h + w))**2 + 1e-8)

        # 13-14. Ellipse fit (major/minor axis ratio, angle)
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (ex, ey), (ma, Ma), angle = ellipse
            ellipse_ratio = Ma / (ma + 1e-8)
            ellipse_angle = angle / 180.0  # Normalize to [0, 1]
        else:
            ellipse_ratio = 1.0
            ellipse_angle = 0.0

        features = np.array([
            area,              # 1. Airway area
            cx, cy,            # 2-3. Centroid
            bbox_top,          # 4. Bounding box top
            bbox_bottom,       # 5. Bounding box bottom
            bbox_left,         # 6. Bounding box left
            bbox_right,        # 7. Bounding box right
            aspect_ratio,      # 8. Aspect ratio
            solidity,          # 9. Solidity
            extent,            # 10. Extent
            perimeter,         # 11. Perimeter
            circularity,       # 12. Circularity
            ellipse_ratio,     # 13. Ellipse major/minor ratio
            ellipse_angle      # 14. Ellipse angle
        ], dtype=np.float32)

        return features

    def fit_pca(self, all_masks: list):
        """
        Fit PCA model on all training masks

        Args:
            all_masks: List of (T, H, W) mask arrays
        """
        self.logger.info("Fitting PCA model on all masks...")

        # Flatten all masks
        flattened = []
        for masks in tqdm(all_masks, desc="Flattening masks"):
            for mask in masks:
                flattened.append(mask.flatten())

        flattened = np.stack(flattened, axis=0)  # (N, H*W)

        # Fit PCA
        self.pca = PCA(n_components=10)
        self.pca.fit(flattened)

        explained_var = self.pca.explained_variance_ratio_.sum()
        self.logger.info(f"PCA fitted. Explained variance: {explained_var:.4f}")

    def extract_pca_features(self, mask: np.ndarray) -> np.ndarray:
        """
        Extract 10 PCA features from binary mask

        Args:
            mask: (H, W) binary mask

        Returns:
            features: (10,) PCA components
        """
        if self.pca is None:
            raise ValueError("PCA model not fitted. Call fit_pca() first.")

        flattened = mask.flatten().reshape(1, -1)
        pca_features = self.pca.transform(flattened)[0]

        return pca_features.astype(np.float32)

    def process_utterance(self, seg_path: Path, use_pca: bool = True) -> dict:
        """
        Process single utterance segmentation

        Args:
            seg_path: Path to segmentation NPZ file
            use_pca: Whether to extract PCA features

        Returns:
            stats: Processing statistics
        """
        # Load segmentation
        data = np.load(seg_path)
        masks = data['segmentations']  # (T, H, W)
        utterance_name = str(data['utterance_name'])

        num_frames = masks.shape[0]

        # Extract geometric features for each frame
        geometric_features = []
        for i in range(num_frames):
            feats = self.extract_geometric_features(masks[i])
            geometric_features.append(feats)

        geometric_features = np.stack(geometric_features, axis=0)  # (T, 14)

        # Extract PCA features if requested
        if use_pca and self.pca is not None:
            pca_features = []
            for i in range(num_frames):
                feats = self.extract_pca_features(masks[i])
                pca_features.append(feats)
            pca_features = np.stack(pca_features, axis=0)  # (T, 10)

            # Combine geometric + PCA
            all_features = np.concatenate([geometric_features, pca_features], axis=1)  # (T, 24)
        else:
            all_features = geometric_features  # (T, 14)

        # Save parameters
        output_path = self.output_dir / f"{utterance_name}_params.npz"
        np.savez_compressed(
            output_path,
            parameters=all_features,
            geometric_features=geometric_features,
            pca_features=pca_features if use_pca and self.pca is not None else None,
            utterance_name=utterance_name,
            num_frames=num_frames,
            feature_names=['area', 'cx', 'cy', 'bbox_top', 'bbox_bottom',
                          'bbox_left', 'bbox_right', 'aspect_ratio', 'solidity',
                          'extent', 'perimeter', 'circularity', 'ellipse_ratio',
                          'ellipse_angle']
        )

        stats = {
            'utterance_name': utterance_name,
            'num_frames': num_frames,
            'output_path': str(output_path),
            'num_features': all_features.shape[1]
        }

        return stats

    def process_dataset(self, segmentation_dir: Path, two_pass: bool = True):
        """
        Process entire dataset

        Args:
            segmentation_dir: Directory with segmentation NPZ files
            two_pass: If True, fit PCA on first pass, extract on second pass
        """
        self.logger.info(f"Processing segmentations from {segmentation_dir}")

        # Find all segmentation files
        seg_files = sorted(segmentation_dir.glob("*/*_segmentations.npz"))

        if len(seg_files) == 0:
            self.logger.error(f"No segmentation files found in {segmentation_dir}")
            return

        self.logger.info(f"Found {len(seg_files)} utterances")

        if two_pass:
            # First pass: Fit PCA
            self.logger.info("First pass: Fitting PCA model...")
            all_masks = []
            for seg_path in tqdm(seg_files, desc="Loading masks"):
                data = np.load(seg_path)
                all_masks.append(data['segmentations'])

            self.fit_pca(all_masks)

            # Save PCA model
            pca_path = self.output_dir / 'pca_model.npz'
            np.savez(
                pca_path,
                components=self.pca.components_,
                mean=self.pca.mean_,
                explained_variance=self.pca.explained_variance_,
                explained_variance_ratio=self.pca.explained_variance_ratio_
            )
            self.logger.info(f"PCA model saved to {pca_path}")

            # Second pass: Extract features
            self.logger.info("Second pass: Extracting features...")

        # Extract features
        all_stats = []
        total_frames = 0

        for seg_path in tqdm(seg_files, desc="Extracting parameters"):
            try:
                stats = self.process_utterance(seg_path, use_pca=two_pass)
                all_stats.append(stats)
                total_frames += stats['num_frames']
            except Exception as e:
                self.logger.error(f"Failed to process {seg_path.name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_utterances': len(all_stats),
            'total_frames': total_frames,
            'num_features': all_stats[0]['num_features'] if all_stats else 0,
            'two_pass': two_pass,
            'pca_fitted': self.pca is not None,
            'output_dir': str(self.output_dir),
            'utterance_stats': all_stats
        }

        summary_path = self.output_dir / 'parameter_extraction_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Parameter extraction complete!")
        self.logger.info(f"  Utterances processed: {len(all_stats)}")
        self.logger.info(f"  Total frames: {total_frames}")
        self.logger.info(f"  Features per frame: {all_stats[0]['num_features'] if all_stats else 0}")
        self.logger.info(f"  Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract articulatory parameters from HDDB segmentations")
    parser.add_argument(
        '--segmentation-dir',
        type=str,
        default='data/processed/segmentations',
        help='Directory with segmentation NPZ files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed/parameters',
        help='Output directory for parameters'
    )
    parser.add_argument(
        '--no-pca',
        action='store_true',
        help='Skip PCA feature extraction (geometric only)'
    )

    args = parser.parse_args()

    # Create extractor
    extractor = ArtParamExtractor(output_dir=args.output_dir)

    # Process dataset
    extractor.process_dataset(
        segmentation_dir=Path(args.segmentation_dir),
        two_pass=not args.no_pca
    )


if __name__ == '__main__':
    main()
