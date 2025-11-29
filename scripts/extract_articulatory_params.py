#!/usr/bin/env python3
"""
Extract Articulatory Parameters from Segmentation Masks

This script extracts both geometric and PCA-based articulatory parameters
from vocal tract segmentation masks.

Usage:
    python scripts/extract_articulatory_params.py --method geometric
    python scripts/extract_articulatory_params.py --method pca --n-components 10
    python scripts/extract_articulatory_params.py --method both
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import time

import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from parameter_extraction.geometric_features import GeometricFeatureExtractor
from parameter_extraction.pca_features import PCAFeatureExtractor
from utils.logger import setup_logger


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


class ArticulatoryParameterExtractor:
    """Main class for extracting articulatory parameters."""

    def __init__(
        self,
        segmentation_dir: str,
        output_dir: str,
        method: str = 'geometric',
        n_pca_components: int = 10,
        logger: logging.Logger = None
    ):
        self.segmentation_dir = Path(segmentation_dir)
        self.output_dir = Path(output_dir)
        self.method = method
        self.n_pca_components = n_pca_components
        self.logger = logger or logging.getLogger(__name__)

        # Initialize extractors
        if method in ['geometric', 'both']:
            self.geometric_extractor = GeometricFeatureExtractor(normalize=True)
            self.logger.info(f"Initialized geometric extractor with {self.geometric_extractor.num_features} features")

        if method in ['pca', 'both']:
            self.pca_extractor = PCAFeatureExtractor(n_components=n_pca_components)
            self.logger.info(f"Initialized PCA extractor with {n_pca_components} components")

        # Output directories
        if method in ['geometric', 'both']:
            self.geometric_output_dir = self.output_dir / 'geometric'
            ensure_dir(self.geometric_output_dir)

        if method in ['pca', 'both']:
            self.pca_output_dir = self.output_dir / 'pca'
            ensure_dir(self.pca_output_dir)

    def find_segmentation_files(self) -> List[Tuple[str, Path]]:
        """Find all segmentation files."""
        seg_files = []

        for utterance_dir in sorted(self.segmentation_dir.iterdir()):
            if not utterance_dir.is_dir():
                continue

            # Find segmentation NPZ file
            npz_files = list(utterance_dir.glob('*_segmentations.npz'))
            if len(npz_files) == 0:
                self.logger.warning(f"No segmentation file found in {utterance_dir}")
                continue

            utterance_name = utterance_dir.name
            seg_files.append((utterance_name, npz_files[0]))

        self.logger.info(f"Found {len(seg_files)} segmentation files")
        return seg_files

    def extract_geometric_features(self, segmentation_files: List[Tuple[str, Path]]):
        """Extract geometric features from all utterances."""
        self.logger.info("Extracting geometric features...")

        stats = {
            'num_utterances': 0,
            'total_frames': 0,
            'feature_names': self.geometric_extractor.feature_names,
            'num_features': self.geometric_extractor.num_features,
            'utterances': {}
        }

        for utterance_name, seg_file in tqdm(segmentation_files, desc="Geometric extraction"):
            # Load segmentations
            data = np.load(seg_file)
            segmentations = data['segmentations']
            num_frames = segmentations.shape[0]

            # Extract features
            features = self.geometric_extractor.extract_batch(segmentations)

            # Save features
            output_file = self.geometric_output_dir / f"{utterance_name}_params.npy"
            np.save(output_file, features)

            # Update stats
            stats['num_utterances'] += 1
            stats['total_frames'] += num_frames
            stats['utterances'][utterance_name] = {
                'num_frames': int(num_frames),
                'feature_shape': list(features.shape),
                'output_file': str(output_file)
            }

        # Save statistics
        stats_file = self.geometric_output_dir / 'extraction_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        self.logger.info(f"Geometric extraction complete:")
        self.logger.info(f"  - Utterances: {stats['num_utterances']}")
        self.logger.info(f"  - Total frames: {stats['total_frames']}")
        self.logger.info(f"  - Features per frame: {stats['num_features']}")
        self.logger.info(f"  - Output dir: {self.geometric_output_dir}")

        return stats

    def extract_pca_features(self, segmentation_files: List[Tuple[str, Path]]):
        """Extract PCA features from all utterances."""
        self.logger.info("Extracting PCA features...")

        # Step 1: Collect all segmentations for PCA fitting
        self.logger.info("Loading all segmentations for PCA fitting...")
        all_segmentations = []
        utterance_info = []

        for utterance_name, seg_file in tqdm(segmentation_files, desc="Loading segmentations"):
            data = np.load(seg_file)
            segmentations = data['segmentations']
            all_segmentations.append(segmentations)
            utterance_info.append((utterance_name, len(segmentations)))

        # Concatenate all segmentations
        all_segmentations = np.concatenate(all_segmentations, axis=0)
        self.logger.info(f"Total segmentations for PCA: {all_segmentations.shape[0]}")

        # Step 2: Fit PCA
        self.logger.info(f"Fitting PCA with {self.n_pca_components} components...")
        self.pca_extractor.fit(all_segmentations)
        self.logger.info(f"PCA explained variance: {self.pca_extractor.total_explained_variance:.4f}")

        # Save PCA model
        pca_model_file = self.pca_output_dir / 'pca_model.pkl'
        self.pca_extractor.save(pca_model_file)
        self.logger.info(f"Saved PCA model to {pca_model_file}")

        # Step 3: Transform each utterance
        self.logger.info("Transforming utterances to PCA features...")
        stats = {
            'num_utterances': 0,
            'total_frames': 0,
            'n_components': self.n_pca_components,
            'explained_variance_ratio': self.pca_extractor.explained_variance_ratio.tolist(),
            'total_explained_variance': float(self.pca_extractor.total_explained_variance),
            'utterances': {}
        }

        for utterance_name, seg_file in tqdm(segmentation_files, desc="PCA transformation"):
            # Load segmentations
            data = np.load(seg_file)
            segmentations = data['segmentations']
            num_frames = segmentations.shape[0]

            # Transform to PCA features
            features = self.pca_extractor.transform(segmentations)

            # Save features
            output_file = self.pca_output_dir / f"{utterance_name}_params.npy"
            np.save(output_file, features)

            # Update stats
            stats['num_utterances'] += 1
            stats['total_frames'] += num_frames
            stats['utterances'][utterance_name] = {
                'num_frames': int(num_frames),
                'feature_shape': list(features.shape),
                'output_file': str(output_file)
            }

        # Save statistics
        stats_file = self.pca_output_dir / 'extraction_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        self.logger.info(f"PCA extraction complete:")
        self.logger.info(f"  - Utterances: {stats['num_utterances']}")
        self.logger.info(f"  - Total frames: {stats['total_frames']}")
        self.logger.info(f"  - Components: {self.n_pca_components}")
        self.logger.info(f"  - Explained variance: {stats['total_explained_variance']:.4f}")
        self.logger.info(f"  - Output dir: {self.pca_output_dir}")

        return stats

    def run(self):
        """Run the extraction pipeline."""
        start_time = time.time()

        # Find segmentation files
        seg_files = self.find_segmentation_files()

        if len(seg_files) == 0:
            self.logger.error("No segmentation files found!")
            return

        # Extract features
        results = {}

        if self.method in ['geometric', 'both']:
            geometric_stats = self.extract_geometric_features(seg_files)
            results['geometric'] = geometric_stats

        if self.method in ['pca', 'both']:
            pca_stats = self.extract_pca_features(seg_files)
            results['pca'] = pca_stats

        # Save overall summary
        summary = {
            'method': self.method,
            'num_utterances': len(seg_files),
            'extraction_time_seconds': time.time() - start_time,
            'results': results
        }

        summary_file = self.output_dir / 'extraction_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        self.logger.info("=" * 60)
        self.logger.info("EXTRACTION COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"Method: {self.method}")
        self.logger.info(f"Utterances processed: {len(seg_files)}")
        self.logger.info(f"Duration: {summary['extraction_time_seconds']:.1f} seconds")
        self.logger.info(f"Summary saved to: {summary_file}")
        self.logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Extract articulatory parameters from segmentation masks"
    )
    parser.add_argument(
        '--segmentation-dir',
        type=str,
        default='data/processed/segmentations',
        help='Directory containing segmentation files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed/parameters',
        help='Output directory for parameters'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['geometric', 'pca', 'both'],
        default='geometric',
        help='Extraction method (geometric, pca, or both)'
    )
    parser.add_argument(
        '--n-components',
        type=int,
        default=10,
        help='Number of PCA components (only for PCA method)'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default='logs/parameter_extraction.log',
        help='Log file path'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logger(
        'ParameterExtraction',
        level=logging.INFO,
        log_file=args.log_file,
        file_level=logging.DEBUG
    )

    logger.info("=" * 60)
    logger.info("ARTICULATORY PARAMETER EXTRACTION")
    logger.info("=" * 60)
    logger.info(f"Segmentation dir: {args.segmentation_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Method: {args.method}")
    if args.method in ['pca', 'both']:
        logger.info(f"PCA components: {args.n_components}")
    logger.info("=" * 60)

    # Create extractor and run
    extractor = ArticulatoryParameterExtractor(
        segmentation_dir=args.segmentation_dir,
        output_dir=args.output_dir,
        method=args.method,
        n_pca_components=args.n_components,
        logger=logger
    )

    extractor.run()


if __name__ == '__main__':
    main()
