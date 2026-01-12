#!/usr/bin/env python3
"""
HDDB Dataset Split Generation
===============================

Create train/val/test splits for HDDB dataset.

Splits:
- Train: 70%
- Val: 15%
- Test: 15%

Ensures all utterances from the same subject stay in the same split.

Author: Claude & Research Assistant
Date: 2026-01-11
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger


class DatasetSplitter:
    """Create train/val/test splits for HDDB dataset"""

    def __init__(
        self,
        segmentation_dir: str,
        parameter_dir: str,
        audio_dir: str,
        output_dir: str,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ):
        """
        Initialize splitter

        Args:
            segmentation_dir: Directory with segmentation NPZ files
            parameter_dir: Directory with parameter NPZ files
            audio_dir: Directory with audio feature NPZ files
            output_dir: Output directory for split manifests
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            seed: Random seed for reproducibility
        """
        self.segmentation_dir = Path(segmentation_dir)
        self.parameter_dir = Path(parameter_dir)
        self.audio_dir = Path(audio_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        self.logger = setup_logger("DatasetSplitter")

        # Set random seed
        np.random.seed(seed)

        # Validate ratios
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    def find_complete_utterances(self) -> dict:
        """
        Find utterances that have all three components:
        - Segmentation
        - Parameters
        - Audio features

        Returns:
            utterances: Dict mapping subject_id -> list of utterance names
        """
        self.logger.info("Finding complete utterances...")

        # Find all segmentation files
        seg_files = set()
        for seg_dir in self.segmentation_dir.glob("*"):
            if seg_dir.is_dir():
                for seg_file in seg_dir.glob("*_segmentations.npz"):
                    utterance_name = seg_file.stem.replace('_segmentations', '')
                    seg_files.add(utterance_name)

        # Find all parameter files
        param_files = set()
        for param_file in self.parameter_dir.glob("*_params.npz"):
            utterance_name = param_file.stem.replace('_params', '')
            param_files.add(utterance_name)

        # Find all audio files
        audio_files = set()
        for audio_file in self.audio_dir.glob("*_audio.npz"):
            utterance_name = audio_file.stem.replace('_audio', '')
            audio_files.add(utterance_name)

        # Find intersection (complete utterances)
        complete = seg_files & param_files & audio_files

        self.logger.info(f"Found {len(seg_files)} segmentations")
        self.logger.info(f"Found {len(param_files)} parameters")
        self.logger.info(f"Found {len(audio_files)} audio features")
        self.logger.info(f"Complete utterances: {len(complete)}")

        # Group by subject
        utterances_by_subject = defaultdict(list)
        for utterance_name in complete:
            # Extract subject ID (e.g., sub010 from sub010_2drt_01_vcv1_r1_recon)
            subject_id = utterance_name.split('_')[0]
            utterances_by_subject[subject_id].append(utterance_name)

        # Sort for reproducibility
        for subject_id in utterances_by_subject:
            utterances_by_subject[subject_id].sort()

        return dict(utterances_by_subject)

    def create_splits(self, utterances_by_subject: dict) -> dict:
        """
        Create train/val/test splits

        Strategy: Split by subject (all utterances from same subject in same split)

        Args:
            utterances_by_subject: Dict mapping subject_id -> list of utterance names

        Returns:
            splits: Dict with 'train', 'val', 'test' lists
        """
        self.logger.info("Creating train/val/test splits...")

        # Get all subjects
        subjects = sorted(utterances_by_subject.keys())
        num_subjects = len(subjects)

        self.logger.info(f"Total subjects: {num_subjects}")

        # Shuffle subjects
        np.random.shuffle(subjects)

        # Calculate split sizes
        num_train = int(num_subjects * self.train_ratio)
        num_val = int(num_subjects * self.val_ratio)
        # num_test = remaining

        # Split subjects
        train_subjects = subjects[:num_train]
        val_subjects = subjects[num_train:num_train + num_val]
        test_subjects = subjects[num_train + num_val:]

        # Collect utterances
        train_utterances = []
        val_utterances = []
        test_utterances = []

        for subject in train_subjects:
            train_utterances.extend(utterances_by_subject[subject])

        for subject in val_subjects:
            val_utterances.extend(utterances_by_subject[subject])

        for subject in test_subjects:
            test_utterances.extend(utterances_by_subject[subject])

        # Sort for reproducibility
        train_utterances.sort()
        val_utterances.sort()
        test_utterances.sort()

        self.logger.info(f"Train: {len(train_subjects)} subjects, {len(train_utterances)} utterances")
        self.logger.info(f"  Subjects: {', '.join(train_subjects)}")
        self.logger.info(f"Val: {len(val_subjects)} subjects, {len(val_utterances)} utterances")
        self.logger.info(f"  Subjects: {', '.join(val_subjects)}")
        self.logger.info(f"Test: {len(test_subjects)} subjects, {len(test_utterances)} utterances")
        self.logger.info(f"  Subjects: {', '.join(test_subjects)}")

        splits = {
            'train': {
                'subjects': train_subjects,
                'utterances': train_utterances
            },
            'val': {
                'subjects': val_subjects,
                'utterances': val_utterances
            },
            'test': {
                'subjects': test_subjects,
                'utterances': test_utterances
            }
        }

        return splits

    def save_splits(self, splits: dict):
        """
        Save splits as JSON manifests

        Args:
            splits: Dict with 'train', 'val', 'test' data
        """
        self.logger.info("Saving split manifests...")

        # Save each split
        for split_name in ['train', 'val', 'test']:
            split_data = splits[split_name]

            # Create manifest with full paths
            manifest = {
                'split': split_name,
                'num_subjects': len(split_data['subjects']),
                'num_utterances': len(split_data['utterances']),
                'subjects': split_data['subjects'],
                'utterances': []
            }

            # Add full paths for each utterance
            for utterance_name in split_data['utterances']:
                utterance_info = {
                    'utterance_name': utterance_name,
                    'segmentation': f"data/processed/segmentations/{utterance_name}/{utterance_name}_segmentations.npz",
                    'parameters': f"data/processed/parameters/{utterance_name}_params.npz",
                    'audio': f"data/processed/audio_features/{utterance_name}_audio.npz"
                }
                manifest['utterances'].append(utterance_info)

            # Save manifest
            manifest_path = self.output_dir / f"{split_name}.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)

            self.logger.info(f"  {split_name}.json: {len(split_data['utterances'])} utterances")

        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'seed': self.seed,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'train_subjects': len(splits['train']['subjects']),
            'val_subjects': len(splits['val']['subjects']),
            'test_subjects': len(splits['test']['subjects']),
            'train_utterances': len(splits['train']['utterances']),
            'val_utterances': len(splits['val']['utterances']),
            'test_utterances': len(splits['test']['utterances']),
            'total_utterances': sum(len(s['utterances']) for s in splits.values())
        }

        summary_path = self.output_dir / 'split_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Summary saved to: {summary_path}")

    def run(self):
        """Run the full splitting process"""
        # Find complete utterances
        utterances_by_subject = self.find_complete_utterances()

        if len(utterances_by_subject) == 0:
            self.logger.error("No complete utterances found!")
            return

        # Create splits
        splits = self.create_splits(utterances_by_subject)

        # Save splits
        self.save_splits(splits)

        self.logger.info("Dataset splitting complete!")


def main():
    parser = argparse.ArgumentParser(description="Create train/val/test splits for HDDB dataset")
    parser.add_argument(
        '--segmentation-dir',
        type=str,
        default='data/processed/segmentations',
        help='Directory with segmentation NPZ files'
    )
    parser.add_argument(
        '--parameter-dir',
        type=str,
        default='data/processed/parameters',
        help='Directory with parameter NPZ files'
    )
    parser.add_argument(
        '--audio-dir',
        type=str,
        default='data/processed/audio_features',
        help='Directory with audio feature NPZ files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed/splits',
        help='Output directory for split manifests'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.70,
        help='Training set ratio (default: 0.70)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation set ratio (default: 0.15)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Test set ratio (default: 0.15)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # Create splitter
    splitter = DatasetSplitter(
        segmentation_dir=args.segmentation_dir,
        parameter_dir=args.parameter_dir,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    # Run splitting
    splitter.run()


if __name__ == '__main__':
    main()
