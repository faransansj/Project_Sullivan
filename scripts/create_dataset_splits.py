#!/usr/bin/env python3
"""
Create Train/Val/Test Dataset Splits

This script creates subject-level splits to prevent data leakage.

Usage:
    python scripts/create_dataset_splits.py --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List
import time

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.logger import setup_logger


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


class DatasetSplitter:
    """Create dataset splits at the subject level."""

    def __init__(
        self,
        segmentation_dir: str,
        parameter_dir: str,
        audio_feature_dir: str,
        output_dir: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        logger: logging.Logger = None
    ):
        self.segmentation_dir = Path(segmentation_dir)
        self.parameter_dir = Path(parameter_dir)
        self.audio_feature_dir = Path(audio_feature_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.logger = logger or logging.getLogger(__name__)

        # Set random seed for reproducibility
        np.random.seed(random_seed)

        # Ensure ratios sum to 1.0
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")

    def get_utterances_by_subject(self) -> Dict[str, List[str]]:
        """Group utterances by subject."""
        subject_utterances = {}

        for utterance_dir in sorted(self.segmentation_dir.iterdir()):
            if not utterance_dir.is_dir():
                continue

            utterance_name = utterance_dir.name
            subject = utterance_name.split('_')[0]  # e.g., "sub001"

            if subject not in subject_utterances:
                subject_utterances[subject] = []

            subject_utterances[subject].append(utterance_name)

        return subject_utterances

    def split_subjects(
        self,
        subject_utterances: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """Split subjects into train/val/test."""
        subjects = sorted(subject_utterances.keys())
        num_subjects = len(subjects)

        self.logger.info(f"Total subjects: {num_subjects}")

        # Calculate split sizes
        num_train = int(num_subjects * self.train_ratio)
        num_val = int(num_subjects * self.val_ratio)
        # num_test is the remainder

        # Shuffle subjects
        np.random.shuffle(subjects)

        # Split
        train_subjects = subjects[:num_train]
        val_subjects = subjects[num_train:num_train + num_val]
        test_subjects = subjects[num_train + num_val:]

        self.logger.info(f"Train subjects: {len(train_subjects)}")
        self.logger.info(f"Val subjects: {len(val_subjects)}")
        self.logger.info(f"Test subjects: {len(test_subjects)}")

        splits = {
            'train': train_subjects,
            'val': val_subjects,
            'test': test_subjects
        }

        return splits

    def create_split_files(
        self,
        splits: Dict[str, List[str]],
        subject_utterances: Dict[str, List[str]]
    ):
        """Create split files with utterance lists."""
        split_info = {
            'random_seed': self.random_seed,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'splits': {}
        }

        for split_name, split_subjects in splits.items():
            # Get all utterances for these subjects
            split_utterances = []
            for subject in split_subjects:
                split_utterances.extend(subject_utterances[subject])

            # Create split directory
            split_dir = self.output_dir / split_name
            ensure_dir(split_dir)

            # Save utterance list
            utterance_list_file = split_dir / 'utterance_list.txt'
            with open(utterance_list_file, 'w') as f:
                for utterance in sorted(split_utterances):
                    f.write(f"{utterance}\n")

            # Save subject list
            subject_list_file = split_dir / 'subject_list.txt'
            with open(subject_list_file, 'w') as f:
                for subject in sorted(split_subjects):
                    f.write(f"{subject}\n")

            # Update split info
            split_info['splits'][split_name] = {
                'num_subjects': len(split_subjects),
                'subjects': sorted(split_subjects),
                'num_utterances': len(split_utterances),
                'utterances': sorted(split_utterances)
            }

            self.logger.info(f"{split_name.capitalize()} split:")
            self.logger.info(f"  - Subjects: {len(split_subjects)}")
            self.logger.info(f"  - Utterances: {len(split_utterances)}")

        # Save split info
        split_info_file = self.output_dir / 'split_info.json'
        with open(split_info_file, 'w') as f:
            json.dump(split_info, f, indent=2)

        self.logger.info(f"Split info saved to: {split_info_file}")

        return split_info

    def run(self):
        """Run the split creation pipeline."""
        start_time = time.time()

        self.logger.info("=" * 60)
        self.logger.info("Creating dataset splits...")
        self.logger.info("=" * 60)

        # Get utterances by subject
        subject_utterances = self.get_utterances_by_subject()

        if len(subject_utterances) == 0:
            self.logger.error("No subjects found!")
            return

        # Split subjects
        splits = self.split_subjects(subject_utterances)

        # Create split files
        split_info = self.create_split_files(splits, subject_utterances)

        # Create summary
        summary = {
            'total_subjects': len(subject_utterances),
            'total_utterances': sum(len(utts) for utts in subject_utterances.values()),
            'split_ratios': {
                'train': self.train_ratio,
                'val': self.val_ratio,
                'test': self.test_ratio
            },
            'actual_splits': {
                split_name: {
                    'num_subjects': split_info['splits'][split_name]['num_subjects'],
                    'num_utterances': split_info['splits'][split_name]['num_utterances']
                }
                for split_name in ['train', 'val', 'test']
            },
            'random_seed': self.random_seed,
            'creation_time_seconds': time.time() - start_time
        }

        summary_file = self.output_dir / 'split_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        self.logger.info("=" * 60)
        self.logger.info("SPLIT CREATION COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"Total subjects: {summary['total_subjects']}")
        self.logger.info(f"Total utterances: {summary['total_utterances']}")
        self.logger.info(f"Train: {summary['actual_splits']['train']['num_subjects']} subjects, "
                        f"{summary['actual_splits']['train']['num_utterances']} utterances")
        self.logger.info(f"Val: {summary['actual_splits']['val']['num_subjects']} subjects, "
                        f"{summary['actual_splits']['val']['num_utterances']} utterances")
        self.logger.info(f"Test: {summary['actual_splits']['test']['num_subjects']} subjects, "
                        f"{summary['actual_splits']['test']['num_utterances']} utterances")
        self.logger.info(f"Duration: {summary['creation_time_seconds']:.1f} seconds")
        self.logger.info(f"Summary saved to: {summary_file}")
        self.logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Create train/val/test dataset splits"
    )
    parser.add_argument(
        '--segmentation-dir',
        type=str,
        default='data/processed/segmentations',
        help='Directory containing segmentation files'
    )
    parser.add_argument(
        '--parameter-dir',
        type=str,
        default='data/processed/parameters',
        help='Directory containing parameter files'
    )
    parser.add_argument(
        '--audio-feature-dir',
        type=str,
        default='data/processed/audio_features',
        help='Directory containing audio feature files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed/splits',
        help='Output directory for splits'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Train split ratio (default: 0.7)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation split ratio (default: 0.15)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Test split ratio (default: 0.15)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default='logs/dataset_splits.log',
        help='Log file path'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logger(
        'DatasetSplitter',
        level=logging.INFO,
        log_file=args.log_file,
        file_level=logging.DEBUG
    )

    logger.info("=" * 60)
    logger.info("DATASET SPLIT CREATION")
    logger.info("=" * 60)
    logger.info(f"Segmentation dir: {args.segmentation_dir}")
    logger.info(f"Parameter dir: {args.parameter_dir}")
    logger.info(f"Audio feature dir: {args.audio_feature_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Train/Val/Test ratios: {args.train_ratio}/{args.val_ratio}/{args.test_ratio}")
    logger.info(f"Random seed: {args.random_seed}")
    logger.info("=" * 60)

    # Create splitter and run
    splitter = DatasetSplitter(
        segmentation_dir=args.segmentation_dir,
        parameter_dir=args.parameter_dir,
        audio_feature_dir=args.audio_feature_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed,
        logger=logger
    )

    splitter.run()


if __name__ == '__main__':
    main()
