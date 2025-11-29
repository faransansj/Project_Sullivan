#!/usr/bin/env python3
"""
Extract Audio Features Synchronized with MRI Frames

This script extracts Mel-spectrogram and MFCC features from audio,
synchronized with MRI frame timestamps.

Usage:
    python scripts/extract_audio_features.py --features mel
    python scripts/extract_audio_features.py --features mfcc
    python scripts/extract_audio_features.py --features both
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple
import time

import h5py
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from audio_features.mel_spectrogram import MelSpectrogramExtractor
from audio_features.mfcc import MFCCExtractor
from utils.logger import setup_logger


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


class AudioFeatureExtractor:
    """Main class for extracting audio features."""

    def __init__(
        self,
        aligned_dir: str,
        segmentation_dir: str,
        output_dir: str,
        feature_type: str = 'both',
        logger: logging.Logger = None
    ):
        self.aligned_dir = Path(aligned_dir)
        self.segmentation_dir = Path(segmentation_dir)
        self.output_dir = Path(output_dir)
        self.feature_type = feature_type
        self.logger = logger or logging.getLogger(__name__)

        # Initialize extractors
        if feature_type in ['mel', 'both']:
            self.mel_extractor = MelSpectrogramExtractor(
                n_mels=80,
                n_fft=512,
                hop_length=160
            )
            self.logger.info("Initialized Mel-spectrogram extractor (80 bins)")

        if feature_type in ['mfcc', 'both']:
            self.mfcc_extractor = MFCCExtractor(
                n_mfcc=13,
                n_fft=512,
                hop_length=160
            )
            self.logger.info("Initialized MFCC extractor (13 coefficients)")

        # Output directories
        if feature_type in ['mel', 'both']:
            self.mel_output_dir = self.output_dir / 'mel_spectrogram'
            ensure_dir(self.mel_output_dir)

        if feature_type in ['mfcc', 'both']:
            self.mfcc_output_dir = self.output_dir / 'mfcc'
            ensure_dir(self.mfcc_output_dir)

    def find_utterance_files(self) -> List[Tuple[str, Path, Path]]:
        """
        Find aligned HDF5 files and corresponding segmentation files.

        Returns
        -------
        files : list of tuples
            List of (utterance_name, aligned_file_path, seg_file_path)
        """
        utterance_files = []

        # Iterate through segmentation directories
        for seg_dir in sorted(self.segmentation_dir.iterdir()):
            if not seg_dir.is_dir():
                continue

            utterance_name = seg_dir.name

            # Find segmentation file
            seg_files = list(seg_dir.glob('*_segmentations.npz'))
            if len(seg_files) == 0:
                self.logger.warning(f"No segmentation file for {utterance_name}")
                continue
            seg_file = seg_files[0]

            # Find corresponding aligned HDF5 file
            # Format: data/processed/aligned/{subject}/{utterance}.h5
            subject = utterance_name.split('_')[0]  # e.g., "sub001"
            aligned_file = self.aligned_dir / subject / f"{utterance_name}.h5"

            if not aligned_file.exists():
                self.logger.warning(f"Aligned file not found: {aligned_file}")
                continue

            utterance_files.append((utterance_name, aligned_file, seg_file))

        self.logger.info(f"Found {len(utterance_files)} utterance pairs")
        return utterance_files

    def extract_features(self, utterance_files: List[Tuple[str, Path, Path]]):
        """Extract features for all utterances."""
        self.logger.info(f"Extracting {self.feature_type} features...")

        stats = {
            'num_utterances': 0,
            'total_frames': 0,
            'feature_type': self.feature_type,
            'utterances': {}
        }

        for utterance_name, aligned_file, seg_file in tqdm(utterance_files, desc="Feature extraction"):
            try:
                # Load segmentation to get number of frames
                seg_data = np.load(seg_file)
                num_mri_frames = seg_data['num_frames']

                # Load aligned HDF5 file
                with h5py.File(aligned_file, 'r') as f:
                    audio = f['audio'][:]
                    audio_sr = f.attrs['audio_sr']
                    mri_fps = f.attrs['mri_fps']

                # Extract Mel-spectrogram
                if self.feature_type in ['mel', 'both']:
                    mel_features = self.mel_extractor.extract(
                        audio,
                        sample_rate=int(audio_sr),
                        num_mri_frames=int(num_mri_frames),
                        mri_fps=float(mri_fps)
                    )

                    # Save
                    mel_output_file = self.mel_output_dir / f"{utterance_name}_mel.npy"
                    np.save(mel_output_file, mel_features)

                # Extract MFCC
                if self.feature_type in ['mfcc', 'both']:
                    mfcc_features = self.mfcc_extractor.extract(
                        audio,
                        sample_rate=int(audio_sr),
                        num_mri_frames=int(num_mri_frames),
                        mri_fps=float(mri_fps)
                    )

                    # Save
                    mfcc_output_file = self.mfcc_output_dir / f"{utterance_name}_mfcc.npy"
                    np.save(mfcc_output_file, mfcc_features)

                # Update stats
                stats['num_utterances'] += 1
                stats['total_frames'] += int(num_mri_frames)
                stats['utterances'][utterance_name] = {
                    'num_frames': int(num_mri_frames),
                    'audio_sr': int(audio_sr),
                    'mri_fps': float(mri_fps)
                }

                if self.feature_type in ['mel', 'both']:
                    stats['utterances'][utterance_name]['mel_shape'] = list(mel_features.shape)
                    stats['utterances'][utterance_name]['mel_file'] = str(mel_output_file)

                if self.feature_type in ['mfcc', 'both']:
                    stats['utterances'][utterance_name]['mfcc_shape'] = list(mfcc_features.shape)
                    stats['utterances'][utterance_name]['mfcc_file'] = str(mfcc_output_file)

            except Exception as e:
                self.logger.error(f"Error processing {utterance_name}: {e}")
                continue

        # Save statistics
        stats_file = self.output_dir / 'audio_feature_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        self.logger.info(f"Audio feature extraction complete:")
        self.logger.info(f"  - Utterances: {stats['num_utterances']}")
        self.logger.info(f"  - Total frames: {stats['total_frames']}")
        self.logger.info(f"  - Feature type: {self.feature_type}")
        if self.feature_type in ['mel', 'both']:
            self.logger.info(f"  - Mel-spectrogram: 80 bins")
        if self.feature_type in ['mfcc', 'both']:
            self.logger.info(f"  - MFCC: 13 coefficients")
        self.logger.info(f"  - Stats saved to: {stats_file}")

        return stats

    def run(self):
        """Run the extraction pipeline."""
        start_time = time.time()

        # Find files
        utterance_files = self.find_utterance_files()

        if len(utterance_files) == 0:
            self.logger.error("No utterance files found!")
            return

        # Extract features
        stats = self.extract_features(utterance_files)

        # Save summary
        summary = {
            'feature_type': self.feature_type,
            'num_utterances': len(utterance_files),
            'extraction_time_seconds': time.time() - start_time,
            'stats': stats
        }

        summary_file = self.output_dir / 'extraction_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        self.logger.info("=" * 60)
        self.logger.info("EXTRACTION COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"Feature type: {self.feature_type}")
        self.logger.info(f"Utterances processed: {len(utterance_files)}")
        self.logger.info(f"Duration: {summary['extraction_time_seconds']:.1f} seconds")
        self.logger.info(f"Summary saved to: {summary_file}")
        self.logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Extract audio features synchronized with MRI frames"
    )
    parser.add_argument(
        '--aligned-dir',
        type=str,
        default='data/processed/aligned',
        help='Directory containing aligned HDF5 files'
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
        default='data/processed/audio_features',
        help='Output directory for audio features'
    )
    parser.add_argument(
        '--features',
        type=str,
        choices=['mel', 'mfcc', 'both'],
        default='both',
        help='Which features to extract (mel, mfcc, or both)'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default='logs/audio_feature_extraction.log',
        help='Log file path'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logger(
        'AudioFeatureExtraction',
        level=logging.INFO,
        log_file=args.log_file,
        file_level=logging.DEBUG
    )

    logger.info("=" * 60)
    logger.info("AUDIO FEATURE EXTRACTION")
    logger.info("=" * 60)
    logger.info(f"Aligned dir: {args.aligned_dir}")
    logger.info(f"Segmentation dir: {args.segmentation_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Features: {args.features}")
    logger.info("=" * 60)

    # Create extractor and run
    extractor = AudioFeatureExtractor(
        aligned_dir=args.aligned_dir,
        segmentation_dir=args.segmentation_dir,
        output_dir=args.output_dir,
        feature_type=args.features,
        logger=logger
    )

    extractor.run()


if __name__ == '__main__':
    main()
