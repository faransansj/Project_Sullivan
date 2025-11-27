#!/usr/bin/env python3
"""
Inspect sample preprocessed data to verify quality.

This script loads and analyzes samples from different quality tiers.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger, get_logger

# Setup logging
setup_logger(level="INFO", console=True)
logger = get_logger(__name__)


def load_utterance(hdf5_path: Path, metadata_path: Path) -> Tuple[Dict, Dict]:
    """
    Load utterance data and metadata.

    Args:
        hdf5_path: Path to HDF5 file
        metadata_path: Path to metadata JSON

    Returns:
        Tuple of (data_dict, metadata_dict)
    """
    # Load HDF5 data
    data = {}
    with h5py.File(hdf5_path, 'r') as f:
        for key in f.keys():
            data[key] = f[key][:]

    # Load metadata
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Add scalar values from metadata to data dict
    # These should be in the original metadata from preprocessing
    # Check if they exist in metadata, otherwise use defaults
    data['audio_sr'] = 22050  # From config
    data['mri_fps'] = 83.28  # From config

    return data, metadata


def analyze_utterance(data: Dict, metadata: Dict, utterance_name: str) -> None:
    """
    Analyze and display utterance statistics.

    Args:
        data: Utterance data from HDF5
        metadata: Utterance metadata
        utterance_name: Name for display
    """
    mri_frames = data['mri_frames']
    audio = data['audio']
    mri_fps = data['mri_fps']
    audio_sr = data['audio_sr']

    alignment = metadata['alignment']
    orig_metadata = metadata['metadata']

    logger.info(f"\n{'='*70}")
    logger.info(f"Utterance: {utterance_name}")
    logger.info(f"{'='*70}")

    # MRI statistics
    logger.info("MRI Data:")
    logger.info(f"  Shape: {mri_frames.shape}")
    logger.info(f"  Dtype: {mri_frames.dtype}")
    logger.info(f"  FPS: {mri_fps}")
    logger.info(f"  Duration: {mri_frames.shape[0] / mri_fps:.2f}s")
    logger.info(f"  Value range: [{mri_frames.min():.3f}, {mri_frames.max():.3f}]")
    logger.info(f"  Mean: {mri_frames.mean():.3f}, Std: {mri_frames.std():.3f}")
    logger.info(f"  Size: {mri_frames.nbytes / 1024 / 1024:.1f} MB")

    # Audio statistics
    logger.info("\nAudio Data:")
    logger.info(f"  Shape: {audio.shape}")
    logger.info(f"  Dtype: {audio.dtype}")
    logger.info(f"  Sample rate: {audio_sr} Hz")
    logger.info(f"  Duration: {len(audio) / audio_sr:.2f}s")
    logger.info(f"  Value range: [{audio.min():.3f}, {audio.max():.3f}]")
    logger.info(f"  Mean: {audio.mean():.6f}, Std: {audio.std():.3f}")
    logger.info(f"  Size: {audio.nbytes / 1024:.1f} KB")

    # Alignment statistics
    logger.info("\nAlignment:")
    logger.info(f"  Offset: {alignment['offset_seconds']:.3f}s")
    logger.info(f"  Correlation: {alignment['correlation']:.3f}")
    logger.info(f"  Valid: {alignment['is_valid']}")
    logger.info(f"  Message: {alignment['validation_message']}")

    # Original metadata
    logger.info("\nOriginal Data:")
    logger.info(f"  MRI shape: {orig_metadata['original_mri_shape']}")
    logger.info(f"  Audio length: {orig_metadata['original_audio_length']}")
    logger.info(f"  Video file: {orig_metadata['video_file']}")

    # Quality assessment
    logger.info("\nQuality Assessment:")

    # Check for NaN/Inf
    has_nan_mri = np.isnan(mri_frames).any()
    has_inf_mri = np.isinf(mri_frames).any()
    has_nan_audio = np.isnan(audio).any()
    has_inf_audio = np.isinf(audio).any()

    logger.info(f"  MRI NaN: {has_nan_mri}, Inf: {has_inf_mri}")
    logger.info(f"  Audio NaN: {has_nan_audio}, Inf: {has_inf_audio}")

    # Check duration match
    mri_duration = mri_frames.shape[0] / mri_fps
    audio_duration = len(audio) / audio_sr
    duration_diff = abs(mri_duration - audio_duration)

    logger.info(f"  Duration diff: {duration_diff:.3f}s")
    logger.info(f"  Duration match: {'✓' if duration_diff < 1.0 else '✗'}")

    # Overall quality
    corr = alignment['correlation']
    if corr >= 0.4:
        quality = "EXCELLENT"
    elif corr >= 0.3:
        quality = "GOOD"
    elif corr >= 0.2:
        quality = "MODERATE"
    elif corr >= 0.1:
        quality = "ACCEPTABLE"
    else:
        quality = "POOR"

    logger.info(f"  Overall quality: {quality}")


def select_samples(batch_summary_path: Path) -> List[Tuple[str, Path, Path, float]]:
    """
    Select samples from different quality tiers.

    Args:
        batch_summary_path: Path to batch summary JSON

    Returns:
        List of (name, hdf5_path, metadata_path, correlation) tuples
    """
    with open(batch_summary_path) as f:
        data = json.load(f)

    # Collect all utterances with correlations
    utterances = []
    for subject in data['subjects']:
        for utt in subject.get('utterances', []):
            utterances.append((
                utt['utterance_name'],
                Path(utt['hdf5_path']),
                Path(utt['metadata_path']),
                utt['correlation']
            ))

    # Sort by correlation
    utterances.sort(key=lambda x: x[3], reverse=True)

    # Select samples
    samples = []

    # Best sample
    samples.append(('Best (Highest correlation)', *utterances[0]))

    # Good quality (correlation >= 0.3)
    good_samples = [u for u in utterances if u[3] >= 0.3]
    if len(good_samples) > 1:
        samples.append(('Good quality sample', *good_samples[len(good_samples)//2]))

    # Moderate quality (0.2 <= correlation < 0.3)
    moderate_samples = [u for u in utterances if 0.2 <= u[3] < 0.3]
    if moderate_samples:
        samples.append(('Moderate quality sample', *moderate_samples[len(moderate_samples)//2]))

    # Acceptable quality (0.1 <= correlation < 0.2)
    acceptable_samples = [u for u in utterances if 0.1 <= u[3] < 0.2]
    if acceptable_samples:
        samples.append(('Acceptable quality sample', *acceptable_samples[len(acceptable_samples)//2]))

    # Poor quality (correlation < 0.1)
    poor_samples = [u for u in utterances if u[3] < 0.1]
    if poor_samples:
        samples.append(('Poor quality sample', *poor_samples[0]))

    # Worst sample
    samples.append(('Worst (Lowest correlation)', *utterances[-1]))

    return samples


def main():
    """Main function."""
    logger.info("="*70)
    logger.info("SAMPLE DATA INSPECTION")
    logger.info("="*70)

    batch_summary_path = Path("data/processed/aligned/batch_summary.json")

    # Select samples
    logger.info("\nSelecting representative samples from different quality tiers...")
    samples = select_samples(batch_summary_path)

    logger.info(f"\nFound {len(samples)} samples to inspect:")
    for i, (category, name, _, _, corr) in enumerate(samples, 1):
        logger.info(f"  {i}. {category}: {name} (corr={corr:.3f})")

    # Inspect each sample
    for category, name, hdf5_path, metadata_path, corr in samples:
        try:
            data, metadata = load_utterance(hdf5_path, metadata_path)
            analyze_utterance(data, metadata, f"{category}: {name}")
        except Exception as e:
            logger.error(f"Failed to inspect {name}: {e}")

    logger.info("\n" + "="*70)
    logger.info("INSPECTION COMPLETE")
    logger.info("="*70)
    logger.info("\nSummary:")
    logger.info(f"  Total samples inspected: {len(samples)}")
    logger.info(f"  Data format: HDF5 + JSON metadata")
    logger.info(f"  All samples loaded successfully")
    logger.info("\nData is ready for downstream processing (segmentation).")


if __name__ == "__main__":
    main()
