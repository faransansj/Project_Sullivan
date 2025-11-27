#!/usr/bin/env python3
"""
Collect comprehensive dataset statistics for USC-TIMIT data.

This script analyzes the USC-TIMIT dataset and generates a detailed
statistics report for planning preprocessing strategies.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.data_loader import USCTIMITLoader
from src.utils.logger import setup_logger, get_logger
from src.utils.io_utils import load_mri_from_video

# Setup logging
setup_logger(level="INFO", console=True)
logger = get_logger(__name__)


def analyze_dataset(data_root: Path, recommended_subjects: List[str] = None) -> Dict[str, Any]:
    """
    Analyze USC-TIMIT dataset and collect statistics.

    Args:
        data_root: Path to USC-TIMIT data root
        recommended_subjects: Optional list of recommended subject IDs to prioritize

    Returns:
        Dict with comprehensive statistics
    """
    logger.info("="*70)
    logger.info("USC-TIMIT Dataset Statistics Collection")
    logger.info("="*70)

    # Initialize loader
    loader = USCTIMITLoader(data_root)

    # Basic stats
    stats = {
        "dataset_name": "USC-TIMIT Speech MRI",
        "data_root": str(data_root),
        "total_subjects": len(loader),
        "subject_ids": loader.get_subject_ids(),
    }

    # Get loader statistics
    loader_stats = loader.get_statistics()
    stats.update(loader_stats)

    # Analyze utterances
    logger.info("\nAnalyzing utterances...")

    total_utterances = 0
    utterance_counts = []

    for subject_id in tqdm(loader.get_subject_ids(), desc="Subjects"):
        try:
            subject_data = loader.load_subject(
                subject_id,
                load_mri=False,
                load_audio=False
            )

            if 'num_utterances' in subject_data:
                num_utt = subject_data['num_utterances']
                total_utterances += num_utt
                utterance_counts.append(num_utt)
        except Exception as e:
            logger.warning(f"Failed to load {subject_id}: {e}")
            continue

    stats['total_utterances'] = total_utterances
    stats['utterances_per_subject'] = {
        'mean': float(np.mean(utterance_counts)),
        'std': float(np.std(utterance_counts)),
        'min': int(np.min(utterance_counts)),
        'max': int(np.max(utterance_counts)),
    }

    # Sample a few subjects for detailed analysis
    logger.info("\nAnalyzing sample subjects in detail...")

    # Use recommended subjects if provided, otherwise use first 3
    sample_subjects = recommended_subjects[:3] if recommended_subjects else loader.get_subject_ids()[:3]

    mri_shapes = []
    mri_durations = []
    audio_durations = []
    audio_srs = []

    for subject_id in tqdm(sample_subjects, desc="Sample subjects"):
        try:
            data = loader.load_subject(subject_id, load_mri=True, load_audio=True)

            if 'mri_frames_example' in data:
                mri = data['mri_frames_example']
                mri_shapes.append(mri.shape)

                fps = data['mri_fps']
                duration = len(mri) / fps
                mri_durations.append(duration)

            if 'audio_example' in data:
                audio_durations.append(data['example_audio_duration'])
                audio_srs.append(data['audio_sr'])

        except Exception as e:
            logger.warning(f"Failed to load detailed data for {subject_id}: {e}")
            continue

    if mri_shapes:
        stats['mri_analysis'] = {
            'sample_shapes': [list(s) for s in mri_shapes],
            'typical_resolution': f"{mri_shapes[0][2]}x{mri_shapes[0][1]}",
            'typical_num_frames': int(np.mean([s[0] for s in mri_shapes])),
            'typical_duration_sec': float(np.mean(mri_durations)),
        }

    if audio_durations:
        stats['audio_analysis'] = {
            'typical_duration_sec': float(np.mean(audio_durations)),
            'sample_rates': list(set(audio_srs)),
        }

    # Recommended subjects info
    if recommended_subjects:
        stats['recommended_subjects'] = recommended_subjects
        stats['num_recommended'] = len(recommended_subjects)

    return stats


def main():
    """Main function."""

    # Paths
    data_root = Path("data/raw/usc_timit_data")
    output_path = Path("docs/dataset_statistics.json")

    # Check if data exists
    if not data_root.exists():
        logger.error(f"Data root not found: {data_root}")
        logger.error("Please download USC-TIMIT data first.")
        return

    # Load recommended subjects if available
    recommended_file = Path("data/raw/recommended_subjects.json")
    recommended_subjects = None

    if recommended_file.exists():
        logger.info(f"Loading recommended subjects from: {recommended_file}")
        with open(recommended_file) as f:
            rec_data = json.load(f)
            recommended_subjects = rec_data.get('subject_ids', [])
            logger.info(f"Found {len(recommended_subjects)} recommended subjects")

    # Analyze dataset
    stats = analyze_dataset(data_root, recommended_subjects)

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    logger.info(f"Total subjects: {stats['total_subjects']}")
    logger.info(f"Total utterances: {stats['total_utterances']}")

    if 'mri_analysis' in stats:
        logger.info(f"MRI resolution: {stats['mri_analysis']['typical_resolution']}")
        logger.info(f"MRI typical frames: {stats['mri_analysis']['typical_num_frames']}")
        logger.info(f"MRI typical duration: {stats['mri_analysis']['typical_duration_sec']:.2f}s")

    if 'audio_analysis' in stats:
        logger.info(f"Audio sample rates: {stats['audio_analysis']['sample_rates']}")
        logger.info(f"Audio typical duration: {stats['audio_analysis']['typical_duration_sec']:.2f}s")

    logger.info(f"\nStatistics saved to: {output_path}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
