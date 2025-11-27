#!/usr/bin/env python3
"""
Batch preprocessing pipeline for USC-TIMIT dataset.

This script processes multiple subjects through the complete Phase 1 pipeline:
1. Load MRI and Audio
2. Denoise both modalities
3. Align audio with MRI
4. Save preprocessed data

Usage:
    python scripts/batch_preprocess.py --subjects sub001 sub007 sub008
    python scripts/batch_preprocess.py --recommended  # Use recommended subjects
    python scripts/batch_preprocess.py --all  # Process all subjects
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.data_loader import USCTIMITLoader
from src.preprocessing.denoising import MRIDenoiser, AudioDenoiser
from src.preprocessing.alignment import AudioMRIAligner, validate_alignment
from src.utils.logger import setup_logger, get_logger
from src.utils.io_utils import (
    ensure_directory,
    load_mri_from_video,
    load_audio,
    save_hdf5,
    save_json,
)
from src.utils.config import PreprocessConfig

# Setup logging
setup_logger(level="INFO", console=True, log_file="logs/batch_preprocess.log")
logger = get_logger(__name__)


def preprocess_utterance(
    video_file: Path,
    mri_fps: float,
    config: PreprocessConfig,
) -> Dict[str, Any]:
    """
    Preprocess a single utterance (MRI + Audio).

    Args:
        video_file: Path to video file containing MRI and audio
        mri_fps: MRI frame rate
        config: Preprocessing configuration

    Returns:
        Dict containing preprocessed data and metadata
    """
    # Load data
    mri_frames = load_mri_from_video(video_file, normalize=True)
    audio, audio_sr = load_audio(video_file, sr=config.audio_sample_rate)

    # Step 1: Denoise MRI
    mri_denoiser = MRIDenoiser(
        method=config.mri_denoise_method,
        spatial_sigma=config.mri_denoise_sigma,
        temporal_window=config.mri_temporal_window,
        apply_temporal=True,
    )
    mri_denoised = mri_denoiser.denoise(mri_frames)

    # Step 2: Denoise Audio
    audio_denoiser = AudioDenoiser(
        noise_sample_duration=0.5,
        prop_decrease=0.8,
        stationary=False,
    )
    audio_denoised = audio_denoiser.denoise(audio, audio_sr)

    # Step 3: Align
    aligner = AudioMRIAligner(
        mri_fps=mri_fps,
        jaw_region=None,
        smooth_sigma=10.0,
    )
    alignment_result = aligner.align(mri_denoised, audio_denoised, audio_sr)

    # Validate alignment
    is_valid, msg = validate_alignment(alignment_result, min_correlation=0.3)

    # Package results
    result = {
        'mri_frames': mri_denoised,
        'audio': alignment_result['aligned_audio'],
        'audio_sr': audio_sr,
        'mri_fps': mri_fps,
        'alignment': {
            'offset_seconds': alignment_result['offset_seconds'],
            'correlation': alignment_result['correlation'],
            'is_valid': is_valid,
            'validation_message': msg,
        },
        'metadata': {
            'original_mri_shape': mri_frames.shape,
            'original_audio_length': len(audio),
            'video_file': str(video_file.name),
        }
    }

    return result


def process_subject(
    subject_id: str,
    loader: USCTIMITLoader,
    config: PreprocessConfig,
    output_dir: Path,
    max_utterances: int = None,
) -> Dict[str, Any]:
    """
    Process all utterances for a subject.

    Args:
        subject_id: Subject ID
        loader: Data loader
        config: Preprocessing configuration
        output_dir: Output directory
        max_utterances: Maximum utterances to process (None = all)

    Returns:
        Dict with processing statistics
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Processing Subject: {subject_id}")
    logger.info(f"{'='*70}")

    # Load subject
    subject_data = loader.load_subject(subject_id, load_mri=False, load_audio=False)

    if 'utterance_files' not in subject_data:
        logger.warning(f"Subject {subject_id} has no utterances, skipping")
        return {'success': False, 'error': 'No utterances'}

    utterance_files = subject_data['utterance_files']
    mri_fps = subject_data['mri_fps']

    # Limit utterances if specified
    if max_utterances is not None:
        utterance_files = utterance_files[:max_utterances]

    logger.info(f"Processing {len(utterance_files)} utterances")

    # Create subject output directory
    subject_output_dir = output_dir / subject_id
    ensure_directory(subject_output_dir)

    # Process each utterance
    processed_utterances = []
    failed_utterances = []

    for idx, video_file in enumerate(tqdm(utterance_files, desc=f"{subject_id}")):
        try:
            # Preprocess utterance
            result = preprocess_utterance(video_file, mri_fps, config)

            # Save preprocessed data
            utterance_name = video_file.stem  # Filename without extension

            # Save to HDF5
            hdf5_path = subject_output_dir / f"{utterance_name}.h5"
            save_hdf5(
                {
                    'mri_frames': result['mri_frames'],
                    'audio': result['audio'],
                    'audio_sr': result['audio_sr'],
                    'mri_fps': result['mri_fps'],
                },
                hdf5_path,
            )

            # Save metadata to JSON
            metadata_path = subject_output_dir / f"{utterance_name}_metadata.json"
            save_json(
                {
                    'alignment': result['alignment'],
                    'metadata': result['metadata'],
                },
                metadata_path,
            )

            processed_utterances.append({
                'utterance_name': utterance_name,
                'hdf5_path': str(hdf5_path),
                'metadata_path': str(metadata_path),
                'alignment_valid': result['alignment']['is_valid'],
                'correlation': result['alignment']['correlation'],
            })

        except Exception as e:
            logger.error(f"Failed to process {video_file.name}: {e}")
            failed_utterances.append({
                'utterance_name': video_file.stem,
                'error': str(e),
            })

    # Save subject summary
    summary = {
        'subject_id': subject_id,
        'total_utterances': len(utterance_files),
        'processed': len(processed_utterances),
        'failed': len(failed_utterances),
        'utterances': processed_utterances,
        'failed_utterances': failed_utterances,
    }

    summary_path = subject_output_dir / 'summary.json'
    save_json(summary, summary_path)

    logger.info(f"\nSubject {subject_id} Summary:")
    logger.info(f"  Processed: {len(processed_utterances)}/{len(utterance_files)}")
    logger.info(f"  Failed: {len(failed_utterances)}")
    logger.info(f"  Output: {subject_output_dir}")

    return summary


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Batch preprocess USC-TIMIT data")
    parser.add_argument(
        '--subjects',
        nargs='+',
        help='List of subject IDs to process'
    )
    parser.add_argument(
        '--recommended',
        action='store_true',
        help='Process recommended subjects only'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all subjects'
    )
    parser.add_argument(
        '--max-utterances',
        type=int,
        default=None,
        help='Maximum utterances per subject (for testing)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/processed/aligned'),
        help='Output directory for preprocessed data'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/preprocess.yaml'),
        help='Path to preprocessing config file'
    )

    args = parser.parse_args()

    # Load configuration
    if args.config.exists():
        from src.utils.config import load_config
        config = load_config(args.config)
        logger.info(f"Loaded config from: {args.config}")
    else:
        config = PreprocessConfig()
        logger.info("Using default configuration")

    # Initialize data loader
    data_root = Path("data/raw/usc_timit_data")
    loader = USCTIMITLoader(data_root)

    # Determine subjects to process
    if args.recommended:
        # Load recommended subjects
        rec_file = Path("data/raw/recommended_subjects.json")
        if rec_file.exists():
            with open(rec_file) as f:
                rec_data = json.load(f)
                subject_ids = rec_data.get('subject_ids', [])
            logger.info(f"Processing {len(subject_ids)} recommended subjects")
        else:
            logger.error(f"Recommended subjects file not found: {rec_file}")
            return
    elif args.all:
        subject_ids = loader.get_subject_ids()
        logger.info(f"Processing all {len(subject_ids)} subjects")
    elif args.subjects:
        subject_ids = args.subjects
        logger.info(f"Processing {len(subject_ids)} specified subjects")
    else:
        logger.error("Must specify --subjects, --recommended, or --all")
        parser.print_help()
        return

    # Create output directory
    ensure_directory(args.output_dir)

    # Process subjects
    logger.info("\n" + "="*70)
    logger.info("BATCH PREPROCESSING")
    logger.info("="*70)
    logger.info(f"Subjects: {len(subject_ids)}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Max utterances per subject: {args.max_utterances or 'All'}")
    logger.info("="*70)

    summaries = []
    for subject_id in subject_ids:
        try:
            summary = process_subject(
                subject_id,
                loader,
                config,
                args.output_dir,
                max_utterances=args.max_utterances,
            )
            summaries.append(summary)
        except Exception as e:
            logger.error(f"Failed to process subject {subject_id}: {e}")

    # Save overall summary
    overall_summary = {
        'total_subjects': len(subject_ids),
        'processed_subjects': len(summaries),
        'total_utterances': sum(s.get('processed', 0) for s in summaries),
        'failed_utterances': sum(s.get('failed', 0) for s in summaries),
        'subjects': summaries,
    }

    summary_path = args.output_dir / 'batch_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(overall_summary, f, indent=2)

    # Final summary
    logger.info("\n" + "="*70)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("="*70)
    logger.info(f"Subjects processed: {overall_summary['processed_subjects']}/{overall_summary['total_subjects']}")
    logger.info(f"Total utterances: {overall_summary['total_utterances']}")
    logger.info(f"Failed utterances: {overall_summary['failed_utterances']}")
    logger.info(f"Summary saved: {summary_path}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
