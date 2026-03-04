#!/usr/bin/env python3
"""
Quick test script to verify data loader works with real USC-TIMIT data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.data_loader import USCTIMITLoader
from src.utils.logger import setup_logger, get_logger

# Setup logging
setup_logger(level="INFO", console=True)
logger = get_logger(__name__)


def main():
    """Test data loader with real USC-TIMIT data."""

    # Path to USC-TIMIT data
    data_root = Path("data/raw/usc_timit_data")

    if not data_root.exists():
        logger.error(f"Data root not found: {data_root}")
        return

    logger.info("=" * 60)
    logger.info("Testing USC-TIMIT Data Loader")
    logger.info("=" * 60)

    # Initialize loader
    logger.info(f"\nInitializing loader from: {data_root}")
    loader = USCTIMITLoader(data_root)

    # Get statistics
    logger.info("\n--- Dataset Statistics ---")
    stats = loader.get_statistics()
    logger.info(f"Total subjects: {stats['num_subjects']}")
    logger.info(f"Formats: {stats['formats']}")
    logger.info(f"Has metadata: {stats['has_metadata']}")

    # List some subjects
    subject_ids = loader.get_subject_ids()[:5]
    logger.info(f"\nFirst 5 subjects: {subject_ids}")

    # Test loading a subject (sub001 from recommended list)
    test_subject = "sub001"
    logger.info(f"\n--- Testing Subject: {test_subject} ---")

    try:
        subject_info = loader.get_subject_info(test_subject)
        logger.info(f"Subject info:")
        logger.info(f"  - MRI format: {subject_info['mri_format']}")
        logger.info(f"  - MRI path: {subject_info['mri_path']}")
        if 'metadata' in subject_info:
            logger.info(f"  - Has metadata: Yes")

        # Load subject data (without loading MRI/audio to save memory)
        logger.info(f"\nLoading subject data (metadata only)...")
        data = loader.load_subject(test_subject, load_mri=False, load_audio=False)

        logger.info(f"Subject data keys: {list(data.keys())}")

        if 'utterance_files' in data:
            logger.info(f"Number of utterances: {data['num_utterances']}")
            logger.info(f"First 3 utterances:")
            for i, utt_file in enumerate(data['utterance_files'][:3]):
                logger.info(f"  {i+1}. {utt_file.name}")

        # Load first utterance (with MRI and audio)
        logger.info(f"\n--- Loading First Utterance (with MRI & Audio) ---")
        data_with_media = loader.load_subject(test_subject, load_mri=True, load_audio=True)

        if 'mri_frames_example' in data_with_media:
            mri = data_with_media['mri_frames_example']
            logger.info(f"MRI frames: {mri.shape} (T, H, W)")
            logger.info(f"  - Num frames: {data_with_media['example_num_frames']}")
            logger.info(f"  - Frame rate: {data_with_media['mri_fps']:.2f} fps")

        if 'audio_example' in data_with_media:
            audio = data_with_media['audio_example']
            logger.info(f"Audio: {audio.shape}")
            logger.info(f"  - Sample rate: {data_with_media['audio_sr']} Hz")
            logger.info(f"  - Duration: {data_with_media['example_audio_duration']:.2f}s")

        logger.info("\n" + "=" * 60)
        logger.info("SUCCESS: Data loader is working correctly!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"\nERROR loading subject: {e}", exc_info=True)
        return


if __name__ == "__main__":
    main()
