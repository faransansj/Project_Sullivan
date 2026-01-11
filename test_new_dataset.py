"""
Test script to verify the new USC-TIMIT dataset is accessible.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.data_loader import USCTIMITLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)


def test_dataset_loading():
    """Test loading the full USC-TIMIT dataset."""

    # Path to the new full dataset
    dataset_path = Path("data/raw/usc_timit_full")

    logger.info("=" * 80)
    logger.info("Testing USC-TIMIT Full Dataset Loading")
    logger.info("=" * 80)

    # Initialize loader
    logger.info(f"\n1. Initializing loader from: {dataset_path}")
    try:
        loader = USCTIMITLoader(dataset_path)
        logger.info(f"   ✓ Loader initialized successfully")
        logger.info(f"   ✓ Found {len(loader)} subjects")
    except Exception as e:
        logger.error(f"   ✗ Failed to initialize loader: {e}")
        return False

    # Get statistics
    logger.info(f"\n2. Getting dataset statistics")
    try:
        stats = loader.get_statistics()
        logger.info(f"   ✓ Number of subjects: {stats['num_subjects']}")
        logger.info(f"   ✓ Subject IDs: {stats['subject_ids'][:5]}... (showing first 5)")
        logger.info(f"   ✓ Data formats: {stats['formats']}")
        if 'fps' in stats:
            logger.info(f"   ✓ FPS range: {stats['fps']['min']:.2f} - {stats['fps']['max']:.2f}")
    except Exception as e:
        logger.error(f"   ✗ Failed to get statistics: {e}")
        return False

    # Try loading one subject
    logger.info(f"\n3. Testing subject loading (first subject)")
    try:
        subject_ids = loader.get_subject_ids()
        if subject_ids:
            first_subject = subject_ids[0]
            logger.info(f"   Loading subject: {first_subject}")

            subject_data = loader.load_subject(
                first_subject,
                load_mri=True,
                load_audio=True
            )

            logger.info(f"   ✓ Subject loaded successfully")
            logger.info(f"   ✓ MRI format: {subject_data.get('mri_format')}")
            logger.info(f"   ✓ Number of utterances: {subject_data.get('num_utterances', 'N/A')}")

            if 'mri_frames_example' in subject_data:
                logger.info(f"   ✓ Example MRI shape: {subject_data['mri_frames_example'].shape}")

            if 'audio_example' in subject_data:
                logger.info(f"   ✓ Example audio: {len(subject_data['audio_example'])} samples at {subject_data['audio_sr']} Hz")
                logger.info(f"   ✓ Audio duration: {subject_data['example_audio_duration']:.2f}s")

            if 'utterance_files' in subject_data:
                logger.info(f"   ✓ Utterance files found: {len(subject_data['utterance_files'])}")
                logger.info(f"   ✓ First file: {subject_data['utterance_files'][0].name}")
        else:
            logger.warning("   ⚠ No subjects found")
            return False

    except Exception as e:
        logger.error(f"   ✗ Failed to load subject: {e}")
        import traceback
        traceback.print_exc()
        return False

    logger.info("\n" + "=" * 80)
    logger.info("✓ All tests passed successfully!")
    logger.info("=" * 80)

    return True


if __name__ == "__main__":
    success = test_dataset_loading()
    sys.exit(0 if success else 1)
