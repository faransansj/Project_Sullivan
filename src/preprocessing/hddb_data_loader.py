"""
HDDB USC-TIMIT Dataset Loader
Specialized loader for HDDB dataset with H5 MRI files and separate WAV audio files.
"""

import h5py
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HDDBLoader:
    """
    Loader for HDDB USC-TIMIT dataset.

    HDDB structure:
        /mnt/HDDB/dataset/my_dataset/dataset/
        ├── sub010/
        │   ├── 2drt/
        │   │   ├── recon/
        │   │   │   ├── sub010_2drt_01_vcv1_r1_recon.h5  # MRI data
        │   │   │   └── ...
        │   │   └── audio/
        │   │       ├── sub010_2drt_01_vcv1_r1_audio.wav  # Audio data
        │   │       └── ...
        │   └── 3d/  # (not used)
        └── ...
    """

    def __init__(self, data_root: str | Path):
        """
        Initialize HDDB dataset loader.

        Args:
            data_root: Path to HDDB dataset root
                      (e.g., /mnt/HDDB/dataset/my_dataset/dataset)
        """
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")

        logger.info(f"Initializing HDDB loader from: {self.data_root}")

        # Discover all subjects and utterances
        self.subjects = self._discover_subjects()
        logger.info(f"Discovered {len(self.subjects)} subjects")

        # Build utterance index
        self.utterances = self._build_utterance_index()
        logger.info(f"Total utterances: {len(self.utterances)}")

    def _discover_subjects(self) -> List[str]:
        """Discover all subject IDs."""
        subjects = sorted([
            d.name for d in self.data_root.iterdir()
            if d.is_dir() and d.name.startswith('sub')
        ])
        return subjects

    def _build_utterance_index(self) -> Dict[str, Dict[str, Path]]:
        """
        Build index of all utterances with their file paths.

        Returns:
            Dict mapping utterance_id to {'h5': path, 'wav': path, 'subject': id}
        """
        utterances = {}

        for subject in self.subjects:
            subject_path = self.data_root / subject
            recon_dir = subject_path / '2drt' / 'recon'
            audio_dir = subject_path / '2drt' / 'audio'

            if not recon_dir.exists() or not audio_dir.exists():
                logger.warning(f"Missing 2drt directories for {subject}")
                continue

            # Get all H5 files
            h5_files = sorted(recon_dir.glob('*.h5'))

            for h5_file in h5_files:
                # Extract utterance ID from filename
                # sub010_2drt_01_vcv1_r1_recon.h5 -> sub010_2drt_01_vcv1_r1
                utterance_id = h5_file.stem.replace('_recon', '')

                # Find corresponding audio file
                wav_file = audio_dir / f"{utterance_id}_audio.wav"

                if not wav_file.exists():
                    logger.warning(f"Audio file not found for {utterance_id}")
                    continue

                utterances[utterance_id] = {
                    'h5': h5_file,
                    'wav': wav_file,
                    'subject': subject
                }

        return utterances

    def load_mri_from_h5(self, h5_path: Path,
                         normalize: bool = True) -> np.ndarray:
        """
        Load MRI frames from H5 file.

        Args:
            h5_path: Path to H5 file
            normalize: Whether to normalize to [0, 1]

        Returns:
            np.ndarray: MRI frames, shape (T, H, W), dtype float32
        """
        with h5py.File(h5_path, 'r') as f:
            # Find the main dataset key
            # H5 structure may vary, typically has one main dataset
            keys = list(f.keys())

            # Try to find the main 3D array (frames, height, width)
            main_key = None
            for key in keys:
                if isinstance(f[key], h5py.Dataset):
                    shape = f[key].shape
                    if len(shape) == 3:  # (T, H, W)
                        main_key = key
                        break

            if main_key is None:
                raise ValueError(f"Could not find 3D dataset in {h5_path}")

            # Load data
            mri_frames = np.array(f[main_key], dtype=np.float32)

        # Normalize to [0, 1] if requested
        if normalize:
            min_val = mri_frames.min()
            max_val = mri_frames.max()
            if max_val > min_val:
                mri_frames = (mri_frames - min_val) / (max_val - min_val)

        return mri_frames

    def load_audio(self, wav_path: Path,
                   target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Load audio from WAV file.

        Args:
            wav_path: Path to WAV file
            target_sr: Target sample rate (None = keep original)

        Returns:
            Tuple of (audio, sample_rate)
            - audio: np.ndarray, shape (N,), dtype float32
            - sample_rate: int
        """
        audio, sr = sf.read(wav_path, dtype='float32')

        # Resample if needed
        if target_sr is not None and sr != target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        return audio, sr

    def load_utterance(self, utterance_id: str,
                       load_mri: bool = True,
                       load_audio: bool = True,
                       target_audio_sr: Optional[int] = None) -> Dict[str, Any]:
        """
        Load MRI and audio for a single utterance.

        Args:
            utterance_id: Utterance identifier (e.g., 'sub010_2drt_01_vcv1_r1')
            load_mri: Whether to load MRI frames
            load_audio: Whether to load audio
            target_audio_sr: Target audio sample rate

        Returns:
            Dict containing:
                - utterance_id: str
                - subject: str
                - mri_frames: np.ndarray (T, H, W) [if load_mri=True]
                - audio: np.ndarray (N,) [if load_audio=True]
                - audio_sr: int [if load_audio=True]
                - num_frames: int [if load_mri=True]
                - duration: float [if load_audio=True]
                - fps: float [if both loaded]
        """
        if utterance_id not in self.utterances:
            raise ValueError(f"Utterance not found: {utterance_id}")

        utterance_info = self.utterances[utterance_id]
        result = {
            'utterance_id': utterance_id,
            'subject': utterance_info['subject']
        }

        # Load MRI
        if load_mri:
            logger.debug(f"Loading MRI for {utterance_id}")
            mri_frames = self.load_mri_from_h5(utterance_info['h5'])
            result['mri_frames'] = mri_frames
            result['num_frames'] = len(mri_frames)
            result['mri_shape'] = mri_frames.shape

        # Load audio
        if load_audio:
            logger.debug(f"Loading audio for {utterance_id}")
            audio, sr = self.load_audio(utterance_info['wav'], target_sr=target_audio_sr)
            result['audio'] = audio
            result['audio_sr'] = sr
            result['duration'] = len(audio) / sr

        # Calculate FPS if both loaded
        if load_mri and load_audio:
            result['fps'] = result['num_frames'] / result['duration']

        return result

    def load_subject_utterances(self, subject_id: str,
                                 max_utterances: Optional[int] = None,
                                 load_mri: bool = True,
                                 load_audio: bool = True,
                                 target_audio_sr: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load all utterances for a subject.

        Args:
            subject_id: Subject ID (e.g., 'sub010')
            max_utterances: Maximum number of utterances to load (None = all)
            load_mri: Whether to load MRI frames
            load_audio: Whether to load audio
            target_audio_sr: Target audio sample rate

        Returns:
            List of utterance data dictionaries
        """
        # Get all utterances for this subject
        subject_utterances = [
            utt_id for utt_id, info in self.utterances.items()
            if info['subject'] == subject_id
        ]

        if max_utterances is not None:
            subject_utterances = subject_utterances[:max_utterances]

        logger.info(f"Loading {len(subject_utterances)} utterances for {subject_id}")

        results = []
        for utt_id in subject_utterances:
            try:
                data = self.load_utterance(
                    utt_id,
                    load_mri=load_mri,
                    load_audio=load_audio,
                    target_audio_sr=target_audio_sr
                )
                results.append(data)
            except Exception as e:
                logger.error(f"Failed to load {utt_id}: {e}")
                continue

        return results

    def get_subject_list(self) -> List[str]:
        """Get list of all subjects."""
        return self.subjects

    def get_utterance_list(self, subject_id: Optional[str] = None) -> List[str]:
        """
        Get list of utterances.

        Args:
            subject_id: If provided, return only utterances for this subject

        Returns:
            List of utterance IDs
        """
        if subject_id is None:
            return list(self.utterances.keys())
        else:
            return [
                utt_id for utt_id, info in self.utterances.items()
                if info['subject'] == subject_id
            ]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.

        Returns:
            Dict with statistics
        """
        stats = {
            'num_subjects': len(self.subjects),
            'num_utterances': len(self.utterances),
            'subjects': self.subjects,
        }

        # Utterances per subject
        utterances_per_subject = {}
        for subject in self.subjects:
            count = sum(1 for info in self.utterances.values()
                       if info['subject'] == subject)
            utterances_per_subject[subject] = count

        stats['utterances_per_subject'] = utterances_per_subject
        stats['avg_utterances_per_subject'] = np.mean(list(utterances_per_subject.values()))

        return stats

    def __len__(self) -> int:
        """Return total number of utterances."""
        return len(self.utterances)

    def __repr__(self) -> str:
        return (
            f"HDDBLoader(data_root={self.data_root}, "
            f"subjects={len(self.subjects)}, "
            f"utterances={len(self.utterances)})"
        )
