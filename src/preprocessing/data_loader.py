"""
Project Sullivan - USC-TIMIT Dataset Loader

This module provides a data loader for the USC-TIMIT Speech MRI dataset.
"""

import json
from pathlib import Path
from typing import Optional, Dict, List, Any

import numpy as np
from scipy.io import loadmat

from src.utils.io_utils import (
    load_mri_sequence,
    load_mri_from_video,
    load_audio as load_audio_file,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class USCTIMITLoader:
    """
    Loader for USC-TIMIT Speech MRI Dataset.

    The USC-TIMIT dataset contains real-time MRI videos and synchronized audio
    recordings from multiple speakers. This loader provides a unified interface
    to access the data.

    Attributes:
        data_root: Root directory of USC-TIMIT dataset
        subjects: List of discovered subjects with metadata
    """

    def __init__(self, data_root: str | Path):
        """
        Initialize USC-TIMIT dataset loader.

        Args:
            data_root: Path to USC-TIMIT dataset root directory

        Raises:
            FileNotFoundError: If data_root does not exist
        """
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.data_root}")

        logger.info(f"Initializing USC-TIMIT loader from: {self.data_root}")

        # Discover subjects
        self.subjects = self._discover_subjects()
        logger.info(f"Discovered {len(self.subjects)} subjects")

    def _discover_subjects(self) -> List[Dict[str, Any]]:
        """
        Scan directory and build subject registry.

        Returns:
            List[Dict]: List of subject metadata dictionaries
                Each dict contains:
                - subject_id: str
                - mri_path: Path to MRI data
                - audio_path: Path to audio data
                - metadata_path: Path to metadata file
                - mri_format: 'video' or 'frames'
                - fps: Frame rate (if available)
        """
        subjects = []

        # USC-TIMIT typically has structure:
        # data_root/
        #   subject_01/
        #     mri_video.mp4 or mri_frames/
        #     audio.wav
        #     metadata.json or metadata.mat

        # Search for subject directories
        subject_dirs = [d for d in self.data_root.iterdir() if d.is_dir()]

        for subject_dir in sorted(subject_dirs):
            subject_id = subject_dir.name

            # Skip hidden directories and common non-subject directories
            if subject_id.startswith(".") or subject_id in ["__pycache__", "logs"]:
                continue

            try:
                subject_info = self._parse_subject_directory(subject_dir, subject_id)
                if subject_info:
                    subjects.append(subject_info)
                    logger.debug(f"Discovered subject: {subject_id}")
            except Exception as e:
                logger.warning(f"Failed to parse subject {subject_id}: {e}")
                continue

        return subjects

    def _parse_subject_directory(
        self, subject_dir: Path, subject_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Parse a single subject directory.

        Args:
            subject_dir: Path to subject directory
            subject_id: Subject identifier

        Returns:
            Dict or None: Subject metadata if valid, None otherwise
        """
        subject_info = {
            "subject_id": subject_id,
            "directory": subject_dir,
        }

        # Look for MRI data (video or frames)
        # USC-TIMIT structure: subject_dir/2drt/video/*.mp4 (video contains audio)
        mri_video_dir = None
        mri_frames_dir = None

        # Check for 2drt/video directory (USC-TIMIT structure)
        video_dir = subject_dir / "2drt" / "video"
        if video_dir.exists() and video_dir.is_dir():
            video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
            if video_files:
                mri_video_dir = video_dir
                logger.debug(f"Found {len(video_files)} video files in {video_dir}")

        # Fallback: Check for video files directly in subject directory
        if not mri_video_dir:
            for video_ext in [".mp4", ".avi", ".mov"]:
                video_files = list(subject_dir.glob(f"*{video_ext}"))
                if video_files:
                    mri_video_dir = subject_dir
                    break

        # Check for frame directories
        if not mri_video_dir:
            for frames_dirname in ["mri_frames", "frames", "mri", "2drt/frames"]:
                frames_dir = subject_dir / frames_dirname
                if frames_dir.exists() and frames_dir.is_dir():
                    # Check if it contains image files
                    if list(frames_dir.glob("*.png")) or list(frames_dir.glob("*.jpg")):
                        mri_frames_dir = frames_dir
                        break

        # Must have either video or frames
        if mri_video_dir:
            subject_info["mri_path"] = mri_video_dir
            subject_info["mri_format"] = "video_dir"
        elif mri_frames_dir:
            subject_info["mri_path"] = mri_frames_dir
            subject_info["mri_format"] = "frames"
        else:
            logger.debug(f"No MRI data found for {subject_id}")
            return None

        # For USC-TIMIT, audio is embedded in video files
        # We'll mark audio_path as the same as mri_path
        subject_info["audio_path"] = mri_video_dir if mri_video_dir else None
        subject_info["audio_embedded"] = True  # Audio is in video files

        # Look for metadata
        metadata_file = None
        for meta_name in ["metadata.json", "info.json", "metadata.mat", "info.mat"]:
            meta_path = subject_dir / meta_name
            if meta_path.exists():
                metadata_file = meta_path
                break

        if metadata_file:
            subject_info["metadata_path"] = metadata_file
            # Try to parse metadata
            try:
                metadata = self._load_metadata(metadata_file)
                subject_info["metadata"] = metadata

                # Extract FPS if available
                if "fps" in metadata:
                    subject_info["fps"] = float(metadata["fps"])
                elif "frame_rate" in metadata:
                    subject_info["fps"] = float(metadata["frame_rate"])
            except Exception as e:
                logger.warning(f"Failed to load metadata for {subject_id}: {e}")

        return subject_info

    def _load_metadata(self, metadata_path: Path) -> Dict[str, Any]:
        """
        Load metadata from JSON or MAT file.

        Args:
            metadata_path: Path to metadata file

        Returns:
            Dict: Metadata dictionary
        """
        if metadata_path.suffix == ".json":
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif metadata_path.suffix == ".mat":
            # Load MATLAB file
            mat_data = loadmat(str(metadata_path))
            # Convert to regular dict (filter out MATLAB internal keys)
            metadata = {
                k: v for k, v in mat_data.items() if not k.startswith("__")
            }
            return metadata
        else:
            raise ValueError(f"Unsupported metadata format: {metadata_path.suffix}")

    def get_subject_ids(self) -> List[str]:
        """
        Get list of all subject IDs.

        Returns:
            List[str]: Subject IDs
        """
        return [s["subject_id"] for s in self.subjects]

    def get_subject_info(self, subject_id: str) -> Dict[str, Any]:
        """
        Get metadata for a specific subject.

        Args:
            subject_id: Subject identifier

        Returns:
            Dict: Subject metadata

        Raises:
            ValueError: If subject not found
        """
        for subject in self.subjects:
            if subject["subject_id"] == subject_id:
                return subject

        raise ValueError(f"Subject not found: {subject_id}")

    def load_subject(
        self,
        subject_id: str,
        load_mri: bool = True,
        load_audio: bool = True,
        target_audio_sr: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Load all data for a subject.

        Args:
            subject_id: Subject identifier
            load_mri: Whether to load MRI frames
            load_audio: Whether to load audio
            target_audio_sr: Target audio sample rate (None = original)

        Returns:
            Dict containing:
                - subject_id: str
                - mri_frames: np.ndarray (T, H, W) if load_mri=True
                - audio: np.ndarray (N,) if load_audio=True
                - audio_sr: int, sample rate
                - mri_fps: float, MRI frame rate
                - metadata: dict

        Raises:
            ValueError: If subject not found
        """
        subject_info = self.get_subject_info(subject_id)

        result = {
            "subject_id": subject_id,
            "metadata": subject_info.get("metadata", {}),
        }

        # For USC-TIMIT, each subject has multiple utterances (video files)
        # Return list of video files instead of loading all at once
        mri_format = subject_info["mri_format"]

        if mri_format == "video_dir":
            # USC-TIMIT: directory of video files
            video_dir = subject_info["mri_path"]
            video_files = sorted(list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi")))

            result["utterance_files"] = video_files
            result["num_utterances"] = len(video_files)
            result["mri_format"] = "video_dir"

            logger.info(f"Found {len(video_files)} utterances for {subject_id}")

            # Load first utterance as example if requested
            if load_mri and video_files:
                logger.info(f"Loading first utterance as example: {video_files[0].name}")
                mri_frames = load_mri_from_video(video_files[0], normalize=True)
                result["mri_frames_example"] = mri_frames
                result["example_num_frames"] = len(mri_frames)
                logger.info(f"  Example: {len(mri_frames)} frames, shape: {mri_frames.shape}")

            # Load first utterance audio if requested
            if load_audio and video_files:
                logger.info(f"Loading first utterance audio from: {video_files[0].name}")
                audio, audio_sr = load_audio_file(video_files[0], sr=target_audio_sr)
                result["audio_example"] = audio
                result["audio_sr"] = audio_sr
                result["example_audio_duration"] = len(audio) / audio_sr
                logger.info(
                    f"  Example audio: {len(audio)} samples at {audio_sr} Hz "
                    f"({result['example_audio_duration']:.2f}s)"
                )
        else:
            # Single video or frames (original behavior)
            if load_mri:
                logger.info(f"Loading MRI for {subject_id}")
                mri_path = subject_info["mri_path"]

                if mri_format == "video":
                    mri_frames = load_mri_from_video(mri_path, normalize=True)
                else:  # frames
                    # Determine image pattern
                    if list(mri_path.glob("*.png")):
                        pattern = "*.png"
                    elif list(mri_path.glob("*.jpg")):
                        pattern = "*.jpg"
                    else:
                        pattern = "*.png"  # default

                    mri_frames = load_mri_sequence(mri_path, pattern=pattern, normalize=True)

                result["mri_frames"] = mri_frames
                result["num_frames"] = len(mri_frames)
                logger.info(f"Loaded {len(mri_frames)} MRI frames, shape: {mri_frames.shape}")

            # Load audio
            if load_audio:
                logger.info(f"Loading audio for {subject_id}")
                audio_path = subject_info["audio_path"]

                audio, audio_sr = load_audio_file(audio_path, sr=target_audio_sr)

                result["audio"] = audio
                result["audio_sr"] = audio_sr
                result["audio_duration"] = len(audio) / audio_sr
                logger.info(
                    f"Loaded audio: {len(audio)} samples at {audio_sr} Hz "
                    f"({result['audio_duration']:.2f}s)"
                )

        # Get or estimate FPS
        if "fps" in subject_info:
            result["mri_fps"] = subject_info["fps"]
        else:
            # Default to 83 fps (measured from USC-TIMIT data)
            result["mri_fps"] = 83.28
            logger.debug("FPS not found in metadata, using measured default: 83.28")

        return result

    def load_batch(
        self,
        subject_ids: List[str],
        load_mri: bool = True,
        load_audio: bool = True,
        target_audio_sr: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load data for multiple subjects.

        Args:
            subject_ids: List of subject identifiers
            load_mri: Whether to load MRI frames
            load_audio: Whether to load audio
            target_audio_sr: Target audio sample rate

        Returns:
            List[Dict]: List of subject data dictionaries
        """
        results = []
        for subject_id in subject_ids:
            try:
                data = self.load_subject(
                    subject_id,
                    load_mri=load_mri,
                    load_audio=load_audio,
                    target_audio_sr=target_audio_sr,
                )
                results.append(data)
            except Exception as e:
                logger.error(f"Failed to load {subject_id}: {e}")
                continue

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.

        Returns:
            Dict: Dataset statistics
        """
        stats = {
            "num_subjects": len(self.subjects),
            "subject_ids": self.get_subject_ids(),
            "formats": {
                "video": sum(1 for s in self.subjects if s.get("mri_format") == "video"),
                "video_dir": sum(1 for s in self.subjects if s.get("mri_format") == "video_dir"),
                "frames": sum(1 for s in self.subjects if s.get("mri_format") == "frames"),
            },
            "has_metadata": sum(1 for s in self.subjects if "metadata" in s),
        }

        # Calculate FPS distribution if available
        fps_values = [s["fps"] for s in self.subjects if "fps" in s]
        if fps_values:
            stats["fps"] = {
                "min": float(np.min(fps_values)),
                "max": float(np.max(fps_values)),
                "mean": float(np.mean(fps_values)),
                "median": float(np.median(fps_values)),
            }

        return stats

    def __len__(self) -> int:
        """Return number of subjects."""
        return len(self.subjects)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"USCTIMITLoader(data_root={self.data_root}, "
            f"num_subjects={len(self.subjects)})"
        )
