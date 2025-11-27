"""
Integration tests for src/preprocessing/data_loader.py
"""

import pytest
import numpy as np
from pathlib import Path
import json

from src.preprocessing.data_loader import USCTIMITLoader
from src.utils.io_utils import save_audio
import cv2


@pytest.fixture
def mock_dataset(tmp_path):
    """
    Create a mock USC-TIMIT dataset structure for testing.

    Creates:
        - 3 subjects with MRI video and audio
        - 2 subjects with MRI frames and audio
        - Various metadata configurations
    """
    dataset_root = tmp_path / "usc_timit_mock"
    dataset_root.mkdir()

    # Subject 1: Video format with metadata
    subject1_dir = dataset_root / "subject_01"
    subject1_dir.mkdir()

    # Create mock MRI video
    video_path = subject1_dir / "mri_video.mp4"
    _create_mock_video(video_path, num_frames=100, fps=50.0)

    # Create mock audio
    audio_path = subject1_dir / "audio.wav"
    audio = np.sin(2 * np.pi * 440 * np.linspace(0, 2, 32000)).astype(np.float32)
    save_audio(audio_path, audio, 16000)

    # Create metadata
    metadata = {
        "subject_id": "subject_01",
        "fps": 50.0,
        "audio_sr": 16000,
        "duration": 2.0,
    }
    with open(subject1_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    # Subject 2: Frames format with metadata
    subject2_dir = dataset_root / "subject_02"
    subject2_dir.mkdir()

    mri_frames_dir = subject2_dir / "mri_frames"
    mri_frames_dir.mkdir()

    # Create mock MRI frames
    for i in range(80):
        frame = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        cv2.imwrite(str(mri_frames_dir / f"frame_{i:04d}.png"), frame)

    # Create mock audio
    audio_path = subject2_dir / "audio.wav"
    audio = np.sin(2 * np.pi * 220 * np.linspace(0, 1.6, 25600)).astype(np.float32)
    save_audio(audio_path, audio, 16000)

    # Create metadata
    metadata = {
        "subject_id": "subject_02",
        "fps": 50.0,
    }
    with open(subject2_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    # Subject 3: Video format without metadata
    subject3_dir = dataset_root / "subject_03"
    subject3_dir.mkdir()

    video_path = subject3_dir / "mri.mp4"
    _create_mock_video(video_path, num_frames=120, fps=60.0)

    audio_path = subject3_dir / "audio.wav"
    audio = np.sin(2 * np.pi * 880 * np.linspace(0, 2, 32000)).astype(np.float32)
    save_audio(audio_path, audio, 16000)

    return dataset_root


def _create_mock_video(video_path: Path, num_frames: int, fps: float):
    """Create a mock video file for testing."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (256, 256), isColor=False)

    for i in range(num_frames):
        # Create a simple pattern
        frame = np.zeros((256, 256), dtype=np.uint8)
        # Add some variation
        cv2.circle(frame, (128, 128), 50 + i % 20, 200, -1)
        out.write(frame)

    out.release()


class TestUSCTIMITLoader:
    """Integration tests for USCTIMITLoader class."""

    def test_initialization(self, mock_dataset):
        """Test loader initialization."""
        loader = USCTIMITLoader(mock_dataset)

        assert loader.data_root == mock_dataset
        assert len(loader.subjects) == 3

    def test_initialization_nonexistent_dir(self, tmp_path):
        """Test initialization with nonexistent directory."""
        nonexistent = tmp_path / "nonexistent"

        with pytest.raises(FileNotFoundError):
            USCTIMITLoader(nonexistent)

    def test_discover_subjects(self, mock_dataset):
        """Test subject discovery."""
        loader = USCTIMITLoader(mock_dataset)

        assert len(loader.subjects) == 3

        # Check subject IDs
        subject_ids = loader.get_subject_ids()
        assert "subject_01" in subject_ids
        assert "subject_02" in subject_ids
        assert "subject_03" in subject_ids

    def test_subject_info_video_format(self, mock_dataset):
        """Test getting info for video format subject."""
        loader = USCTIMITLoader(mock_dataset)

        subject1_info = loader.get_subject_info("subject_01")

        assert subject1_info["subject_id"] == "subject_01"
        assert subject1_info["mri_format"] == "video"
        assert "mri_path" in subject1_info
        assert "audio_path" in subject1_info
        assert "metadata" in subject1_info
        assert subject1_info["fps"] == 50.0

    def test_subject_info_frames_format(self, mock_dataset):
        """Test getting info for frames format subject."""
        loader = USCTIMITLoader(mock_dataset)

        subject2_info = loader.get_subject_info("subject_02")

        assert subject2_info["subject_id"] == "subject_02"
        assert subject2_info["mri_format"] == "frames"
        assert "mri_path" in subject2_info
        assert "audio_path" in subject2_info

    def test_get_subject_info_not_found(self, mock_dataset):
        """Test getting info for nonexistent subject."""
        loader = USCTIMITLoader(mock_dataset)

        with pytest.raises(ValueError, match="Subject not found"):
            loader.get_subject_info("nonexistent")

    def test_load_subject_video_format(self, mock_dataset):
        """Test loading subject with video format MRI."""
        loader = USCTIMITLoader(mock_dataset)

        data = loader.load_subject("subject_01")

        assert "mri_frames" in data
        assert "audio" in data
        assert data["mri_frames"].ndim == 3  # (T, H, W)
        assert data["audio"].ndim == 1
        assert data["mri_fps"] == 50.0
        assert "audio_sr" in data
        assert "frame_timestamps" in data

    def test_load_subject_frames_format(self, mock_dataset):
        """Test loading subject with frames format MRI."""
        loader = USCTIMITLoader(mock_dataset)

        data = loader.load_subject("subject_02")

        assert "mri_frames" in data
        assert data["mri_frames"].shape[0] == 80  # 80 frames
        assert data["mri_frames"].shape[1:] == (256, 256)

    def test_load_subject_mri_only(self, mock_dataset):
        """Test loading MRI only."""
        loader = USCTIMITLoader(mock_dataset)

        data = loader.load_subject("subject_01", load_mri=True, load_audio=False)

        assert "mri_frames" in data
        assert "audio" not in data

    def test_load_subject_audio_only(self, mock_dataset):
        """Test loading audio only."""
        loader = USCTIMITLoader(mock_dataset)

        data = loader.load_subject("subject_01", load_mri=False, load_audio=True)

        assert "mri_frames" not in data
        assert "audio" in data

    def test_load_subject_with_resampling(self, mock_dataset):
        """Test loading subject with audio resampling."""
        loader = USCTIMITLoader(mock_dataset)

        data = loader.load_subject("subject_01", target_audio_sr=22050)

        assert data["audio_sr"] == 22050

    def test_load_batch(self, mock_dataset):
        """Test loading multiple subjects."""
        loader = USCTIMITLoader(mock_dataset)

        batch_data = loader.load_batch(["subject_01", "subject_02"])

        assert len(batch_data) == 2
        assert batch_data[0]["subject_id"] == "subject_01"
        assert batch_data[1]["subject_id"] == "subject_02"

    def test_load_batch_partial_failure(self, mock_dataset):
        """Test loading batch with some invalid subjects."""
        loader = USCTIMITLoader(mock_dataset)

        # Include one invalid subject
        batch_data = loader.load_batch(
            ["subject_01", "nonexistent", "subject_02"]
        )

        # Should successfully load the valid subjects
        assert len(batch_data) == 2
        loaded_ids = [d["subject_id"] for d in batch_data]
        assert "subject_01" in loaded_ids
        assert "subject_02" in loaded_ids

    def test_get_statistics(self, mock_dataset):
        """Test getting dataset statistics."""
        loader = USCTIMITLoader(mock_dataset)

        stats = loader.get_statistics()

        assert stats["num_subjects"] == 3
        assert len(stats["subject_ids"]) == 3
        assert "formats" in stats
        assert stats["formats"]["video"] == 2  # subject_01 and subject_03
        assert stats["formats"]["frames"] == 1  # subject_02
        assert stats["has_metadata"] == 2  # subject_01 and subject_02 have metadata

    def test_len(self, mock_dataset):
        """Test __len__ method."""
        loader = USCTIMITLoader(mock_dataset)

        assert len(loader) == 3

    def test_repr(self, mock_dataset):
        """Test __repr__ method."""
        loader = USCTIMITLoader(mock_dataset)

        repr_str = repr(loader)

        assert "USCTIMITLoader" in repr_str
        assert str(mock_dataset) in repr_str
        assert "3" in repr_str  # num_subjects


class TestDataLoaderEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataset(self, tmp_path):
        """Test loader with empty dataset directory."""
        empty_dir = tmp_path / "empty_dataset"
        empty_dir.mkdir()

        loader = USCTIMITLoader(empty_dir)

        assert len(loader.subjects) == 0
        assert len(loader.get_subject_ids()) == 0

    def test_subject_missing_audio(self, tmp_path):
        """Test subject with MRI but no audio."""
        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()

        subject_dir = dataset_root / "subject_no_audio"
        subject_dir.mkdir()

        # Only create MRI video, no audio
        video_path = subject_dir / "mri.mp4"
        _create_mock_video(video_path, num_frames=50, fps=50.0)

        loader = USCTIMITLoader(dataset_root)

        # Subject should not be discovered (requires both MRI and audio)
        assert len(loader.subjects) == 0

    def test_subject_missing_mri(self, tmp_path):
        """Test subject with audio but no MRI."""
        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()

        subject_dir = dataset_root / "subject_no_mri"
        subject_dir.mkdir()

        # Only create audio, no MRI
        audio_path = subject_dir / "audio.wav"
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)
        save_audio(audio_path, audio, 16000)

        loader = USCTIMITLoader(dataset_root)

        # Subject should not be discovered
        assert len(loader.subjects) == 0

    def test_load_subject_metadata_fallback(self, tmp_path):
        """Test loading subject without metadata (uses defaults)."""
        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()

        subject_dir = dataset_root / "subject_no_meta"
        subject_dir.mkdir()

        # Create MRI and audio, but no metadata
        video_path = subject_dir / "mri.mp4"
        _create_mock_video(video_path, num_frames=100, fps=50.0)

        audio_path = subject_dir / "audio.wav"
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 2, 32000)).astype(np.float32)
        save_audio(audio_path, audio, 16000)

        loader = USCTIMITLoader(dataset_root)
        data = loader.load_subject("subject_no_meta")

        # Should use default FPS
        assert data["mri_fps"] == 50.0
        assert "mri_frames" in data
        assert "audio" in data
