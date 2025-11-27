"""
Unit tests for src/preprocessing/alignment.py
"""

import pytest
import numpy as np

from src.preprocessing.alignment import (
    AudioMRIAligner,
    align_audio_mri,
    validate_alignment,
)


class TestAudioMRIAligner:
    """Tests for AudioMRIAligner class."""

    def test_initialization(self):
        """Test aligner initialization."""
        aligner = AudioMRIAligner(mri_fps=83.28)
        assert aligner.mri_fps == 83.28
        assert aligner.jaw_region is None
        assert aligner.smooth_sigma == 10.0

    def test_initialization_with_jaw_region(self):
        """Test initialization with jaw region."""
        jaw_region = (20, 60, 10, 70)
        aligner = AudioMRIAligner(mri_fps=50.0, jaw_region=jaw_region)
        assert aligner.jaw_region == jaw_region

    def test_align_synthetic_data(self):
        """Test alignment with synthetic synchronized data."""
        # Create synthetic MRI frames with motion
        np.random.seed(42)
        num_frames = 100
        fps = 50.0
        sr = 16000

        # Create frames with temporal pattern
        frames = np.zeros((num_frames, 64, 64), dtype=np.float32)
        for i in range(num_frames):
            # Simulate jaw movement (sine wave)
            motion = np.sin(2 * np.pi * i / 20)
            frames[i, 30:40, :] = motion * 0.5 + 0.5

        # Create audio with similar pattern
        duration = num_frames / fps
        t = np.arange(0, duration, 1/sr)
        audio = np.sin(2 * np.pi * t / (20/fps))
        audio = audio.astype(np.float32)

        # Align
        aligner = AudioMRIAligner(mri_fps=fps, smooth_sigma=5.0)
        result = aligner.align(frames, audio, sr)

        # Check result structure
        assert 'offset_samples' in result
        assert 'offset_seconds' in result
        assert 'correlation' in result
        assert 'mri_energy' in result
        assert 'audio_envelope' in result
        assert 'aligned_audio' in result
        assert 'frame_timestamps' in result

        # Check shapes
        assert len(result['mri_energy']) == num_frames
        assert len(result['audio_envelope']) == len(audio)
        assert len(result['frame_timestamps']) == num_frames

        # Correlation should be reasonable for synchronized data
        assert result['correlation'] > 0.0

    def test_align_with_offset(self):
        """Test alignment detects offset."""
        np.random.seed(42)
        num_frames = 50
        fps = 50.0
        sr = 16000

        # Create frames
        frames = np.random.rand(num_frames, 32, 32).astype(np.float32)

        # Create audio that's longer (simulating delayed start)
        audio_duration = (num_frames / fps) + 0.5  # Extra 0.5s
        audio = np.random.rand(int(audio_duration * sr)).astype(np.float32)

        aligner = AudioMRIAligner(mri_fps=fps)
        result = aligner.align(frames, audio, sr)

        # Should detect some offset
        assert isinstance(result['offset_seconds'], float)

    def test_extract_motion_energy(self):
        """Test motion energy extraction."""
        # Create frames with known motion
        frames = np.zeros((50, 64, 64), dtype=np.float32)
        # Frame 25 has big change
        frames[25:, 20:40, 20:40] = 1.0

        aligner = AudioMRIAligner(mri_fps=50.0)
        energy = aligner._extract_motion_energy(frames)

        # Check shape
        assert len(energy) == 50

        # Energy should spike at frame 25
        assert energy[25] > energy[0]
        assert energy[25] > energy[49]

    def test_extract_motion_energy_with_jaw_region(self):
        """Test motion energy with jaw region."""
        frames = np.random.rand(30, 64, 64).astype(np.float32)
        jaw_region = (20, 50, 10, 54)

        aligner = AudioMRIAligner(mri_fps=50.0, jaw_region=jaw_region)
        energy = aligner._extract_motion_energy(frames)

        assert len(energy) == 30
        assert np.all(energy >= 0)
        assert np.all(energy <= 1)

    def test_extract_audio_envelope(self):
        """Test audio envelope extraction."""
        sr = 16000
        duration = 1.0

        # Create audio with amplitude modulation
        t = np.arange(0, duration, 1/sr)
        carrier = np.sin(2 * np.pi * 440 * t)
        envelope_signal = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)  # 2 Hz modulation
        audio = (carrier * envelope_signal).astype(np.float32)

        aligner = AudioMRIAligner(mri_fps=50.0, smooth_sigma=5.0)
        envelope = aligner._extract_audio_envelope(audio, sr)

        # Check shape
        assert len(envelope) == len(audio)

        # Envelope should be normalized
        assert np.min(envelope) >= 0
        assert np.max(envelope) <= 1

    def test_invalid_mri_shape(self):
        """Test that invalid MRI shape raises error."""
        frames = np.random.rand(10, 10)  # 2D instead of 3D
        audio = np.random.rand(16000).astype(np.float32)

        aligner = AudioMRIAligner()
        with pytest.raises(ValueError, match="Expected 3D MRI frames"):
            aligner.align(frames, audio, 16000)

    def test_invalid_audio_shape(self):
        """Test that invalid audio shape raises error."""
        frames = np.random.rand(50, 32, 32).astype(np.float32)
        audio = np.random.rand(10, 16000).astype(np.float32)  # 2D instead of 1D

        aligner = AudioMRIAligner()
        with pytest.raises(ValueError, match="Expected 1D audio"):
            aligner.align(frames, audio, 16000)


class TestConvenienceFunction:
    """Tests for align_audio_mri convenience function."""

    def test_align_audio_mri(self):
        """Test convenience function."""
        np.random.seed(42)
        frames = np.random.rand(50, 32, 32).astype(np.float32)
        audio = np.random.rand(40000).astype(np.float32)

        result = align_audio_mri(frames, audio, audio_sr=16000, mri_fps=50.0)

        assert 'offset_seconds' in result
        assert 'correlation' in result
        assert 'aligned_audio' in result


class TestValidateAlignment:
    """Tests for validate_alignment function."""

    def test_valid_alignment(self):
        """Test validation of good alignment."""
        result = {
            'correlation': 0.75,
            'offset_seconds': 0.1,
            'mri_energy': np.random.rand(100),
        }

        is_valid, msg = validate_alignment(result, min_correlation=0.5)

        assert is_valid is True
        assert "Valid alignment" in msg
        assert "0.75" in msg

    def test_low_correlation(self):
        """Test rejection of low correlation."""
        result = {
            'correlation': 0.3,
            'offset_seconds': 0.1,
            'mri_energy': np.random.rand(100),
        }

        is_valid, msg = validate_alignment(result, min_correlation=0.5)

        assert is_valid is False
        assert "Low correlation" in msg
        assert "0.3" in msg

    def test_large_offset(self):
        """Test rejection of large offset."""
        result = {
            'correlation': 0.8,
            'offset_seconds': 10.0,  # Very large offset
            'mri_energy': np.random.rand(100),  # ~1.2 seconds at 83fps
        }

        is_valid, msg = validate_alignment(result, min_correlation=0.5)

        assert is_valid is False
        assert "Large offset" in msg

    def test_custom_threshold(self):
        """Test custom correlation threshold."""
        result = {
            'correlation': 0.6,
            'offset_seconds': 0.05,
            'mri_energy': np.random.rand(100),
        }

        # Should pass with 0.5 threshold
        is_valid, _ = validate_alignment(result, min_correlation=0.5)
        assert is_valid is True

        # Should fail with 0.7 threshold
        is_valid, _ = validate_alignment(result, min_correlation=0.7)
        assert is_valid is False


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_short_mri(self):
        """Test with very short MRI sequence."""
        frames = np.random.rand(5, 32, 32).astype(np.float32)
        audio = np.random.rand(8000).astype(np.float32)

        aligner = AudioMRIAligner(mri_fps=50.0)
        result = aligner.align(frames, audio, 16000)

        assert 'aligned_audio' in result

    def test_audio_shorter_than_mri(self):
        """Test when audio is shorter than MRI."""
        frames = np.random.rand(100, 32, 32).astype(np.float32)  # 2 seconds at 50fps
        audio = np.random.rand(8000).astype(np.float32)  # 0.5 seconds at 16kHz

        aligner = AudioMRIAligner(mri_fps=50.0)
        result = aligner.align(frames, audio, 16000)

        # Should still produce result (but may warn)
        assert 'aligned_audio' in result

    def test_audio_much_longer_than_mri(self):
        """Test when audio is much longer than MRI."""
        frames = np.random.rand(50, 32, 32).astype(np.float32)  # 1 second at 50fps
        audio = np.random.rand(80000).astype(np.float32)  # 5 seconds at 16kHz

        aligner = AudioMRIAligner(mri_fps=50.0)
        result = aligner.align(frames, audio, 16000)

        # Aligned audio should be trimmed to match MRI duration
        expected_len = int((len(frames) / 50.0) * 16000)
        assert len(result['aligned_audio']) == expected_len

    def test_silent_audio(self):
        """Test with silent audio."""
        frames = np.random.rand(50, 32, 32).astype(np.float32)
        audio = np.zeros(16000, dtype=np.float32)

        aligner = AudioMRIAligner()
        result = aligner.align(frames, audio, 16000)

        # Should complete without error
        assert 'aligned_audio' in result

    def test_static_frames(self):
        """Test with static (no motion) frames."""
        frames = np.ones((50, 32, 32), dtype=np.float32)
        audio = np.random.rand(20000).astype(np.float32)

        aligner = AudioMRIAligner(mri_fps=50.0)
        result = aligner.align(frames, audio, 16000)

        # Should complete without error
        assert 'aligned_audio' in result
        # Correlation might be low but shouldn't crash
        assert isinstance(result['correlation'], float)
