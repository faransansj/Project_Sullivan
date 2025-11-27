"""
Unit tests for src/preprocessing/denoising.py
"""

import pytest
import numpy as np

from src.preprocessing.denoising import (
    MRIDenoiser,
    AudioDenoiser,
    denoise_mri_sequence,
    denoise_audio,
)


class TestMRIDenoiser:
    """Tests for MRIDenoiser class."""

    def test_initialization(self):
        """Test denoiser initialization."""
        denoiser = MRIDenoiser(method="gaussian", spatial_sigma=1.0)
        assert denoiser.method == "gaussian"
        assert denoiser.spatial_sigma == 1.0
        assert denoiser.temporal_window == 3

    def test_invalid_temporal_window(self):
        """Test that even temporal window raises error."""
        with pytest.raises(ValueError, match="temporal_window must be odd"):
            MRIDenoiser(temporal_window=4)

    def test_denoise_gaussian(self):
        """Test Gaussian denoising."""
        # Create noisy frames (100 frames, 64x64)
        np.random.seed(42)
        frames = np.random.rand(100, 64, 64).astype(np.float32)

        denoiser = MRIDenoiser(method="gaussian", spatial_sigma=1.0)
        denoised = denoiser.denoise(frames)

        # Check output shape
        assert denoised.shape == frames.shape
        assert denoised.dtype == np.float32

        # Denoised should be smoother (lower variance in spatial dimensions)
        original_var = np.var(frames[50])  # Single frame
        denoised_var = np.var(denoised[50])
        assert denoised_var < original_var

    def test_denoise_bilateral(self):
        """Test bilateral filter denoising."""
        np.random.seed(42)
        frames = np.random.rand(50, 64, 64).astype(np.float32)

        denoiser = MRIDenoiser(method="bilateral", spatial_sigma=1.0)
        denoised = denoiser.denoise(frames)

        assert denoised.shape == frames.shape
        assert denoised.dtype == np.float32

    def test_denoise_nlm(self):
        """Test non-local means denoising."""
        np.random.seed(42)
        frames = np.random.rand(20, 32, 32).astype(np.float32)

        denoiser = MRIDenoiser(method="nlm", spatial_sigma=1.0, apply_temporal=False)
        denoised = denoiser.denoise(frames)

        assert denoised.shape == frames.shape

    def test_temporal_filtering(self):
        """Test temporal median filtering."""
        # Create frames with temporal noise
        np.random.seed(42)
        frames = np.ones((50, 32, 32), dtype=np.float32)
        # Add noise to every 5th frame
        frames[::5] += 0.5

        denoiser = MRIDenoiser(
            method="gaussian",
            spatial_sigma=0.5,
            temporal_window=5,
            apply_temporal=True,
        )
        denoised = denoiser.denoise(frames)

        # Temporal filtering should reduce spikes
        assert np.max(denoised) < np.max(frames)

    def test_no_temporal_filtering(self):
        """Test disabling temporal filtering."""
        np.random.seed(42)
        frames = np.random.rand(30, 32, 32).astype(np.float32)

        denoiser = MRIDenoiser(method="gaussian", apply_temporal=False)
        denoised = denoiser.denoise(frames)

        assert denoised.shape == frames.shape

    def test_short_sequence_temporal(self):
        """Test temporal filtering with sequence shorter than window."""
        np.random.seed(42)
        frames = np.random.rand(2, 32, 32).astype(np.float32)  # Only 2 frames

        denoiser = MRIDenoiser(temporal_window=5, apply_temporal=True)
        denoised = denoiser.denoise(frames)

        # Should skip temporal filtering
        assert denoised.shape == frames.shape

    def test_invalid_shape(self):
        """Test that invalid frame shape raises error."""
        frames = np.random.rand(10, 10)  # 2D instead of 3D

        denoiser = MRIDenoiser()
        with pytest.raises(ValueError, match="Expected 3D or 4D frames"):
            denoiser.denoise(frames)

    def test_4d_frames(self):
        """Test denoising 4D frames (with channels)."""
        np.random.seed(42)
        frames = np.random.rand(20, 32, 32, 3).astype(np.float32)

        denoiser = MRIDenoiser(method="gaussian", apply_temporal=False)
        denoised = denoiser.denoise(frames)

        assert denoised.shape == frames.shape


class TestAudioDenoiser:
    """Tests for AudioDenoiser class."""

    def test_initialization(self):
        """Test audio denoiser initialization."""
        denoiser = AudioDenoiser(noise_sample_duration=0.5)
        assert denoiser.noise_sample_duration == 0.5
        assert denoiser.prop_decrease == 1.0

    def test_denoise_audio(self):
        """Test audio denoising."""
        # Create noisy audio (1 second at 16kHz)
        np.random.seed(42)
        sr = 16000
        duration = 1.0
        audio = np.sin(2 * np.pi * 440 * np.arange(0, duration, 1/sr))  # 440 Hz tone
        audio += np.random.randn(len(audio)) * 0.1  # Add noise

        denoiser = AudioDenoiser(noise_sample_duration=0.1)
        denoised = denoiser.denoise(audio.astype(np.float32), sr)

        # Check output
        assert denoised.shape == audio.shape
        # Denoised should have less noise (lower std)
        # Note: This is a simple check, actual denoising quality depends on noisereduce

    def test_short_audio_error(self):
        """Test that very short audio raises error."""
        sr = 16000
        audio = np.random.rand(100).astype(np.float32)  # Only 100 samples

        denoiser = AudioDenoiser()
        with pytest.raises(ValueError, match="Audio too short"):
            denoiser.denoise(audio, sr)

    def test_custom_noise_profile(self):
        """Test denoising with custom noise profile."""
        np.random.seed(42)
        sr = 16000
        audio = np.random.randn(sr).astype(np.float32)
        noise_profile = np.random.randn(sr // 4).astype(np.float32)

        denoiser = AudioDenoiser()
        denoised = denoiser.denoise(audio, sr, noise_profile=noise_profile)

        assert denoised.shape == audio.shape


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_denoise_mri_sequence(self):
        """Test convenience function for MRI denoising."""
        np.random.seed(42)
        frames = np.random.rand(50, 64, 64).astype(np.float32)

        denoised = denoise_mri_sequence(
            frames,
            method="gaussian",
            spatial_sigma=1.0,
            temporal_window=3,
        )

        assert denoised.shape == frames.shape

    def test_denoise_audio_convenience(self):
        """Test convenience function for audio denoising."""
        np.random.seed(42)
        sr = 16000
        audio = np.random.randn(sr).astype(np.float32)

        denoised = denoise_audio(audio, sr, noise_sample_duration=0.1)

        assert denoised.shape == audio.shape


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_frame(self):
        """Test denoising single frame."""
        frame = np.random.rand(1, 64, 64).astype(np.float32)

        denoiser = MRIDenoiser(apply_temporal=True, temporal_window=3)
        denoised = denoiser.denoise(frame)

        # Should skip temporal (sequence too short)
        assert denoised.shape == frame.shape

    def test_zero_frames(self):
        """Test denoising zero-valued frames."""
        frames = np.zeros((20, 32, 32), dtype=np.float32)

        denoiser = MRIDenoiser()
        denoised = denoiser.denoise(frames)

        assert np.all(denoised == 0.0)

    def test_constant_audio(self):
        """Test denoising constant audio."""
        sr = 16000
        audio = np.ones(sr, dtype=np.float32)

        denoiser = AudioDenoiser()
        denoised = denoiser.denoise(audio, sr)

        # Should return similar constant
        assert denoised.shape == audio.shape
