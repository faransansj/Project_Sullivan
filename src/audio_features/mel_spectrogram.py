"""
Mel-Spectrogram Feature Extraction

This module extracts mel-spectrogram features from audio synchronized with MRI frames.
"""

import numpy as np
import librosa
from typing import Optional
import warnings


class MelSpectrogramExtractor:
    """
    Extract mel-spectrogram features synchronized with MRI frames.

    Parameters
    ----------
    n_mels : int, default=80
        Number of mel filterbanks
    n_fft : int, default=512
        FFT window size
    hop_length : int, default=160
        Hop length between frames
    window : str, default='hann'
        Window function
    fmin : float, default=0.0
        Minimum frequency
    fmax : float, optional
        Maximum frequency (None = sample_rate / 2)
    power : float, default=2.0
        Exponent for magnitude spectrogram (2.0 = power, 1.0 = magnitude)
    """

    def __init__(
        self,
        n_mels: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,
        window: str = 'hann',
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        power: float = 2.0
    ):
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window
        self.fmin = fmin
        self.fmax = fmax
        self.power = power

    def extract(
        self,
        audio: np.ndarray,
        sample_rate: int,
        num_mri_frames: int,
        mri_fps: float
    ) -> np.ndarray:
        """
        Extract mel-spectrogram features synchronized with MRI frames.

        Parameters
        ----------
        audio : np.ndarray
            Audio waveform, shape (num_samples,)
        sample_rate : int
            Audio sample rate in Hz
        num_mri_frames : int
            Number of MRI frames to synchronize with
        mri_fps : float
            MRI frame rate (frames per second)

        Returns
        -------
        features : np.ndarray
            Mel-spectrogram features, shape (num_mri_frames, n_mels)
        """
        # Extract mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            fmin=self.fmin,
            fmax=self.fmax,
            power=self.power
        )

        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Transpose to (time, freq)
        mel_spec_db = mel_spec_db.T  # Shape: (num_audio_frames, n_mels)

        # Synchronize with MRI frames
        synced_features = self._sync_to_mri_frames(
            mel_spec_db,
            sample_rate,
            num_mri_frames,
            mri_fps
        )

        return synced_features.astype(np.float32)

    def _sync_to_mri_frames(
        self,
        audio_features: np.ndarray,
        sample_rate: int,
        num_mri_frames: int,
        mri_fps: float
    ) -> np.ndarray:
        """
        Synchronize audio features with MRI frames.

        Uses linear interpolation to align audio frames with MRI frames.

        Parameters
        ----------
        audio_features : np.ndarray
            Audio features, shape (num_audio_frames, n_features)
        sample_rate : int
            Audio sample rate
        num_mri_frames : int
            Target number of MRI frames
        mri_fps : float
            MRI frame rate

        Returns
        -------
        synced_features : np.ndarray
            Synchronized features, shape (num_mri_frames, n_features)
        """
        num_audio_frames, n_features = audio_features.shape

        # Compute time for each audio frame
        audio_times = np.arange(num_audio_frames) * self.hop_length / sample_rate

        # Compute time for each MRI frame
        mri_times = np.arange(num_mri_frames) / mri_fps

        # Interpolate audio features to MRI times
        synced_features = np.zeros((num_mri_frames, n_features), dtype=np.float32)

        for i in range(n_features):
            synced_features[:, i] = np.interp(
                mri_times,
                audio_times,
                audio_features[:, i]
            )

        return synced_features

    def __repr__(self) -> str:
        return (f"MelSpectrogramExtractor(n_mels={self.n_mels}, "
                f"n_fft={self.n_fft}, hop_length={self.hop_length})")
