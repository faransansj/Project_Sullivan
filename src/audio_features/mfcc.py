"""
MFCC Feature Extraction

This module extracts MFCC features from audio synchronized with MRI frames.
"""

import numpy as np
import librosa
from typing import Optional


class MFCCExtractor:
    """
    Extract MFCC features synchronized with MRI frames.

    Parameters
    ----------
    n_mfcc : int, default=13
        Number of MFCC coefficients to extract
    n_fft : int, default=512
        FFT window size
    hop_length : int, default=160
        Hop length between frames
    n_mels : int, default=40
        Number of mel filterbanks (for computing MFCC)
    fmin : float, default=0.0
        Minimum frequency
    fmax : float, optional
        Maximum frequency (None = sample_rate / 2)
    """

    def __init__(
        self,
        n_mfcc: int = 13,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: int = 40,
        fmin: float = 0.0,
        fmax: Optional[float] = None
    ):
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

    def extract(
        self,
        audio: np.ndarray,
        sample_rate: int,
        num_mri_frames: int,
        mri_fps: float
    ) -> np.ndarray:
        """
        Extract MFCC features synchronized with MRI frames.

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
            MFCC features, shape (num_mri_frames, n_mfcc)
        """
        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )

        # Transpose to (time, features)
        mfcc = mfcc.T  # Shape: (num_audio_frames, n_mfcc)

        # Synchronize with MRI frames
        synced_features = self._sync_to_mri_frames(
            mfcc,
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
        return (f"MFCCExtractor(n_mfcc={self.n_mfcc}, "
                f"n_fft={self.n_fft}, hop_length={self.hop_length})")
