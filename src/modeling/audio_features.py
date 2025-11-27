"""
Audio Feature Extraction for Articulatory Parameter Prediction

This module extracts acoustic features from audio waveforms
for use in audio-to-parameter models.
"""

import numpy as np
import librosa
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import warnings


@dataclass
class AudioFeatureConfig:
    """Configuration for audio feature extraction"""

    # Sampling rate
    sr: int = 20000  # USC-TIMIT audio sampling rate

    # STFT parameters
    n_fft: int = 512
    hop_length: int = 200  # ~10ms hop at 20kHz
    win_length: int = 400  # ~20ms window

    # Mel-spectrogram parameters
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float = 8000.0  # Nyquist frequency

    # MFCC parameters
    n_mfcc: int = 13
    use_delta: bool = True
    use_delta_delta: bool = True

    # Normalization
    normalize: bool = True


class AudioFeatureExtractor:
    """Extract acoustic features from audio waveforms"""

    def __init__(self, config: Optional[AudioFeatureConfig] = None):
        """
        Args:
            config: Audio feature extraction configuration
        """
        self.config = config or AudioFeatureConfig()

        # Normalization statistics (fitted on training data)
        self.feature_mean = None
        self.feature_std = None

    def extract_mel_spectrogram(self,
                               audio: np.ndarray,
                               sr: Optional[int] = None) -> np.ndarray:
        """
        Extract log mel-spectrogram from audio

        Args:
            audio: (num_samples,) audio waveform
            sr: Sampling rate (if None, uses config.sr)

        Returns:
            (num_frames, n_mels) log mel-spectrogram
        """
        sr = sr or self.config.sr

        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            n_mels=self.config.n_mels,
            fmin=self.config.fmin,
            fmax=self.config.fmax
        )

        # Convert to log scale (dB)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Transpose to (time, freq)
        return log_mel_spec.T

    def extract_mfcc(self,
                    audio: np.ndarray,
                    sr: Optional[int] = None,
                    include_deltas: bool = True) -> np.ndarray:
        """
        Extract MFCC features from audio

        Args:
            audio: (num_samples,) audio waveform
            sr: Sampling rate
            include_deltas: Include delta and delta-delta features

        Returns:
            (num_frames, n_features) MFCC features
            n_features = n_mfcc * (1 + delta + delta_delta)
        """
        sr = sr or self.config.sr

        # Extract MFCCs
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=self.config.n_mfcc,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            n_mels=self.config.n_mels
        )

        features = [mfcc]

        if include_deltas and self.config.use_delta:
            delta = librosa.feature.delta(mfcc)
            features.append(delta)

        if include_deltas and self.config.use_delta_delta:
            delta2 = librosa.feature.delta(mfcc, order=2)
            features.append(delta2)

        # Concatenate and transpose to (time, features)
        features = np.concatenate(features, axis=0).T

        return features

    def extract_full_features(self,
                             audio: np.ndarray,
                             sr: Optional[int] = None,
                             feature_type: str = "mel") -> np.ndarray:
        """
        Extract full feature set from audio

        Args:
            audio: (num_samples,) audio waveform
            sr: Sampling rate
            feature_type: Type of features ("mel", "mfcc", "both")

        Returns:
            (num_frames, n_features) acoustic features
        """
        sr = sr or self.config.sr

        if feature_type == "mel":
            return self.extract_mel_spectrogram(audio, sr)

        elif feature_type == "mfcc":
            return self.extract_mfcc(audio, sr, include_deltas=True)

        elif feature_type == "both":
            mel = self.extract_mel_spectrogram(audio, sr)
            mfcc = self.extract_mfcc(audio, sr, include_deltas=True)

            # Ensure same number of frames
            min_frames = min(len(mel), len(mfcc))
            mel = mel[:min_frames]
            mfcc = mfcc[:min_frames]

            return np.concatenate([mel, mfcc], axis=1)

        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

    def synchronize_with_mri(self,
                            audio_features: np.ndarray,
                            mri_fps: float,
                            audio_fps: Optional[float] = None) -> np.ndarray:
        """
        Synchronize audio features with MRI frame rate

        Args:
            audio_features: (num_audio_frames, n_features) audio features
            mri_fps: MRI frame rate (frames per second)
            audio_fps: Audio feature frame rate (if None, computed from config)

        Returns:
            (num_mri_frames, n_features) synchronized features
        """
        if audio_fps is None:
            # Compute from hop_length
            audio_fps = self.config.sr / self.config.hop_length

        # Compute number of MRI frames
        duration = len(audio_features) / audio_fps
        num_mri_frames = int(duration * mri_fps)

        # Resample audio features to match MRI frame rate
        from scipy.interpolate import interp1d

        audio_times = np.arange(len(audio_features)) / audio_fps
        mri_times = np.arange(num_mri_frames) / mri_fps

        # Interpolate each feature dimension
        synchronized = np.zeros((num_mri_frames, audio_features.shape[1]))

        for i in range(audio_features.shape[1]):
            interpolator = interp1d(
                audio_times,
                audio_features[:, i],
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )
            synchronized[:, i] = interpolator(mri_times)

        return synchronized

    def fit_normalization(self, all_features: np.ndarray):
        """
        Fit normalization statistics from training data

        Args:
            all_features: (num_samples, n_features) all training features
        """
        self.feature_mean = np.mean(all_features, axis=0)
        self.feature_std = np.std(all_features, axis=0)

        # Prevent division by zero
        self.feature_std = np.where(self.feature_std < 1e-6, 1.0, self.feature_std)

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using fitted statistics

        Args:
            features: (num_frames, n_features) features

        Returns:
            Normalized features
        """
        if self.feature_mean is None or self.feature_std is None:
            raise ValueError("Must call fit_normalization() first")

        return (features - self.feature_mean) / self.feature_std

    def denormalize_features(self, normalized_features: np.ndarray) -> np.ndarray:
        """
        Denormalize features back to original scale

        Args:
            normalized_features: (num_frames, n_features) normalized features

        Returns:
            Original scale features
        """
        if self.feature_mean is None or self.feature_std is None:
            raise ValueError("Must call fit_normalization() first")

        return normalized_features * self.feature_std + self.feature_mean


def load_audio(audio_path: str,
               sr: int = 20000,
               mono: bool = True) -> Tuple[np.ndarray, int]:
    """
    Load audio file

    Args:
        audio_path: Path to audio file
        sr: Target sampling rate
        mono: Convert to mono

    Returns:
        (audio, sr) tuple
    """
    audio, loaded_sr = librosa.load(audio_path, sr=sr, mono=mono)
    return audio, loaded_sr


def validate_audio_features(features: np.ndarray,
                           expected_dims: Optional[int] = None) -> Dict[str, any]:
    """
    Validate extracted audio features

    Args:
        features: (num_frames, n_features) audio features
        expected_dims: Expected feature dimensionality

    Returns:
        Validation results dictionary
    """
    results = {
        'valid': True,
        'warnings': [],
        'statistics': {}
    }

    # Check for NaN or Inf
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        results['valid'] = False
        results['warnings'].append("Contains NaN or Inf values")

    # Check dimensionality
    if expected_dims is not None and features.shape[1] != expected_dims:
        results['warnings'].append(
            f"Expected {expected_dims} features, got {features.shape[1]}"
        )

    # Check for silence (all zeros)
    if np.all(features == 0):
        results['valid'] = False
        results['warnings'].append("All features are zero (possible silence)")

    # Statistical checks
    results['statistics']['shape'] = features.shape
    results['statistics']['mean'] = np.mean(features, axis=0)
    results['statistics']['std'] = np.std(features, axis=0)
    results['statistics']['min'] = np.min(features, axis=0)
    results['statistics']['max'] = np.max(features, axis=0)

    # Check dynamic range
    dynamic_range = np.max(features) - np.min(features)
    if dynamic_range < 1.0:
        results['warnings'].append(
            f"Low dynamic range ({dynamic_range:.2f}), possible audio issue"
        )

    results['statistics']['dynamic_range'] = dynamic_range

    return results
