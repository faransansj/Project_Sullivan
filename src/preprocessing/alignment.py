"""
Project Sullivan - Audio-MRI Alignment

This module provides algorithms for synchronizing audio signals with MRI frames.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from scipy import signal, interpolate
from scipy.ndimage import gaussian_filter1d

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AudioMRIAligner:
    """
    Aligner for synchronizing audio signals with MRI frame sequences.

    Uses cross-correlation between MRI motion energy and audio envelope
    to find optimal temporal alignment.
    """

    def __init__(
        self,
        mri_fps: float = 83.28,
        jaw_region: Optional[Tuple[int, int, int, int]] = None,
        smooth_sigma: float = 10.0,
    ):
        """
        Initialize audio-MRI aligner.

        Args:
            mri_fps: MRI frame rate (frames per second)
            jaw_region: Optional (y1, y2, x1, x2) to focus on jaw movement.
                If None, uses entire frame.
            smooth_sigma: Gaussian smoothing sigma (in ms) for signals
        """
        self.mri_fps = mri_fps
        self.jaw_region = jaw_region
        self.smooth_sigma = smooth_sigma

        logger.info(
            f"AudioMRIAligner initialized: fps={mri_fps}, "
            f"jaw_region={jaw_region}, smooth_sigma={smooth_sigma}ms"
        )

    def align(
        self,
        mri_frames: np.ndarray,
        audio: np.ndarray,
        audio_sr: int,
        method: str = "cross_correlation",
    ) -> Dict[str, Any]:
        """
        Align audio with MRI frames.

        Args:
            mri_frames: MRI frames (T, H, W)
            audio: Audio signal (N,)
            audio_sr: Audio sample rate
            method: Alignment method ("cross_correlation")

        Returns:
            Dict containing:
                - offset_samples: int, audio sample offset
                - offset_seconds: float, time offset in seconds
                - correlation: float, correlation coefficient (quality metric)
                - mri_energy: np.ndarray, MRI motion energy
                - audio_envelope: np.ndarray, audio envelope
                - aligned_audio: np.ndarray, time-shifted audio
                - frame_timestamps: np.ndarray, MRI frame timestamps

        Raises:
            ValueError: If inputs are invalid
        """
        if mri_frames.ndim != 3:
            raise ValueError(f"Expected 3D MRI frames, got shape {mri_frames.shape}")
        if audio.ndim != 1:
            raise ValueError(f"Expected 1D audio, got shape {audio.shape}")

        logger.info(
            f"Aligning {len(mri_frames)} MRI frames with "
            f"{len(audio)} audio samples ({len(audio)/audio_sr:.2f}s)"
        )

        if method == "cross_correlation":
            result = self._align_cross_correlation(mri_frames, audio, audio_sr)
        else:
            raise ValueError(f"Unknown method: {method}")

        logger.info(
            f"Alignment completed: offset={result['offset_seconds']:.3f}s, "
            f"correlation={result['correlation']:.3f}"
        )

        return result

    def _align_cross_correlation(
        self,
        mri_frames: np.ndarray,
        audio: np.ndarray,
        audio_sr: int,
    ) -> Dict[str, Any]:
        """
        Align using cross-correlation between motion energy and audio envelope.

        Args:
            mri_frames: MRI frames (T, H, W)
            audio: Audio signal (N,)
            audio_sr: Audio sample rate

        Returns:
            Dict: Alignment results
        """
        # Step 1: Extract MRI motion energy
        mri_energy = self._extract_motion_energy(mri_frames)

        # Step 2: Extract audio envelope
        audio_envelope = self._extract_audio_envelope(audio, audio_sr)

        # Step 3: Resample MRI energy to audio sample rate
        frame_timestamps = np.arange(len(mri_energy)) / self.mri_fps
        audio_timestamps = np.arange(len(audio_envelope)) / audio_sr

        # Interpolate MRI energy to audio timestamps
        interp_func = interpolate.interp1d(
            frame_timestamps,
            mri_energy,
            kind='linear',
            bounds_error=False,
            fill_value=0.0,
        )

        # Only interpolate where we have MRI data
        mri_duration = len(mri_energy) / self.mri_fps
        audio_duration = len(audio_envelope) / audio_sr

        if audio_duration < mri_duration * 0.5:
            logger.warning(
                f"Audio ({audio_duration:.2f}s) much shorter than "
                f"MRI ({mri_duration:.2f}s), alignment may be poor"
            )

        # Create resampled MRI energy at audio sample rate
        max_duration = max(mri_duration, audio_duration)
        resample_timestamps = np.arange(0, max_duration, 1/audio_sr)
        mri_energy_resampled = interp_func(resample_timestamps)

        # Step 4: Normalize signals for correlation
        mri_norm = (mri_energy_resampled - np.mean(mri_energy_resampled)) / (
            np.std(mri_energy_resampled) + 1e-8
        )
        audio_norm = (audio_envelope - np.mean(audio_envelope)) / (
            np.std(audio_envelope) + 1e-8
        )

        # Step 5: Cross-correlation
        # Pad shorter signal
        max_len = max(len(mri_norm), len(audio_norm))
        mri_padded = np.pad(mri_norm, (0, max_len - len(mri_norm)))
        audio_padded = np.pad(audio_norm, (0, max_len - len(audio_norm)))

        # Compute cross-correlation
        correlation = signal.correlate(audio_padded, mri_padded, mode='same')

        # Find peak
        peak_idx = np.argmax(correlation)
        offset_samples = peak_idx - len(correlation) // 2

        # Correlation coefficient at peak
        peak_corr = correlation[peak_idx] / len(correlation)

        # Step 6: Align audio
        if offset_samples > 0:
            # Audio needs to be delayed (pad beginning)
            aligned_audio = np.pad(audio, (offset_samples, 0), mode='constant')
        else:
            # Audio needs to be advanced (trim beginning)
            aligned_audio = audio[-offset_samples:]

        # Trim to match MRI duration
        expected_audio_len = int(mri_duration * audio_sr)
        if len(aligned_audio) > expected_audio_len:
            aligned_audio = aligned_audio[:expected_audio_len]
        elif len(aligned_audio) < expected_audio_len:
            aligned_audio = np.pad(
                aligned_audio,
                (0, expected_audio_len - len(aligned_audio)),
                mode='constant'
            )

        return {
            'offset_samples': offset_samples,
            'offset_seconds': offset_samples / audio_sr,
            'correlation': float(peak_corr),
            'mri_energy': mri_energy,
            'audio_envelope': audio_envelope,
            'aligned_audio': aligned_audio,
            'frame_timestamps': frame_timestamps,
            'audio_sr': audio_sr,
        }

    def _extract_motion_energy(self, frames: np.ndarray) -> np.ndarray:
        """
        Extract motion energy from MRI frames.

        Computes frame-to-frame difference to capture articulator movement.

        Args:
            frames: MRI frames (T, H, W)

        Returns:
            np.ndarray: Motion energy per frame (T,)
        """
        # Apply jaw region if specified
        if self.jaw_region is not None:
            y1, y2, x1, x2 = self.jaw_region
            frames_roi = frames[:, y1:y2, x1:x2]
        else:
            frames_roi = frames

        # Compute frame differences
        diff = np.diff(frames_roi, axis=0, prepend=frames_roi[0:1])

        # Sum absolute differences per frame
        energy = np.sum(np.abs(diff), axis=(1, 2))

        # Smooth with Gaussian
        if self.smooth_sigma > 0:
            # Convert sigma from ms to frames
            sigma_frames = (self.smooth_sigma / 1000.0) * self.mri_fps
            energy = gaussian_filter1d(energy, sigma=sigma_frames)

        # Normalize to [0, 1]
        energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)

        return energy

    def _extract_audio_envelope(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract audio envelope using Hilbert transform.

        Args:
            audio: Audio signal (N,)
            sr: Sample rate

        Returns:
            np.ndarray: Audio envelope (N,)
        """
        # Hilbert transform to get analytic signal
        analytic_signal = signal.hilbert(audio)

        # Envelope is absolute value
        envelope = np.abs(analytic_signal)

        # Smooth with Gaussian
        if self.smooth_sigma > 0:
            # Convert sigma from ms to samples
            sigma_samples = (self.smooth_sigma / 1000.0) * sr
            envelope = gaussian_filter1d(envelope, sigma=sigma_samples)

        # Normalize to [0, 1]
        envelope = (envelope - envelope.min()) / (envelope.max() - envelope.min() + 1e-8)

        return envelope


def align_audio_mri(
    mri_frames: np.ndarray,
    audio: np.ndarray,
    audio_sr: int,
    mri_fps: float = 83.28,
) -> Dict[str, Any]:
    """
    Convenience function to align audio with MRI frames.

    Args:
        mri_frames: MRI frames (T, H, W)
        audio: Audio signal (N,)
        audio_sr: Audio sample rate
        mri_fps: MRI frame rate

    Returns:
        Dict: Alignment results

    Example:
        >>> result = align_audio_mri(mri_frames, audio, audio_sr=16000)
        >>> aligned_audio = result['aligned_audio']
        >>> correlation = result['correlation']
    """
    aligner = AudioMRIAligner(mri_fps=mri_fps)
    return aligner.align(mri_frames, audio, audio_sr)


def validate_alignment(
    alignment_result: Dict[str, Any],
    min_correlation: float = 0.5,
) -> Tuple[bool, str]:
    """
    Validate alignment quality.

    Args:
        alignment_result: Result from AudioMRIAligner.align()
        min_correlation: Minimum acceptable correlation

    Returns:
        Tuple[bool, str]: (is_valid, message)

    Example:
        >>> result = align_audio_mri(mri_frames, audio, sr)
        >>> is_valid, msg = validate_alignment(result)
        >>> if not is_valid:
        >>>     print(f"Alignment failed: {msg}")
    """
    correlation = alignment_result['correlation']

    if correlation < min_correlation:
        return False, f"Low correlation: {correlation:.3f} < {min_correlation}"

    offset_seconds = abs(alignment_result['offset_seconds'])
    mri_duration = len(alignment_result['mri_energy']) / 83.28  # Approximate

    if offset_seconds > mri_duration * 0.5:
        return (
            False,
            f"Large offset: {offset_seconds:.2f}s (>{mri_duration*0.5:.2f}s)"
        )

    return True, f"Valid alignment: correlation={correlation:.3f}"
