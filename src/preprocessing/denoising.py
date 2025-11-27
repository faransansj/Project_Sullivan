"""
Project Sullivan - MRI and Audio Denoising

This module provides denoising algorithms for MRI frames and audio signals.
"""

import numpy as np
from typing import Optional, Literal
import cv2

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MRIDenoiser:
    """
    Denoiser for MRI frame sequences.

    Supports spatial and temporal denoising methods:
    - Spatial: Gaussian blur, bilateral filter
    - Temporal: Median filter across time
    """

    def __init__(
        self,
        method: Literal["gaussian", "bilateral", "nlm"] = "gaussian",
        spatial_sigma: float = 1.0,
        temporal_window: int = 3,
        apply_temporal: bool = True,
    ):
        """
        Initialize MRI denoiser.

        Args:
            method: Spatial denoising method
                - "gaussian": Gaussian blur (fast, simple)
                - "bilateral": Bilateral filter (edge-preserving)
                - "nlm": Non-local means (slow, high quality)
            spatial_sigma: Sigma for spatial filtering
            temporal_window: Window size for temporal median filter (must be odd)
            apply_temporal: Whether to apply temporal filtering
        """
        self.method = method
        self.spatial_sigma = spatial_sigma
        self.temporal_window = temporal_window
        self.apply_temporal = apply_temporal

        # Validate temporal window
        if temporal_window % 2 == 0:
            raise ValueError("temporal_window must be odd")

        logger.info(
            f"MRIDenoiser initialized: method={method}, "
            f"spatial_sigma={spatial_sigma}, temporal_window={temporal_window}"
        )

    def denoise(self, frames: np.ndarray) -> np.ndarray:
        """
        Denoise MRI frame sequence.

        Args:
            frames: MRI frames (T, H, W) or (T, H, W, C)

        Returns:
            np.ndarray: Denoised frames with same shape

        Raises:
            ValueError: If frames shape is invalid
        """
        if frames.ndim not in [3, 4]:
            raise ValueError(f"Expected 3D or 4D frames, got shape {frames.shape}")

        logger.info(f"Denoising {len(frames)} MRI frames, shape: {frames.shape}")

        # Apply spatial denoising
        denoised = self._spatial_denoise(frames)

        # Apply temporal denoising
        if self.apply_temporal and len(frames) >= self.temporal_window:
            denoised = self._temporal_denoise(denoised)

        logger.info("Denoising completed")
        return denoised

    def _spatial_denoise(self, frames: np.ndarray) -> np.ndarray:
        """
        Apply spatial denoising to each frame.

        Args:
            frames: Input frames (T, H, W) or (T, H, W, C)

        Returns:
            np.ndarray: Spatially denoised frames
        """
        denoised = np.zeros_like(frames)

        for i, frame in enumerate(frames):
            if self.method == "gaussian":
                # Gaussian blur
                kernel_size = int(2 * np.ceil(3 * self.spatial_sigma) + 1)
                denoised[i] = cv2.GaussianBlur(
                    frame, (kernel_size, kernel_size), self.spatial_sigma
                )

            elif self.method == "bilateral":
                # Bilateral filter (edge-preserving)
                # d: diameter of pixel neighborhood
                # sigmaColor: filter sigma in color space
                # sigmaSpace: filter sigma in coordinate space
                d = int(2 * np.ceil(3 * self.spatial_sigma) + 1)
                sigma_color = 75
                sigma_space = self.spatial_sigma * 10

                # Convert to uint8 for bilateral filter
                frame_uint8 = (frame * 255).astype(np.uint8)
                denoised_uint8 = cv2.bilateralFilter(
                    frame_uint8, d, sigma_color, sigma_space
                )
                denoised[i] = denoised_uint8.astype(np.float32) / 255.0

            elif self.method == "nlm":
                # Non-local means denoising
                # h: filter strength
                # templateWindowSize: size of template patch
                # searchWindowSize: size of search area
                h = 10
                template_size = 7
                search_size = 21

                # Convert to uint8
                frame_uint8 = (frame * 255).astype(np.uint8)
                denoised_uint8 = cv2.fastNlMeansDenoising(
                    frame_uint8, None, h, template_size, search_size
                )
                denoised[i] = denoised_uint8.astype(np.float32) / 255.0

            else:
                raise ValueError(f"Unknown method: {self.method}")

        return denoised

    def _temporal_denoise(self, frames: np.ndarray) -> np.ndarray:
        """
        Apply temporal median filtering.

        Args:
            frames: Input frames (T, H, W) or (T, H, W, C)

        Returns:
            np.ndarray: Temporally smoothed frames
        """
        logger.debug(f"Applying temporal median filter (window={self.temporal_window})")

        # Use scipy for efficient median filtering
        from scipy.ndimage import median_filter

        # Apply median filter along time axis (axis=0)
        # Size: (temporal_window, 1, 1) for (T, H, W)
        if frames.ndim == 3:
            size = (self.temporal_window, 1, 1)
        else:  # 4D
            size = (self.temporal_window, 1, 1, 1)

        denoised = median_filter(frames, size=size, mode='reflect')

        return denoised


class AudioDenoiser:
    """
    Denoiser for audio signals using spectral subtraction.

    Uses noisereduce library for spectral noise reduction.
    """

    def __init__(
        self,
        noise_sample_duration: float = 0.5,
        prop_decrease: float = 1.0,
        stationary: bool = False,
    ):
        """
        Initialize audio denoiser.

        Args:
            noise_sample_duration: Duration (seconds) of noise profile to use
            prop_decrease: Proportion of noise to reduce (0-1, 1=full reduction)
            stationary: Whether noise is stationary (True) or non-stationary (False)
        """
        self.noise_sample_duration = noise_sample_duration
        self.prop_decrease = prop_decrease
        self.stationary = stationary

        logger.info(
            f"AudioDenoiser initialized: "
            f"noise_duration={noise_sample_duration}s, "
            f"prop_decrease={prop_decrease}, stationary={stationary}"
        )

    def denoise(
        self,
        audio: np.ndarray,
        sr: int,
        noise_profile: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Denoise audio signal.

        Args:
            audio: Audio signal (N,)
            sr: Sample rate
            noise_profile: Optional noise profile to use. If None, uses
                beginning of audio as noise sample.

        Returns:
            np.ndarray: Denoised audio with same shape

        Raises:
            ValueError: If audio is too short
        """
        if len(audio) < sr * 0.1:  # At least 0.1 seconds
            raise ValueError(f"Audio too short: {len(audio)} samples")

        logger.info(f"Denoising audio: {len(audio)} samples at {sr} Hz")

        try:
            import noisereduce as nr
        except ImportError:
            logger.warning(
                "noisereduce not installed, returning original audio. "
                "Install with: pip install noisereduce"
            )
            return audio

        # If no noise profile provided, use beginning of audio
        if noise_profile is None:
            noise_samples = int(self.noise_sample_duration * sr)
            noise_samples = min(noise_samples, len(audio) // 4)  # Max 25% of audio
            noise_profile = audio[:noise_samples]

        # Apply noise reduction
        denoised = nr.reduce_noise(
            y=audio,
            sr=sr,
            y_noise=noise_profile,
            prop_decrease=self.prop_decrease,
            stationary=self.stationary,
        )

        logger.info("Audio denoising completed")
        return denoised


def denoise_mri_sequence(
    frames: np.ndarray,
    method: Literal["gaussian", "bilateral", "nlm"] = "gaussian",
    spatial_sigma: float = 1.0,
    temporal_window: int = 3,
    apply_temporal: bool = True,
) -> np.ndarray:
    """
    Convenience function to denoise MRI frames.

    Args:
        frames: MRI frames (T, H, W) or (T, H, W, C)
        method: Denoising method
        spatial_sigma: Spatial filtering sigma
        temporal_window: Temporal median filter window
        apply_temporal: Whether to apply temporal filtering

    Returns:
        np.ndarray: Denoised frames

    Example:
        >>> frames = load_mri_sequence("path/to/frames")
        >>> denoised = denoise_mri_sequence(frames, method="bilateral")
    """
    denoiser = MRIDenoiser(
        method=method,
        spatial_sigma=spatial_sigma,
        temporal_window=temporal_window,
        apply_temporal=apply_temporal,
    )
    return denoiser.denoise(frames)


def denoise_audio(
    audio: np.ndarray,
    sr: int,
    noise_sample_duration: float = 0.5,
    prop_decrease: float = 1.0,
) -> np.ndarray:
    """
    Convenience function to denoise audio.

    Args:
        audio: Audio signal (N,)
        sr: Sample rate
        noise_sample_duration: Duration of noise profile
        prop_decrease: Noise reduction proportion

    Returns:
        np.ndarray: Denoised audio

    Example:
        >>> audio, sr = load_audio("path/to/audio.wav")
        >>> denoised = denoise_audio(audio, sr)
    """
    denoiser = AudioDenoiser(
        noise_sample_duration=noise_sample_duration,
        prop_decrease=prop_decrease,
    )
    return denoiser.denoise(audio, sr)
