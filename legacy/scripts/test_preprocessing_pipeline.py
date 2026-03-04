#!/usr/bin/env python3
"""
Test preprocessing pipeline on real USC-TIMIT data.

This script tests denoising and alignment on actual MRI-audio pairs
to validate the Phase 1 preprocessing implementation.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.data_loader import USCTIMITLoader
from src.preprocessing.denoising import MRIDenoiser, AudioDenoiser
from src.preprocessing.alignment import AudioMRIAligner, validate_alignment
from src.utils.logger import setup_logger, get_logger
from src.utils.io_utils import ensure_directory

# Setup logging
setup_logger(level="INFO", console=True)
logger = get_logger(__name__)


def visualize_denoising_results(
    original_frames: np.ndarray,
    denoised_frames: np.ndarray,
    original_audio: np.ndarray,
    denoised_audio: np.ndarray,
    audio_sr: int,
    output_path: Path,
):
    """
    Visualize denoising results.

    Args:
        original_frames: Original MRI frames
        denoised_frames: Denoised MRI frames
        original_audio: Original audio
        denoised_audio: Denoised audio
        audio_sr: Audio sample rate
        output_path: Path to save visualization
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Sample frame indices
    frame_idx = len(original_frames) // 2  # Middle frame

    # MRI comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_frames[frame_idx], cmap='gray')
    ax1.set_title(f'Original MRI Frame {frame_idx}', fontsize=12, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(denoised_frames[frame_idx], cmap='gray')
    ax2.set_title(f'Denoised MRI Frame {frame_idx}', fontsize=12, fontweight='bold')
    ax2.axis('off')

    # MRI noise comparison (difference)
    ax3 = fig.add_subplot(gs[1, 0])
    noise = np.abs(original_frames[frame_idx] - denoised_frames[frame_idx])
    im = ax3.imshow(noise, cmap='hot')
    ax3.set_title('Removed Noise (MRI)', fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im, ax=ax3, fraction=0.046)

    # MRI temporal comparison (3 consecutive frames)
    ax4 = fig.add_subplot(gs[1, 1])
    frame_range = slice(frame_idx-1, frame_idx+2)
    temporal_profile_orig = original_frames[frame_range, 42, :].T  # Middle row
    temporal_profile_den = denoised_frames[frame_range, 42, :].T

    ax4.plot(temporal_profile_orig.mean(axis=1), label='Original', alpha=0.7)
    ax4.plot(temporal_profile_den.mean(axis=1), label='Denoised', alpha=0.7)
    ax4.set_title('Temporal Profile (Center Row)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Column')
    ax4.set_ylabel('Intensity')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Audio waveform comparison
    ax5 = fig.add_subplot(gs[2, 0])
    time = np.arange(len(original_audio)) / audio_sr
    # Plot first 2 seconds for clarity
    samples_to_plot = min(len(original_audio), int(2 * audio_sr))
    ax5.plot(time[:samples_to_plot], original_audio[:samples_to_plot],
             alpha=0.7, label='Original', linewidth=0.5)
    ax5.plot(time[:samples_to_plot], denoised_audio[:samples_to_plot],
             alpha=0.7, label='Denoised', linewidth=0.5)
    ax5.set_title('Audio Waveform (First 2s)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Amplitude')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Audio spectrum comparison
    ax6 = fig.add_subplot(gs[2, 1])
    from scipy import signal as scipy_signal

    # Compute PSD
    freqs_orig, psd_orig = scipy_signal.welch(original_audio, audio_sr, nperseg=1024)
    freqs_den, psd_den = scipy_signal.welch(denoised_audio, audio_sr, nperseg=1024)

    ax6.semilogy(freqs_orig, psd_orig, alpha=0.7, label='Original')
    ax6.semilogy(freqs_den, psd_den, alpha=0.7, label='Denoised')
    ax6.set_title('Audio Power Spectral Density', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Frequency (Hz)')
    ax6.set_ylabel('PSD')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle('Denoising Results: MRI & Audio', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization: {output_path}")
    plt.close()


def visualize_alignment_results(
    alignment_result: dict,
    output_path: Path,
):
    """
    Visualize alignment results.

    Args:
        alignment_result: Result from AudioMRIAligner.align()
        output_path: Path to save visualization
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # MRI energy vs Audio envelope
    mri_energy = alignment_result['mri_energy']
    audio_envelope = alignment_result['audio_envelope']
    frame_timestamps = alignment_result['frame_timestamps']
    audio_sr = alignment_result['audio_sr']

    # Time axes
    mri_time = frame_timestamps
    audio_time = np.arange(len(audio_envelope)) / audio_sr

    # Before alignment
    ax = axes[0]
    ax.plot(mri_time, mri_energy, label='MRI Motion Energy', linewidth=2)
    ax.plot(audio_time, audio_envelope, label='Audio Envelope', alpha=0.7)
    ax.set_title('Before Alignment', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalized Energy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # After alignment (shift audio envelope)
    ax = axes[1]
    offset_sec = alignment_result['offset_seconds']
    audio_time_aligned = audio_time + offset_sec

    ax.plot(mri_time, mri_energy, label='MRI Motion Energy', linewidth=2)
    ax.plot(audio_time_aligned, audio_envelope, label='Audio Envelope (Aligned)', alpha=0.7)
    ax.axvline(offset_sec, color='red', linestyle='--', alpha=0.5,
               label=f'Offset: {offset_sec:.3f}s')

    correlation = alignment_result['correlation']
    ax.set_title(f'After Alignment (Correlation: {correlation:.3f})',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalized Energy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization: {output_path}")
    plt.close()


def test_preprocessing_pipeline(
    subject_id: str = "sub001",
    utterance_idx: int = 0,
    output_dir: Path = Path("results/preprocessing_test"),
):
    """
    Test full preprocessing pipeline on a single utterance.

    Args:
        subject_id: Subject ID to test
        utterance_idx: Utterance index to test
        output_dir: Output directory for results
    """
    logger.info("="*70)
    logger.info("Testing Preprocessing Pipeline")
    logger.info("="*70)

    # Create output directory
    ensure_directory(output_dir)

    # Initialize data loader
    data_root = Path("data/raw/usc_timit_data")
    loader = USCTIMITLoader(data_root)

    # Load subject
    logger.info(f"\n--- Loading Subject: {subject_id} ---")
    subject_data = loader.load_subject(subject_id, load_mri=False, load_audio=False)

    if 'utterance_files' not in subject_data:
        logger.error(f"Subject {subject_id} has no utterance files")
        return

    utterance_files = subject_data['utterance_files']
    if utterance_idx >= len(utterance_files):
        logger.error(f"Utterance index {utterance_idx} out of range (max: {len(utterance_files)-1})")
        return

    # Load specific utterance
    utterance_file = utterance_files[utterance_idx]
    logger.info(f"Testing utterance: {utterance_file.name}")

    # Load MRI and audio
    from src.utils.io_utils import load_mri_from_video, load_audio

    logger.info("\nLoading MRI frames...")
    mri_frames = load_mri_from_video(utterance_file, normalize=True)
    logger.info(f"  Shape: {mri_frames.shape}")

    logger.info("\nLoading audio...")
    audio, audio_sr = load_audio(utterance_file, sr=None)
    logger.info(f"  Shape: {audio.shape}, SR: {audio_sr} Hz")

    # Step 1: Denoising
    logger.info("\n" + "="*70)
    logger.info("Step 1: Denoising")
    logger.info("="*70)

    # MRI denoising
    logger.info("\n[MRI Denoising]")
    mri_denoiser = MRIDenoiser(
        method="gaussian",
        spatial_sigma=1.0,
        temporal_window=3,
        apply_temporal=True,
    )
    mri_denoised = mri_denoiser.denoise(mri_frames)

    # Audio denoising
    logger.info("\n[Audio Denoising]")
    audio_denoiser = AudioDenoiser(
        noise_sample_duration=0.5,
        prop_decrease=0.8,  # 80% noise reduction
        stationary=False,
    )
    audio_denoised = audio_denoiser.denoise(audio, audio_sr)

    # Visualize denoising results
    logger.info("\nVisualizing denoising results...")
    visualize_denoising_results(
        mri_frames,
        mri_denoised,
        audio,
        audio_denoised,
        audio_sr,
        output_dir / f"{subject_id}_{utterance_idx:02d}_denoising.png",
    )

    # Step 2: Alignment
    logger.info("\n" + "="*70)
    logger.info("Step 2: Alignment")
    logger.info("="*70)

    aligner = AudioMRIAligner(
        mri_fps=subject_data['mri_fps'],
        jaw_region=None,  # Use full frame
        smooth_sigma=10.0,
    )

    logger.info("\nAligning audio with MRI frames...")
    alignment_result = aligner.align(
        mri_denoised,
        audio_denoised,
        audio_sr,
    )

    # Validate alignment
    is_valid, msg = validate_alignment(alignment_result, min_correlation=0.3)
    logger.info(f"\nAlignment validation: {msg}")

    # Visualize alignment
    logger.info("\nVisualizing alignment results...")
    visualize_alignment_results(
        alignment_result,
        output_dir / f"{subject_id}_{utterance_idx:02d}_alignment.png",
    )

    # Summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    logger.info(f"Subject: {subject_id}")
    logger.info(f"Utterance: {utterance_file.name}")
    logger.info(f"MRI frames: {len(mri_frames)} ({mri_frames.shape[1]}x{mri_frames.shape[2]})")
    logger.info(f"Audio: {len(audio)} samples @ {audio_sr} Hz ({len(audio)/audio_sr:.2f}s)")
    logger.info(f"\nAlignment:")
    logger.info(f"  - Offset: {alignment_result['offset_seconds']:.3f}s")
    logger.info(f"  - Correlation: {alignment_result['correlation']:.3f}")
    logger.info(f"  - Valid: {is_valid}")
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("="*70)


def main():
    """Main function."""

    # Test on first recommended subject
    test_preprocessing_pipeline(
        subject_id="sub001",
        utterance_idx=0,
        output_dir=Path("results/preprocessing_test"),
    )


if __name__ == "__main__":
    main()
