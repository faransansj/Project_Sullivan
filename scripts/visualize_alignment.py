#!/usr/bin/env python3
"""
Generate visualizations of alignment results.

This script creates comprehensive visualizations showing alignment quality
across different samples and quality tiers.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger, get_logger
from src.utils.io_utils import ensure_directory

# Setup logging
setup_logger(level="INFO", console=True)
logger = get_logger(__name__)


def load_sample(hdf5_path: Path, metadata_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load sample data."""
    with h5py.File(hdf5_path, 'r') as f:
        mri_frames = f['mri_frames'][:]
        audio = f['audio'][:]

    with open(metadata_path) as f:
        metadata = json.load(f)

    return mri_frames, audio, metadata


def compute_alignment_signals(mri_frames: np.ndarray, audio: np.ndarray,
                              mri_fps: float, audio_sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute alignment signals (MRI energy and audio envelope).

    Args:
        mri_frames: MRI frames array (T, H, W)
        audio: Audio signal
        mri_fps: MRI frame rate
        audio_sr: Audio sample rate

    Returns:
        Tuple of (mri_time, mri_energy, audio_time, audio_envelope)
    """
    # Compute MRI motion energy (frame-to-frame difference)
    mri_diff = np.diff(mri_frames, axis=0)
    mri_energy = np.sqrt(np.mean(mri_diff ** 2, axis=(1, 2)))

    # Smooth MRI energy
    from scipy.ndimage import gaussian_filter1d
    sigma_frames = (10.0 / 1000.0) * mri_fps  # 10ms smoothing
    mri_energy = gaussian_filter1d(mri_energy, sigma=sigma_frames)

    # MRI time axis
    mri_time = np.arange(len(mri_energy)) / mri_fps

    # Compute audio envelope
    analytic_signal = signal.hilbert(audio)
    audio_envelope = np.abs(analytic_signal)

    # Smooth audio envelope
    sigma_samples = (10.0 / 1000.0) * audio_sr  # 10ms smoothing
    audio_envelope = gaussian_filter1d(audio_envelope, sigma=sigma_samples)

    # Audio time axis
    audio_time = np.arange(len(audio_envelope)) / audio_sr

    return mri_time, mri_energy, audio_time, audio_envelope


def plot_alignment(mri_frames: np.ndarray, audio: np.ndarray, metadata: Dict,
                  title: str, save_path: Path) -> None:
    """
    Plot alignment visualization for a single sample.

    Args:
        mri_frames: MRI frames
        audio: Audio signal
        metadata: Metadata dict
        title: Plot title
        save_path: Save path for figure
    """
    mri_fps = 83.28
    audio_sr = 22050

    # Compute signals
    mri_time, mri_energy, audio_time, audio_envelope = compute_alignment_signals(
        mri_frames, audio, mri_fps, audio_sr
    )

    # Get alignment info
    alignment = metadata['alignment']
    offset = alignment['offset_seconds']
    correlation = alignment['correlation']

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Plot 1: MRI motion energy
    axes[0].plot(mri_time, mri_energy, 'b-', linewidth=1, label='MRI Motion Energy')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('MRI Energy')
    axes[0].set_title('MRI Motion Energy (Frame-to-Frame Difference)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot 2: Audio envelope
    axes[1].plot(audio_time, audio_envelope, 'r-', linewidth=1, label='Audio Envelope')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Audio Amplitude')
    axes[1].set_title('Audio Envelope')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Plot 3: Aligned signals (overlay)
    # Normalize for comparison
    mri_energy_norm = (mri_energy - mri_energy.min()) / (mri_energy.max() - mri_energy.min() + 1e-8)
    audio_envelope_norm = (audio_envelope - audio_envelope.min()) / (audio_envelope.max() - audio_envelope.min() + 1e-8)

    # Shift audio by offset
    audio_time_shifted = audio_time + offset

    axes[2].plot(mri_time, mri_energy_norm, 'b-', linewidth=1.5, label='MRI Energy (normalized)', alpha=0.7)
    axes[2].plot(audio_time_shifted, audio_envelope_norm, 'r-', linewidth=1.5, label='Audio Envelope (shifted, normalized)', alpha=0.7)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Normalized Signal')
    axes[2].set_title(f'Aligned Signals (Offset: {offset:.3f}s, Correlation: {correlation:.3f})')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    # Add info box
    info_text = f"Correlation: {correlation:.3f}\nOffset: {offset:.3f}s\nValid: {alignment['is_valid']}"
    axes[2].text(0.02, 0.98, info_text, transform=axes[2].transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved alignment visualization: {save_path}")


def plot_correlation_distribution(batch_summary_path: Path, save_path: Path) -> None:
    """
    Plot correlation distribution across all samples.

    Args:
        batch_summary_path: Path to batch summary JSON
        save_path: Save path for figure
    """
    with open(batch_summary_path) as f:
        data = json.load(f)

    # Collect correlations
    correlations = []
    for subject in data['subjects']:
        for utt in subject.get('utterances', []):
            if 'correlation' in utt:
                correlations.append(utt['correlation'])

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Alignment Quality Distribution', fontsize=14, fontweight='bold')

    # Histogram
    axes[0].hist(correlations, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].axvline(0.3, color='red', linestyle='--', linewidth=2, label='Validation Threshold (0.3)')
    axes[0].axvline(np.median(correlations), color='orange', linestyle='--', linewidth=2, label=f'Median ({np.median(correlations):.3f})')
    axes[0].axvline(np.mean(correlations), color='green', linestyle='--', linewidth=2, label=f'Mean ({np.mean(correlations):.3f})')
    axes[0].set_xlabel('Correlation')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Correlation Distribution (N={len(correlations)})')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Cumulative distribution
    sorted_corr = np.sort(correlations)
    cumulative = np.arange(1, len(sorted_corr) + 1) / len(sorted_corr) * 100
    axes[1].plot(sorted_corr, cumulative, 'b-', linewidth=2)
    axes[1].axvline(0.3, color='red', linestyle='--', linewidth=2, label='Threshold (0.3)')
    axes[1].axhline(50, color='gray', linestyle=':', linewidth=1)
    axes[1].set_xlabel('Correlation')
    axes[1].set_ylabel('Cumulative Percentage (%)')
    axes[1].set_title('Cumulative Distribution Function')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Add statistics box
    stats_text = (
        f"Statistics:\n"
        f"Mean: {np.mean(correlations):.3f}\n"
        f"Median: {np.median(correlations):.3f}\n"
        f"Std Dev: {np.std(correlations):.3f}\n"
        f"Min: {np.min(correlations):.3f}\n"
        f"Max: {np.max(correlations):.3f}\n"
        f"Above 0.3: {sum(c >= 0.3 for c in correlations)}/{len(correlations)} ({sum(c >= 0.3 for c in correlations)/len(correlations)*100:.1f}%)"
    )
    axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                fontsize=9, family='monospace')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved correlation distribution: {save_path}")


def plot_quality_comparison(samples: List[Tuple], output_dir: Path) -> None:
    """
    Plot side-by-side comparison of different quality samples.

    Args:
        samples: List of (category, name, hdf5_path, metadata_path, correlation) tuples
        output_dir: Output directory for plots
    """
    # Load data for all samples
    sample_data = []
    for category, name, hdf5_path, metadata_path, corr in samples:
        mri_frames, audio, metadata = load_sample(hdf5_path, metadata_path)
        sample_data.append((category, name, mri_frames, audio, metadata, corr))

    # Create comparison figure
    n_samples = len(sample_data)
    fig, axes = plt.subplots(n_samples, 1, figsize=(15, 4 * n_samples))
    if n_samples == 1:
        axes = [axes]
    fig.suptitle('Quality Tier Comparison', fontsize=14, fontweight='bold')

    mri_fps = 83.28
    audio_sr = 22050

    for idx, (category, name, mri_frames, audio, metadata, corr) in enumerate(sample_data):
        # Compute signals
        mri_time, mri_energy, audio_time, audio_envelope = compute_alignment_signals(
            mri_frames, audio, mri_fps, audio_sr
        )

        # Normalize
        mri_energy_norm = (mri_energy - mri_energy.min()) / (mri_energy.max() - mri_energy.min() + 1e-8)
        audio_envelope_norm = (audio_envelope - audio_envelope.min()) / (audio_envelope.max() - audio_envelope.min() + 1e-8)

        # Get offset
        offset = metadata['alignment']['offset_seconds']
        audio_time_shifted = audio_time + offset

        # Plot
        axes[idx].plot(mri_time, mri_energy_norm, 'b-', linewidth=1.5, label='MRI Energy', alpha=0.7)
        axes[idx].plot(audio_time_shifted, audio_envelope_norm, 'r-', linewidth=1.5, label='Audio Envelope (shifted)', alpha=0.7)
        axes[idx].set_xlabel('Time (s)')
        axes[idx].set_ylabel('Normalized Signal')
        axes[idx].set_title(f'{category}: {name} (Correlation: {corr:.3f})')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend(loc='upper right')
        axes[idx].set_xlim([0, min(15, max(mri_time.max(), audio_time_shifted.max()))])  # Show first 15s

    plt.tight_layout()
    save_path = output_dir / 'quality_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved quality comparison: {save_path}")


def main():
    """Main function."""
    logger.info("="*70)
    logger.info("ALIGNMENT VISUALIZATION")
    logger.info("="*70)

    # Setup paths
    batch_summary_path = Path("data/processed/aligned/batch_summary.json")
    vis_dir = Path("data/processed/aligned/visualizations")
    ensure_directory(vis_dir)

    # Select samples (same as inspection script)
    with open(batch_summary_path) as f:
        data = json.load(f)

    utterances = []
    for subject in data['subjects']:
        for utt in subject.get('utterances', []):
            utterances.append((
                utt['utterance_name'],
                Path(utt['hdf5_path']),
                Path(utt['metadata_path']),
                utt['correlation']
            ))

    utterances.sort(key=lambda x: x[3], reverse=True)

    # Select representative samples
    samples = []
    samples.append(('Best', *utterances[0]))

    good_samples = [u for u in utterances if u[3] >= 0.3]
    if len(good_samples) > 1:
        samples.append(('Good', *good_samples[len(good_samples)//2]))

    moderate_samples = [u for u in utterances if 0.2 <= u[3] < 0.3]
    if moderate_samples:
        samples.append(('Moderate', *moderate_samples[len(moderate_samples)//2]))

    acceptable_samples = [u for u in utterances if 0.1 <= u[3] < 0.2]
    if acceptable_samples:
        samples.append(('Acceptable', *acceptable_samples[len(acceptable_samples)//2]))

    logger.info(f"\nGenerating visualizations for {len(samples)} representative samples...")

    # Generate individual alignment plots
    for category, name, hdf5_path, metadata_path, corr in samples:
        logger.info(f"Processing {category}: {name} (corr={corr:.3f})...")
        mri_frames, audio, metadata = load_sample(hdf5_path, metadata_path)

        title = f"{category} Quality: {name}\nCorrelation: {corr:.3f}"
        save_path = vis_dir / f"alignment_{category.lower()}_{name}.png"

        plot_alignment(mri_frames, audio, metadata, title, save_path)

    # Generate correlation distribution plot
    logger.info("\nGenerating correlation distribution plot...")
    plot_correlation_distribution(batch_summary_path, vis_dir / 'correlation_distribution.png')

    # Generate quality comparison plot
    logger.info("\nGenerating quality comparison plot...")
    plot_quality_comparison(samples, vis_dir)

    logger.info("\n" + "="*70)
    logger.info("VISUALIZATION COMPLETE")
    logger.info("="*70)
    logger.info(f"\nAll visualizations saved to: {vis_dir}")
    logger.info(f"  - Individual alignment plots: {len(samples)} files")
    logger.info(f"  - Correlation distribution: 1 file")
    logger.info(f"  - Quality comparison: 1 file")
    logger.info(f"\nTotal files generated: {len(samples) + 2}")


if __name__ == "__main__":
    main()
