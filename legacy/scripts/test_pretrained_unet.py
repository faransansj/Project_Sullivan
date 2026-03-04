#!/usr/bin/env python3
"""
Test pre-trained U-Net model on USC-TIMIT data.

This script evaluates whether the BartsMRIPhysics pre-trained weights
can be directly applied or fine-tuned for our dataset (the "잭팟" test).
"""

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.segmentation.unet import load_pretrained_unet
from src.utils.logger import setup_logger, get_logger
from src.utils.io_utils import ensure_directory

# Setup logging
setup_logger(level="INFO", console=True)
logger = get_logger(__name__)


def load_sample(hdf5_path: Path, metadata_path: Path) -> Tuple[np.ndarray, Dict]:
    """Load sample MRI data and metadata."""
    with h5py.File(hdf5_path, 'r') as f:
        mri_frames = f['mri_frames'][:]

    with open(metadata_path) as f:
        metadata = json.load(f)

    return mri_frames, metadata


def visualize_segmentation(mri_frame: np.ndarray, segmentation: np.ndarray,
                          frame_idx: int, utterance_name: str,
                          save_path: Path) -> None:
    """
    Visualize MRI frame with segmentation overlay.

    Args:
        mri_frame: Original MRI frame (H, W)
        segmentation: Predicted segmentation map (H, W)
        frame_idx: Frame index
        utterance_name: Utterance name
        save_path: Save path for visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{utterance_name} - Frame {frame_idx}', fontsize=14, fontweight='bold')

    # Plot 1: Original MRI
    axes[0].imshow(mri_frame, cmap='gray')
    axes[0].set_title('Original MRI Frame')
    axes[0].axis('off')

    # Plot 2: Segmentation
    # Use a colormap that shows different classes distinctly
    axes[1].imshow(segmentation, cmap='tab10', vmin=0, vmax=9)
    axes[1].set_title('Predicted Segmentation')
    axes[1].axis('off')

    # Plot 3: Overlay
    axes[2].imshow(mri_frame, cmap='gray')
    axes[2].imshow(segmentation, cmap='tab10', alpha=0.5, vmin=0, vmax=9)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    # Add colorbar
    from matplotlib.patches import Patch
    class_labels = [
        'Background',
        'Tongue',
        'Jaw',
        'Lips',
        'Velum',
        'Pharynx',
        'Vocal Folds'
    ]
    unique_classes = np.unique(segmentation)
    legend_elements = [Patch(facecolor=plt.cm.tab10(i/10), label=f'{i}: {class_labels[i] if i < len(class_labels) else "Unknown"}')
                      for i in unique_classes]
    axes[2].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved segmentation visualization: {save_path}")


def test_pretrained_model(model_path: Path, sample_hdf5: Path,
                         sample_metadata: Path, output_dir: Path,
                         n_frames: int = 5) -> None:
    """
    Test pre-trained model on sample data.

    Args:
        model_path: Path to pre-trained model weights
        sample_hdf5: Path to sample HDF5 file
        sample_metadata: Path to sample metadata
        output_dir: Output directory for visualizations
        n_frames: Number of frames to test
    """
    logger.info(f"Loading pre-trained model from: {model_path}")

    # Check if CUDA is available and compatible
    # GTX 750 Ti has compute capability 5.0, but current PyTorch needs >= 7.0
    if torch.cuda.is_available():
        try:
            # Try to create a simple tensor on GPU
            torch.tensor([1.0]).cuda()
            device = 'cuda'
            logger.info(f"Using device: cuda (GPU acceleration enabled)")
        except Exception as e:
            logger.warning(f"CUDA available but not compatible: {e}")
            device = 'cpu'
            logger.info(f"Using device: cpu (CUDA not compatible with GPU)")
    else:
        device = 'cpu'
        logger.info(f"Using device: cpu (CUDA not available)")

    # Load model
    model = load_pretrained_unet(str(model_path), n_classes=7, device=device)
    logger.info(f"Model loaded successfully with 7 classes")

    # Load sample data
    logger.info(f"Loading sample data from: {sample_hdf5}")
    mri_frames, metadata = load_sample(sample_hdf5, sample_metadata)
    logger.info(f"Loaded {len(mri_frames)} MRI frames with shape {mri_frames.shape}")

    # Get sample name
    utterance_name = sample_hdf5.stem.replace('_metadata', '')

    # Ensure output directory exists
    ensure_directory(output_dir)

    # Test on n_frames evenly spaced throughout the sequence
    frame_indices = np.linspace(0, len(mri_frames) - 1, n_frames, dtype=int)

    logger.info(f"\nTesting on {n_frames} frames: {frame_indices.tolist()}")

    with torch.no_grad():
        for i, frame_idx in enumerate(frame_indices, 1):
            logger.info(f"\nProcessing frame {i}/{n_frames} (index {frame_idx})...")

            # Get frame
            mri_frame = mri_frames[frame_idx]  # (H, W)

            # Prepare input tensor (B, C, H, W)
            # The model expects normalized input
            input_tensor = torch.from_numpy(mri_frame).float()
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims

            # Pad to 96x96 (next multiple of 16 after 84)
            # U-Net uses 4 pooling layers (divides by 2^4 = 16)
            h, w = input_tensor.shape[2], input_tensor.shape[3]
            target_size = 96  # Next multiple of 16 >= 84
            pad_h = target_size - h
            pad_w = target_size - w
            # Pad symmetrically (top, bottom, left, right)
            input_tensor = F.pad(input_tensor, (pad_w//2, pad_w - pad_w//2,
                                               pad_h//2, pad_h - pad_h//2),
                                mode='constant', value=0)

            input_tensor = input_tensor.to(device)

            logger.info(f"  Input shape: {input_tensor.shape}, range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")

            # Forward pass
            output = model(input_tensor)  # (B, n_classes, H_padded, W_padded)
            logger.info(f"  Output shape: {output.shape}")

            # Crop output back to original size (84x84)
            # Remove padding added before
            output = output[:, :, pad_h//2:pad_h//2+h, pad_w//2:pad_w//2+w]

            # Get predicted segmentation
            pred_seg = torch.argmax(output, dim=1)[0]  # (H, W)
            pred_seg_np = pred_seg.cpu().numpy()

            # Get prediction statistics
            unique_classes, counts = np.unique(pred_seg_np, return_counts=True)
            logger.info(f"  Predicted classes: {unique_classes.tolist()}")
            for cls, count in zip(unique_classes, counts):
                percentage = count / pred_seg_np.size * 100
                logger.info(f"    Class {cls}: {count} pixels ({percentage:.1f}%)")

            # Visualize
            save_path = output_dir / f'{utterance_name}_frame_{frame_idx:04d}_seg.png'
            visualize_segmentation(mri_frame, pred_seg_np, frame_idx, utterance_name, save_path)

    logger.info(f"\n{'='*70}")
    logger.info("TEST COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Visualizations saved to: {output_dir}")


def main():
    """Main function."""
    logger.info("="*70)
    logger.info("PRE-TRAINED U-NET TEST (잭팟 Verification)")
    logger.info("="*70)

    # Setup paths
    model_path = Path("models/pretrained_unet/Network Weights/train_subj_1_2_3_4_l_rate_0.0003_mb_size_4_epochs_200/unet_parameters.pth")
    batch_summary_path = Path("data/processed/aligned/batch_summary.json")
    output_dir = Path("data/processed/aligned/segmentation_test")

    # Select best quality sample
    logger.info("\nSelecting highest quality sample for testing...")
    with open(batch_summary_path) as f:
        data = json.load(f)

    best_sample = None
    best_corr = -1

    for subject in data['subjects']:
        for utt in subject.get('utterances', []):
            if utt['correlation'] > best_corr:
                best_corr = utt['correlation']
                best_sample = utt

    if best_sample is None:
        logger.error("No valid samples found!")
        return

    sample_hdf5 = Path(best_sample['hdf5_path'])
    sample_metadata = Path(best_sample['metadata_path'])

    logger.info(f"Selected sample: {best_sample['utterance_name']}")
    logger.info(f"  Correlation: {best_corr:.3f}")
    logger.info(f"  HDF5: {sample_hdf5}")
    logger.info(f"  Metadata: {sample_metadata}")

    # Test model
    logger.info(f"\n{'='*70}")
    logger.info("Running inference...")
    logger.info(f"{'='*70}")

    test_pretrained_model(
        model_path=model_path,
        sample_hdf5=sample_hdf5,
        sample_metadata=sample_metadata,
        output_dir=output_dir,
        n_frames=5
    )

    logger.info("\n" + "="*70)
    logger.info("ANALYSIS")
    logger.info("="*70)
    logger.info("\nIf the segmentations show meaningful vocal tract structures,")
    logger.info("we hit the 잭팟! We can fine-tune these weights for USC-TIMIT.")
    logger.info("\nIf not, we'll proceed with Option 1-D (synthetic pseudo-labels).")
    logger.info(f"\nPlease review visualizations in: {output_dir}")


if __name__ == "__main__":
    main()
