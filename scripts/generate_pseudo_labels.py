#!/usr/bin/env python3
"""
Generate pseudo-labels for U-Net training using traditional CV methods.

This script processes selected frames from the USC-TIMIT dataset and
generates initial segmentation labels using classical computer vision
techniques (Otsu, GrabCut, Active Contours, Watershed).

The generated pseudo-labels serve as training data for training U-Net
from scratch on USC-TIMIT data.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.segmentation.traditional_cv import VocalTractSegmenter, create_initial_contour, estimate_tongue_region
from src.utils.logger import setup_logger, get_logger
from src.utils.io_utils import ensure_directory

# Setup logging
setup_logger(level="INFO", console=True)
logger = get_logger(__name__)


def load_frames(hdf5_path: Path, frame_indices: List[int]) -> np.ndarray:
    """
    Load specific frames from HDF5 file.

    Args:
        hdf5_path: Path to HDF5 file
        frame_indices: List of frame indices to load

    Returns:
        Array of frames (N, H, W)
    """
    with h5py.File(hdf5_path, 'r') as f:
        mri_frames = f['mri_frames'][:]

    selected_frames = mri_frames[frame_indices]
    return selected_frames


def generate_pseudo_label(frame: np.ndarray,
                          segmenter: VocalTractSegmenter,
                          method: str = 'combined') -> Dict[str, np.ndarray]:
    """
    Generate pseudo-label for a single frame.

    Args:
        frame: MRI frame (H, W), normalized [0, 1]
        segmenter: VocalTractSegmenter instance
        method: Segmentation method ('otsu', 'grabcut', 'watershed', 'combined')

    Returns:
        Dictionary containing:
        - 'segmentation': Final segmentation map (H, W)
        - 'tissue_mask': Binary tissue/air mask
        - 'vocal_tract_mask': Vocal tract region mask
        - 'method': Method used
    """
    result = {'method': method}

    # Step 1: Tissue/air separation
    tissue_mask, threshold = segmenter.segment_tissue_air(frame)
    result['tissue_mask'] = tissue_mask
    result['threshold'] = threshold

    # Step 2: Vocal tract region mask
    vocal_tract_mask = segmenter.create_vocal_tract_mask(frame)
    result['vocal_tract_mask'] = vocal_tract_mask

    # Step 3: Generate segmentation based on method
    if method == 'otsu':
        # Multi-level Otsu (3 classes: background, air, tissue)
        seg_map = segmenter.segment_multilevel_otsu(frame, n_classes=3)
        result['segmentation'] = seg_map

    elif method == 'grabcut':
        # GrabCut for foreground/background
        fg_mask = segmenter.segment_grabcut(frame)

        # Combine with tissue mask
        seg_map = np.zeros_like(frame, dtype=np.uint8)
        seg_map[vocal_tract_mask & ~tissue_mask] = 0  # Air (background)
        seg_map[vocal_tract_mask & tissue_mask & fg_mask] = 1  # Tissue (foreground)
        seg_map[~vocal_tract_mask] = 0  # External background

        result['segmentation'] = seg_map
        result['grabcut_mask'] = fg_mask

    elif method == 'watershed':
        # Watershed segmentation
        labels = segmenter.segment_watershed(frame)

        # Map watershed labels to semantic classes
        # Label 0 = background, others = different tissue regions
        seg_map = np.clip(labels, 0, 5).astype(np.uint8)
        seg_map[~vocal_tract_mask] = 0  # External background

        result['segmentation'] = seg_map
        result['watershed_labels'] = labels

    elif method == 'combined':
        # Combined approach: Multi-Otsu + GrabCut + Watershed
        # This provides the best initial pseudo-labels

        # 1. Multi-level Otsu for basic tissue classes
        otsu_seg = segmenter.segment_multilevel_otsu(frame, n_classes=4)

        # 2. GrabCut for refined foreground
        grabcut_fg = segmenter.segment_grabcut(frame)

        # 3. Create combined segmentation
        seg_map = np.zeros_like(frame, dtype=np.uint8)

        # Class 0: External background
        seg_map[~vocal_tract_mask] = 0

        # Within vocal tract:
        # Class 0: Air (dark regions, not tissue)
        seg_map[vocal_tract_mask & ~tissue_mask] = 0

        # Class 1: Tongue (bright tissue in GrabCut foreground, lower region)
        h, w = frame.shape
        lower_region = np.zeros_like(frame, dtype=bool)
        lower_region[int(h*0.4):, :] = True  # Lower 60% of image

        seg_map[vocal_tract_mask & tissue_mask & grabcut_fg & lower_region] = 1

        # Class 2: Jaw/Hard palate (bright tissue, upper region)
        upper_region = ~lower_region
        seg_map[vocal_tract_mask & tissue_mask & grabcut_fg & upper_region] = 2

        # Class 3: Lips (tissue at edges)
        edge_region = np.zeros_like(frame, dtype=bool)
        edge_region[:, :int(w*0.2)] = True  # Left edge
        edge_region[:, int(w*0.8):] = True  # Right edge

        seg_map[vocal_tract_mask & tissue_mask & edge_region] = 3

        result['segmentation'] = seg_map
        result['otsu_seg'] = otsu_seg
        result['grabcut_mask'] = grabcut_fg

    else:
        raise ValueError(f"Unknown method: {method}")

    return result


def visualize_pseudo_label(frame: np.ndarray,
                          result: Dict[str, np.ndarray],
                          save_path: Path,
                          title: str = "") -> None:
    """
    Visualize pseudo-label generation results.

    Args:
        frame: Original MRI frame
        result: Dictionary from generate_pseudo_label()
        save_path: Path to save visualization
        title: Plot title
    """
    seg_map = result['segmentation']
    n_classes = seg_map.max() + 1

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Plot 1: Original frame
    axes[0, 0].imshow(frame, cmap='gray')
    axes[0, 0].set_title('Original MRI Frame')
    axes[0, 0].axis('off')

    # Plot 2: Tissue mask
    axes[0, 1].imshow(result['tissue_mask'], cmap='gray')
    axes[0, 1].set_title(f"Tissue Mask (threshold={result['threshold']:.3f})")
    axes[0, 1].axis('off')

    # Plot 3: Vocal tract mask
    axes[0, 2].imshow(result['vocal_tract_mask'], cmap='gray')
    axes[0, 2].set_title('Vocal Tract Region')
    axes[0, 2].axis('off')

    # Plot 4: Segmentation map
    axes[1, 0].imshow(seg_map, cmap='tab10', vmin=0, vmax=9)
    axes[1, 0].set_title(f"Pseudo-label ({n_classes} classes)")
    axes[1, 0].axis('off')

    # Plot 5: Overlay
    axes[1, 1].imshow(frame, cmap='gray')
    axes[1, 1].imshow(seg_map, cmap='tab10', alpha=0.5, vmin=0, vmax=9)
    axes[1, 1].set_title('Overlay')
    axes[1, 1].axis('off')

    # Plot 6: Class distribution
    unique, counts = np.unique(seg_map, return_counts=True)
    class_labels = ['Background/Air', 'Tongue', 'Jaw/Palate', 'Lips', 'Class 4', 'Class 5']
    axes[1, 2].bar([class_labels[i] if i < len(class_labels) else f'Class {i}'
                    for i in unique],
                   counts / seg_map.size * 100)
    axes[1, 2].set_ylabel('Percentage (%)')
    axes[1, 2].set_title('Class Distribution')
    axes[1, 2].tick_params(axis='x', rotation=45)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=plt.cm.tab10(i/10), label=class_labels[i] if i < len(class_labels) else f'Class {i}')
        for i in unique
    ]
    axes[1, 0].legend(handles=legend_elements, loc='center left',
                     bbox_to_anchor=(1, 0.5), fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved visualization: {save_path}")


def save_pseudo_label(seg_map: np.ndarray,
                     metadata: Dict,
                     save_path: Path) -> None:
    """
    Save pseudo-label to NPZ file.

    Args:
        seg_map: Segmentation map (H, W)
        metadata: Metadata dictionary
        save_path: Path to save NPZ file
    """
    np.savez_compressed(
        save_path,
        segmentation=seg_map.astype(np.uint8),
        **metadata
    )
    logger.info(f"Saved pseudo-label: {save_path}")


def select_frames_for_labeling(batch_summary_path: Path,
                               frames_per_subject: int = 10,
                               max_subjects: int = 15) -> List[Tuple[str, Path, List[int]]]:
    """
    Select frames for pseudo-label generation.

    Args:
        batch_summary_path: Path to batch summary JSON
        frames_per_subject: Number of frames to sample per subject
        max_subjects: Maximum number of subjects to process

    Returns:
        List of (utterance_name, hdf5_path, frame_indices) tuples
    """
    with open(batch_summary_path) as f:
        data = json.load(f)

    selected_samples = []

    for subject_idx, subject in enumerate(data['subjects'][:max_subjects]):
        subject_id = subject['subject_id']
        utterances = subject.get('utterances', [])

        if not utterances:
            continue

        # Select best quality utterance for this subject
        best_utt = max(utterances, key=lambda u: u.get('correlation', 0))

        hdf5_path = Path(best_utt['hdf5_path'])

        # Load frame count
        with h5py.File(hdf5_path, 'r') as f:
            n_frames = len(f['mri_frames'])

        # Sample frames evenly throughout utterance
        frame_indices = np.linspace(0, n_frames - 1, frames_per_subject, dtype=int).tolist()

        selected_samples.append((
            best_utt['utterance_name'],
            hdf5_path,
            frame_indices
        ))

        logger.info(f"Subject {subject_id}: {best_utt['utterance_name']} "
                   f"(corr={best_utt['correlation']:.3f}, {n_frames} frames, "
                   f"sampling {frames_per_subject} frames)")

    return selected_samples


def main():
    """Main function."""
    logger.info("="*70)
    logger.info("PSEUDO-LABEL GENERATION")
    logger.info("="*70)

    # Configuration
    batch_summary_path = Path("data/processed/aligned/batch_summary.json")
    output_dir = Path("data/processed/pseudo_labels")
    vis_dir = output_dir / "visualizations"
    ensure_directory(output_dir)
    ensure_directory(vis_dir)

    frames_per_subject = 10
    max_subjects = 15
    method = 'combined'  # Use combined approach for best results

    # Initialize segmenter
    logger.info(f"\nInitializing VocalTractSegmenter...")
    segmenter = VocalTractSegmenter(
        gaussian_sigma=1.0,
        min_tissue_area=50
    )

    # Select frames
    logger.info(f"\nSelecting frames for labeling...")
    logger.info(f"  Frames per subject: {frames_per_subject}")
    logger.info(f"  Max subjects: {max_subjects}")
    logger.info(f"  Method: {method}")

    selected_samples = select_frames_for_labeling(
        batch_summary_path,
        frames_per_subject=frames_per_subject,
        max_subjects=max_subjects
    )

    logger.info(f"\nTotal frames to process: {len(selected_samples) * frames_per_subject}")
    logger.info(f"Total samples (utterances): {len(selected_samples)}")

    # Process each sample
    logger.info(f"\n{'='*70}")
    logger.info("PROCESSING FRAMES")
    logger.info(f"{'='*70}\n")

    total_frames = 0
    for sample_idx, (utt_name, hdf5_path, frame_indices) in enumerate(selected_samples, 1):
        logger.info(f"\n[{sample_idx}/{len(selected_samples)}] Processing: {utt_name}")
        logger.info(f"  HDF5: {hdf5_path}")
        logger.info(f"  Frames: {frame_indices}")

        # Load frames
        frames = load_frames(hdf5_path, frame_indices)
        logger.info(f"  Loaded {len(frames)} frames, shape: {frames.shape}")

        # Create output directory for this utterance
        utt_output_dir = output_dir / utt_name
        ensure_directory(utt_output_dir)

        # Process each frame
        for frame_idx, global_frame_idx in enumerate(frame_indices):
            frame = frames[frame_idx]

            # Generate pseudo-label
            result = generate_pseudo_label(frame, segmenter, method=method)
            seg_map = result['segmentation']

            # Get class distribution
            unique, counts = np.unique(seg_map, return_counts=True)
            class_dist = {int(c): int(count) for c, count in zip(unique, counts)}

            logger.info(f"    Frame {global_frame_idx}: "
                       f"Classes: {unique.tolist()}, "
                       f"Dominant: {unique[counts.argmax()]} ({counts.max()/seg_map.size*100:.1f}%)")

            # Save pseudo-label
            metadata = {
                'utterance_name': utt_name,
                'hdf5_path': str(hdf5_path),
                'frame_index': global_frame_idx,
                'method': method,
                'class_distribution': class_dist,
                'n_classes': len(unique),
                'threshold': float(result['threshold'])
            }

            save_path = utt_output_dir / f"frame_{global_frame_idx:04d}_label.npz"
            save_pseudo_label(seg_map, metadata, save_path)

            # Visualize (every 5th frame to save space)
            if frame_idx % 5 == 0 or frame_idx == 0:
                vis_path = vis_dir / f"{utt_name}_frame_{global_frame_idx:04d}.png"
                title = f"{utt_name} - Frame {global_frame_idx}"
                visualize_pseudo_label(frame, result, vis_path, title=title)

            total_frames += 1

    logger.info(f"\n{'='*70}")
    logger.info("GENERATION COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"\nTotal frames processed: {total_frames}")
    logger.info(f"Total samples: {len(selected_samples)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Visualizations: {vis_dir}")

    # Create summary
    summary = {
        'total_frames': total_frames,
        'total_samples': len(selected_samples),
        'frames_per_subject': frames_per_subject,
        'max_subjects': max_subjects,
        'method': method,
        'samples': [
            {
                'utterance_name': utt_name,
                'hdf5_path': str(hdf5_path),
                'frame_indices': frame_indices
            }
            for utt_name, hdf5_path, frame_indices in selected_samples
        ]
    }

    summary_path = output_dir / 'generation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nGeneration summary saved: {summary_path}")


if __name__ == "__main__":
    main()
