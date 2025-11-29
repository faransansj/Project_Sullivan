#!/usr/bin/env python3
"""
Selective Dataset Segmentation Script
======================================

Segment a selected subset of USC-TIMIT utterances for efficient processing.

Author: AI Research Assistant
Date: 2025-11-29
"""

import os
import sys
import json
import h5py
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.segmentation.unet import UNet
from src.utils.logger import setup_logger
from src.utils.io_utils import ensure_directory


def select_utterances(batch_summary: dict, max_per_subject: int = 10) -> list:
    """
    Select subset of utterances for segmentation

    Args:
        batch_summary: Loaded batch_summary.json
        max_per_subject: Maximum utterances per subject

    Returns:
        List of selected utterance info dicts
    """
    selected = []

    for subject_info in batch_summary['subjects']:
        subject_id = subject_info['subject_id']
        utterances = subject_info['utterances']

        # Select up to max_per_subject utterances per subject
        # Prioritize utterances with good alignment (high correlation)
        sorted_utts = sorted(
            utterances,
            key=lambda x: x.get('correlation', 0),
            reverse=True
        )

        selected_utts = sorted_utts[:max_per_subject]

        for utt in selected_utts:
            selected.append({
                'subject_id': subject_id,
                'utterance_name': utt['utterance_name'],
                'hdf5_path': utt['hdf5_path'],
                'correlation': utt.get('correlation', 0)
            })

    return selected


def main():
    parser = argparse.ArgumentParser(description="Segment subset of USC-TIMIT dataset")
    parser.add_argument(
        '--model',
        type=str,
        default='models/unet_scratch/unet_final.pth',
        help='Path to trained U-Net model'
    )
    parser.add_argument(
        '--batch-summary',
        type=str,
        default='data/processed/aligned/batch_summary.json',
        help='Path to batch_summary.json'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed/segmentations',
        help='Output directory for segmentations'
    )
    parser.add_argument(
        '--max-per-subject',
        type=int,
        default=10,
        help='Maximum utterances per subject'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use for inference'
    )
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Test mode: process only 2 utterances'
    )

    args = parser.parse_args()

    logger = setup_logger("SelectiveSegmenter")

    # Load batch summary
    logger.info(f"Loading batch summary from {args.batch_summary}")
    with open(args.batch_summary, 'r') as f:
        batch_summary = json.load(f)

    # Select utterances
    if args.test_only:
        logger.info("TEST MODE: Processing only 2 utterances")
        selected = select_utterances(batch_summary, max_per_subject=1)[:2]
    else:
        selected = select_utterances(batch_summary, max_per_subject=args.max_per_subject)

    logger.info(f"Selected {len(selected)} utterances for segmentation")

    # Create output directory
    output_dir = Path(args.output_dir)
    ensure_directory(output_dir)

    # Save selection info
    selection_info = {
        'timestamp': datetime.now().isoformat(),
        'total_selected': len(selected),
        'max_per_subject': args.max_per_subject,
        'test_mode': args.test_only,
        'selected_utterances': selected
    }

    selection_path = output_dir / 'selection_info.json'
    with open(selection_path, 'w') as f:
        json.dump(selection_info, f, indent=2)

    logger.info(f"Selection info saved to {selection_path}")

    # Load model
    logger.info(f"Loading model from {args.model}")
    device = torch.device(args.device)
    model = UNet(n_classes=4)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully")

    # Segment selected utterances
    logger.info("Starting segmentation...")

    stats_list = []
    total_frames = 0
    start_time = datetime.now()

    for idx, utt_info in enumerate(tqdm(selected, desc="Segmenting")):
        utterance_name = utt_info['utterance_name']
        hdf5_path = utt_info['hdf5_path']

        # Check if file exists
        if not os.path.exists(hdf5_path):
            logger.warning(f"HDF5 file not found: {hdf5_path}")
            continue

        # Create utterance output directory
        utt_output_dir = output_dir / utterance_name
        ensure_directory(utt_output_dir)

        try:
            # Load MRI frames
            with h5py.File(hdf5_path, 'r') as f:
                mri_frames = f['mri_frames'][:]
                num_frames = mri_frames.shape[0]

            # Segment frames
            segmentations = []
            class_distributions = []

            for frame_idx in range(num_frames):
                frame = mri_frames[frame_idx]

                # Normalize
                frame_norm = (frame - frame.mean()) / (frame.std() + 1e-8)

                # Pad to 96x96
                h, w = frame.shape
                pad_h = (96 - h) // 2
                pad_w = (96 - w) // 2
                frame_padded = np.pad(
                    frame_norm,
                    ((pad_h, 96 - h - pad_h), (pad_w, 96 - w - pad_w)),
                    mode='constant',
                    constant_values=0
                )

                # Convert to tensor
                frame_tensor = torch.FloatTensor(frame_padded).unsqueeze(0).unsqueeze(0).to(device)

                # Inference
                with torch.no_grad():
                    output = model(frame_tensor)
                    pred = output.argmax(dim=1).squeeze(0).cpu().numpy()

                # Unpad
                seg = pred[pad_h:pad_h+h, pad_w:pad_w+w]

                # Class distribution
                class_dist = np.bincount(seg.flatten(), minlength=4).astype(float)
                class_dist /= seg.size

                segmentations.append(seg)
                class_distributions.append(class_dist)

            # Stack and save
            segmentations = np.stack(segmentations, axis=0)
            class_distributions = np.stack(class_distributions, axis=0)

            output_path = utt_output_dir / f"{utterance_name}_segmentations.npz"
            np.savez_compressed(
                output_path,
                segmentations=segmentations.astype(np.uint8),
                class_distributions=class_distributions.astype(np.float32),
                utterance_name=utterance_name,
                hdf5_path=hdf5_path,
                num_frames=num_frames,
                class_names=['background', 'tongue', 'jaw', 'lips']
            )

            # Stats
            stats = {
                'utterance_name': utterance_name,
                'subject_id': utt_info['subject_id'],
                'num_frames': num_frames,
                'output_path': str(output_path),
                'mean_class_distribution': class_distributions.mean(axis=0).tolist(),
                'correlation': utt_info.get('correlation', 0)
            }
            stats_list.append(stats)
            total_frames += num_frames

            logger.info(f"  {utterance_name}: {num_frames} frames segmented")

        except Exception as e:
            logger.error(f"Failed to segment {utterance_name}: {e}")
            continue

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Save summary
    summary = {
        'timestamp': end_time.isoformat(),
        'total_utterances': len(stats_list),
        'total_frames': total_frames,
        'duration_seconds': duration,
        'frames_per_second': total_frames / duration if duration > 0 else 0,
        'model_path': args.model,
        'device': str(device),
        'output_dir': str(output_dir),
        'test_mode': args.test_only,
        'utterance_stats': stats_list
    }

    summary_path = output_dir / 'segmentation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "="*60)
    logger.info("SEGMENTATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Utterances processed: {len(stats_list)}")
    logger.info(f"Total frames: {total_frames:,}")
    logger.info(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    logger.info(f"Speed: {total_frames/duration:.2f} frames/sec")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info("="*60)


if __name__ == '__main__':
    main()
