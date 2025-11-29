#!/usr/bin/env python3
"""
Full Dataset Segmentation Script
==================================

Apply trained U-Net model to all 468 USC-TIMIT utterances.

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


class FullDatasetSegmenter:
    """Apply trained U-Net to full dataset"""

    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize segmenter

        Args:
            model_path: Path to trained U-Net model weights
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        self.logger = setup_logger("FullDatasetSegmenter")

        # Load model
        self.logger.info(f"Loading model from {model_path}")
        self.model = UNet(n_classes=4)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.logger.info("Model loaded successfully")

    def pad_frame(self, frame: np.ndarray, target_size: int = 96) -> np.ndarray:
        """
        Pad frame to target size (U-Net requires size divisible by 16)

        Args:
            frame: (H, W) array
            target_size: Target size (default 96)

        Returns:
            Padded frame (target_size, target_size)
        """
        h, w = frame.shape
        pad_h = (target_size - h) // 2
        pad_w = (target_size - w) // 2

        padded = np.pad(
            frame,
            ((pad_h, target_size - h - pad_h), (pad_w, target_size - w - pad_w)),
            mode='constant',
            constant_values=0
        )

        return padded

    def unpad_segmentation(self, seg: np.ndarray, original_size: int = 84) -> np.ndarray:
        """
        Remove padding from segmentation

        Args:
            seg: (96, 96) segmentation
            original_size: Original size (default 84)

        Returns:
            Unpadded segmentation (84, 84)
        """
        padded_size = seg.shape[0]
        crop = (padded_size - original_size) // 2
        return seg[crop:crop+original_size, crop:crop+original_size]

    def segment_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Segment single MRI frame

        Args:
            frame: (H, W) MRI frame

        Returns:
            segmentation: (H, W) class indices (0-3)
        """
        # Normalize frame
        frame_norm = (frame - frame.mean()) / (frame.std() + 1e-8)

        # Pad to 96x96
        frame_padded = self.pad_frame(frame_norm)

        # Convert to tensor (1, 1, 96, 96)
        frame_tensor = torch.FloatTensor(frame_padded).unsqueeze(0).unsqueeze(0)
        frame_tensor = frame_tensor.to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(frame_tensor)  # (1, 4, 96, 96)
            pred = output.argmax(dim=1).squeeze(0).cpu().numpy()  # (96, 96)

        # Unpad to original size
        seg_unpadded = self.unpad_segmentation(pred)

        return seg_unpadded

    def segment_utterance(self, hdf5_path: str, output_dir: Path) -> dict:
        """
        Segment all frames in an utterance

        Args:
            hdf5_path: Path to HDF5 file with MRI frames
            output_dir: Output directory for segmentations

        Returns:
            stats: Dictionary with processing statistics
        """
        utterance_name = Path(hdf5_path).stem

        # Create output directory
        utterance_output_dir = output_dir / utterance_name
        ensure_directory(utterance_output_dir)

        # Load MRI frames
        with h5py.File(hdf5_path, 'r') as f:
            mri_frames = f['mri_frames'][:]  # (num_frames, H, W)
            num_frames = mri_frames.shape[0]

        # Segment each frame
        segmentations = []
        class_distributions = []

        for frame_idx in range(num_frames):
            frame = mri_frames[frame_idx]
            seg = self.segment_frame(frame)

            # Compute class distribution
            class_dist = np.bincount(seg.flatten(), minlength=4).astype(float)
            class_dist /= seg.size

            segmentations.append(seg)
            class_distributions.append(class_dist)

        # Stack all segmentations
        segmentations = np.stack(segmentations, axis=0)  # (num_frames, H, W)
        class_distributions = np.stack(class_distributions, axis=0)  # (num_frames, 4)

        # Save as NPZ
        output_path = utterance_output_dir / f"{utterance_name}_segmentations.npz"
        np.savez_compressed(
            output_path,
            segmentations=segmentations.astype(np.uint8),
            class_distributions=class_distributions.astype(np.float32),
            utterance_name=utterance_name,
            hdf5_path=hdf5_path,
            num_frames=num_frames,
            class_names=['background', 'tongue', 'jaw', 'lips']
        )

        stats = {
            'utterance_name': utterance_name,
            'num_frames': num_frames,
            'output_path': str(output_path),
            'mean_class_distribution': class_distributions.mean(axis=0).tolist()
        }

        return stats

    def segment_dataset(self, batch_summary_path: str, output_dir: str):
        """
        Segment entire dataset

        Args:
            batch_summary_path: Path to batch_summary.json
            output_dir: Output directory for segmentations
        """
        output_dir = Path(output_dir)
        ensure_directory(output_dir)

        # Load batch summary
        with open(batch_summary_path, 'r') as f:
            batch_summary = json.load(f)

        total_utterances = batch_summary['total_utterances']
        self.logger.info(f"Starting segmentation of {total_utterances} utterances")

        # Process each subject
        all_stats = []
        total_frames_processed = 0

        for subject_info in tqdm(batch_summary['subjects'], desc="Subjects"):
            subject_id = subject_info['subject_id']
            self.logger.info(f"Processing {subject_id}")

            for utterance_info in tqdm(subject_info['utterances'],
                                      desc=f"{subject_id} utterances",
                                      leave=False):
                hdf5_path = utterance_info['hdf5_path']

                # Skip if HDF5 file doesn't exist
                if not os.path.exists(hdf5_path):
                    self.logger.warning(f"HDF5 file not found: {hdf5_path}")
                    continue

                # Segment utterance
                try:
                    stats = self.segment_utterance(hdf5_path, output_dir)
                    all_stats.append(stats)
                    total_frames_processed += stats['num_frames']
                except Exception as e:
                    self.logger.error(f"Failed to segment {utterance_info['utterance_name']}: {e}")
                    continue

        # Save processing summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_utterances': len(all_stats),
            'total_frames': total_frames_processed,
            'model_path': str(self.model.state_dict()),
            'device': str(self.device),
            'output_dir': str(output_dir),
            'utterance_stats': all_stats
        }

        summary_path = output_dir / 'segmentation_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Segmentation complete!")
        self.logger.info(f"  Utterances processed: {len(all_stats)}")
        self.logger.info(f"  Total frames: {total_frames_processed}")
        self.logger.info(f"  Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Segment full USC-TIMIT dataset")
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
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use for inference'
    )

    args = parser.parse_args()

    # Create segmenter
    segmenter = FullDatasetSegmenter(
        model_path=args.model,
        device=args.device
    )

    # Segment dataset
    segmenter.segment_dataset(
        batch_summary_path=args.batch_summary,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
