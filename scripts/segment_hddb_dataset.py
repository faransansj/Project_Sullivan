#!/usr/bin/env python3
"""
HDDB Dataset Segmentation Script
==================================

Apply trained U-Net model to HDDB dataset (27 subjects, ~800 utterances).

Author: Claude & Research Assistant
Date: 2026-01-11
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

from src.segmentation.unet_simple import UNet
from src.utils.logger import setup_logger

def ensure_directory(path):
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)


class HDDBSegmenter:
    """Apply trained U-Net to HDDB dataset"""

    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize segmenter

        Args:
            model_path: Path to trained U-Net model weights
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        self.logger = setup_logger("HDDBSegmenter")

        # Load model
        self.logger.info(f"Loading model from {model_path}")
        # Note: Loaded model is binary segmentation (airway vs tissue)
        self.model = UNet(n_channels=1, n_classes=1, bilinear=True)

        # Load state dict
        state_dict = torch.load(model_path, map_location=self.device)

        # Handle PyTorch Lightning checkpoint format
        if 'model.' in list(state_dict.keys())[0]:
            # Remove 'model.' prefix from keys
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('model.', '')
                new_state_dict[new_key] = value
            state_dict = new_state_dict

        self.model.load_state_dict(state_dict)
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
            segmentation: (H, W) binary mask (0=background, 1=airway)
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
            output = self.model(frame_tensor)  # (1, 1, 96, 96) for binary
            pred = torch.sigmoid(output)  # Apply sigmoid for binary segmentation
            pred = (pred > 0.5).float().squeeze(0).squeeze(0).cpu().numpy()  # (96, 96)

        # Unpad to original size
        seg_unpadded = self.unpad_segmentation(pred).astype(np.uint8)

        return seg_unpadded

    def segment_utterance(self, h5_path: str, output_dir: Path) -> dict:
        """
        Segment all frames in an utterance

        Args:
            h5_path: Path to HDF5 file with MRI frames
            output_dir: Output directory for segmentations

        Returns:
            stats: Dictionary with processing statistics
        """
        utterance_name = Path(h5_path).stem

        # Create output directory
        utterance_output_dir = output_dir / utterance_name
        ensure_directory(utterance_output_dir)

        # Load MRI frames from HDDB recon format
        with h5py.File(h5_path, 'r') as f:
            # HDDB recon format uses 'recon' key
            mri_frames = f['recon'][:]  # (T, H, W) format
            num_frames = mri_frames.shape[0]

        # Segment each frame
        segmentations = []
        class_distributions = []

        for frame_idx in tqdm(range(num_frames), desc=f"Segmenting {utterance_name}", leave=False):
            frame = mri_frames[frame_idx]
            seg = self.segment_frame(frame)

            # Compute class distribution (binary: 0=background, 1=airway)
            class_dist = np.bincount(seg.flatten(), minlength=2).astype(float)
            class_dist /= seg.size

            segmentations.append(seg)
            class_distributions.append(class_dist)

        # Stack all segmentations
        segmentations = np.stack(segmentations, axis=0)  # (num_frames, H, W)
        class_distributions = np.stack(class_distributions, axis=0)  # (num_frames, 2)

        # Save as NPZ
        output_path = utterance_output_dir / f"{utterance_name}_segmentations.npz"
        np.savez_compressed(
            output_path,
            segmentations=segmentations.astype(np.uint8),
            class_distributions=class_distributions.astype(np.float32),
            utterance_name=utterance_name,
            h5_path=h5_path,
            num_frames=num_frames,
            class_names=['background', 'airway'],  # Binary segmentation
            segmentation_type='binary'  # Note: not multi-class
        )

        stats = {
            'utterance_name': utterance_name,
            'num_frames': num_frames,
            'output_path': str(output_path),
            'mean_class_distribution': class_distributions.mean(axis=0).tolist()
        }

        return stats

    def segment_dataset(self, data_root: str, subjects: list, output_dir: str):
        """
        Segment entire HDDB dataset

        Args:
            data_root: Root path to HDDB dataset
            subjects: List of subject IDs to process
            output_dir: Output directory for segmentations
        """
        data_root = Path(data_root)
        output_dir = Path(output_dir)
        ensure_directory(output_dir)

        self.logger.info(f"Starting segmentation of {len(subjects)} subjects")

        # Process each subject
        all_stats = []
        total_frames_processed = 0

        for subject_id in tqdm(subjects, desc="Subjects"):
            # Use recon directory instead of raw
            subject_dir = data_root / subject_id / "2drt" / "recon"

            if not subject_dir.exists():
                self.logger.warning(f"Subject directory not found: {subject_dir}")
                continue

            self.logger.info(f"Processing {subject_id}")

            # Find all H5 files
            h5_files = sorted(subject_dir.glob("*.h5"))

            for h5_path in tqdm(h5_files, desc=f"{subject_id} utterances", leave=False):
                try:
                    stats = self.segment_utterance(str(h5_path), output_dir)
                    all_stats.append(stats)
                    total_frames_processed += stats['num_frames']
                except Exception as e:
                    self.logger.error(f"Failed to segment {h5_path.name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        # Save processing summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_subjects': len(subjects),
            'total_utterances': len(all_stats),
            'total_frames': total_frames_processed,
            'model_path': 'models/unet_scratch/unet_best.pth',
            'device': str(self.device),
            'output_dir': str(output_dir),
            'subjects': subjects,
            'utterance_stats': all_stats
        }

        summary_path = output_dir / 'segmentation_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Segmentation complete!")
        self.logger.info(f"  Subjects processed: {len(subjects)}")
        self.logger.info(f"  Utterances processed: {len(all_stats)}")
        self.logger.info(f"  Total frames: {total_frames_processed}")
        self.logger.info(f"  Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Segment HDDB dataset")
    parser.add_argument(
        '--model',
        type=str,
        default='models/unet_scratch/unet_best.pth',
        help='Path to trained U-Net model'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='/mnt/HDDB/dataset/my_dataset/dataset',
        help='Root path to HDDB dataset'
    )
    parser.add_argument(
        '--subjects',
        type=str,
        default='all',
        help='Comma-separated subject IDs or "all"'
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

    # Get list of subjects
    if args.subjects == 'all':
        # Load from dataset_stats.json
        stats_path = Path('data/hddb_dataset_stats.json')
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            subjects = stats['subjects']
        else:
            # Find subjects manually
            data_root = Path(args.data_root)
            subjects = sorted([d.name for d in data_root.iterdir() if d.is_dir() and d.name.startswith('sub')])
    else:
        subjects = args.subjects.split(',')

    print(f"Processing {len(subjects)} subjects: {', '.join(subjects[:5])}{'...' if len(subjects) > 5 else ''}")

    # Create segmenter
    segmenter = HDDBSegmenter(
        model_path=args.model,
        device=args.device
    )

    # Segment dataset
    segmenter.segment_dataset(
        data_root=args.data_root,
        subjects=subjects,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
