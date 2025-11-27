"""
PyTorch Dataset for U-Net training with pseudo-labeled rtMRI data.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.logger import get_logger

logger = get_logger(__name__)


class SegmentationDataset(Dataset):
    """
    Dataset for vocal tract segmentation with pseudo-labels.

    Loads MRI frames and corresponding pseudo-label segmentation maps.
    """

    def __init__(
        self,
        pseudo_labels_dir: Path,
        transform: Optional[callable] = None,
        augment: bool = False
    ):
        """
        Initialize segmentation dataset.

        Args:
            pseudo_labels_dir: Directory containing pseudo-label NPZ files
            transform: Optional transform to apply to both image and mask
            augment: Whether to apply data augmentation
        """
        self.pseudo_labels_dir = Path(pseudo_labels_dir)
        self.transform = transform
        self.augment = augment

        # Collect all pseudo-label files
        self.samples = self._collect_samples()

        logger.info(f"Loaded {len(self.samples)} pseudo-labeled samples")

    def _collect_samples(self) -> List[Dict]:
        """
        Collect all pseudo-label files.

        Returns:
            List of sample dictionaries with paths and metadata
        """
        samples = []

        # Find all NPZ files
        for npz_path in sorted(self.pseudo_labels_dir.rglob("frame_*_label.npz")):
            # Load metadata
            try:
                data = np.load(npz_path, allow_pickle=True)

                sample = {
                    'label_path': npz_path,
                    'hdf5_path': Path(str(data['hdf5_path'])),
                    'frame_index': int(data['frame_index']),
                    'utterance_name': str(data['utterance_name']),
                }

                samples.append(sample)

            except Exception as e:
                logger.warning(f"Failed to load {npz_path}: {e}")
                continue

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image, mask, metadata)
            - image: (1, H, W) tensor, normalized [0, 1]
            - mask: (H, W) tensor, class indices [0, num_classes-1]
            - metadata: Dictionary with sample info
        """
        sample = self.samples[idx]

        # Load MRI frame
        with h5py.File(sample['hdf5_path'], 'r') as f:
            frame = f['mri_frames'][sample['frame_index']]

        # Normalize to [0, 1]
        if frame.max() > 1.0:
            frame = frame / frame.max()

        # Load segmentation mask
        label_data = np.load(sample['label_path'])
        mask = label_data['segmentation']

        # Pad to 96x96 for U-Net compatibility (requires size divisible by 16)
        target_size = 96
        h, w = frame.shape
        if h != target_size or w != target_size:
            pad_h = (target_size - h) // 2
            pad_w = (target_size - w) // 2
            frame = np.pad(frame, ((pad_h, target_size - h - pad_h), (pad_w, target_size - w - pad_w)), mode='constant')
            mask = np.pad(mask, ((pad_h, target_size - h - pad_h), (pad_w, target_size - w - pad_w)), mode='constant')

        # Convert to tensors
        image = torch.from_numpy(frame).float().unsqueeze(0)  # (1, H, W)
        mask = torch.from_numpy(mask).long()  # (H, W)

        # Apply augmentation if enabled
        if self.augment:
            image, mask = self._augment(image, mask)

        # Apply custom transform if provided
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        # Metadata
        metadata = {
            'utterance_name': sample['utterance_name'],
            'frame_index': sample['frame_index'],
            'label_path': str(sample['label_path']),
        }

        return image, mask, metadata

    def _augment(
        self,
        image: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply data augmentation.

        Args:
            image: (1, H, W) image tensor
            mask: (H, W) mask tensor

        Returns:
            Augmented (image, mask) tuple
        """
        # Random horizontal flip (50%)
        if torch.rand(1) > 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[1])

        # Random rotation (-10 to +10 degrees)
        if torch.rand(1) > 0.5:
            angle = (torch.rand(1) - 0.5) * 20  # -10 to +10
            image = self._rotate(image, angle)
            mask = self._rotate(mask.unsqueeze(0).float(), angle).squeeze(0).long()

        # Random brightness adjustment (0.8 to 1.2)
        if torch.rand(1) > 0.5:
            brightness_factor = 0.8 + torch.rand(1) * 0.4
            image = torch.clamp(image * brightness_factor, 0, 1)

        # Random contrast adjustment (0.8 to 1.2)
        if torch.rand(1) > 0.5:
            contrast_factor = 0.8 + torch.rand(1) * 0.4
            mean = image.mean()
            image = torch.clamp((image - mean) * contrast_factor + mean, 0, 1)

        return image, mask

    def _rotate(self, tensor: torch.Tensor, angle: float) -> torch.Tensor:
        """
        Rotate tensor by given angle.

        Args:
            tensor: (C, H, W) tensor
            angle: Rotation angle in degrees

        Returns:
            Rotated tensor
        """
        import torch.nn.functional as F

        # Convert angle to radians
        angle_rad = angle * np.pi / 180.0

        # Create rotation matrix
        cos_val = np.cos(angle_rad)
        sin_val = np.sin(angle_rad)

        # Affine transformation matrix (2x3)
        theta = torch.tensor([
            [cos_val, -sin_val, 0],
            [sin_val, cos_val, 0]
        ], dtype=torch.float32)

        # Apply affine transformation
        grid = F.affine_grid(
            theta.unsqueeze(0),
            tensor.unsqueeze(0).size(),
            align_corners=False
        )

        rotated = F.grid_sample(
            tensor.unsqueeze(0),
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )

        return rotated.squeeze(0)


def create_train_val_test_splits(
    pseudo_labels_dir: Path,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Create train/val/test splits from pseudo-labeled data.

    Splits by subject (utterance) to avoid data leakage.

    Args:
        pseudo_labels_dir: Directory containing pseudo-label subdirectories
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_paths, val_paths, test_paths)
    """
    np.random.seed(random_seed)

    # Get all utterance directories
    utterance_dirs = sorted([
        d for d in Path(pseudo_labels_dir).iterdir()
        if d.is_dir() and not d.name == 'visualizations'
    ])

    # Shuffle utterances
    utterance_dirs = np.random.permutation(utterance_dirs).tolist()

    # Calculate split sizes
    n_total = len(utterance_dirs)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_test - n_val

    # Split
    train_dirs = utterance_dirs[:n_train]
    val_dirs = utterance_dirs[n_train:n_train + n_val]
    test_dirs = utterance_dirs[n_train + n_val:]

    # Collect all NPZ files for each split
    train_paths = []
    val_paths = []
    test_paths = []

    for d in train_dirs:
        train_paths.extend(sorted(d.glob("frame_*_label.npz")))

    for d in val_dirs:
        val_paths.extend(sorted(d.glob("frame_*_label.npz")))

    for d in test_dirs:
        test_paths.extend(sorted(d.glob("frame_*_label.npz")))

    logger.info(f"Dataset splits:")
    logger.info(f"  Train: {len(train_dirs)} subjects, {len(train_paths)} frames")
    logger.info(f"  Val: {len(val_dirs)} subjects, {len(val_paths)} frames")
    logger.info(f"  Test: {len(test_dirs)} subjects, {len(test_paths)} frames")

    return train_paths, val_paths, test_paths


class SegmentationDatasetSplit(Dataset):
    """
    Segmentation dataset with pre-defined file list (for train/val/test splits).
    """

    def __init__(
        self,
        label_paths: List[Path],
        transform: Optional[callable] = None,
        augment: bool = False
    ):
        """
        Initialize dataset with specific label files.

        Args:
            label_paths: List of pseudo-label NPZ file paths
            transform: Optional transform
            augment: Whether to apply augmentation
        """
        self.label_paths = label_paths
        self.transform = transform
        self.augment = augment

        # Load metadata for all samples
        self.samples = []
        for label_path in label_paths:
            try:
                data = np.load(label_path, allow_pickle=True)
                sample = {
                    'label_path': label_path,
                    'hdf5_path': Path(str(data['hdf5_path'])),
                    'frame_index': int(data['frame_index']),
                    'utterance_name': str(data['utterance_name']),
                }
                self.samples.append(sample)
            except Exception as e:
                logger.warning(f"Failed to load {label_path}: {e}")
                continue

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Get a single sample (same as SegmentationDataset)."""
        sample = self.samples[idx]

        # Load MRI frame
        with h5py.File(sample['hdf5_path'], 'r') as f:
            frame = f['mri_frames'][sample['frame_index']]

        # Normalize to [0, 1]
        if frame.max() > 1.0:
            frame = frame / frame.max()

        # Load segmentation mask
        label_data = np.load(sample['label_path'])
        mask = label_data['segmentation']

        # Pad to 96x96 for U-Net compatibility (requires size divisible by 16)
        target_size = 96
        h, w = frame.shape
        if h != target_size or w != target_size:
            pad_h = (target_size - h) // 2
            pad_w = (target_size - w) // 2
            frame = np.pad(frame, ((pad_h, target_size - h - pad_h), (pad_w, target_size - w - pad_w)), mode='constant')
            mask = np.pad(mask, ((pad_h, target_size - h - pad_h), (pad_w, target_size - w - pad_w)), mode='constant')

        # Convert to tensors
        image = torch.from_numpy(frame).float().unsqueeze(0)  # (1, H, W)
        mask = torch.from_numpy(mask).long()  # (H, W)

        # Apply augmentation if enabled
        if self.augment:
            image, mask = self._augment(image, mask)

        # Apply custom transform if provided
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        # Metadata
        metadata = {
            'utterance_name': sample['utterance_name'],
            'frame_index': sample['frame_index'],
            'label_path': str(sample['label_path']),
        }

        return image, mask, metadata

    def _augment(
        self,
        image: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply data augmentation (same as SegmentationDataset)."""
        # Random horizontal flip (50%)
        if torch.rand(1) > 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[1])

        # Random brightness adjustment (0.8 to 1.2)
        if torch.rand(1) > 0.5:
            brightness_factor = 0.8 + torch.rand(1) * 0.4
            image = torch.clamp(image * brightness_factor, 0, 1)

        # Random contrast adjustment (0.8 to 1.2)
        if torch.rand(1) > 0.5:
            contrast_factor = 0.8 + torch.rand(1) * 0.4
            mean = image.mean()
            image = torch.clamp((image - mean) * contrast_factor + mean, 0, 1)

        return image, mask
