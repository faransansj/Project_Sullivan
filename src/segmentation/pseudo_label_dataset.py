"""
PyTorch Dataset for Pseudo-Labels
"""

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import json
from typing import Optional, Tuple, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PseudoLabelDataset(Dataset):
    """
    Dataset for loading pseudo-labeled MRI segmentation data.
    
    Args:
        data_dir: Path to pseudo_labels directory
        split: 'train', 'val', or 'test'
        transform: Optional albumentations transform
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Load metadata
        metadata_path = self.data_dir / 'metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.samples = metadata['samples']
        
        # Split data (70/15/15 train/val/test)
        np.random.seed(42)  # Fixed seed for reproducibility
        indices = np.random.permutation(len(self.samples))
        
        n_train = int(0.70 * len(self.samples))
        n_val = int(0.15 * len(self.samples))
        
        if split == 'train':
            self.indices = indices[:n_train]
        elif split == 'val':
            self.indices = indices[n_train:n_train + n_val]
        elif split == 'test':
            self.indices = indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {split}")
        
        print(f"[{split.upper()}] Loaded {len(self.indices)} samples")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get sample metadata
        sample_idx = self.indices[idx]
        sample = self.samples[sample_idx]
        
        # Load image and mask
        image_path = self.data_dir / sample['image_path']
        mask_path = self.data_dir / sample['mask_path']
        
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Normalize mask to binary {0, 1}
        mask = (mask > 127).astype(np.float32)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            # Add channel dimension to mask if needed
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)  # (1, H, W)
        else:
            # Default: convert to tensor
            image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
            mask = torch.from_numpy(mask).unsqueeze(0)    # (1, H, W)

        return image, mask


def get_train_transform():
    """Data augmentation for training"""
    return A.Compose([
        # Geometric augmentation
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5,
            border_mode=cv2.BORDER_CONSTANT,
            value=0
        ),
        
        # Intensity augmentation
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.GaussNoise(var_limit=(0.0, 0.01), p=0.3),
        
        # Convert to tensor
        ToTensorV2()
    ])


def get_val_transform():
    """No augmentation for validation/test"""
    return A.Compose([
        ToTensorV2()
    ])


if __name__ == "__main__":
    # Test dataset
    data_dir = "data/pseudo_labels"
    
    # Test train set
    train_dataset = PseudoLabelDataset(data_dir, split='train', transform=get_train_transform())
    print(f"\nTrain dataset: {len(train_dataset)} samples")
    
    image, mask = train_dataset[0]
    print(f"Image shape: {image.shape}, dtype: {image.dtype}")
    print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
    print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"Mask unique values: {torch.unique(mask).tolist()}")
    
    # Test val set
    val_dataset = PseudoLabelDataset(data_dir, split='val', transform=get_val_transform())
    print(f"\nVal dataset: {len(val_dataset)} samples")
    
    # Test test set
    test_dataset = PseudoLabelDataset(data_dir, split='test', transform=get_val_transform())
    print(f"Test dataset: {len(test_dataset)} samples")
    
    print(f"\nTotal: {len(train_dataset) + len(val_dataset) + len(test_dataset)} samples")
    print("✅ Dataset test passed!")
