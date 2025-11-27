"""
Dataset classes for Phase 2: Audio-to-Articulatory Parameter Modeling

This module provides PyTorch Dataset classes for loading paired
audio features and articulatory parameters.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from dataclasses import dataclass


@dataclass
class DataSample:
    """Single training/validation sample"""
    audio_features: torch.Tensor  # (time, audio_dim)
    parameters: torch.Tensor  # (time, 10)
    utterance_name: str
    subject_id: str
    duration: float  # seconds


class ArticulatoryDataset(Dataset):
    """
    Dataset for audio-to-articulatory parameter prediction

    Data format:
        Each sample consists of:
        - Audio features: (time, audio_dim) - Mel-spectrogram or MFCC
        - Articulatory parameters: (time, 10) - Extracted from segmentations
    """

    def __init__(self,
                 data_dir: str,
                 split: str = "train",
                 audio_feature_type: str = "mel",
                 normalize: bool = True,
                 max_duration: Optional[float] = None,
                 min_duration: Optional[float] = 0.1):
        """
        Args:
            data_dir: Root directory containing processed data
            split: Dataset split ("train", "val", "test")
            audio_feature_type: Type of audio features ("mel", "mfcc", "both")
            normalize: Whether features are normalized
            max_duration: Maximum utterance duration (seconds)
            min_duration: Minimum utterance duration (seconds)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.audio_feature_type = audio_feature_type
        self.normalize = normalize
        self.max_duration = max_duration
        self.min_duration = min_duration

        # Load file lists
        self.samples = self._load_sample_list()

        # Statistics
        self.audio_dim = self._get_audio_dim()
        self.param_dim = 10  # Fixed: 10 articulatory parameters

        print(f"Loaded {len(self.samples)} samples for split '{split}'")
        print(f"Audio feature dim: {self.audio_dim}, Parameter dim: {self.param_dim}")

    def _load_sample_list(self) -> List[Dict[str, str]]:
        """Load list of samples for this split"""
        split_file = self.data_dir / f"{self.split}_samples.json"

        if not split_file.exists():
            # Fallback: scan directory for files
            return self._scan_directory()

        with open(split_file, 'r') as f:
            sample_list = json.load(f)

        # Filter by duration if specified
        filtered_samples = []
        for sample in sample_list:
            duration = sample.get('duration', float('inf'))

            if self.min_duration is not None and duration < self.min_duration:
                continue

            if self.max_duration is not None and duration > self.max_duration:
                continue

            filtered_samples.append(sample)

        return filtered_samples

    def _scan_directory(self) -> List[Dict[str, str]]:
        """Scan directory for audio-parameter pairs"""
        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        samples = []

        # Find all parameter files
        param_files = sorted(split_dir.glob("*_parameters.npy"))

        for param_file in param_files:
            # Construct audio feature filename
            base_name = param_file.stem.replace("_parameters", "")
            audio_file = split_dir / f"{base_name}_audio_{self.audio_feature_type}.npy"

            if audio_file.exists():
                samples.append({
                    'audio_path': str(audio_file),
                    'param_path': str(param_file),
                    'utterance_name': base_name,
                    'subject_id': base_name.split('_')[0] if '_' in base_name else 'unknown'
                })

        return samples

    def _get_audio_dim(self) -> int:
        """Infer audio feature dimensionality from first sample"""
        if len(self.samples) == 0:
            # Default dimensions
            if self.audio_feature_type == "mel":
                return 80
            elif self.audio_feature_type == "mfcc":
                return 39  # 13 MFCCs + 13 deltas + 13 delta-deltas
            else:
                return 119  # mel + mfcc

        # Load first sample to get dimensions
        sample = self.samples[0]
        audio = np.load(sample['audio_path'])
        return audio.shape[1]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample

        Returns:
            (audio_features, parameters) tuple
            - audio_features: (time, audio_dim)
            - parameters: (time, 10)
        """
        sample = self.samples[idx]

        # Load audio features
        audio = np.load(sample['audio_path'])  # (time, audio_dim)

        # Load parameters
        params = np.load(sample['param_path'])  # (time, 10)

        # Ensure same sequence length
        min_len = min(len(audio), len(params))
        audio = audio[:min_len]
        params = params[:min_len]

        # Convert to tensors
        audio_tensor = torch.FloatTensor(audio)
        param_tensor = torch.FloatTensor(params)

        return audio_tensor, param_tensor

    def get_sample_info(self, idx: int) -> Dict:
        """Get metadata for a sample"""
        return self.samples[idx]


def collate_fn_pad(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function with padding for variable-length sequences

    Args:
        batch: List of (audio, params) tuples

    Returns:
        Dictionary with:
            - audio: (batch, max_time, audio_dim) padded audio features
            - params: (batch, max_time, 10) padded parameters
            - lengths: (batch,) sequence lengths
            - mask: (batch, max_time) padding mask
    """
    audio_list, param_list = zip(*batch)

    # Get sequence lengths
    lengths = torch.LongTensor([len(audio) for audio in audio_list])
    max_len = torch.max(lengths).item()

    # Get dimensions
    batch_size = len(batch)
    audio_dim = audio_list[0].shape[1]
    param_dim = param_list[0].shape[1]

    # Initialize padded tensors
    audio_padded = torch.zeros(batch_size, max_len, audio_dim)
    param_padded = torch.zeros(batch_size, max_len, param_dim)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    # Fill in data
    for i, (audio, params, length) in enumerate(zip(audio_list, param_list, lengths)):
        audio_padded[i, :length] = audio
        param_padded[i, :length] = params
        mask[i, :length] = True

    return {
        'audio': audio_padded,
        'params': param_padded,
        'lengths': lengths,
        'mask': mask
    }


def create_dataloaders(data_dir: str,
                      batch_size: int = 32,
                      num_workers: int = 4,
                      audio_feature_type: str = "mel",
                      max_duration: Optional[float] = None) -> Dict[str, DataLoader]:
    """
    Create train/val/test dataloaders

    Args:
        data_dir: Root directory containing data
        batch_size: Batch size
        num_workers: Number of dataloader workers
        audio_feature_type: Type of audio features
        max_duration: Maximum utterance duration

    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        dataset = ArticulatoryDataset(
            data_dir=data_dir,
            split=split,
            audio_feature_type=audio_feature_type,
            max_duration=max_duration
        )

        # Shuffle only for training
        shuffle = (split == 'train')

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn_pad,
            pin_memory=True
        )

        dataloaders[split] = dataloader

    return dataloaders


def compute_dataset_statistics(data_dir: str,
                              audio_feature_type: str = "mel") -> Dict[str, np.ndarray]:
    """
    Compute normalization statistics from training set

    Args:
        data_dir: Root directory containing data
        audio_feature_type: Type of audio features

    Returns:
        Dictionary with 'audio_mean', 'audio_std', 'param_mean', 'param_std'
    """
    dataset = ArticulatoryDataset(
        data_dir=data_dir,
        split='train',
        audio_feature_type=audio_feature_type,
        normalize=False
    )

    # Collect all features
    all_audio = []
    all_params = []

    print("Computing dataset statistics...")
    for i in range(len(dataset)):
        audio, params = dataset[i]
        all_audio.append(audio.numpy())
        all_params.append(params.numpy())

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(dataset)} samples")

    # Concatenate
    all_audio = np.concatenate(all_audio, axis=0)
    all_params = np.concatenate(all_params, axis=0)

    # Compute statistics
    stats = {
        'audio_mean': np.mean(all_audio, axis=0).astype(np.float32),
        'audio_std': np.std(all_audio, axis=0).astype(np.float32),
        'param_mean': np.mean(all_params, axis=0).astype(np.float32),
        'param_std': np.std(all_params, axis=0).astype(np.float32)
    }

    # Prevent division by zero
    stats['audio_std'] = np.where(stats['audio_std'] < 1e-6, 1.0, stats['audio_std'])
    stats['param_std'] = np.where(stats['param_std'] < 1e-6, 1.0, stats['param_std'])

    print("\nDataset Statistics:")
    print(f"Audio features: mean={np.mean(stats['audio_mean']):.4f}, std={np.mean(stats['audio_std']):.4f}")
    print(f"Parameters: mean={np.mean(stats['param_mean']):.4f}, std={np.mean(stats['param_std']):.4f}")

    return stats


def save_dataset_statistics(stats: Dict[str, np.ndarray], output_path: str):
    """Save normalization statistics to file"""
    np.savez(output_path, **stats)
    print(f"Saved statistics to {output_path}")


def load_dataset_statistics(stats_path: str) -> Dict[str, np.ndarray]:
    """Load normalization statistics from file"""
    data = np.load(stats_path)
    return {
        'audio_mean': data['audio_mean'],
        'audio_std': data['audio_std'],
        'param_mean': data['param_mean'],
        'param_std': data['param_std']
    }
