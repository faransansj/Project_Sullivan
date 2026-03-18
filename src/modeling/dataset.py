"""
PyTorch Dataset for Audio-to-Articulatory Parameter Pairs

This module provides dataset classes and utilities for loading synchronized
audio features and articulatory parameters.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import zipfile
import io
from scipy import interpolate


class AudioAugmentation:
    """SpecAugment-style augmentation for audio features (HuBERT or Mel).

    Applies time masking, feature masking, and Gaussian noise.
    Only intended for training — do not apply to val/test.

    Parameters
    ----------
    time_mask_max_len : int
        Maximum number of consecutive time frames to zero out per mask.
    time_mask_num : int
        Number of independent time masks to apply.
    freq_mask_max_len : int
        Maximum number of consecutive feature dims to zero out per mask.
    freq_mask_num : int
        Number of independent feature masks to apply.
    noise_std : float
        Std of Gaussian noise added to features (0 = disabled).
    """

    def __init__(
        self,
        time_mask_max_len: int = 30,
        time_mask_num: int = 2,
        freq_mask_max_len: int = 64,
        freq_mask_num: int = 2,
        noise_std: float = 0.01,
    ):
        self.time_mask_max_len = time_mask_max_len
        self.time_mask_num = time_mask_num
        self.freq_mask_max_len = freq_mask_max_len
        self.freq_mask_num = freq_mask_num
        self.noise_std = noise_std

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to feature tensor of shape (T, F)."""
        T, F = features.shape
        features = features.clone()

        # Time masking
        for _ in range(self.time_mask_num):
            if self.time_mask_max_len > 0 and T > 1:
                t = np.random.randint(1, max(2, min(self.time_mask_max_len, T)))
                t0 = np.random.randint(0, max(1, T - t))
                features[t0:t0 + t, :] = 0.0

        # Feature masking
        for _ in range(self.freq_mask_num):
            if self.freq_mask_max_len > 0 and F > 1:
                f = np.random.randint(1, max(2, min(self.freq_mask_max_len, F)))
                f0 = np.random.randint(0, max(1, F - f))
                features[:, f0:f0 + f] = 0.0

        # Gaussian noise
        if self.noise_std > 0:
            features = features + torch.randn_like(features) * self.noise_std

        return features


@dataclass
class DataSample:
    """Single training/validation sample (Legacy support from Phase 2)"""
    audio_features: torch.Tensor  # (time, audio_dim)
    parameters: torch.Tensor  # (time, 10)
    utterance_name: str
    subject_id: str
    duration: float  # seconds


class ArticulatoryDataset(Dataset):
    """
    Dataset for audio features and articulatory parameters.

    Loads pre-extracted audio features (mel-spectrogram or MFCC) and
    corresponding articulatory parameters (geometric or PCA).

    Parameters
    ----------
    utterance_list : list of str
        List of utterance names to include
    audio_feature_dir : Path
        Directory containing audio features
    parameter_dir : Path
        Directory containing articulatory parameters
    audio_feature_type : str, default='mel'
        Type of audio features ('mel' or 'mfcc')
    parameter_type : str, default='geometric'
        Type of parameters ('geometric' or 'pca')
    normalize_params : bool, default=True
        Whether to normalize parameters to [0, 1] range
    sequence_length : int, optional
        If provided, split utterances into fixed-length sequences
    """

    def __init__(
        self,
        utterance_list: List[str],
        audio_feature_dir: Path,
        parameter_dir: Path,
        audio_feature_type: str = 'mel',
        parameter_type: str = 'geometric',
        normalize_params: bool = True,
        normalization_type: str = 'minmax',  # 'minmax' or 'zscore'
        sequence_length: Optional[int] = None,
        streaming: bool = False,
        zip_file_path: Optional[Path] = None,
        augmentation: Optional['AudioAugmentation'] = None,
    ):
        self.utterance_list = utterance_list
        self.audio_feature_dir = Path(audio_feature_dir)
        self.parameter_dir = Path(parameter_dir)
        self.audio_feature_type = audio_feature_type
        self.parameter_type = parameter_type
        self.normalize_params = normalize_params
        self.normalization_type = normalization_type
        self.sequence_length = sequence_length
        self.streaming = streaming
        self.zip_file_path = Path(zip_file_path) if zip_file_path else None
        self.augmentation = augmentation

        # Load data (metadata only if streaming)
        self.data = self._load_data()

        # Compute or set normalization statistics if needed
        if self.normalize_params:
            self._compute_normalization_stats()

    def _load_data(self) -> List[Dict]:
        """Load utterances (metadata or full data)."""
        data = []

        for utterance_name in self.utterance_list:
            # Paths - try NPZ first (legacy), then NPY (current pipeline)
            audio_file_npz = self.audio_feature_dir / f'{utterance_name}_audio.npz'
            param_file_npz = self.parameter_dir / f'{utterance_name}_params.npz'
            
            # Construct NPY paths
            if self.audio_feature_type == 'mel':
                audio_file_npy = self.audio_feature_dir / 'mel_spectrogram' / f'{utterance_name}_mel.npy'
                if not audio_file_npy.exists():
                    audio_file_npy = self.audio_feature_dir / f'{utterance_name}_mel.npy'
            elif self.audio_feature_type == 'hubert':
                audio_file_npy = self.audio_feature_dir / 'hubert' / f'{utterance_name}_hubert.npy'
            else:
                audio_file_npy = self.audio_feature_dir / 'mfcc' / f'{utterance_name}_mfcc.npy'
                if not audio_file_npy.exists():
                    audio_file_npy = self.audio_feature_dir / f'{utterance_name}_mfcc.npy'

            if self.parameter_type == 'geometric':
                param_file_npy = self.parameter_dir / 'geometric' / f'{utterance_name}_params.npy'
                if not param_file_npy.exists():
                    param_file_npy = self.parameter_dir / f'{utterance_name}_params.npy'
            else:
                param_file_npy = self.parameter_dir / 'pca' / f'{utterance_name}_params.npy'
                if not param_file_npy.exists():
                    param_file_npy = self.parameter_dir / f'{utterance_name}_params.npy'

            # Determine which files to use
            if audio_file_npz.exists():
                audio_file = audio_file_npz
                is_npy_audio = False
            elif audio_file_npy.exists():
                audio_file = audio_file_npy
                is_npy_audio = True
            else:
                continue

            if param_file_npz.exists():
                param_file = param_file_npz
                is_npy_param = False
            elif param_file_npy.exists():
                param_file = param_file_npy
                is_npy_param = True
            else:
                continue

            if self.streaming:
                # Streaming mode: store metadata only
                if self.sequence_length is not None:
                    try:
                        if is_npy_param:
                            params_mmap = np.load(param_file, mmap_mode='r')
                            num_frames = params_mmap.shape[0]
                        else:
                            param_data = np.load(param_file)
                            num_frames = param_data['geometric_features'].shape[0]
                        
                        num_sequences = num_frames // self.sequence_length
                        for i in range(num_sequences):
                            start_idx = i * self.sequence_length
                            end_idx = start_idx + self.sequence_length
                            data.append({
                                'utterance_name': f"{utterance_name}_seq{i}",
                                'audio_file': audio_file,
                                'param_file': param_file,
                                'is_npy_audio': is_npy_audio,
                                'is_npy_param': is_npy_param,
                                'start_idx': start_idx,
                                'end_idx': end_idx
                            })
                    except Exception as e:
                        print(f"Warning: Could not determine length for {utterance_name}: {e}")
                else:
                    data.append({
                        'utterance_name': utterance_name,
                        'audio_file': audio_file,
                        'param_file': param_file,
                        'is_npy_audio': is_npy_audio,
                        'is_npy_param': is_npy_param,
                        'start_idx': None,
                        'end_idx': None
                    })
                continue

            # Load full data if not streaming
            if is_npy_audio:
                audio_features = np.load(audio_file)
            else:
                audio_data = np.load(audio_file)
                audio_features = audio_data['mel_spectrogram'] if self.audio_feature_type == 'mel' else audio_data['mfcc']

            if is_npy_param:
                parameters = np.load(param_file)
            else:
                param_data = np.load(param_file)
                if self.parameter_type == 'geometric':
                    parameters = param_data['geometric_features']
                elif self.parameter_type == 'pca':
                    parameters = param_data['pca_features']
                else:
                    parameters = param_data['parameters']

            if audio_features.shape[0] != parameters.shape[0]:
                audio_features = self._interpolate_features(audio_features, parameters.shape[0])

            if self.sequence_length is not None:
                num_frames = audio_features.shape[0]
                num_sequences = num_frames // self.sequence_length
                for i in range(num_sequences):
                    start_idx = i * self.sequence_length
                    end_idx = start_idx + self.sequence_length
                    data.append({
                        'utterance_name': f"{utterance_name}_seq{i}",
                        'audio_features': audio_features[start_idx:end_idx],
                        'parameters': parameters[start_idx:end_idx]
                    })
            else:
                data.append({
                    'utterance_name': utterance_name,
                    'audio_features': audio_features,
                    'parameters': parameters
                })

        return data

    def _compute_normalization_stats(self):
        """Compute statistics for normalization."""
        if self.streaming:
            num_samples = min(len(self.data), 100)
            sampled_params = []
            for i in range(num_samples):
                item = self.data[i]
                param_data = np.load(item['param_file'])
                if item['is_npy_param']:
                    params = param_data
                else:
                    params = param_data['geometric_features'] if self.parameter_type == 'geometric' else param_data['pca_features']
                
                if item['start_idx'] is not None:
                    params = params[item['start_idx']:item['end_idx']]
                sampled_params.append(params)
            all_params = np.concatenate(sampled_params, axis=0) if sampled_params else np.zeros((1, 14))
        else:
            all_params = np.concatenate([item['parameters'] for item in self.data], axis=0)

        if self.normalization_type == 'minmax':
            self.param_min = np.min(all_params, axis=0, keepdims=True)
            self.param_max = np.max(all_params, axis=0, keepdims=True)
            self.param_range = np.where((self.param_max - self.param_min) < 1e-8, 1.0, self.param_max - self.param_min)
        elif self.normalization_type == 'zscore':
            self.param_mean = np.mean(all_params, axis=0, keepdims=True)
            self.param_std = np.where(np.std(all_params, axis=0, keepdims=True) < 1e-8, 1.0, np.std(all_params, axis=0, keepdims=True))

    def _interpolate_features(self, features: np.ndarray, target_length: int) -> np.ndarray:
        """Interpolate features to target length."""
        source_length, feature_dim = features.shape
        source_indices = np.arange(source_length)
        target_indices = np.linspace(0, source_length - 1, target_length)
        interpolated = np.zeros((target_length, feature_dim))
        for i in range(feature_dim):
            f = interpolate.interp1d(source_indices, features[:, i], kind='linear')
            interpolated[:, i] = f(target_indices)
        return interpolated

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        item = self.data[idx]
        if self.streaming:
            audio_data = np.load(item['audio_file'])
            param_data = np.load(item['param_file'])
            audio_features = audio_data if item['is_npy_audio'] else audio_data['mel_spectrogram']
            parameters = param_data if item['is_npy_param'] else param_data['geometric_features']
            if audio_features.shape[0] != parameters.shape[0]:
                audio_features = self._interpolate_features(audio_features, parameters.shape[0])
            if item['start_idx'] is not None:
                audio_features = audio_features[item['start_idx']:item['end_idx']]
                parameters = parameters[item['start_idx']:item['end_idx']]
        else:
            audio_features = item['audio_features']
            parameters = item['parameters']

        audio_features = torch.FloatTensor(audio_features)
        parameters = torch.FloatTensor(parameters)

        if self.augmentation is not None:
            audio_features = self.augmentation(audio_features)

        if self.normalize_params:
            if self.normalization_type == 'minmax':
                parameters = (parameters - torch.FloatTensor(self.param_min)) / torch.FloatTensor(self.param_range)
            elif self.normalization_type == 'zscore':
                parameters = (parameters - torch.FloatTensor(self.param_mean)) / torch.FloatTensor(self.param_std)

        return audio_features, parameters, item['utterance_name']

    def denormalize_parameters(self, normalized_params: torch.Tensor) -> torch.Tensor:
        if not self.normalize_params:
            return normalized_params
        if self.normalization_type == 'minmax':
            return normalized_params * torch.FloatTensor(self.param_range) + torch.FloatTensor(self.param_min)
        elif self.normalization_type == 'zscore':
            return normalized_params * torch.FloatTensor(self.param_std) + torch.FloatTensor(self.param_mean)
        return normalized_params


def collate_fn(batch):
    """Custom collate function for variable-length sequences."""
    audio_features, parameters, utterance_names = zip(*batch)
    max_len = max(audio.shape[0] for audio in audio_features)
    padded_audio = []
    padded_params = []
    lengths = []
    for audio, params in zip(audio_features, parameters):
        seq_len = audio.shape[0]
        lengths.append(seq_len)
        pad_len = max_len - seq_len
        if pad_len > 0:
            audio = torch.cat([audio, torch.zeros(pad_len, audio.shape[1])], dim=0)
            params = torch.cat([params, torch.zeros(pad_len, params.shape[1])], dim=0)
        padded_audio.append(audio)
        padded_params.append(params)
    return torch.stack(padded_audio), torch.stack(padded_params), torch.LongTensor(lengths), utterance_names


def create_dataloaders(
    splits_dir: Path,
    audio_feature_dir: Path,
    parameter_dir: Path,
    audio_feature_type: str = 'mel',
    parameter_type: str = 'geometric',
    batch_size: int = 16,
    num_workers: int = 4,
    sequence_length: Optional[int] = None,
    normalization_type: str = 'minmax',
    streaming: bool = False,
    zip_file_path: Optional[str] = None,
    train_augmentation: Optional['AudioAugmentation'] = None,
) -> Dict[str, DataLoader]:
    dataloaders = {}
    for split in ['train', 'val', 'test']:
        utterance_list_file = splits_dir / f'{split}.json'
        if not utterance_list_file.exists():
            continue
        with open(utterance_list_file, 'r') as f:
            split_data = json.load(f)
            if isinstance(split_data, list):
                utterance_list = split_data
            elif isinstance(split_data, dict) and 'utterances' in split_data:
                utterance_list = [u['utterance_name'] if isinstance(u, dict) else u for u in split_data['utterances']]
            else:
                continue

        dataset = ArticulatoryDataset(
            utterance_list=utterance_list,
            audio_feature_dir=audio_feature_dir,
            parameter_dir=parameter_dir,
            audio_feature_type=audio_feature_type,
            parameter_type=parameter_type,
            normalize_params=True,
            normalization_type=normalization_type,
            sequence_length=sequence_length,
            streaming=streaming,
            zip_file_path=zip_file_path,
            augmentation=train_augmentation if split == 'train' else None,
        )
        dataloaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=(split == 'train'),
            num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
        )
    return dataloaders