"""
PyTorch Dataset for Audio-to-Articulatory Parameter Pairs

This module provides dataset classes and utilities for loading synchronized
audio features and articulatory parameters.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import zipfile
import io


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
        sequence_length: Optional[int] = None,
        streaming: bool = False,
        zip_file_path: Optional[Path] = None
    ):
        self.utterance_list = utterance_list
        self.audio_feature_dir = Path(audio_feature_dir)
        self.parameter_dir = Path(parameter_dir)
        self.audio_feature_type = audio_feature_type
        self.parameter_type = parameter_type
        self.normalize_params = normalize_params
        self.parameter_type = parameter_type
        self.normalize_params = normalize_params
        self.sequence_length = sequence_length
        self.streaming = streaming
        self.zip_file_path = Path(zip_file_path) if zip_file_path else None
        
        # If using zip, we open it once per worker or on demand
        # For simplicity in 'streaming', we open on demand or keep open if possible (but pickling issues)
        # We'll open on demand in getitem for safety with multi-processing workers

        # Load data (metadata only if streaming)
        self.data = self._load_data()

        # Compute or set normalization statistics if needed
        if self.normalize_params:
            self._compute_normalization_stats()

    def _load_data(self) -> List[Dict]:
        """Load utterances (metadata or full data)."""
        data = []

        for utterance_name in self.utterance_list:
            # Paths
            if self.audio_feature_type == 'mel':
                audio_file = self.audio_feature_dir / 'mel_spectrogram' / f'{utterance_name}_mel.npy'
            else:  # mfcc
                audio_file = self.audio_feature_dir / 'mfcc' / f'{utterance_name}_mfcc.npy'

            param_file = self.parameter_dir / self.parameter_type / f'{utterance_name}_params.npy'

            # Check files exist (skip check if zip)
            if self.zip_file_path:
                # In zip mode, we construct internal paths
                # Assuming structure: Dataset/audio_features/... if referencing root of zip?
                # Or just basic relative paths.
                # Let's assume audio_feature_dir relative to zip root if zip is used.
                # Actually, if zip is used, audio_feature_dir should be the internal path prefix.
                pass
            elif not audio_file.exists():
                print(f"Warning: Audio file not found: {audio_file}")
                continue
            if not param_file.exists():
                print(f"Warning: Parameter file not found: {param_file}")
                continue

            if self.streaming:
                # Get shape for sequence splitting without loading full data if possible
                if self.zip_file_path:
                     # For zip, we might need to load one to get shape, or rely on metadata if available
                     # To avoid opening zip 1000s times here, we could optimize, but for now:
                     # Just add metadata entry.
                     # We assume full file retrieval for shape isn't too expensive once, or we just store metadata.
                     # WARNING: Opening zip for every item in init is slow. 
                     # Better to have a metadata file.
                     # For now, let's assume one-to-one mapping and NO sequence splitting for Zip Streaming 
                     # UNLESS we read one.
                     
                     data.append({
                        'utterance_name': utterance_name,
                        'audio_path': f"audio_features/{'mel_spectrogram' if self.audio_feature_type == 'mel' else 'mfcc'}/{utterance_name}_{'mel' if self.audio_feature_type == 'mel' else 'mfcc'}.npy",
                        'param_path': f"parameters/{self.parameter_type}/{utterance_name}_params.npy",
                        'start_idx': None,
                        'end_idx': None
                     })
                     continue

                if self.sequence_length is not None:
                    # In streaming mode with sequence_length, we still need the length
                    # We can load just to get shape
                    audio_features = np.load(audio_file, mmap_mode='r')
                    num_frames = audio_features.shape[0]
                    num_sequences = num_frames // self.sequence_length
                    for i in range(num_sequences):
                        data.append({
                            'utterance_name': f"{utterance_name}_seq{i}",
                            'audio_file': audio_file,
                            'param_file': param_file,
                            'start_idx': i * self.sequence_length,
                            'end_idx': (i + 1) * self.sequence_length
                        })
                else:
                    data.append({
                        'utterance_name': utterance_name,
                        'audio_file': audio_file,
                        'param_file': param_file,
                        'start_idx': None,
                        'end_idx': None
                    })
            else:
                # Non-streaming: load full data
                audio_features = np.load(audio_file)
                parameters = np.load(param_file)

                assert audio_features.shape[0] == parameters.shape[0]

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
        """Compute mean and std for parameter normalization."""
        if self.streaming:
            # For streaming, we take a subset to estimate stats or load if pre-calculated
            # Here we take up to 100 samples to estimate
            num_samples = min(len(self.data), 100)
            sampled_params = []
            
            if self.zip_file_path:
                 with zipfile.ZipFile(self.zip_file_path, 'r') as z:
                    for i in range(num_samples):
                        item = self.data[i]
                        with z.open(item['param_path']) as f:
                            params = np.load(io.BytesIO(f.read()))
                        sampled_params.append(params)
            else:
                for i in range(num_samples):
                    item = self.data[i]
                    params = np.load(item['param_file'])
                    if item['start_idx'] is not None:
                        params = params[item['start_idx']:item['end_idx']]
                    sampled_params.append(params)
            
            if len(sampled_params) > 0:
                all_params = np.concatenate(sampled_params, axis=0)
            else:
                 # Fallback if empty
                all_params = np.zeros((1, 14)) # Dummy shape
        else:
            all_params = np.concatenate([item['parameters'] for item in self.data], axis=0)

        self.param_mean = np.mean(all_params, axis=0, keepdims=True)
        self.param_std = np.std(all_params, axis=0, keepdims=True)

        # Avoid division by zero
        self.param_std = np.where(self.param_std < 1e-8, 1.0, self.param_std)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Get a single data sample.
        """
        item = self.data[idx]

        if self.streaming:
            if self.zip_file_path:
                # Read from zip
                with zipfile.ZipFile(self.zip_file_path, 'r') as z:
                    with z.open(item['audio_path']) as f:
                        audio_features = np.load(io.BytesIO(f.read()))
                    with z.open(item['param_path']) as f:
                        parameters = np.load(io.BytesIO(f.read()))
            else:
                audio_features = np.load(item['audio_file'])
                parameters = np.load(item['param_file'])

            if item['start_idx'] is not None:
                audio_features = audio_features[item['start_idx']:item['end_idx']]
                parameters = parameters[item['start_idx']:item['end_idx']]
        else:
            audio_features = item['audio_features']
            parameters = item['parameters']

        audio_features = torch.FloatTensor(audio_features)
        parameters = torch.FloatTensor(parameters)

        # Normalize parameters
        if self.normalize_params:
            parameters = (parameters - torch.FloatTensor(self.param_mean)) / torch.FloatTensor(self.param_std)

        return audio_features, parameters, item['utterance_name']

    def denormalize_parameters(self, normalized_params: torch.Tensor) -> torch.Tensor:
        """
        Denormalize parameters back to original scale.

        Parameters
        ----------
        normalized_params : torch.Tensor
            Normalized parameters

        Returns
        -------
        params : torch.Tensor
            Parameters in original scale
        """
        if not self.normalize_params:
            return normalized_params

        return normalized_params * torch.FloatTensor(self.param_std) + torch.FloatTensor(self.param_mean)


def collate_fn(batch):
    """
    Custom collate function for variable-length sequences.

    Pads sequences to the same length within a batch.
    """
    audio_features, parameters, utterance_names = zip(*batch)

    # Get max length in batch
    max_len = max(audio.shape[0] for audio in audio_features)

    # Pad sequences
    padded_audio = []
    padded_params = []
    lengths = []

    for audio, params in zip(audio_features, parameters):
        seq_len = audio.shape[0]
        lengths.append(seq_len)

        # Pad audio
        audio_dim = audio.shape[1]
        pad_len = max_len - seq_len
        if pad_len > 0:
            padding = torch.zeros(pad_len, audio_dim)
            audio = torch.cat([audio, padding], dim=0)
        padded_audio.append(audio)

        # Pad parameters
        param_dim = params.shape[1]
        if pad_len > 0:
            padding = torch.zeros(pad_len, param_dim)
            params = torch.cat([params, padding], dim=0)
        padded_params.append(params)

    # Stack
    audio_batch = torch.stack(padded_audio, dim=0)  # (batch, max_len, audio_dim)
    param_batch = torch.stack(padded_params, dim=0)  # (batch, max_len, param_dim)
    lengths = torch.LongTensor(lengths)

    return audio_batch, param_batch, lengths, utterance_names


def create_dataloaders(
    splits_dir: Path,
    audio_feature_dir: Path,
    parameter_dir: Path,
    audio_feature_type: str = 'mel',
    parameter_type: str = 'geometric',
    batch_size: int = 16,
    num_workers: int = 4,
    sequence_length: Optional[int] = None,
    streaming: bool = False,
    zip_file_path: Optional[str] = None
) -> Dict[str, DataLoader]:
    """
    Create train, val, and test dataloaders.

    Parameters
    ----------
    splits_dir : Path
        Directory containing split files
    audio_feature_dir : Path
        Directory containing audio features
    parameter_dir : Path
        Directory containing parameters
    audio_feature_type : str
        Type of audio features ('mel' or 'mfcc')
    parameter_type : str
        Type of parameters ('geometric' or 'pca')
    batch_size : int
        Batch size
    num_workers : int
        Number of data loading workers
    sequence_length : int, optional
        Fixed sequence length (None = use full utterances)

    Returns
    -------
    dataloaders : dict
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        # Load utterance list
        utterance_list_file = splits_dir / split / 'utterance_list.txt'
        with open(utterance_list_file, 'r') as f:
            utterance_list = [line.strip() for line in f]

        # Create dataset
        dataset = ArticulatoryDataset(
            utterance_list=utterance_list,
            audio_feature_dir=audio_feature_dir,
            parameter_dir=parameter_dir,
            audio_feature_type=audio_feature_type,
            parameter_type=parameter_type,
            normalize_params=True,
            sequence_length=sequence_length,
            streaming=streaming,
            zip_file_path=zip_file_path
        )

        # Create dataloader
        shuffle = (split == 'train')
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

        dataloaders[split] = dataloader

        print(f"{split.capitalize()} dataset: {len(dataset)} samples")

    return dataloaders
