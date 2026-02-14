"""
HuBERT Feature Extraction for High-Performance Articulatory Inversion

This module uses pre-trained HuBERT models to extract deep acoustic features,
which significantly outperform traditional Mel-spectrograms for inversion tasks.
"""

import torch
import torchaudio
import numpy as np
from typing import Optional


class HuBERTExtractor:
    """
    Extract features from pre-trained HuBERT-Large model.
    HuBERT features are synchronized with MRI frames via interpolation.

    Parameters
    ----------
    model_name : str, default='hubert_large_ll60k'
        Name of the HuBERT model to use.
    layer_index : int, default=12
        Which layer's output to use (12 is often optimal for articulatory tasks).
    device : str, default='cpu'
        Device to run the model on ('cpu' or 'cuda').
    """

    def __init__(
        self,
        model_name: str = "hubert_large_ll60k",
        layer_index: int = 12,
        device: str = "cpu"
    ):
        self.device = device
        self.layer_index = layer_index
        
        # Load pre-trained HuBERT model via torchaudio
        print(f"Loading HuBERT model: {model_name} on {device}...")
        bundle = getattr(torchaudio.pipelines, model_name.upper())
        self.model = bundle.get_model().to(device)
        self.model.eval()
        
        # Audio must be 16kHz for HuBERT
        self.sample_rate = 16000
        
    def extract(
        self,
        audio: np.ndarray,
        num_mri_frames: int,
        mri_fps: float
    ) -> np.ndarray:
        """
        Extract HuBERT features synchronized with MRI frames.

        Parameters
        ----------
        audio : np.ndarray
            Audio waveform (must be 16kHz), shape (num_samples,)
        num_mri_frames : int
            Number of MRI frames to synchronize with
        mri_fps : float
            MRI frame rate

        Returns
        -------
        features : np.ndarray
            HuBERT features, shape (num_mri_frames, 1024)
        """
        waveform = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Extract all layer features
            features_list, _ = self.model.extract_features(waveform)
            # Select target layer (Large has 24 layers)
            layer_features = features_list[self.layer_index].squeeze(0).cpu().numpy()
            
        # HuBERT typically outputs at 50Hz (20ms hop)
        hubert_fps = 50.0
        
        # Synchronize with MRI frames
        synced_features = self._sync_to_mri_frames(
            layer_features,
            hubert_fps,
            num_mri_frames,
            mri_fps
        )
        
        return synced_features

    def _sync_to_mri_frames(
        self,
        features: np.ndarray,
        source_fps: float,
        num_mri_frames: int,
        mri_fps: float
    ) -> np.ndarray:
        """Linear interpolation to align HuBERT frames with MRI frames."""
        num_source_frames, n_features = features.shape
        source_times = np.arange(num_source_frames) / source_fps
        mri_times = np.arange(num_mri_frames) / mri_fps
        
        synced = np.zeros((num_mri_frames, n_features), dtype=np.float32)
        for i in range(n_features):
            synced[:, i] = np.interp(mri_times, source_times, features[:, i])
            
        return synced
