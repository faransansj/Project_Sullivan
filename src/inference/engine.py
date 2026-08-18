"""
Inference Engine for Project Sullivan

Provides a unified interface for loading trained models (Transformer or Conformer)
and running end-to-end audio → articulatory parameter prediction.

Phase 4: Supports both Mel-spectrogram and HuBERT features.
"""

import sys
import json
from pathlib import Path
from typing import Optional, Union

import yaml
import torch
import numpy as np
import librosa

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.modeling.transformer import TransformerModel
from src.modeling.conformer_model import ConformerInversionModel


class InferenceEngine:
    """
    Unified inference engine for articulatory inversion models.

    Supports:
    - Transformer (Phase 3 baseline)
    - Conformer (Phase 4 accuracy improvement)
    - Mel-spectrogram and HuBERT feature extraction

    Parameters
    ----------
    model_path : str
        Path to trained model checkpoint (.ckpt)
    config_path : str
        Path to YAML config used during training
    stats_path : str, optional
        Path to normalization statistics (JSON)
    model_type : str
        'transformer' or 'conformer'
    feature_type : str
        'mel' or 'hubert'
    device : str
        'cpu' or 'cuda'
    """

    def __init__(
        self,
        model_path: str,
        config_path: str,
        stats_path: Optional[str] = None,
        model_type: str = "transformer",
        feature_type: str = "mel",
        device: str = "cpu",
    ):
        print("Initializing Inference Engine...")
        if model_type not in {'transformer', 'conformer'}:
            raise ValueError("model_type must be 'transformer' or 'conformer'")
        if feature_type not in {'mel', 'hubert'}:
            raise ValueError("feature_type must be 'mel' or 'hubert'")
        self.device = torch.device(device)
        self.feature_type = feature_type

        # Load config
        print(f"Loading config from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load normalization stats
        self.stats = None
        if stats_path:
            print(f"Loading statistics from {stats_path}")
            with open(stats_path, 'r') as f:
                self.stats = json.load(f)
                normalization_type = self.stats.get('normalization_type', 'minmax')
                required = ('min', 'max') if normalization_type == 'minmax' else ('mean', 'std')
                if normalization_type not in {'minmax', 'zscore'} or any(
                    key not in self.stats for key in required
                ):
                    raise ValueError(f"Invalid normalization statistics: {stats_path}")
                for key in required:
                    self.stats[key] = np.asarray(self.stats[key])

        # Load model
        print(f"Loading {model_type} model from {model_path}")
        ModelClass = {
            'transformer': TransformerModel,
            'conformer': ConformerInversionModel,
        }[model_type]
        self.model = ModelClass.load_from_checkpoint(
            model_path, map_location=self.device
        )
        self.model.to(self.device).eval()
        expected_input_dim = 1024 if feature_type == 'hubert' else self.config.get('model', {}).get('input_dim', 80)
        checkpoint_input_dim = getattr(self.model.hparams, 'input_dim', expected_input_dim)
        if checkpoint_input_dim != expected_input_dim:
            raise ValueError(
                f"{feature_type} features produce {expected_input_dim} dimensions, "
                f"but checkpoint expects {checkpoint_input_dim}"
            )

        # Initialize HuBERT extractor lazily (heavy import)
        self._hubert_extractor = None

        print(f"Inference Engine Ready ({model_type}, {feature_type} features).")

    def _get_hubert_extractor(self):
        """Lazy-load HuBERT extractor."""
        if self._hubert_extractor is None:
            from src.audio_features.hubert_extractor import HuBERTExtractor
            self._hubert_extractor = HuBERTExtractor(device=str(self.device))
        return self._hubert_extractor

    def _extract_mel(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract Mel-spectrogram features."""
        data_conf = self.config.get('data', {})
        model_conf = self.config.get('model', {})
        n_fft = data_conf.get('n_fft', 512)
        hop_length = data_conf.get('hop_length', 160)
        n_mels = model_conf.get('input_dim', 80)

        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db.T  # (Time, Mels)

    def _extract_hubert(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract HuBERT features."""
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        extractor = self._get_hubert_extractor()
        return extractor.extract(
            audio,
            None,
            mri_fps=83.3,
            feature_time_offset_seconds=0.0,
            timeline_policy="truncate_to_hubert_support",
        )

    def _preprocess_audio(self, audio_path: str) -> np.ndarray:
        """Load audio and extract features."""
        data_conf = self.config.get('data', {})
        sr = data_conf.get('sr', 16000)

        try:
            audio, original_sr = librosa.load(audio_path, sr=None)
        except Exception as e:
            raise ValueError(f"Failed to load audio file {audio_path}: {e}")

        if original_sr != sr:
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=sr)

        if self.feature_type == "hubert":
            return self._extract_hubert(audio, sr)
        else:
            return self._extract_mel(audio, sr)

    def _denormalize(self, predictions: np.ndarray) -> np.ndarray:
        """Denormalize predictions using loaded statistics."""
        if self.stats is None:
            return predictions

        normalization_type = self.stats.get('normalization_type', 'minmax')
        reference = self.stats['min'] if normalization_type == 'minmax' else self.stats['mean']
        if len(reference) != predictions.shape[1]:
            raise ValueError(
                f"Stats dim ({len(reference)}) != prediction dim ({predictions.shape[1]})"
            )

        if normalization_type == 'zscore':
            return predictions * self.stats['std'] + self.stats['mean']

        p_range = self.stats['max'] - self.stats['min']
        p_range[p_range == 0] = 1.0
        return predictions * p_range + self.stats['min']

    def predict(self, audio_path: str) -> np.ndarray:
        """
        Run full inference pipeline.

        Parameters
        ----------
        audio_path : str
            Path to audio file (WAV)

        Returns
        -------
        np.ndarray
            Denormalized articulatory parameters, shape (Time, output_dim)
        """
        features = self._preprocess_audio(audio_path)

        audio_tensor = (
            torch.FloatTensor(features).unsqueeze(0).to(self.device)
        )

        with torch.no_grad():
            preds_norm = self.model(audio_tensor)

        preds_np = preds_norm.squeeze(0).cpu().numpy()
        preds_denorm = self._denormalize(preds_np)

        return preds_denorm

    def predict_with_details(self, audio_path: str) -> dict:
        """
        Predict with additional detail (features, raw predictions, denormalized).

        Returns
        -------
        dict
            'features': input features
            'predictions_raw': normalized predictions
            'predictions': denormalized predictions
            'param_names': list of parameter names
        """
        features = self._preprocess_audio(audio_path)
        audio_tensor = (
            torch.FloatTensor(features).unsqueeze(0).to(self.device)
        )

        with torch.no_grad():
            preds_norm = self.model(audio_tensor)

        preds_np = preds_norm.squeeze(0).cpu().numpy()
        preds_denorm = self._denormalize(preds_np)

        # Standard articulatory parameter names
        output_dim = preds_np.shape[1]
        geo_names = [
            'tongue_area', 'tongue_cx', 'tongue_cy', 'tongue_tip',
            'tongue_dorsum', 'tongue_curvature', 'jaw_opening',
            'jaw_position', 'lip_aperture', 'lip_protrusion',
            'constriction_degree', 'constriction_loc',
            'velum_height', 'hyoid_position',
        ]
        pca_names = [f'pca_{i}' for i in range(10)]
        param_names = (geo_names + pca_names)[:output_dim]

        return {
            'features': features,
            'predictions_raw': preds_np,
            'predictions': preds_denorm,
            'param_names': param_names,
        }