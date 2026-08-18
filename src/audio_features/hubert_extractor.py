"""HuBERT extraction with explicit audio/MRI feature-center timing."""

from typing import Optional

import numpy as np
import torch
import torchaudio

if __package__ == "src.audio_features":
    from ..research.alignment_diagnostic import supported_features_for_timestamps
else:
    from research.alignment_diagnostic import supported_features_for_timestamps

HUBERT_SAMPLE_RATE = 16000
HUBERT_STRIDE_SAMPLES = 320
HUBERT_RECEPTIVE_FIELD_SAMPLES = 400
HUBERT_FEATURE_STRIDE_SECONDS = HUBERT_STRIDE_SAMPLES / HUBERT_SAMPLE_RATE
HUBERT_FEATURE_ORIGIN_SECONDS = (HUBERT_RECEPTIVE_FIELD_SAMPLES - 1) / (2 * HUBERT_SAMPLE_RATE)


class HuBERTExtractor:
    """Extract HuBERT features and map only MRI frames with centered support."""

    def __init__(
        self,
        model_name: str = "hubert_large",
        layer_index: int = 12,
        device: str = "cpu",
    ):
        self.device = device
        self.layer_index = layer_index
        print(f"Loading HuBERT model: {model_name} on {device}...")
        bundle = getattr(torchaudio.pipelines, model_name.upper())
        self.model = bundle.get_model().to(device)
        self.model.eval()
        self.sample_rate = HUBERT_SAMPLE_RATE

    def extract(
        self,
        audio: np.ndarray,
        num_mri_frames: Optional[int],
        mri_fps: float,
        *,
        mri_timestamps: Optional[np.ndarray] = None,
        feature_time_offset_seconds: float = 0.0,
        timeline_policy: str = "strict",
        return_frame_indices: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Extract features, excluding MRI timestamps outside receptive-field centers.

        ``return_frame_indices=True`` is required by supervised callers so the
        target sequence can be sliced with exactly the returned source indices.
        Audio-only inference uses the same deterministic support calculation but
        does not need the absolute frame indices.
        """
        waveform = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            features_list, _ = self.model.extract_features(waveform)
            layer_features = features_list[self.layer_index].squeeze(0).cpu().numpy()

        synced, frame_indices = self._sync_to_mri_frames(
            layer_features,
            num_mri_frames,
            mri_fps,
            mri_timestamps=mri_timestamps,
            feature_time_offset_seconds=feature_time_offset_seconds,
            timeline_policy=timeline_policy,
        )
        self.last_supported_mri_frame_indices = frame_indices
        self.last_alignment_provenance = {
            "feature_time_offset_seconds": feature_time_offset_seconds,
            "feature_time_sign_convention": "feature_time = MRI_time + offset",
            "feature_origin_seconds": HUBERT_FEATURE_ORIGIN_SECONDS,
            "feature_stride_seconds": HUBERT_FEATURE_STRIDE_SECONDS,
            "receptive_field_samples": HUBERT_RECEPTIVE_FIELD_SAMPLES,
            "supported_mri_frame_indices": frame_indices.tolist(),
            "timeline_policy": timeline_policy,
        }
        if timeline_policy == "strict" and not return_frame_indices:
            raise ValueError(
                "supervised strict extraction requires return_frame_indices=True for target slicing"
            )
        return (synced, frame_indices) if return_frame_indices else synced

    def _sync_to_mri_frames(
        self,
        features: np.ndarray,
        num_mri_frames: Optional[int],
        mri_fps: float,
        *,
        mri_timestamps: Optional[np.ndarray] = None,
        feature_time_offset_seconds: float = 0.0,
        timeline_policy: str = "strict",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return supported features and indices into the requested MRI timeline."""
        if mri_fps <= 0 or not np.isfinite([mri_fps, feature_time_offset_seconds]).all():
            raise ValueError("mri_fps and feature_time_offset_seconds must be finite and valid")
        last_center = HUBERT_FEATURE_ORIGIN_SECONDS + (len(features) - 1) * (
            HUBERT_FEATURE_STRIDE_SECONDS
        )
        if timeline_policy == "strict":
            if num_mri_frames is None:
                raise ValueError("strict timeline policy requires num_mri_frames")
            timestamps = (
                np.arange(num_mri_frames, dtype=np.float64) / mri_fps
                if mri_timestamps is None
                else np.asarray(mri_timestamps, dtype=np.float64)
            )
            if timestamps.shape != (num_mri_frames,):
                raise ValueError("mri_timestamps must match num_mri_frames")
        elif timeline_policy == "truncate_to_hubert_support":
            if num_mri_frames is not None or mri_timestamps is not None:
                raise ValueError("audio-only truncate policy owns its output timeline")
            count = int(np.floor((last_center - feature_time_offset_seconds) * mri_fps)) + 1
            timestamps = np.arange(max(0, count), dtype=np.float64) / mri_fps
        else:
            raise ValueError(f"Unknown timeline_policy: {timeline_policy}")

        return supported_features_for_timestamps(
            features,
            timestamps,
            feature_stride_seconds=HUBERT_FEATURE_STRIDE_SECONDS,
            feature_time_offset_seconds=feature_time_offset_seconds,
            feature_origin_seconds=HUBERT_FEATURE_ORIGIN_SECONDS,
        )
