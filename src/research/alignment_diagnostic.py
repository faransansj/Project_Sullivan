"""Validation-only prediction/target lag sweep diagnostics."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .metrics import dimension_mean_pcc, rmse


def frame_timestamp(frame_index: int, frame_rate: float) -> float:
    """Return the zero-based MRI frame timestamp in seconds."""
    if (
        not isinstance(frame_index, (int, np.integer))
        or isinstance(frame_index, (bool, np.bool_))
        or frame_index < 0
        or frame_rate <= 0
        or not np.isfinite(frame_rate)
    ):
        raise ValueError(
            "frame_index must be non-negative and frame_rate must be finite and positive"
        )
    return frame_index / frame_rate


def audio_sample_indices(timestamps: np.ndarray, sample_rate: int) -> np.ndarray:
    """Map non-negative stream timestamps to nearest decoded sample indices."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if (
        timestamps.ndim != 1
        or not np.isfinite(timestamps).all()
        or (timestamps < 0).any()
        or not isinstance(sample_rate, (int, np.integer))
        or isinstance(sample_rate, (bool, np.bool_))
        or sample_rate <= 0
    ):
        raise ValueError("timestamps and integer sample_rate must be finite and non-negative")
    return np.floor(timestamps * sample_rate + 0.5).astype(np.int64)


def hubert_feature_center_timestamps(
    feature_count: int,
    *,
    sample_rate: int = 16000,
    stride_samples: int = 320,
    receptive_field_samples: int = 400,
) -> np.ndarray:
    """Return HuBERT convolution-output receptive-field center timestamps."""
    if (
        not isinstance(feature_count, (int, np.integer))
        or isinstance(feature_count, (bool, np.bool_))
        or feature_count < 0
        or min(sample_rate, stride_samples, receptive_field_samples) <= 0
    ):
        raise ValueError("feature count and HuBERT timing parameters must be valid integers")
    centers = np.arange(feature_count, dtype=np.float64) * stride_samples
    return (centers + (receptive_field_samples - 1) / 2) / sample_rate


def interpolate_feature_timestamps(
    features: np.ndarray,
    timestamps: np.ndarray,
    *,
    feature_stride_seconds: float,
    feature_time_offset_seconds: float,
    feature_origin_seconds: float = 0.0,
) -> np.ndarray:
    """Linearly sample regularly spaced features without endpoint clamping.

    Sign convention: ``feature_time = MRI_time + feature_time_offset_seconds``.
    The named offset is mandatory so alignment assumptions cannot be hidden in
    a default. Legacy motion-correlation offsets use a different, unverified
    convention and must not be passed here without an independent conversion.
    """
    features = np.asarray(features)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if features.ndim != 2 or not len(features) or not np.issubdtype(features.dtype, np.number):
        raise ValueError("features must have numeric shape [time, dimension] and be non-empty")
    if not np.isfinite(features).all():
        raise ValueError("features must be finite")
    if timestamps.ndim != 1 or not np.isfinite(timestamps).all():
        raise ValueError("timestamps must be a finite one-dimensional array")
    if (
        feature_stride_seconds <= 0
        or not np.isfinite(feature_stride_seconds)
        or not np.isfinite([feature_time_offset_seconds, feature_origin_seconds]).all()
    ):
        raise ValueError("feature timing values must be finite and stride must be positive")

    query = timestamps + feature_time_offset_seconds
    last_timestamp = feature_origin_seconds + (len(features) - 1) * feature_stride_seconds
    if len(query) and (query.min() < feature_origin_seconds or query.max() > last_timestamp):
        raise ValueError(
            f"requested feature timestamp outside [{feature_origin_seconds}, {last_timestamp}] seconds"
        )
    source = (
        feature_origin_seconds + np.arange(len(features), dtype=np.float64) * feature_stride_seconds
    )
    sampled = np.column_stack(
        [np.interp(query, source, features[:, dimension]) for dimension in range(features.shape[1])]
    )
    return (
        sampled.astype(features.dtype, copy=False)
        if np.issubdtype(features.dtype, np.floating)
        else sampled
    )


def supported_features_for_timestamps(
    features: np.ndarray,
    timestamps: np.ndarray,
    *,
    feature_stride_seconds: float,
    feature_time_offset_seconds: float,
    feature_origin_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate only timestamps inside feature-center support.

    Returns the sampled features and indices into the original timestamp/target
    sequence so supervision can be sliced by exactly the same indices.
    """
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.ndim != 1 or not np.isfinite(timestamps).all():
        raise ValueError("timestamps must be a finite one-dimensional array")
    last = feature_origin_seconds + (len(features) - 1) * feature_stride_seconds
    query = timestamps + feature_time_offset_seconds
    indices = np.flatnonzero((query >= feature_origin_seconds) & (query <= last))
    if not len(indices):
        raise ValueError("no MRI timestamps overlap feature-center support")
    return (
        interpolate_feature_timestamps(
            features,
            timestamps[indices],
            feature_stride_seconds=feature_stride_seconds,
            feature_time_offset_seconds=feature_time_offset_seconds,
            feature_origin_seconds=feature_origin_seconds,
        ),
        indices,
    )


def features_for_mri_frames(
    features: np.ndarray,
    frame_indices: np.ndarray,
    *,
    mri_frame_rate: float,
    feature_stride_seconds: float,
    feature_time_offset_seconds: float,
    feature_origin_seconds: float = 0.0,
) -> np.ndarray:
    """Map frames using ``feature_time = MRI_time + feature_time_offset_seconds``."""
    frame_indices = np.asarray(frame_indices)
    if frame_indices.ndim != 1 or not np.issubdtype(frame_indices.dtype, np.integer):
        raise ValueError("frame_indices must be a one-dimensional integer array")
    timestamps = np.asarray(
        [frame_timestamp(int(index), mri_frame_rate) for index in frame_indices],
        dtype=np.float64,
    )
    return interpolate_feature_timestamps(
        features,
        timestamps,
        feature_stride_seconds=feature_stride_seconds,
        feature_time_offset_seconds=feature_time_offset_seconds,
        feature_origin_seconds=feature_origin_seconds,
    )


def lag_sweep(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    frame_rate: float,
    split: str,
    mask: Optional[np.ndarray] = None,
    min_lag_ms: int = -300,
    max_lag_ms: int = 300,
    step_ms: int = 20,
    metric: str = "rmse",
) -> List[dict]:
    """Compare target[t] with prediction[t + lag]; positive lag delays predictions."""
    if split not in {"validation", "val"}:
        raise ValueError("Lag selection is validation-only; test and train splits are rejected")
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    if predictions.shape != targets.shape or predictions.ndim != 3:
        raise ValueError("predictions and targets must have shape [utterance, time, dimension]")
    if frame_rate <= 0 or step_ms <= 0 or min_lag_ms > max_lag_ms:
        raise ValueError("invalid frame rate or lag range")
    valid = (
        np.ones(predictions.shape[:2], dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    )
    if valid.shape != predictions.shape[:2]:
        raise ValueError("mask must have shape [utterance, time]")

    scorer = (
        rmse if metric == "rmse" else dimension_mean_pcc if metric == "dimension_mean_pcc" else None
    )
    if scorer is None:
        raise ValueError("metric must be rmse or dimension_mean_pcc")

    results = []
    time = predictions.shape[1]
    for lag_ms in range(min_lag_ms, max_lag_ms + 1, step_ms):
        lag_frames = int(round(lag_ms * frame_rate / 1000.0))
        if abs(lag_frames) >= time:
            continue
        if lag_frames >= 0:
            pred_slice = predictions[:, lag_frames:]
            target_slice = targets[:, : time - lag_frames or None]
            pair_mask = valid[:, lag_frames:] & valid[:, : time - lag_frames or None]
        else:
            offset = -lag_frames
            pred_slice = predictions[:, : time - offset]
            target_slice = targets[:, offset:]
            pair_mask = valid[:, : time - offset] & valid[:, offset:]
        if not pair_mask.any():
            continue
        results.append(
            {
                "lag_ms": lag_ms,
                "lag_frames": lag_frames,
                "metric": metric,
                "value": scorer(pred_slice, target_slice, pair_mask),
                "valid_frames": int(pair_mask.sum()),
            }
        )
    if not results:
        raise ValueError("lag sweep produced no valid comparisons")
    return results


def best_lag(results: List[dict]) -> dict:
    if not results:
        raise ValueError("results cannot be empty")
    metric = results[0]["metric"]
    finite = [result for result in results if np.isfinite(result["value"])]
    if not finite:
        raise ValueError("no lag has a finite metric value")
    selector = min if metric == "rmse" else max
    return selector(finite, key=lambda result: result["value"])
