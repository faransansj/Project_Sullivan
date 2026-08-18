"""Validation-only prediction/target lag sweep diagnostics."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .metrics import dimension_mean_pcc, rmse


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
