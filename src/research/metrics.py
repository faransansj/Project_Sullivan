"""Explicit masked scalar and contour metrics for AAI evaluation."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


def _validated(prediction: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray]):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    if prediction.ndim < 2:
        raise ValueError("inputs must have a final target dimension")
    valid = (
        np.ones(prediction.shape[:-1], dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    )
    if valid.shape != prediction.shape[:-1]:
        raise ValueError("mask must match inputs excluding the target dimension")
    if not valid.any():
        raise ValueError("metric has no valid observations")
    return prediction, target, valid


def rmse(prediction: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    prediction, target, valid = _validated(prediction, target, mask)
    return float(np.sqrt(np.mean((prediction[valid] - target[valid]) ** 2)))


def mae(prediction: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    prediction, target, valid = _validated(prediction, target, mask)
    return float(np.mean(np.abs(prediction[valid] - target[valid])))


def _pcc(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def global_pcc(
    prediction: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None
) -> float:
    """PCC after flattening all valid frames and target dimensions."""
    prediction, target, valid = _validated(prediction, target, mask)
    return _pcc(prediction[valid], target[valid])


def dimension_mean_pcc(
    prediction: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None
) -> float:
    """Mean PCC across target dimensions; undefined dimensions are excluded."""
    prediction, target, valid = _validated(prediction, target, mask)
    values = [
        _pcc(prediction[..., index][valid], target[..., index][valid])
        for index in range(prediction.shape[-1])
    ]
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def utterance_mean_pcc(
    prediction: np.ndarray,
    target: np.ndarray,
    utterance_ids: np.ndarray,
    mask: Optional[np.ndarray] = None,
    speaker_ids: Optional[np.ndarray] = None,
) -> float:
    """Mean of per-utterance PCC values, giving each utterance equal weight."""
    prediction, target, valid = _validated(prediction, target, mask)
    utterance_ids = np.asarray(utterance_ids)
    if utterance_ids.shape != valid.shape:
        raise ValueError("utterance_ids must match inputs excluding the target dimension")
    if speaker_ids is None:
        groups = utterance_ids.astype(str)
    else:
        speaker_ids = np.asarray(speaker_ids)
        if speaker_ids.shape != valid.shape:
            raise ValueError("speaker_ids must match inputs excluding the target dimension")
        groups = np.char.add(
            np.char.add(speaker_ids.astype(str), "::"), utterance_ids.astype(str)
        )
    values = []
    for utterance_id in np.unique(groups[valid]):
        utterance_mask = valid & (groups == utterance_id)
        value = _pcc(prediction[utterance_mask], target[utterance_mask])
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def speaker_mean_pcc(
    prediction: np.ndarray,
    target: np.ndarray,
    speaker_ids: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Mean dimension-wise PCC across speakers, giving each speaker equal weight."""
    prediction, target, valid = _validated(prediction, target, mask)
    speaker_ids = np.asarray(speaker_ids)
    if speaker_ids.shape != valid.shape:
        raise ValueError("speaker_ids must match inputs excluding the target dimension")
    values = []
    for speaker_id in np.unique(speaker_ids[valid]):
        speaker_mask = valid & (speaker_ids == speaker_id)
        value = dimension_mean_pcc(prediction, target, speaker_mask)
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def _scaled_coordinates(coordinates: np.ndarray, pixel_spacing: Optional[Tuple[float, float]]):
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if pixel_spacing is None:
        return coordinates
    x_spacing, y_spacing = pixel_spacing
    if x_spacing <= 0 or y_spacing <= 0:
        raise ValueError("pixel spacing must be positive")
    return coordinates * np.asarray([x_spacing, y_spacing])


def point_euclidean_rmse(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
    pixel_spacing: Optional[Tuple[float, float]] = None,
) -> float:
    """sqrt(mean squared Euclidean point distance), in pixels or mm when spacing is supplied."""
    prediction = _scaled_coordinates(prediction, pixel_spacing)
    target = _scaled_coordinates(target, pixel_spacing)
    if prediction.shape != target.shape or prediction.shape[-1] != 2:
        raise ValueError("contours must have matching [..., points, 2] shapes")
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.shape != prediction.shape[:-1] or not valid.any():
        raise ValueError("valid_mask must match contour points and contain a valid point")
    squared_distance = np.sum((prediction - target) ** 2, axis=-1)
    return float(np.sqrt(np.mean(squared_distance[valid])))


contour_rmse = point_euclidean_rmse


def speaker_macro_contour_rmse(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
    speaker_ids: np.ndarray,
    pixel_spacing: Optional[Tuple[float, float]] = None,
) -> float:
    """Macro-average contour RMSE so each unseen speaker has equal weight."""
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    valid = np.asarray(valid_mask, dtype=bool)
    speaker_ids = np.asarray(speaker_ids)
    if prediction.shape != target.shape or prediction.shape[-1] != 2:
        raise ValueError("contours must have matching [..., 2] shapes")
    if valid.shape != prediction.shape[:-1]:
        raise ValueError("valid_mask must match contour points")
    if speaker_ids.shape != prediction.shape[:2]:
        raise ValueError("speaker_ids must have shape [batch, time]")
    expanded = np.broadcast_to(
        speaker_ids[(...,) + (None,) * (valid.ndim - speaker_ids.ndim)], valid.shape
    )
    values = [
        point_euclidean_rmse(prediction, target, valid & (expanded == speaker), pixel_spacing)
        for speaker in np.unique(speaker_ids)
        if (valid & (expanded == speaker)).any()
    ]
    if not values:
        raise ValueError("no speaker has valid contour points")
    return float(np.mean(values))


def _point_distances(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((first[:, None, :] - second[None, :, :]) ** 2, axis=-1))


def symmetric_chamfer_distance(
    prediction: np.ndarray,
    target: np.ndarray,
    prediction_mask: np.ndarray,
    target_mask: Optional[np.ndarray] = None,
    pixel_spacing: Optional[Tuple[float, float]] = None,
) -> float:
    if np.asarray(prediction).ndim != 2 or np.asarray(target).ndim != 2:
        raise ValueError("Chamfer distance accepts one [points, 2] contour at a time")
    prediction = _scaled_coordinates(prediction, pixel_spacing)[
        np.asarray(prediction_mask, dtype=bool)
    ]
    target_mask = prediction_mask if target_mask is None else target_mask
    target = _scaled_coordinates(target, pixel_spacing)[np.asarray(target_mask, dtype=bool)]
    if not len(prediction) or not len(target):
        raise ValueError("Chamfer distance requires valid points in both contours")
    distances = _point_distances(prediction, target)
    return float((distances.min(axis=1).mean() + distances.min(axis=0).mean()) / 2)


def hausdorff_distance(
    prediction: np.ndarray,
    target: np.ndarray,
    prediction_mask: np.ndarray,
    target_mask: Optional[np.ndarray] = None,
    pixel_spacing: Optional[Tuple[float, float]] = None,
) -> float:
    if np.asarray(prediction).ndim != 2 or np.asarray(target).ndim != 2:
        raise ValueError("Hausdorff distance accepts one [points, 2] contour at a time")
    prediction = _scaled_coordinates(prediction, pixel_spacing)[
        np.asarray(prediction_mask, dtype=bool)
    ]
    target_mask = prediction_mask if target_mask is None else target_mask
    target = _scaled_coordinates(target, pixel_spacing)[np.asarray(target_mask, dtype=bool)]
    if not len(prediction) or not len(target):
        raise ValueError("Hausdorff distance requires valid points in both contours")
    distances = _point_distances(prediction, target)
    return float(max(distances.min(axis=1).max(), distances.min(axis=0).max()))


def articulator_rmse(
    predictions: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    valid_masks: Mapping[str, np.ndarray],
    pixel_spacing: Optional[Tuple[float, float]] = None,
) -> Dict[str, float]:
    if predictions.keys() != targets.keys() or predictions.keys() != valid_masks.keys():
        raise ValueError("prediction, target, and mask articulators must match")
    return {
        name: point_euclidean_rmse(
            predictions[name], targets[name], valid_masks[name], pixel_spacing
        )
        for name in predictions
    }
