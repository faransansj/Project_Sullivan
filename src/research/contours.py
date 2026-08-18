"""Canonical contour contract and synthetic-compatible JSON loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Protocol

import numpy as np


@dataclass(frozen=True)
class PixelSpacing:
    x_mm_per_pixel: float
    y_mm_per_pixel: float

    def __post_init__(self) -> None:
        if self.x_mm_per_pixel <= 0 or self.y_mm_per_pixel <= 0:
            raise ValueError("pixel spacing must be positive")


@dataclass
class ArticulatorContour:
    coordinates: np.ndarray
    valid_mask: np.ndarray
    is_static: bool = False

    def __post_init__(self) -> None:
        self.coordinates = np.asarray(self.coordinates, dtype=np.float32)
        self.valid_mask = np.asarray(self.valid_mask, dtype=bool)
        if self.coordinates.ndim != 2 or self.coordinates.shape[-1] != 2:
            raise ValueError("coordinates must have shape [points, 2]")
        if self.valid_mask.shape != self.coordinates.shape[:-1]:
            raise ValueError("valid_mask must have shape [points]")


@dataclass
class ContourSample:
    sample_id: str
    speaker_id: str
    utterance_id: str
    frame_index: int
    timestamp: float
    audio_path: str
    mri_path: str
    articulators: Dict[str, ArticulatorContour]
    pixel_spacing: Optional[PixelSpacing] = None
    coordinate_convention: str = "image_xy_anterior_unspecified"

    def __post_init__(self) -> None:
        if not self.sample_id or not self.speaker_id or not self.utterance_id:
            raise ValueError("sample, speaker, and utterance IDs are required")
        if self.frame_index < 0 or self.timestamp < 0:
            raise ValueError("frame_index and timestamp must be non-negative")
        if not self.articulators:
            raise ValueError("at least one articulator is required")


class ContourLoader(Protocol):
    def load(self, sample_id: str) -> ContourSample: ...


class JsonContourLoader:
    """Load canonical samples from a JSON document keyed by sample_id."""

    def __init__(self, path: Path):
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        records = document.get("samples", document)
        if not isinstance(records, list):
            raise ValueError("contour JSON must contain a samples list")
        self._records = {record["sample_id"]: record for record in records}
        if len(self._records) != len(records):
            raise ValueError("duplicate contour sample IDs")

    def load(self, sample_id: str) -> ContourSample:
        try:
            record = self._records[sample_id]
        except KeyError as error:
            raise KeyError(f"Unknown contour sample: {sample_id}") from error
        spacing = record.get("pixel_spacing")
        return ContourSample(
            sample_id=record["sample_id"],
            speaker_id=record["speaker_id"],
            utterance_id=record["utterance_id"],
            frame_index=int(record["frame_index"]),
            timestamp=float(record["timestamp"]),
            audio_path=record["audio_path"],
            mri_path=record["mri_path"],
            articulators={
                name: ArticulatorContour(
                    value["coordinates"], value["valid_mask"], value.get("is_static", False)
                )
                for name, value in record["articulators"].items()
            },
            pixel_spacing=PixelSpacing(**spacing) if spacing else None,
            coordinate_convention=record.get(
                "coordinate_convention", "image_xy_anterior_unspecified"
            ),
        )


def masked_point_loss(
    prediction: np.ndarray, target: np.ndarray, valid_mask: np.ndarray
) -> float:
    """Mean Euclidean error over annotated contour points only."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if prediction.shape != target.shape or prediction.shape[-1] != 2:
        raise ValueError("prediction and target must have matching [..., 2] shapes")
    if valid.shape != prediction.shape[:-1] or not valid.any():
        raise ValueError("valid_mask must match points and contain a valid point")
    return float(np.linalg.norm(prediction - target, axis=-1)[valid].mean())


def masked_velocity_loss(
    prediction: np.ndarray, target: np.ndarray, valid_mask: np.ndarray
) -> float:
    """Mean first-order contour velocity error; adjacent frames must both be valid."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if prediction.shape != target.shape or prediction.ndim != 5 or prediction.shape[-1] != 2:
        raise ValueError("prediction and target must have shape [batch, time, articulator, point, 2]")
    if valid.shape != prediction.shape[:-1]:
        raise ValueError("valid_mask must match points")
    pair_valid = valid[:, 1:] & valid[:, :-1]
    if not pair_valid.any():
        return 0.0
    velocity_error = (prediction[:, 1:] - prediction[:, :-1]) - (
        target[:, 1:] - target[:, :-1]
    )
    return float(np.linalg.norm(velocity_error, axis=-1)[pair_valid].mean())


def flatten_contours(coordinates: np.ndarray) -> np.ndarray:
    """Flatten [..., articulator, point, xy] into a model output dimension."""
    coordinates = np.asarray(coordinates)
    if coordinates.ndim < 4 or coordinates.shape[-1] != 2:
        raise ValueError("coordinates must end with [articulator, point, 2]")
    return coordinates.reshape(*coordinates.shape[:-3], -1)


def reshape_contours(values: np.ndarray, articulators: int, points: int) -> np.ndarray:
    values = np.asarray(values)
    expected = articulators * points * 2
    if values.shape[-1] != expected:
        raise ValueError(f"Expected flat dimension {expected}, got {values.shape[-1]}")
    return values.reshape(*values.shape[:-1], articulators, points, 2)


def resample_ordered_contour(contour: ArticulatorContour, point_count: int) -> ArticulatorContour:
    """Linearly resample valid ordered points at uniform arc-length intervals."""
    if point_count < 1:
        raise ValueError("point_count must be positive")
    fully_annotated = bool(contour.valid_mask.all())
    points = contour.coordinates[contour.valid_mask]
    if not len(points):
        return ArticulatorContour(
            np.zeros((point_count, 2), dtype=np.float32),
            np.zeros(point_count, dtype=bool),
            contour.is_static,
        )
    if len(points) == 1:
        return ArticulatorContour(
            np.repeat(points, point_count, axis=0),
            np.full(point_count, fully_annotated, dtype=bool),
            contour.is_static,
        )
    segment_lengths = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    distance = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if distance[-1] == 0:
        sampled = np.repeat(points[:1], point_count, axis=0)
    else:
        targets = np.linspace(0.0, distance[-1], point_count)
        sampled = np.column_stack(
            [np.interp(targets, distance, points[:, dimension]) for dimension in range(2)]
        )
    # Partial contours have unknown correspondence across gaps. Coordinates are
    # available for inspection, but remain excluded from supervision.
    return ArticulatorContour(
        sampled,
        np.full(point_count, fully_annotated, dtype=bool),
        contour.is_static,
    )
