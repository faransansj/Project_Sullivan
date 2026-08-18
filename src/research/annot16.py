"""Minimal adapter for Annot-16 handmade ground-truth contour JSON files."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np

from .contours import ArticulatorContour, ContourSample, PixelSpacing

_FILENAME = re.compile(
    r"^(?P<speaker>sub\d+)_2drt_(?P<task>.+)_track_frame-(?P<frame>\d+)\.json$"
)

HAND_GROUND_TRUTH_ARTICULATORS = frozenset(
    {
        "epiglottis",
        "tongue",
        "lower_lip",
        "chin",
        "arytenoid",
        "pharyngeal_wall",
        "hard_palate",
        "velum",
        "upper_lip",
    }
)


class Annot16GroundTruthAdapter:
    """Convert one official handmade Annot-16 JSON frame to the canonical contract.

    The source frame number is MATLAB/trackdata 1-based. Canonical ``frame_index``
    is zero-based. Timestamp and physical spacing are populated only when their
    externally verified acquisition metadata are supplied.
    """

    def __init__(
        self,
        dataset_root: Path,
        *,
        frame_rate: Optional[float] = None,
        pixel_spacing: Optional[tuple[float, float]] = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.frame_rate = frame_rate
        if frame_rate is not None and frame_rate <= 0:
            raise ValueError("frame_rate must be positive")
        self.pixel_spacing = (
            PixelSpacing(*pixel_spacing) if pixel_spacing is not None else None
        )

    def load(self, annotation_path: Path) -> ContourSample:
        annotation_path = Path(annotation_path)
        match = _FILENAME.match(annotation_path.name)
        if not match:
            raise ValueError(f"Unrecognized Annot-16 ground-truth filename: {annotation_path.name}")

        source_frame_number = int(match.group("frame"))
        frame_index = source_frame_number - 1
        speaker_id = match.group("speaker")
        utterance_id = f"{speaker_id}_2drt_{match.group('task')}"
        image_path = (
            self.dataset_root
            / "hand_ground_truth"
            / "extracted_frames_jpg"
            / f"{utterance_id}_video.mp4_frame-{source_frame_number}.jpg"
        )
        if not image_path.exists():
            raise FileNotFoundError(f"Corresponding Annot-16 MRI frame not found: {image_path}")

        document = json.loads(annotation_path.read_text(encoding="utf-8"))
        unknown = set(document) - HAND_GROUND_TRUTH_ARTICULATORS
        if unknown:
            raise ValueError(f"Unknown Annot-16 articulators: {sorted(unknown)}")
        articulators = {}
        for name, values in document.items():
            coordinates = np.asarray(values, dtype=np.float32)
            if coordinates.ndim != 2 or coordinates.shape[-1] != 2:
                raise ValueError(f"{name} coordinates must have shape [points, 2]")
            articulators[name] = ArticulatorContour(
                coordinates=coordinates,
                valid_mask=np.isfinite(coordinates).all(axis=1),
                is_static=(name == "hard_palate"),
            )

        return ContourSample(
            sample_id=f"annot16:{utterance_id}:{source_frame_number}",
            speaker_id=speaker_id,
            utterance_id=utterance_id,
            frame_index=frame_index,
            timestamp=(frame_index / self.frame_rate if self.frame_rate is not None else None),
            audio_path="",
            mri_path=str(image_path),
            articulators=articulators,
            pixel_spacing=self.pixel_spacing,
            coordinate_convention="image_xy_origin_top_left_x_right_y_down",
            source_provenance={
                "dataset": "75-Speaker Annot-16",
                "zenodo_record": 18931763,
                "annotation_path": str(annotation_path),
                "source_frame_number": source_frame_number,
                "source_frame_numbering": "MATLAB trackdata frameNo, 1-based",
            },
        )
