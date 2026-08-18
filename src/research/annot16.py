"""Minimal adapter for Annot-16 handmade ground-truth contour JSON files."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.io import loadmat

from .contours import ArticulatorContour, ContourSample, PixelSpacing

_FILENAME = re.compile(r"^(?P<speaker>sub\d+)_2drt_(?P<task>.+)_track_frame-(?P<frame>\d+)\.json$")

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

# PARTIAL: only identities supported by same-frame dense/handmade/MRI overlay evidence.
DENSE_ARTICULATOR_MAPPING = (
    (0, 1, "epiglottis"),
    (0, 2, "tongue"),
    (0, 4, "lower_lip"),
    (0, 5, "chin"),
    (1, 1, "arytenoid"),
    (1, 2, "pharyngeal_wall"),
    (2, 1, "hard_palate"),
    (2, 2, "velum"),
    (2, 5, "upper_lip"),
)
_DENSE_FILENAME = re.compile(r"^(?P<speaker>sub\d+)_(?P<task>.+)_track\.mat$")


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
        if frame_rate is not None and (frame_rate <= 0 or not np.isfinite(frame_rate)):
            raise ValueError("frame_rate must be finite and positive")
        self.pixel_spacing = PixelSpacing(*pixel_spacing) if pixel_spacing is not None else None

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


class Annot16DenseAdapter:
    """Load one dense MAT frame using a PARTIAL, evidence-bounded mapping.

    This contract is not corpus validation: identities outside
    ``DENSE_ARTICULATOR_MAPPING`` remain unresolved and are not exposed.
    """

    def __init__(
        self,
        dataset_root: Path,
        *,
        image_size: tuple[int, int] = (84, 84),
        frame_rate: Optional[float] = None,
        pixel_spacing: Optional[tuple[float, float]] = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.width, self.height = image_size
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")
        if frame_rate is not None and (frame_rate <= 0 or not np.isfinite(frame_rate)):
            raise ValueError("frame_rate must be finite and positive")
        self.frame_rate = frame_rate
        self.pixel_spacing = PixelSpacing(*pixel_spacing) if pixel_spacing is not None else None

    def load(self, track_path: Path, source_frame_number: int) -> ContourSample:
        track_path = Path(track_path)
        match = _DENSE_FILENAME.match(track_path.name)
        if not match:
            raise ValueError(f"Unrecognized Annot-16 dense filename: {track_path.name}")
        if (
            not isinstance(source_frame_number, (int, np.integer))
            or isinstance(source_frame_number, (bool, np.bool_))
            or source_frame_number < 1
        ):
            raise ValueError("source_frame_number must be a one-based positive integer")
        try:
            records = loadmat(track_path, simplify_cells=True)["trackdata"]
        except (KeyError, OSError, ValueError, TypeError) as error:
            raise ValueError(f"Malformed Annot-16 MAT file: {track_path}") from error
        if isinstance(records, dict):
            records = [records]
        if not isinstance(records, (list, np.ndarray)):
            raise ValueError("trackdata must be a sequence of frame records")

        by_frame = {}
        for record in records:
            if not isinstance(record, dict) or "frameNo" not in record:
                raise ValueError("each trackdata record must contain frameNo")
            frame_value = np.asarray(record["frameNo"]).squeeze()
            try:
                finite_frame = bool(np.isfinite(frame_value))
                frame_number = int(frame_value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("frameNo must be a finite scalar integer") from error
            if frame_value.ndim or not finite_frame:
                raise ValueError("frameNo must be a finite scalar integer")
            if frame_number != frame_value or frame_number < 1:
                raise ValueError("frameNo must be a positive integer")
            if frame_number in by_frame:
                raise ValueError(f"duplicate frameNo: {frame_number}")
            by_frame[frame_number] = record
        if source_frame_number not in by_frame:
            raise KeyError(f"Dense frameNo not found: {source_frame_number}")

        record = by_frame[source_frame_number]
        try:
            segments = record["contours"]["segment"]
            template_value = np.asarray(record["template"]).squeeze()
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError("dense record requires contours.segment and template") from error
        if isinstance(segments, dict):
            segments = [segments]
        if not isinstance(segments, (list, np.ndarray)) or len(segments) < 3:
            raise ValueError("dense record must contain at least three segments")
        try:
            finite_template = bool(np.isfinite(template_value))
            template_number = int(template_value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("template must be a finite scalar integer") from error
        if template_value.ndim or not finite_template or template_number != template_value:
            raise ValueError("template must be a finite scalar integer")

        parsed_segments = []
        for segment in segments[:3]:
            if not isinstance(segment, dict) or "v" not in segment or "i" not in segment:
                raise ValueError("each segment requires v coordinates and i identities")
            try:
                coordinates = np.asarray(segment["v"], dtype=np.float32)
                identities = np.asarray(segment["i"], dtype=np.float64).reshape(-1)
            except (TypeError, ValueError) as error:
                raise ValueError("segment v and i must be numeric arrays") from error
            if coordinates.ndim != 2 or coordinates.shape[1] != 2:
                raise ValueError("segment v must have shape [points, 2]")
            if len(identities) != len(coordinates) or not np.isfinite(identities).all():
                raise ValueError("segment i must contain one finite identity per point")
            integer_identities = identities.astype(np.int64)
            if not np.array_equal(integer_identities, identities):
                raise ValueError("segment identities must be integers")
            image_coordinates = coordinates.copy()
            image_coordinates[:, 0] += self.width / 2
            image_coordinates[:, 1] = -image_coordinates[:, 1] + self.height / 2
            parsed_segments.append((image_coordinates, integer_identities))

        articulators = {}
        for segment_index, identity, name in DENSE_ARTICULATOR_MAPPING:
            coordinates, identities = parsed_segments[segment_index]
            selected = coordinates[identities == identity]
            articulators[name] = ArticulatorContour(
                coordinates=selected,
                valid_mask=np.isfinite(selected).all(axis=1),
                is_static=(name == "hard_palate"),
            )

        speaker_id = match.group("speaker")
        utterance_id = f"{speaker_id}_{match.group('task')}"
        image_path = (
            self.dataset_root
            / "hand_ground_truth"
            / "extracted_frames_jpg"
            / f"{utterance_id}_video.mp4_frame-{source_frame_number}.jpg"
        )
        frame_index = source_frame_number - 1
        mapping = [
            {"segment_index": segment, "identity": identity, "articulator": name}
            for segment, identity, name in DENSE_ARTICULATOR_MAPPING
        ]
        return ContourSample(
            sample_id=f"annot16-dense:{utterance_id}:{source_frame_number}",
            speaker_id=speaker_id,
            utterance_id=utterance_id,
            frame_index=frame_index,
            timestamp=(frame_index / self.frame_rate if self.frame_rate is not None else None),
            audio_path="",
            mri_path=str(image_path) if image_path.exists() else "",
            articulators=articulators,
            pixel_spacing=self.pixel_spacing,
            coordinate_convention="image_xy_origin_top_left_x_right_y_down",
            source_provenance={
                "dataset": "75-Speaker Annot-16",
                "zenodo_record": 18931763,
                "track_path": str(track_path),
                "source_frame_number": source_frame_number,
                "source_frame_numbering": "MATLAB trackdata frameNo, 1-based",
                "template": template_number,
                "dense_mapping": mapping,
                "point_order": "source segment v row order filtered by identity i",
                "coordinate_transform": "x=v[:,0]+width/2; y=-v[:,1]+height/2",
                "semantic_mapping_status": "PARTIAL",
                "validation_scope": "handmade-overlap evidence only; not corpus-validated",
            },
        )
