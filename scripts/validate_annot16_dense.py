#!/usr/bin/env python3
"""Regenerate evidence-bounded Dense Annot-16 validation artifacts locally."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.io import whosmat

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.research.annot16 import (  # noqa: E402
    Annot16DenseAdapter,
    Annot16GroundTruthAdapter,
    DENSE_ARTICULATOR_MAPPING,
)

COLORS = {
    "epiglottis": "#ff3030",
    "tongue": "#30ff30",
    "lower_lip": "#3090ff",
    "chin": "#ff30ff",
    "arytenoid": "#30ffff",
    "pharyngeal_wall": "#ffff30",
    "hard_palate": "#ff9030",
    "velum": "#ffffff",
    "upper_lip": "#b060ff",
}
DEFAULT_OVERLAYS = {530, 1005, 1638, 2516}


def dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def point_error(dense: np.ndarray, handmade: np.ndarray) -> float:
    distances = np.linalg.norm(dense[:, None] - handmade[None, :], axis=2)
    return float((distances.min(axis=1).mean() + distances.min(axis=0).mean()) / 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-rate", type=float, default=83.28)
    args = parser.parse_args()

    output = args.output_dir
    overlay_dir = output / "overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for old_overlay in overlay_dir.glob("*.png"):
        old_overlay.unlink()

    dense_adapter = Annot16DenseAdapter(args.dataset_root, frame_rate=args.frame_rate)
    handmade_adapter = Annot16GroundTruthAdapter(args.dataset_root, frame_rate=args.frame_rate)
    annotations = sorted(
        (args.dataset_root / "hand_ground_truth" / "ground_truth_json").glob("*.json")
    )
    metrics = defaultdict(list)
    bounds = []
    samples = []
    templates = Counter()
    failures = []

    for annotation in annotations:
        try:
            handmade = handmade_adapter.load(annotation)
            frame_number = handmade.source_provenance["source_frame_number"]
            track = (
                args.dataset_root
                / handmade.speaker_id
                / "track"
                / f"{handmade.utterance_id}_track.mat"
            )
            dense = dense_adapter.load(track, frame_number)
            frame_metrics = {}
            for name, contour in dense.articulators.items():
                dense_points = contour.coordinates[contour.valid_mask]
                handmade_contour = handmade.articulators.get(name)
                if handmade_contour is None or not len(dense_points):
                    continue
                handmade_points = handmade_contour.coordinates[handmade_contour.valid_mask]
                if not len(handmade_points):
                    continue
                error = point_error(dense_points, handmade_points)
                metrics[name].append(error)
                frame_metrics[name] = round(error, 4)
                bounds.extend(dense_points.tolist())
            templates[dense.source_provenance["template"]] += 1
            samples.append(
                {
                    "sample_id": dense.sample_id,
                    "track_path": str(track),
                    "handmade_annotation_path": str(annotation),
                    "source_frame_number": frame_number,
                    "template": dense.source_provenance["template"],
                    "symmetric_nearest_point_error_pixels": frame_metrics,
                }
            )

            if dense.utterance_id == "sub061_2drt_17_topic1" and frame_number in DEFAULT_OVERLAYS:
                image = Image.open(handmade.mri_path).convert("RGB")
                draw = ImageDraw.Draw(image)
                for name, contour in dense.articulators.items():
                    points = contour.coordinates[contour.valid_mask]
                    if len(points) > 1:
                        draw.line([tuple(point) for point in points], fill=COLORS[name], width=1)
                    hand = handmade.articulators.get(name)
                    if hand is not None:
                        for x, y in hand.coordinates[hand.valid_mask]:
                            draw.ellipse(
                                (x - 1, y - 1, x + 1, y + 1), fill="white", outline="black"
                            )
                overlay_path = (
                    overlay_dir / f"sub061_topic1_frame-{frame_number}_dense-handmade.png"
                )
                image.resize((504, 504), Image.Resampling.NEAREST).save(overlay_path)
        except (FileNotFoundError, KeyError, ValueError) as error:
            failures.append({"annotation_path": str(annotation), "error": str(error)})

    track_files = sorted(args.dataset_root.glob("sub*/track/*_track.mat"))
    dense_frames = sum(
        dict((name, shape) for name, shape, _ in whosmat(path))["trackdata"][1]
        for path in track_files
    )
    coordinate_array = np.asarray(bounds)
    mapped = [
        {"segment_index": segment, "identity": identity, "canonical_articulator": name}
        for segment, identity, name in DENSE_ARTICULATOR_MAPPING
    ]
    means = {name: round(float(np.mean(values)), 4) for name, values in metrics.items()}
    within_bounds = bool(((coordinate_array >= 0) & (coordinate_array < 84)).all())
    overlay_count = len(list(overlay_dir.glob("*.png")))

    dump(
        output / "inventory.json",
        {
            "dataset": "75-Speaker Annot-16",
            "dense_track_files": len(track_files),
            "dense_annotated_frames": dense_frames,
            "handmade_frames": len(annotations),
            "evaluated_handmade_overlaps": len(samples),
            "failed_handmade_overlaps": failures,
            "raw_contour_arrays_redistributed": False,
        },
    )
    dump(
        output / "dense_schema.json",
        {
            "mat_variable": "trackdata",
            "frame_record_fields": ["contours", "frameNo", "template"],
            "frame_numbering": (
                "frameNo is 1-based; canonical frame_index = frameNo - 1; " "lookup uses equality"
            ),
            "segments": (
                "four zero-based contours.segment entries; no fields literally " "named R1/R2/R3"
            ),
            "coordinates": "x_image=v[:,0]+width/2; y_image=-v[:,1]+height/2",
            "point_order": "source v row order retained after filtering by i equality",
            "missing_contour": (
                "absent mapped identity becomes an empty contour; absent frame is an error"
            ),
            "confidence": (
                "segment mu exists but is undocumented; no verified per-point confidence"
            ),
            "tracking_failure_encoding": (
                "unresolved; non-finite coordinates receive valid_mask=false"
            ),
        },
    )
    dump(
        output / "mapping_evidence.json",
        {
            "status": "PARTIAL",
            "mapping": mapped,
            "evidence_scope": "all locally available handmade overlaps; not corpus validation",
            "evaluated_handmade_overlaps": len(samples),
            "speakers": sorted(
                {sample["sample_id"].split(":")[1].split("_")[0] for sample in samples}
            ),
            "mean_symmetric_nearest_point_error_pixels": means,
            "metric_warning": (
                "nearest-point error is diagnostic, not semantic proof or point correspondence"
            ),
            "unresolved": (
                "all other segment/identity semantics, segment 3, mu, and failure encoding"
            ),
        },
    )
    dump(output / "sample_manifest.json", {"samples": samples, "failures": failures})
    dump(
        output / "validation_report.json",
        {
            "status": "PARTIAL",
            "adapter_parse": "success" if not failures else "partial",
            "evaluated_handmade_overlaps": len(samples),
            "failed_handmade_overlaps": len(failures),
            "coordinate_range": {
                "min_xy": coordinate_array.min(axis=0).round(4).tolist(),
                "max_xy": coordinate_array.max(axis=0).round(4).tolist(),
                "within_84x84": within_bounds,
            },
            "templates": dict(sorted(templates.items())),
            "representative_overlays": overlay_count,
            "claim_limit": "evidence-bounded handmade overlaps only; not corpus-validated",
            "raw_contour_arrays_redistributed": False,
        },
    )
    print(f"evaluated={len(samples)} failed={len(failures)} overlays={overlay_count}")


if __name__ == "__main__":
    main()
