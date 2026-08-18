#!/usr/bin/env python3
"""Inventory Annot-16 and render canonical handmade-contour overlays."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw
from scipy.io import whosmat

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.research.annot16 import Annot16GroundTruthAdapter

COLORS = {
    "epiglottis": "red",
    "tongue": "lime",
    "lower_lip": "blue",
    "chin": "magenta",
    "arytenoid": "cyan",
    "pharyngeal_wall": "yellow",
    "hard_palate": "orange",
    "velum": "white",
    "upper_lip": "purple",
}


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--repository-aligned-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--speaker", default="sub061")
    parser.add_argument("--utterance", default="sub061_2drt_17_topic1")
    parser.add_argument("--frame-rate", type=float, default=83.28)
    parser.add_argument("--pixel-spacing", type=float, nargs=2, default=(2.4, 2.4))
    args = parser.parse_args()

    speakers = sorted(path.name for path in args.dataset_root.glob("sub*") if path.is_dir())
    track_files = sorted(args.dataset_root.glob("sub*/track/*_track.mat"))
    alignment_files = sorted(args.dataset_root.glob("sub*/alignment/*.TextGrid"))
    ground_truth_files = sorted(
        (args.dataset_root / "hand_ground_truth" / "ground_truth_json").glob("*.json")
    )
    dense_frames = sum(
        dict((name, shape) for name, shape, _ in whosmat(path))["trackdata"][1]
        for path in track_files
    )
    repository_speakers = sorted(
        path.name for path in args.repository_aligned_root.glob("sub*") if path.is_dir()
    )
    exact = sorted(set(speakers) & set(repository_speakers))

    adapter = Annot16GroundTruthAdapter(
        args.dataset_root,
        frame_rate=args.frame_rate,
        pixel_spacing=tuple(args.pixel_spacing),
    )
    selected = [
        path for path in ground_truth_files if path.name.startswith(args.utterance + "_track_")
    ]
    if not selected:
        raise ValueError(f"No handmade ground truth for {args.utterance}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.output_dir / "overlay"
    overlay_dir.mkdir(exist_ok=True)
    canonical_samples = []
    point_counts = Counter()
    for annotation_path in selected:
        sample = adapter.load(annotation_path)
        image = Image.open(sample.mri_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for name, contour in sample.articulators.items():
            points = [tuple(map(float, point)) for point in contour.coordinates[contour.valid_mask]]
            point_counts[name] += len(points)
            if len(points) >= 2:
                draw.line(points, fill=COLORS.get(name, "white"), width=1)
            for x, y in points:
                draw.ellipse((x - 0.7, y - 0.7, x + 0.7, y + 0.7), fill=COLORS.get(name, "white"))
        output_path = overlay_dir / f"{sample.sample_id.replace(':', '_')}.png"
        image.resize((504, 504), Image.Resampling.NEAREST).save(output_path)
        canonical_samples.append(
            {
                "sample_id": sample.sample_id,
                "speaker_id": sample.speaker_id,
                "utterance_id": sample.utterance_id,
                "frame_index": sample.frame_index,
                "timestamp": sample.timestamp,
                "audio_path": sample.audio_path,
                "mri_path": sample.mri_path,
                "pixel_spacing": {
                    "x_mm_per_pixel": sample.pixel_spacing.x_mm_per_pixel,
                    "y_mm_per_pixel": sample.pixel_spacing.y_mm_per_pixel,
                },
                "coordinate_convention": sample.coordinate_convention,
                "source_provenance": sample.source_provenance,
                "overlay_path": str(output_path),
                "articulators": {
                    name: {
                        "coordinates": contour.coordinates.tolist(),
                        "valid_mask": contour.valid_mask.tolist(),
                        "is_static": contour.is_static,
                    }
                    for name, contour in sample.articulators.items()
                },
            }
        )

    inventory = {
        "dataset_title": "75-Speaker Annot-16",
        "zenodo_record": 18931763,
        "archive": str(args.archive),
        "archive_size_bytes": args.archive.stat().st_size,
        "archive_md5": md5(args.archive),
        "speakers": speakers,
        "speaker_count": len(speakers),
        "dense_track_utterances": len(track_files),
        "dense_annotated_frames": dense_frames,
        "phonetic_alignment_files": len(alignment_files),
        "hand_ground_truth_frames": len(ground_truth_files),
        "hand_ground_truth_articulators": sorted(point_counts),
        "repository_aligned_speakers": repository_speakers,
        "exact_speaker_matches": exact,
        "annot16_only_speakers": sorted(set(speakers) - set(repository_speakers)),
        "repository_only_speakers": sorted(set(repository_speakers) - set(speakers)),
    }
    transform = {
        "source_image_size_pixels": [84, 84],
        "origin": "top-left",
        "x_direction": "right",
        "y_direction": "down",
        "coordinate_units": "source image pixels",
        "crop": None,
        "resize_before_overlay": None,
        "flip": None,
        "rotation_degrees": 0,
        "canonical_transform": "identity",
        "frame_rate_hz": args.frame_rate,
        "frame_numbering": "source MAT frameNo is 1-based; canonical index is frameNo-1",
        "timestamp_formula": "(source_frame_number - 1) / frame_rate_hz",
        "pixel_spacing_mm": list(args.pixel_spacing),
        "pixel_spacing_source": "Lim et al., Scientific Data 2021, USC 75-speaker acquisition",
    }
    report = {
        "selected_speaker": args.speaker,
        "selected_utterance": args.utterance,
        "frames_inspected": len(canonical_samples),
        "source_repository_metadata_match": str(
            args.repository_aligned_root / args.speaker / f"{args.utterance}_video_metadata.json"
        ),
        "coordinate_bounds_valid": all(
            0 <= value < 84
            for sample in canonical_samples
            for contour in sample["articulators"].values()
            for point in contour["coordinates"]
            for value in point
        ),
        "canonical_conversion": "success",
        "overlay_generation": "success",
        "manual_review_required": True,
        "samples": canonical_samples,
    }
    for name, payload in (
        ("inventory.json", inventory),
        ("coordinate_transform.json", transform),
        ("sample_manifest.json", {"samples": canonical_samples}),
        ("validation_report.json", report),
    ):
        (args.output_dir / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
