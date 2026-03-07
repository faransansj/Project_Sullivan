#!/usr/bin/env python3
"""
Generate Pseudo-Labels for U-Net Training from HDF5 files

Reads MRI frames from data/processed/aligned/ HDF5 files (output of
extract_frames_from_mp4.py), applies adaptive thresholding to produce
binary airway masks, and saves the result as PNG images + metadata.json
in the format expected by PseudoLabelDataset.

Output structure:
  data/pseudo_labels/
  ├── images/  ← ROI-cropped grayscale frames (PNG)
  ├── masks/   ← binary airway masks (PNG)
  └── metadata.json

Usage:
    python scripts/generate_pseudo_labels_from_hdf5.py
    python scripts/generate_pseudo_labels_from_hdf5.py --subjects sub001,sub002
    python scripts/generate_pseudo_labels_from_hdf5.py --frames-per-utterance 20
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils.logger import setup_logger


# ---------------------------------------------------------------------------
# Image processing helpers (same logic as generate_pseudo_labels.py)
# ---------------------------------------------------------------------------

ROI_PARAMS = {"top": 0.25, "bottom": 0.95, "left": 0.15, "right": 0.85}


def extract_roi(frame: np.ndarray) -> tuple[np.ndarray, tuple]:
    h, w = frame.shape
    y1, y2 = int(h * ROI_PARAMS["top"]),  int(h * ROI_PARAMS["bottom"])
    x1, x2 = int(w * ROI_PARAMS["left"]), int(w * ROI_PARAMS["right"])
    return frame[y1:y2, x1:x2], (y1, y2, x1, x2)


def segment_adaptive(frame_uint8: np.ndarray) -> np.ndarray:
    """Binary airway mask via adaptive Gaussian threshold."""
    blurred = cv2.GaussianBlur(frame_uint8, (5, 5), 0)
    mask = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11, C=2,
    )
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def quality_score(mask: np.ndarray) -> int:
    """
    Score 0–100. Checks airway ratio (10–40%) and component structure.
    """
    total = mask.size
    airway = int(np.sum(mask == 255))
    ratio = airway / total

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    n_comp = n_labels - 1

    score = 0
    if 0.10 <= ratio <= 0.40:
        score += 50
    elif 0.05 <= ratio <= 0.50:
        score += 30

    if n_comp < 5:
        score += 30
    elif n_comp < 10:
        score += 20
    elif n_comp < 20:
        score += 10

    if n_comp > 0 and airway > 0:
        largest = int(np.max(stats[1:, cv2.CC_STAT_AREA]))
        if largest / airway > 0.7:
            score += 20
        elif largest / airway > 0.5:
            score += 10

    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate pseudo-labels from HDF5 MRI frames"
    )
    parser.add_argument(
        "--aligned-dir",
        type=str,
        default="data/processed/aligned",
        help="Directory containing HDF5 files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/pseudo_labels",
        help="Output directory for pseudo-labels",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated subject list (default: all)",
    )
    parser.add_argument(
        "--max-utterances-per-subject",
        type=int,
        default=None,
        help="Max HDF5 files per subject (default: all)",
    )
    parser.add_argument(
        "--frames-per-utterance",
        type=int,
        default=30,
        help="Number of evenly-spaced frames to sample per utterance (default: 30)",
    )
    parser.add_argument(
        "--min-quality",
        type=int,
        default=50,
        help="Minimum quality score 0-100 to accept a mask (default: 50)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="logs/generate_pseudo_labels.log",
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    aligned_dir  = project_root / args.aligned_dir
    output_dir   = project_root / args.output_dir
    log_file     = project_root / args.log_file

    log_file.parent.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    masks_dir  = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("PseudoLabels", log_file=str(log_file))

    subjects = [s.strip() for s in args.subjects.split(",")] if args.subjects else None

    logger.info("=" * 60)
    logger.info("GENERATE PSEUDO LABELS FROM HDF5")
    logger.info("=" * 60)
    logger.info(f"Aligned dir : {aligned_dir}")
    logger.info(f"Output dir  : {output_dir}")
    logger.info(f"Subjects    : {subjects or 'all'}")
    logger.info(f"Frames/utt  : {args.frames_per_utterance}")
    logger.info(f"Min quality : {args.min_quality}")
    logger.info("=" * 60)

    # Collect HDF5 files
    h5_files = []
    for subject_dir in sorted(aligned_dir.iterdir()):
        if not subject_dir.is_dir():
            continue
        sid = subject_dir.name
        if subjects and sid not in subjects:
            continue
        files = sorted(subject_dir.glob("*.h5"))
        if args.max_utterances_per_subject:
            files = files[: args.max_utterances_per_subject]
        for f in files:
            h5_files.append((sid, f))

    logger.info(f"HDF5 files found: {len(h5_files)}")

    samples = []
    n_generated = 0
    n_rejected  = 0

    for sid, h5_path in tqdm(h5_files, desc="Processing utterances"):
        utterance_name = h5_path.stem

        with h5py.File(h5_path, "r") as f:
            frames = f["mri_frames"][:]   # (T, H, W) float32 [0,1]

        T = frames.shape[0]
        indices = np.linspace(0, T - 1, args.frames_per_utterance, dtype=int)

        for idx in indices:
            raw = frames[idx]

            # float32 [0,1] → uint8 [0,255]
            frame_u8 = cv2.normalize(
                (raw * 255).astype(np.float32), None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)

            roi, roi_coords = extract_roi(frame_u8)
            mask = segment_adaptive(roi)
            score = quality_score(mask)

            if score < args.min_quality:
                n_rejected += 1
                continue

            sample_id = f"{utterance_name}_f{idx:05d}"
            img_path  = images_dir / f"{sample_id}.png"
            msk_path  = masks_dir  / f"{sample_id}.png"

            cv2.imwrite(str(img_path), roi)
            cv2.imwrite(str(msk_path), mask)

            samples.append({
                "sample_id":    sample_id,
                "subject":      sid,
                "utterance":    utterance_name,
                "frame_idx":    int(idx),
                "roi_coords":   [int(c) for c in roi_coords],
                "image_path":   str(img_path.relative_to(output_dir)),
                "mask_path":    str(msk_path.relative_to(output_dir)),
                "quality_score": score,
            })
            n_generated += 1

    metadata = {
        "config": {
            "aligned_dir":            str(aligned_dir),
            "frames_per_utterance":   args.frames_per_utterance,
            "min_quality_score":      args.min_quality,
        },
        "summary": {
            "total_generated": n_generated,
            "total_rejected":  n_rejected,
            "acceptance_rate": n_generated / max(n_generated + n_rejected, 1),
        },
        "samples": samples,
    }

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("=" * 60)
    logger.info("PSEUDO-LABEL GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Generated   : {n_generated}")
    logger.info(f"Rejected    : {n_rejected}")
    logger.info(f"Accept rate : {metadata['summary']['acceptance_rate']*100:.1f}%")
    logger.info(f"Metadata    : {meta_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
