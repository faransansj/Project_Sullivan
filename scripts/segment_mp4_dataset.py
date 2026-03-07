#!/usr/bin/env python3
"""
Segment Vocal Tract from MP4-derived HDF5 files

Scans data/processed/aligned/ for HDF5 files and runs U-Net segmentation
on the extracted MRI frames. Does NOT require batch_summary.json — it
discovers files by directly walking the aligned directory.

Output structure:
  data/processed/segmentations/
  └── {utterance_name}/
      └── {utterance_name}_segmentations.npz

Usage:
    python scripts/segment_mp4_dataset.py
    python scripts/segment_mp4_dataset.py --subjects sub001,sub002
    python scripts/segment_mp4_dataset.py --device cuda --skip-existing
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.segmentation.unet_simple import UNet
from src.utils.logger import setup_logger
from src.utils.io_utils import ensure_directory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_hdf5_files(
    aligned_dir: Path,
    subjects: list[str] | None,
    max_per_subject: int | None,
) -> list[dict]:
    """Walk aligned_dir and collect HDF5 file info."""
    items = []

    for subject_dir in sorted(aligned_dir.iterdir()):
        if not subject_dir.is_dir():
            continue
        subject_id = subject_dir.name
        if subject_id == "batch_summary.json":
            continue
        if subjects and subject_id not in subjects:
            continue

        h5_files = sorted(subject_dir.glob("*.h5"))
        if max_per_subject:
            h5_files = h5_files[:max_per_subject]

        for h5_path in h5_files:
            utterance_name = h5_path.stem
            items.append(
                {
                    "subject_id": subject_id,
                    "utterance_name": utterance_name,
                    "hdf5_path": h5_path,
                }
            )

    return items


def load_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    """Load U-Net checkpoint, handling Lightning 'model.' prefix."""
    model = UNet(n_channels=1, n_classes=1)
    state_dict = torch.load(model_path, map_location=device)

    # Handle Lightning checkpoint wrapper
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k[6:] if k.startswith("model.") else k] = v

    model.load_state_dict(cleaned)
    model.to(device)
    model.eval()
    return model


def segment_frames(
    mri_frames: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    pad_size: int = 96,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run U-Net on all frames of an utterance.

    Parameters
    ----------
    mri_frames : (T, H, W) float32
    model      : loaded U-Net
    device     : torch device
    pad_size   : spatial size to pad frames to (must be divisible by 16)

    Returns
    -------
    segmentations       : (T, H, W) uint8
    class_distributions : (T, 2) float32  [background, airway]
    """
    T, H, W = mri_frames.shape
    pad_h = (pad_size - H) // 2
    pad_w = (pad_size - W) // 2

    segmentations = np.zeros((T, H, W), dtype=np.uint8)
    class_dists = np.zeros((T, 2), dtype=np.float32)

    for i in range(T):
        frame = mri_frames[i]
        frame_norm = (frame - frame.mean()) / (frame.std() + 1e-8)

        frame_padded = np.pad(
            frame_norm,
            ((pad_h, pad_size - H - pad_h), (pad_w, pad_size - W - pad_w)),
            mode="constant",
            constant_values=0,
        )

        tensor = torch.FloatTensor(frame_padded).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(tensor)
            pred = (torch.sigmoid(out) > 0.5).int().squeeze(0).squeeze(0).cpu().numpy()

        seg = pred[pad_h: pad_h + H, pad_w: pad_w + W].astype(np.uint8)
        segmentations[i] = seg

        dist = np.bincount(seg.flatten(), minlength=2).astype(np.float32)
        dist /= seg.size
        class_dists[i] = dist

    return segmentations, class_dists


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Segment vocal tract from MP4-derived HDF5 files"
    )
    parser.add_argument(
        "--aligned-dir",
        type=str,
        default="data/processed/aligned",
        help="Directory containing HDF5 files (output of extract_frames_from_mp4.py)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/segmentations",
        help="Output directory for segmentation NPZ files",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/unet_scratch/unet_best.pth",
        help="Path to U-Net checkpoint",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated list of subjects (default: all)",
    )
    parser.add_argument(
        "--max-per-subject",
        type=int,
        default=None,
        help="Max utterances per subject (default: all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Inference device",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip utterances whose segmentation NPZ already exists",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="logs/segment_mp4.log",
        help="Log file path",
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    aligned_dir = project_root / args.aligned_dir
    output_dir = project_root / args.output_dir
    model_path = project_root / args.model
    log_file = project_root / args.log_file

    log_file.parent.mkdir(parents=True, exist_ok=True)
    ensure_directory(output_dir)

    logger = setup_logger("SegmentMP4", log_file=str(log_file))

    subjects = [s.strip() for s in args.subjects.split(",")] if args.subjects else None
    device = torch.device(args.device)

    logger.info("=" * 60)
    logger.info("SEGMENT MP4 DATASET")
    logger.info("=" * 60)
    logger.info(f"Aligned dir : {aligned_dir}")
    logger.info(f"Output dir  : {output_dir}")
    logger.info(f"Model       : {model_path}")
    logger.info(f"Device      : {device}")
    logger.info(f"Subjects    : {subjects or 'all'}")
    logger.info("=" * 60)

    # Discover HDF5 files
    items = discover_hdf5_files(aligned_dir, subjects, args.max_per_subject)
    logger.info(f"Found {len(items)} HDF5 utterances")

    if len(items) == 0:
        logger.error("No HDF5 files found. Run extract_frames_from_mp4.py first.")
        return

    # Load model
    logger.info(f"Loading model from {model_path}")
    try:
        model = load_model(model_path, device)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    stats_list = []
    total_frames = 0
    start_time = datetime.now()

    for item in tqdm(items, desc="Segmenting"):
        utterance_name = item["utterance_name"]
        hdf5_path = item["hdf5_path"]
        subject_id = item["subject_id"]

        utt_out_dir = output_dir / utterance_name
        ensure_directory(utt_out_dir)
        npz_path = utt_out_dir / f"{utterance_name}_segmentations.npz"

        if args.skip_existing and npz_path.exists():
            logger.debug(f"Skipping {utterance_name} (already exists)")
            continue

        try:
            with h5py.File(hdf5_path, "r") as f:
                mri_frames = f["mri_frames"][:]

            segmentations, class_dists = segment_frames(mri_frames, model, device)
            num_frames = segmentations.shape[0]

            np.savez_compressed(
                npz_path,
                segmentations=segmentations,
                class_distributions=class_dists,
                utterance_name=utterance_name,
                hdf5_path=str(hdf5_path),
                num_frames=num_frames,
                class_names=["background", "airway"],
            )

            stats_list.append(
                {
                    "utterance_name": utterance_name,
                    "subject_id": subject_id,
                    "num_frames": int(num_frames),
                    "mean_airway_fraction": float(class_dists[:, 1].mean()),
                    "output_path": str(npz_path),
                }
            )
            total_frames += num_frames

        except Exception as e:
            logger.error(f"Failed to segment {utterance_name}: {e}")
            continue

    duration = (datetime.now() - start_time).total_seconds()

    summary = {
        "created_at": datetime.now().isoformat(),
        "total_utterances": len(stats_list),
        "total_frames": total_frames,
        "duration_seconds": duration,
        "fps_throughput": total_frames / max(duration, 1),
        "model_path": str(model_path),
        "device": str(device),
        "utterance_stats": stats_list,
    }

    summary_path = output_dir / "segmentation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("SEGMENTATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Utterances  : {len(stats_list)}")
    logger.info(f"Total frames: {total_frames:,}")
    logger.info(f"Duration    : {duration:.1f}s ({duration / 60:.1f} min)")
    logger.info(f"Throughput  : {summary['fps_throughput']:.1f} frames/sec")
    logger.info(f"Summary     : {summary_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
