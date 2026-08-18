#!/usr/bin/env python3
"""
Extract HuBERT Features from HDF5 Audio (Synchronized with MRI Frames)

Reads audio from data/processed/aligned/ HDF5 files, resamples to 16kHz,
extracts HuBERT-Large layer-12 features (1024-dim), and interpolates to
match MRI frame count.

Output: data/processed/audio_features/hubert/{utterance_name}_hubert.npy
        data/processed/audio_features/hubert/{utterance_name}_hubert_mri_frame_indices.npy
        Features and targets must both be indexed by the stored MRI indices.

Usage:
    python scripts/extract_hubert_features.py
    python scripts/extract_hubert_features.py --device cuda
    python scripts/extract_hubert_features.py --subjects sub001,sub002
    python scripts/extract_hubert_features.py --skip-existing
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from audio_features.hubert_extractor import (  # noqa: E402
    HUBERT_FEATURE_ORIGIN_SECONDS,
    HUBERT_FEATURE_STRIDE_SECONDS,
    HUBERT_SAMPLE_RATE,
)
from research.alignment_diagnostic import supported_features_for_timestamps  # noqa: E402
from utils.logger import setup_logger  # noqa: E402

HUBERT_SR = HUBERT_SAMPLE_RATE


def resample_audio(audio: np.ndarray, src_sr: int) -> np.ndarray:
    """Resample audio to 16kHz if needed."""
    if src_sr == HUBERT_SR:
        return audio
    waveform = torch.from_numpy(audio).float().unsqueeze(0)
    resampler = torchaudio.transforms.Resample(src_sr, HUBERT_SR)
    return resampler(waveform).squeeze(0).numpy()


def sync_to_mri_frames(
    features: np.ndarray,
    num_mri_frames: int,
    mri_fps: float,
    *,
    mri_timestamps: np.ndarray | None = None,
    feature_time_offset_seconds: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return HuBERT features and exact supported target-frame indices."""
    timestamps = (
        np.arange(num_mri_frames, dtype=np.float64) / mri_fps
        if mri_timestamps is None
        else np.asarray(mri_timestamps, dtype=np.float64)
    )
    if timestamps.shape != (num_mri_frames,):
        raise ValueError("mri_timestamps must match num_mri_frames")
    return supported_features_for_timestamps(
        features,
        timestamps,
        feature_stride_seconds=HUBERT_FEATURE_STRIDE_SECONDS,
        feature_time_offset_seconds=feature_time_offset_seconds,
        feature_origin_seconds=HUBERT_FEATURE_ORIGIN_SECONDS,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract HuBERT features from HDF5 audio → aligned with MRI"
    )
    parser.add_argument("--aligned-dir", type=str, default="data/processed/aligned")
    parser.add_argument("--segmentation-dir", type=str, default="data/processed/segmentations")
    parser.add_argument("--output-dir", type=str, default="data/processed/audio_features")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--layer-index", type=int, default=12, help="HuBERT layer to extract (default: 12)"
    )
    parser.add_argument("--subjects", type=str, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--feature-time-offset-seconds",
        type=float,
        default=0.0,
        help="feature_time = MRI_time + offset; do not reuse legacy alignment offsets",
    )
    parser.add_argument("--log-file", type=str, default="logs/extract_hubert.log")

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    aligned_dir = project_root / args.aligned_dir
    seg_dir = project_root / args.segmentation_dir
    output_dir = project_root / args.output_dir / "hubert"
    log_file = project_root / args.log_file

    log_file.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("HuBERT", log_file=str(log_file))
    subjects = [s.strip() for s in args.subjects.split(",")] if args.subjects else None

    logger.info("=" * 60)
    logger.info("HUBERT FEATURE EXTRACTION")
    logger.info("=" * 60)
    logger.info(f"Device     : {args.device}")
    logger.info(f"Layer      : {args.layer_index}")
    logger.info(f"Subjects   : {subjects or 'all'}")
    logger.info(f"Output dir : {output_dir}")
    logger.info(f"Feature time offset: {args.feature_time_offset_seconds:.6f}s")
    logger.info("=" * 60)

    # Load HuBERT-Large model
    logger.info("Loading HuBERT-Large model (first run downloads ~1.4GB)...")
    bundle = torchaudio.pipelines.HUBERT_LARGE
    model = bundle.get_model().to(args.device)
    model.eval()
    logger.info("HuBERT-Large loaded.")

    # Collect utterances from segmentation directory
    utterances = []
    for utt_dir in sorted(seg_dir.iterdir()):
        if not utt_dir.is_dir():
            continue
        utt_name = utt_dir.name
        subject = utt_name.split("_")[0]
        if subjects and subject not in subjects:
            continue
        seg_files = list(utt_dir.glob("*_segmentations.npz"))
        if not seg_files:
            continue
        aligned_file = aligned_dir / subject / f"{utt_name}.h5"
        if not aligned_file.exists():
            continue
        out_file = output_dir / f"{utt_name}_hubert.npy"
        index_file = output_dir / f"{utt_name}_hubert_mri_frame_indices.npy"
        utterances.append((utt_name, aligned_file, seg_files[0], out_file, index_file))

    logger.info(f"Utterances found: {len(utterances)}")

    n_success = 0
    n_skipped = 0
    n_failed = 0
    mappings = []

    for utt_name, aligned_file, seg_file, out_file, index_file in tqdm(utterances, desc="HuBERT"):
        if args.skip_existing and out_file.exists() and index_file.exists():
            n_skipped += 1
            continue
        try:
            seg_data = np.load(seg_file)
            num_mri_frames = int(seg_data["num_frames"])

            with h5py.File(aligned_file, "r") as f:
                audio = f["audio"][:]
                audio_sr = int(f.attrs["audio_sr"])
                mri_fps = float(f.attrs["mri_fps"])
                mri_timestamps = f["mri_timestamps"][:] if "mri_timestamps" in f else None

            # Resample to 16kHz
            audio_16k = resample_audio(audio, audio_sr)

            # Extract HuBERT features
            waveform = torch.from_numpy(audio_16k).float().unsqueeze(0).to(args.device)
            with torch.no_grad():
                features_list, _ = model.extract_features(waveform)
                layer_feat = features_list[args.layer_index].squeeze(0).cpu().numpy()

            # Sync to MRI frame rate
            synced, frame_indices = sync_to_mri_frames(
                layer_feat,
                num_mri_frames,
                mri_fps,
                mri_timestamps=mri_timestamps,
                feature_time_offset_seconds=args.feature_time_offset_seconds,
            )
            np.save(out_file, synced)
            np.save(index_file, frame_indices)
            mappings.append(
                {
                    "utterance_id": utt_name,
                    "feature_file": str(out_file),
                    "mri_frame_indices_file": str(index_file),
                    "supported_frame_count": len(frame_indices),
                    "first_last_supported_frame": [
                        int(frame_indices[0]),
                        int(frame_indices[-1]),
                    ],
                    "timestamp_source": (
                        "hdf5:mri_timestamps"
                        if mri_timestamps is not None
                        else "frame_index/mri_fps"
                    ),
                }
            )
            n_success += 1

        except Exception as e:
            logger.warning(f"FAILED {utt_name}: {e}")
            n_failed += 1

    # Save summary
    summary = {
        "total": len(utterances),
        "success": n_success,
        "skipped": n_skipped,
        "failed": n_failed,
        "layer_index": args.layer_index,
        "feature_dim": 1024,
        "model": "HuBERT-Large",
        "feature_time_offset_seconds": args.feature_time_offset_seconds,
        "feature_origin_seconds": HUBERT_FEATURE_ORIGIN_SECONDS,
        "feature_stride_seconds": HUBERT_FEATURE_STRIDE_SECONDS,
        "feature_time_sign_convention": ("feature_time = MRI_time + feature_time_offset_seconds"),
        "supervision_contract": "slice target frames by each stored mri_frame_indices file",
        "mappings": mappings,
        "legacy_alignment_offset_automatically_reused": False,
        "legacy_alignment_offset_conversion": None,
    }
    with open(output_dir.parent / "hubert_extraction_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("HUBERT EXTRACTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Success  : {n_success}")
    logger.info(f"Skipped  : {n_skipped}")
    logger.info(f"Failed   : {n_failed}")
    logger.info(f"Output   : {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
