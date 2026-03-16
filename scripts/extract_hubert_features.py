#!/usr/bin/env python3
"""
Extract HuBERT Features from HDF5 Audio (Synchronized with MRI Frames)

Reads audio from data/processed/aligned/ HDF5 files, resamples to 16kHz,
extracts HuBERT-Large layer-12 features (1024-dim), and interpolates to
match MRI frame count.

Output: data/processed/audio_features/hubert/{utterance_name}_hubert.npy
        shape: (num_mri_frames, 1024) float32

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
from utils.logger import setup_logger


# HuBERT outputs at 50Hz (20ms hop), audio must be 16kHz
HUBERT_FPS = 50.0
HUBERT_SR = 16000


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
) -> np.ndarray:
    """Linear interpolation: HuBERT 50fps → MRI fps."""
    num_hubert_frames, n_dim = features.shape
    source_times = np.arange(num_hubert_frames) / HUBERT_FPS
    mri_times = np.arange(num_mri_frames) / mri_fps

    synced = np.zeros((num_mri_frames, n_dim), dtype=np.float32)
    for i in range(n_dim):
        synced[:, i] = np.interp(mri_times, source_times, features[:, i])
    return synced


def main():
    parser = argparse.ArgumentParser(
        description="Extract HuBERT features from HDF5 audio → aligned with MRI"
    )
    parser.add_argument("--aligned-dir", type=str, default="data/processed/aligned")
    parser.add_argument("--segmentation-dir", type=str, default="data/processed/segmentations")
    parser.add_argument("--output-dir", type=str, default="data/processed/audio_features")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--layer-index", type=int, default=12, help="HuBERT layer to extract (default: 12)")
    parser.add_argument("--subjects", type=str, default=None)
    parser.add_argument("--skip-existing", action="store_true")
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
        utterances.append((utt_name, aligned_file, seg_files[0], out_file))

    logger.info(f"Utterances found: {len(utterances)}")

    n_success = 0
    n_skipped = 0
    n_failed = 0

    for utt_name, aligned_file, seg_file, out_file in tqdm(utterances, desc="HuBERT"):
        if args.skip_existing and out_file.exists():
            n_skipped += 1
            continue
        try:
            seg_data = np.load(seg_file)
            num_mri_frames = int(seg_data["num_frames"])

            with h5py.File(aligned_file, "r") as f:
                audio = f["audio"][:]
                audio_sr = int(f.attrs["audio_sr"])
                mri_fps = float(f.attrs["mri_fps"])

            # Resample to 16kHz
            audio_16k = resample_audio(audio, audio_sr)

            # Extract HuBERT features
            waveform = torch.from_numpy(audio_16k).float().unsqueeze(0).to(args.device)
            with torch.no_grad():
                features_list, _ = model.extract_features(waveform)
                layer_feat = features_list[args.layer_index].squeeze(0).cpu().numpy()

            # Sync to MRI frame rate
            synced = sync_to_mri_frames(layer_feat, num_mri_frames, mri_fps)
            np.save(out_file, synced)
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
