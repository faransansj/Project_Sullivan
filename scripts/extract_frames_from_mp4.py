#!/usr/bin/env python3
"""
Extract Frames and Audio from MP4 (USC-TIMIT 2D RT dataset)

For each MP4 file in the dataset, extract:
  - Video frames → (T, H, W) float32 numpy array
  - Audio       → (N,) float32 numpy array

Saves everything as HDF5 files in the same format expected by the
existing segmentation and audio-feature extraction scripts.

Output structure:
  data/processed/aligned/
  ├── sub001/
  │   ├── sub001_2drt_01_vcv1_r1.h5
  │   └── ...
  └── batch_summary.json

Usage:
    python scripts/extract_frames_from_mp4.py
    python scripts/extract_frames_from_mp4.py --subjects sub001,sub002
    python scripts/extract_frames_from_mp4.py --max-per-subject 5
    python scripts/extract_frames_from_mp4.py --workers 8 --skip-existing
"""

import argparse
import json
import logging
import subprocess
import sys

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils.logger import setup_logger


# ---------------------------------------------------------------------------
# Core extraction helpers
# ---------------------------------------------------------------------------

def extract_video_frames(video_path: Path) -> tuple[np.ndarray, float]:
    """
    Extract all frames from a video file.

    Returns
    -------
    frames : np.ndarray  shape (T, H, W) float32, values in [0, 1]
    fps    : float
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray.astype(np.float32) / 255.0)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames extracted from {video_path}")

    return np.stack(frames, axis=0), fps


def _get_ffmpeg_binary() -> str:
    """Return ffmpeg binary path: system ffmpeg or imageio-ffmpeg fallback."""
    import shutil
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError(
            "ffmpeg not found in PATH and imageio-ffmpeg not installed. "
            "Run: uv pip install 'imageio[ffmpeg]'"
        )


def extract_audio_ffmpeg(video_path: Path, target_sr: int = 22050) -> tuple[np.ndarray, int]:
    """
    Extract audio from a video file using ffmpeg piped directly to numpy.
    Uses raw PCM output — no libsndfile dependency.
    Falls back to imageio-ffmpeg bundled binary if ffmpeg is not in PATH.

    Returns
    -------
    audio : np.ndarray  shape (N,) float32
    sr    : int         sample rate
    """
    ffmpeg_bin = _get_ffmpeg_binary()
    cmd = [
        ffmpeg_bin, "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-vn",                   # no video
        "-ar", str(target_sr),   # resample
        "-ac", "1",              # mono
        "-f", "f32le",           # raw 32-bit float little-endian PCM → stdout
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {video_path}: {result.stderr.decode()}"
        )

    audio = np.frombuffer(result.stdout, dtype=np.float32).copy()
    return audio, target_sr


# ---------------------------------------------------------------------------
# Per-utterance worker (runs in subprocess pool)
# ---------------------------------------------------------------------------

def process_one_utterance(args_tuple) -> dict:
    """Process a single MP4 → HDF5. Returns a result dict."""
    mp4_path, hdf5_path, utterance_name, subject_id, skip_existing = args_tuple

    result = {
        "utterance_name": utterance_name,
        "subject_id": subject_id,
        "hdf5_path": str(hdf5_path),
        "success": False,
        "error": None,
        "num_frames": 0,
        "audio_duration": 0.0,
        "fps": 0.0,
    }

    if skip_existing and hdf5_path.exists():
        result["success"] = True
        result["skipped"] = True
        # Read metadata from existing file for summary
        try:
            with h5py.File(hdf5_path, "r") as f:
                result["num_frames"] = int(f["mri_frames"].shape[0])
                result["fps"] = float(f.attrs.get("mri_fps", 0))
                result["audio_duration"] = (
                    float(f["audio"].shape[0]) / float(f.attrs.get("audio_sr", 1))
                )
        except Exception:
            pass
        return result

    try:
        frames, fps = extract_video_frames(mp4_path)
        audio, audio_sr = extract_audio_ffmpeg(mp4_path)

        hdf5_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(hdf5_path, "w") as f:
            f.create_dataset("mri_frames", data=frames, compression="gzip", compression_opts=4)
            f.create_dataset("audio", data=audio, compression="gzip", compression_opts=4)
            f.attrs["mri_fps"] = fps
            f.attrs["audio_sr"] = audio_sr
            f.attrs["subject_id"] = subject_id
            f.attrs["utterance_name"] = utterance_name
            f.attrs["source_mp4"] = str(mp4_path)
            f.attrs["created_at"] = datetime.now().isoformat()

        result["success"] = True
        result["num_frames"] = int(frames.shape[0])
        result["fps"] = float(fps)
        result["audio_duration"] = float(len(audio) / audio_sr)

    except Exception as e:
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_jobs(
    data_root: Path,
    output_dir: Path,
    subjects: Optional[list[str]],
    max_per_subject: Optional[int],
    metafile: Path,
) -> list[tuple]:
    """Build the list of (mp4, hdf5, utterance_name, subject_id) jobs."""
    # Load metafile to know which files exist
    with open(metafile) as f:
        meta = json.load(f)

    jobs = []

    subject_dirs = sorted(
        [d for d in data_root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name

        if subjects and subject_id not in subjects:
            continue

        video_dir = subject_dir / "2drt" / "video"
        if not video_dir.exists():
            continue

        mp4_files = sorted(video_dir.glob("*.mp4"))

        if max_per_subject:
            mp4_files = mp4_files[:max_per_subject]

        for mp4_path in mp4_files:
            # Derive utterance name: strip "_video.mp4" suffix
            stem = mp4_path.stem  # e.g. sub001_2drt_01_vcv1_r1_video
            if stem.endswith("_video"):
                utterance_name = stem[: -len("_video")]
            else:
                utterance_name = stem

            hdf5_path = output_dir / subject_id / f"{utterance_name}.h5"
            jobs.append((mp4_path, hdf5_path, utterance_name, subject_id))

    return jobs


def build_batch_summary(results: list[dict], output_dir: Path) -> dict:
    """Build batch_summary.json compatible with segment_subset.py."""
    by_subject: dict[str, list] = {}
    for r in results:
        if not r["success"]:
            continue
        sid = r["subject_id"]
        by_subject.setdefault(sid, []).append(
            {
                "utterance_name": r["utterance_name"],
                "hdf5_path": r["hdf5_path"],
                "correlation": 1.0,  # all aligned by definition (same file)
                "num_frames": r["num_frames"],
            }
        )

    subjects_list = [
        {"subject_id": sid, "utterances": utts}
        for sid, utts in sorted(by_subject.items())
    ]

    summary = {
        "created_at": datetime.now().isoformat(),
        "total_subjects": len(subjects_list),
        "total_utterances": sum(len(s["utterances"]) for s in subjects_list),
        "subjects": subjects_list,
    }

    summary_path = output_dir / "batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames and audio from USC-TIMIT MP4 files → HDF5"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="dl_data/dataset_2drt_video_only",
        help="Root directory of the 2D RT video dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/aligned",
        help="Output directory for HDF5 files",
    )
    parser.add_argument(
        "--metafile",
        type=str,
        default="dl_data/dataset_2drt_video_only/metafile_public_20210129.json",
        help="Path to the dataset metafile JSON",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated list of subjects to process (default: all)",
    )
    parser.add_argument(
        "--max-per-subject",
        type=int,
        default=None,
        help="Maximum number of utterances per subject (default: all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes (default: 4)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip MP4 files whose HDF5 already exists",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="logs/extract_frames.log",
        help="Log file path",
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_root = project_root / args.data_root
    output_dir = project_root / args.output_dir
    metafile = project_root / args.metafile
    log_file = project_root / args.log_file

    log_file.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("ExtractFrames", log_file=str(log_file))

    subjects = [s.strip() for s in args.subjects.split(",")] if args.subjects else None

    logger.info("=" * 60)
    logger.info("EXTRACT FRAMES FROM MP4")
    logger.info("=" * 60)
    logger.info(f"Data root  : {data_root}")
    logger.info(f"Output dir : {output_dir}")
    logger.info(f"Subjects   : {subjects or 'all'}")
    logger.info(f"Max/subj   : {args.max_per_subject or 'all'}")
    logger.info(f"Workers    : {args.workers}")
    logger.info(f"Skip exist : {args.skip_existing}")
    logger.info("=" * 60)

    jobs = collect_jobs(data_root, output_dir, subjects, args.max_per_subject, metafile)
    logger.info(f"Total utterances to process: {len(jobs)}")

    if len(jobs) == 0:
        logger.error("No jobs found. Check --data-root and --subjects.")
        return

    # Inject skip_existing flag into each job tuple
    jobs_with_flag = [(*j, args.skip_existing) for j in jobs]

    results = []
    failed = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one_utterance, j): j[2] for j in jobs_with_flag}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
            r = future.result()
            results.append(r)
            if not r["success"]:
                failed.append(r)
                logger.warning(f"FAILED {r['utterance_name']}: {r['error']}")

    # Build batch_summary.json
    summary = build_batch_summary(results, output_dir)

    success_count = sum(1 for r in results if r["success"])
    skipped_count = sum(1 for r in results if r.get("skipped"))
    total_frames = sum(r["num_frames"] for r in results if r["success"])

    logger.info("=" * 60)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total     : {len(jobs)}")
    logger.info(f"Success   : {success_count}")
    logger.info(f"Skipped   : {skipped_count}")
    logger.info(f"Failed    : {len(failed)}")
    logger.info(f"Total frames : {total_frames:,}")
    logger.info(f"Subjects  : {summary['total_subjects']}")
    logger.info(f"Utterances: {summary['total_utterances']}")
    logger.info(f"Summary   : {output_dir / 'batch_summary.json'}")
    logger.info("=" * 60)

    if failed:
        logger.warning("Failed utterances:")
        for r in failed:
            logger.warning(f"  {r['utterance_name']}: {r['error']}")


if __name__ == "__main__":
    main()
