#!/usr/bin/env python3
"""Reproduce the Phase 2 audio/MRI timing audit from one source-corpus MP4."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torchaudio
from scipy import signal
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.research.alignment_diagnostic import (  # noqa: E402
    audio_sample_indices,
    hubert_feature_center_timestamps,
)

DEFAULT_MEDIA = (
    PROJECT_ROOT / "data/raw/usc_timit_data/sub061/2drt/video/sub061_2drt_17_topic1_video.mp4"
)
DEFAULT_TRACK = (
    PROJECT_ROOT / "data/annot16/75SpeakerAnnot16/sub061/track/sub061_2drt_17_topic1_track.mat"
)
DEFAULT_TEXTGRID = (
    PROJECT_ROOT
    / "data/annot16/75SpeakerAnnot16/sub061/alignment/sub061_2drt_17_topic1_text.TextGrid"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/research/aai_phase2_alignment"


def run(*command: str) -> bytes:
    return subprocess.run(command, check=True, stdout=subprocess.PIPE).stdout


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalized_overlap_correlation(
    left_times: np.ndarray,
    left: np.ndarray,
    right_times: np.ndarray,
    right: np.ndarray,
    offset_seconds: float,
) -> tuple[float, int]:
    """Correlate left(t) with right(t + offset), using overlap only."""
    query = left_times + offset_seconds
    valid = (query >= right_times[0]) & (query <= right_times[-1])
    x = left[valid].astype(np.float64)
    y = np.interp(query[valid], right_times, right).astype(np.float64)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan"), len(x)
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    return float(np.mean(x * y)), len(x)


def parse_word_intervals(path: Path) -> list[tuple[float, float, str]]:
    text = path.read_text(encoding="utf-16")
    words_tier = text.split("item [2]:", 1)[0]
    matches = re.findall(r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "(.*?)"', words_tier, re.S)
    if not matches:
        raise ValueError(f"No word intervals found in {path}")
    return [(float(start), float(end), label.lstrip("\ufeff")) for start, end, label in matches]


def contour_displacement(path: Path) -> np.ndarray:
    records = loadmat(path, simplify_cells=True)["trackdata"]
    result = np.zeros(len(records), dtype=np.float64)
    for frame in range(1, len(records)):
        distances = []
        previous = records[frame - 1]["contours"]["segment"][:3]
        current = records[frame]["contours"]["segment"][:3]
        for old, new in zip(previous, current):
            old_points, new_points = np.asarray(old["v"]), np.asarray(new["v"])
            old_ids, new_ids = np.asarray(old["i"]), np.asarray(new["i"])
            for identity in np.intersect1d(old_ids, new_ids):
                before, after = old_points[old_ids == identity], new_points[new_ids == identity]
                if before.shape == after.shape:
                    distances.extend(np.linalg.norm(after - before, axis=1))
        result[frame] = float(np.mean(distances)) if distances else np.nan
    result[np.isnan(result)] = np.nanmedian(result)
    return result


def peak_summary(offsets: np.ndarray, values: np.ndarray) -> dict:
    best = int(np.nanargmax(values))
    outside = np.abs(offsets - offsets[best]) > 0.05
    runner = int(np.nanargmax(np.where(outside, values, np.nan)))
    zero = int(np.argmin(np.abs(offsets)))
    return {
        "best_offset_seconds": round(float(offsets[best]), 6),
        "best_correlation": round(float(values[best]), 6),
        "zero_offset_correlation": round(float(values[zero]), 6),
        "best_distant_competitor_offset_seconds": round(float(offsets[runner]), 6),
        "best_distant_competitor_correlation": round(float(values[runner]), 6),
        "distant_peak_margin": round(float(values[best] - values[runner]), 6),
        "ambiguous": bool(values[best] - values[runner] < 0.03),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media", type=Path, default=DEFAULT_MEDIA)
    parser.add_argument("--track", type=Path, default=DEFAULT_TRACK)
    parser.add_argument("--textgrid", type=Path, default=DEFAULT_TEXTGRID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    probe = json.loads(
        run(
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(args.media),
        )
    )
    frames = json.loads(
        run(
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp,best_effort_timestamp_time,pkt_duration,pkt_duration_time",
            "-of",
            "json",
            str(args.media),
        )
    )["frames"]
    packets = json.loads(
        run(
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_packets",
            "-show_entries",
            "packet=pts,pts_time,duration,duration_time,side_data_list",
            "-of",
            "json",
            str(args.media),
        )
    )["packets"]
    audio_frames = json.loads(
        run(
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_frames",
            "-show_entries",
            "frame=pts,pts_time,best_effort_timestamp,nb_samples,side_data_list",
            "-of",
            "json",
            str(args.media),
        )
    )["frames"]
    ffmpeg_version = run("ffmpeg", "-version").decode().splitlines()[0]
    ffprobe_version = run("ffprobe", "-version").decode().splitlines()[0]
    video_stream, audio_stream = probe["streams"]
    width, height = int(video_stream["width"]), int(video_stream["height"])
    video = (
        np.frombuffer(
            run(
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(args.media),
                "-map",
                "0:v:0",
                "-pix_fmt",
                "gray",
                "-f",
                "rawvideo",
                "-",
            ),
            dtype=np.uint8,
        )
        .reshape(-1, height, width)
        .astype(np.float32)
    )
    audio_rate = int(audio_stream["sample_rate"])
    audio = np.frombuffer(
        run(
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(args.media),
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            str(audio_rate),
            "-f",
            "f32le",
            "-",
        ),
        dtype="<f4",
    )
    time_base_numerator, time_base_denominator = map(int, video_stream["time_base"].split("/"))
    video_pts = np.asarray([int(frame["best_effort_timestamp"]) for frame in frames])
    video_times = video_pts.astype(np.float64) * time_base_numerator / time_base_denominator
    if len(video) != len(video_times):
        raise RuntimeError("Decoded frame count does not match ffprobe frame PTS count")

    differences = np.abs(np.diff(video, axis=0, prepend=video[:1]))
    motion = {
        "global_motion": differences.mean(axis=(1, 2)),
        "lower_face_motion": differences[:, 25:78, :55].mean(axis=(1, 2)),
        "mouth_roi_motion": differences[:, 25:65, :45].mean(axis=(1, 2)),
        "dense_contour_displacement": contour_displacement(args.track),
    }
    motion = {name: gaussian_filter1d(values, 2) for name, values in motion.items()}

    hop, window = 220, 551
    starts = np.arange(0, len(audio) - window + 1, hop)
    audio_times = (starts + (window - 1) / 2) / audio_rate
    rms = np.asarray([np.sqrt(np.mean(audio[start : start + window] ** 2)) for start in starts])
    frequencies, flux_times, spectrum = signal.stft(
        audio,
        audio_rate,
        nperseg=window,
        noverlap=window - hop,
        nfft=1024,
        boundary=None,
        padded=False,
    )
    del frequencies
    spectral_flux = np.r_[0.0, np.maximum(0, np.diff(np.abs(spectrum), axis=1)).sum(axis=0)]
    audio_proxies = {
        "rms": (audio_times, gaussian_filter1d(rms, 2)),
        "spectral_flux": (flux_times, gaussian_filter1d(spectral_flux, 2)),
    }

    offsets = np.linspace(-1.0, 1.0, 401)
    metric_values: dict[str, np.ndarray] = {}
    overlap_counts: dict[str, np.ndarray] = {}
    for motion_name, motion_values in motion.items():
        for audio_name, (times, values) in audio_proxies.items():
            name = f"{motion_name}__{audio_name}"
            scores = [
                normalized_overlap_correlation(video_times, motion_values, times, values, offset)
                for offset in offsets
            ]
            metric_values[name] = np.asarray([score for score, _ in scores])
            overlap_counts[name] = np.asarray([count for _, count in scores])

    intervals = parse_word_intervals(args.textgrid)
    speech = np.zeros_like(audio_times, dtype=np.float64)
    for start, end, label in intervals:
        if label.strip():
            speech[(audio_times >= start) & (audio_times < end)] = 1
    log_rms = gaussian_filter1d(np.log(rms + 1e-8), 2)
    textgrid_scores = [
        normalized_overlap_correlation(audio_times, speech, audio_times, log_rms, offset)
        for offset in offsets
    ]
    textgrid_name = "textgrid_speech__audio_log_rms"
    metric_values[textgrid_name] = np.asarray([score for score, _ in textgrid_scores])
    overlap_counts[textgrid_name] = np.asarray([count for _, count in textgrid_scores])

    landscape = []
    for index, offset in enumerate(offsets):
        landscape.append(
            {
                "feature_time_offset_seconds": round(float(offset), 6),
                "correlations": {
                    name: round(float(values[index]), 6) for name, values in metric_values.items()
                },
                "overlap_counts": {
                    name: int(values[index]) for name, values in overlap_counts.items()
                },
            }
        )
    peaks = {name: peak_summary(offsets, values) for name, values in metric_values.items()}

    textgrid_local_anchors = []
    local_offsets = np.linspace(-0.3, 0.3, 121)
    for start, end in ((0.0, 8.0), (8.0, 16.0), (16.0, 24.0), (24.0, 33.0)):
        mask = (audio_times >= start) & (audio_times < end)
        local_scores = [
            normalized_overlap_correlation(
                audio_times[mask], speech[mask], audio_times, log_rms, offset
            )
            for offset in local_offsets
        ]
        values = np.asarray([score for score, _ in local_scores])
        summary = peak_summary(local_offsets, values)
        summary["window_seconds"] = [start, end]
        summary["center_seconds"] = (start + end) / 2
        summary["overlap_count_at_peak"] = int(local_scores[int(np.nanargmax(values))][1])
        summary["informative_for_drift"] = bool(
            summary["best_correlation"] >= 0.5 and summary["distant_peak_margin"] >= 0.03
        )
        textgrid_local_anchors.append(summary)
    informative_anchors = [
        anchor for anchor in textgrid_local_anchors if anchor["informative_for_drift"]
    ]
    textgrid_drift_fit = None
    if len(informative_anchors) >= 3:
        centers = np.asarray([anchor["center_seconds"] for anchor in informative_anchors])
        lags = np.asarray([anchor["best_offset_seconds"] for anchor in informative_anchors])
        slope, intercept = np.polyfit(centers, lags, 1)
        textgrid_drift_fit = {
            "slope_seconds_per_second": float(slope),
            "intercept_seconds": float(intercept),
            "max_fit_residual_seconds": float(np.max(np.abs(lags - (slope * centers + intercept)))),
        }

    nominal_step = float(np.median(np.diff(video_times)))
    fit = np.polyfit(np.arange(len(video_times)), video_times, 1)
    residual = video_times - np.polyval(fit, np.arange(len(video_times)))
    skip_side_data = [
        side
        for packet in packets
        for side in packet.get("side_data_list", [])
        if side.get("side_data_type") == "Skip Samples"
    ]
    if len(skip_side_data) != 2:
        raise RuntimeError(
            f"Expected initial skip and final discard side data, got {skip_side_data}"
        )
    initial_side = next(side for side in skip_side_data if int(side["skip_samples"]) > 0)
    final_side = next(side for side in skip_side_data if int(side["discard_padding"]) > 0)
    skip_samples = int(initial_side["skip_samples"])
    discard_padding = int(final_side["discard_padding"])
    first_audio_packet = packets[0]
    decoded_frame_samples = np.asarray([int(frame["nb_samples"]) for frame in audio_frames])
    decoded_frame_pts = np.asarray([int(frame["pts"]) for frame in audio_frames])
    if decoded_frame_samples.sum() != len(audio):
        raise RuntimeError("Decoded audio frame sample counts do not equal emitted sample count")
    if not np.array_equal(
        decoded_frame_pts[1:], decoded_frame_pts[:-1] + decoded_frame_samples[:-1]
    ):
        raise RuntimeError("Decoded audio frame PTS are not sample-contiguous")
    if decoded_frame_pts[0] != 0 or decoded_frame_pts[-1] + decoded_frame_samples[-1] != len(audio):
        raise RuntimeError("Decoded audio frames do not span exactly [0, emitted_sample_count)")
    source_hash = hashlib.sha256(args.media.read_bytes()).hexdigest()
    conv_config = torchaudio.pipelines.HUBERT_LARGE._params["extractor_conv_layer_config"]
    conv_kernel = [layer[1] for layer in conv_config]
    conv_stride = [layer[2] for layer in conv_config]
    if conv_kernel != [10, 3, 3, 3, 3, 2, 2] or conv_stride != [5, 2, 2, 2, 2, 2, 2]:
        raise RuntimeError(f"Unexpected installed HuBERT convolution config: {conv_config}")
    audio_16k = np.frombuffer(
        run(
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(args.media),
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "f32le",
            "-",
        ),
        dtype="<f4",
    )
    resampled_count = len(audio_16k)
    hubert_stride_samples, hubert_receptive_samples = 320, 400
    hubert_count = (resampled_count - hubert_receptive_samples) // hubert_stride_samples + 1
    hubert_centers = hubert_feature_center_timestamps(hubert_count)
    hubert_center, hubert_last_center = float(hubert_centers[0]), float(hubert_centers[-1])
    supported_frames = np.flatnonzero(
        (video_times >= hubert_center) & (video_times <= hubert_last_center)
    )
    mapped_audio_samples = audio_sample_indices(video_times, audio_rate)
    audio_packet_pts = np.asarray([int(packet["pts"]) for packet in packets], dtype=np.int64)
    audio_packet_steps = np.diff(audio_packet_pts)

    inventory = {
        "dataset_identity": "local 75-Speaker-layout source candidate (not legacy USC-TIMIT)",
        "utterance_id": "sub061_2drt_17_topic1",
        "source_media_path": str(args.media.resolve()),
        "source_release_chain_verified": False,
        "source_media_sha256": source_hash,
        "tool_versions": {
            "ffmpeg": ffmpeg_version,
            "ffprobe": ffprobe_version,
        },
        "decode_commands": {
            "video": "ffmpeg -v error -i INPUT -map 0:v:0 -pix_fmt gray -f rawvideo -",
            "audio": f"ffmpeg -v error -i INPUT -map 0:a:0 -ac 1 -ar {audio_rate} -f f32le -",
            "audio_16k": "ffmpeg -v error -i INPUT -map 0:a:0 -ac 1 -ar 16000 -f f32le -",
        },
        "video": {
            "codec": video_stream["codec_name"],
            "frame_count": len(video),
            "width": width,
            "height": height,
            "time_base": video_stream["time_base"],
            "average_frame_rate_rational": video_stream["avg_frame_rate"],
            "start_time_seconds": float(video_stream["start_time"]),
            "stream_duration_seconds": float(video_stream["duration"]),
            "first_pts_seconds": float(video_times[0]),
            "last_pts_seconds": float(video_times[-1]),
            "pts_integer_step": int(np.median(np.diff(video_pts))),
            "median_pts_step_seconds": nominal_step,
        },
        "audio": {
            "codec": audio_stream["codec_name"],
            "sample_rate_hz": audio_rate,
            "channels": int(audio_stream["channels"]),
            "time_base": audio_stream["time_base"],
            "stream_start_time_seconds": float(audio_stream["start_time"]),
            "stream_duration_seconds": float(audio_stream["duration"]),
            "initial_padding_samples": int(audio_stream.get("initial_padding", 0)),
            "first_packet_pts_seconds": float(first_audio_packet["pts_time"]),
            "first_packet_skip_samples": skip_samples,
            "final_packet_discard_padding_samples": discard_padding,
            "decoded_audio_frame_count": len(audio_frames),
            "decoded_audio_frame_sample_sum": int(decoded_frame_samples.sum()),
            "first_decoded_frame_pts_samples": int(decoded_frame_pts[0]),
            "last_decoded_frame_pts_samples": int(decoded_frame_pts[-1]),
            "last_decoded_frame_nb_samples": int(decoded_frame_samples[-1]),
            "decoded_sample_count": len(audio),
            "decoded_duration_seconds": len(audio) / audio_rate,
            "decoded_sample_zero_container_time_seconds": 0.0,
            "packet_count": len(packets),
            "packet_pts_step_samples_unique": sorted(set(audio_packet_steps.tolist())),
            "mri_first_last_mapped_sample_indices": [
                int(mapped_audio_samples[0]),
                int(mapped_audio_samples[-1]),
            ],
        },
        "textgrid": {
            "path": str(args.textgrid.relative_to(PROJECT_ROOT)),
            "encoding": "UTF-16",
            "word_intervals": len(intervals),
            "xmin_seconds": intervals[0][0],
            "xmax_seconds": intervals[-1][1],
        },
        "acquisition_provenance": {
            "corpus_page": "https://sail.usc.edu/span/75speakers/",
            "dataset_paper": "https://doi.org/10.1038/s41597-021-00976-x",
            "acquisition_system_paper": "https://sail.usc.edu/span/pdfs/narayanan2014realtime.pdf",
            "synchronization_hardware_paper": "https://sail.usc.edu/span/pdfs/bresch2006synchronized.pdf",
            "evidence": "Primary sources document synchronized audio/rtMRI for the 75-speaker corpus and the shared-clock USC acquisition system. The local MP4 path/layout and content are consistent with that corpus, but no primary manifest proves this file's release/transcoding chain.",
            "local_file_claim_limit": "Do not treat the mounted MP4 as an authenticated official derivative without a primary release manifest.",
        },
    }
    diagnostics = {
        "gate2": "NO-GO",
        "gate2_criteria": {
            "verified_source_identity": "partial: corpus layout/content match but release manifest absent",
            "deterministic_decoder_and_pts_mapping": "pass",
            "aac_priming_resolved": "pass",
            "stream_origin_resolved": "pass: 0 seconds",
            "drift_bounded": "fail: container clocks are regular, but content anchors do not bound audio-to-MRI drift through the unknown transcode",
            "hubert_center_and_boundary_treatment": "pass",
            "weak_proxy_peak_required": False,
        },
        "gate_basis": "NO-GO because acquisition/container evidence gives a zero-offset candidate but informative content anchors cannot bound audio-to-MRI drift after an unverified transcode",
        "timestamp_contract": {
            "mri_frame_time": "ffprobe best_effort_timestamp_time for decoded frame i",
            "decoded_audio_sample_time": f"sample_index / {audio_rate}; ffmpeg applies AAC Skip Samples before emitted sample 0",
            "audio_sample_for_mri_frame": f"floor(frame_pts_seconds * {audio_rate} + 0.5)",
            "feature_time_offset_seconds": 0.0,
        },
        "legacy_metadata_correction": {
            "previous_audio_sample_count": 728064,
            "decoded_pts_trimmed_audio_sample_count": len(audio),
            "difference_samples": 728064 - len(audio),
            "interpretation": "legacy count included AAC frame padding; decoded PTS/Skip Samples timeline is authoritative",
            "previous_assumed_fps": 83.28,
            "exact_container_fps": int(video_stream["avg_frame_rate"].split("/")[0])
            / int(video_stream["avg_frame_rate"].split("/")[1]),
            "last_frame_timestamp_error_from_83_28_seconds": float(
                video_times[-1] - (len(video_times) - 1) / 83.28
            ),
        },
        "stream_clock_evidence": {
            "video_pts_affine_step_seconds": float(fit[0]),
            "video_pts_max_affine_residual_seconds": float(np.max(np.abs(residual))),
            "audio_packet_pts_step_samples_unique": sorted(set(audio_packet_steps.tolist())),
            "audio_and_video_stream_start_seconds": [
                float(audio_stream["start_time"]),
                float(video_stream["start_time"]),
            ],
            "audio_minus_video_stream_duration_seconds": float(audio_stream["duration"])
            - float(video_stream["duration"]),
            "drift_assessment": "video and decoded-audio PTS are internally regular, but PTS regularity alone cannot prove that the unverified transcode preserved relative audio/MRI timing",
        },
        "offset_diagnostics": peaks,
        "textgrid_audio_local_anchors": textgrid_local_anchors,
        "textgrid_audio_drift_fit": textgrid_drift_fit,
        "drift_evidence_limit": "Only one local TextGrid/audio window met the predeclared correlation>=0.5 and distant-peak-margin>=0.03 rule, so no slope was fit. TextGrid/audio anchors do not directly anchor MRI content in any case; motion proxies were ambiguous.",
        "textgrid_evidence": {
            "best_offset_seconds": peaks["textgrid_speech__audio_log_rms"]["best_offset_seconds"],
            "best_correlation": peaks["textgrid_speech__audio_log_rms"]["best_correlation"],
            "zero_offset_correlation": peaks["textgrid_speech__audio_log_rms"][
                "zero_offset_correlation"
            ],
            "speech_log_rms_mean": float(log_rms[speech.astype(bool)].mean()),
            "silence_log_rms_mean": float(log_rms[~speech.astype(bool)].mean()),
        },
        "hubert_mapping": {
            "resampling": f"decoded {audio_rate} Hz sample timeline deterministically resampled to 16000 Hz",
            "resampled_sample_count": resampled_count,
            "installed_torchaudio_version": torchaudio.__version__,
            "installed_bundle": "torchaudio.pipelines.HUBERT_LARGE",
            "installed_extractor_conv_layer_config": [list(layer) for layer in conv_config],
            "conv_kernel": conv_kernel,
            "conv_stride": conv_stride,
            "stride_samples": hubert_stride_samples,
            "stride_seconds": 0.02,
            "receptive_field_samples": hubert_receptive_samples,
            "first_feature_center_seconds": hubert_center,
            "feature_center_formula": "(feature_index*320 + 199.5) / 16000",
            "feature_count_for_this_audio": hubert_count,
            "last_feature_center_seconds": hubert_last_center,
            "supported_mri_frame_range_inclusive": [
                int(supported_frames[0]),
                int(supported_frames[-1]),
            ],
            "excluded_boundary_frames": [0, 1],
            "mapping": "interpolate feature vectors by their receptive-field center timestamps at exact MRI PTS; do not extrapolate",
            "supervised_extraction_contract": "HuBERTExtractor strict mode returns frame indices; extract_hubert_features.py stores *_hubert_mri_frame_indices.npy beside features and targets must use that slice",
        },
        "candidate_alignment": {
            "feature_time_offset_seconds": 0.0,
            "validation_resolution_seconds": 0.005,
            "textgrid_peak_difference_from_zero_seconds": peaks["textgrid_speech__audio_log_rms"][
                "best_offset_seconds"
            ],
            "interpretation": "TextGrid/audio peak is consistent with zero but does not validate MRI timing or drift",
        },
        "residual_uncertainty": [
            "Mounted MP4 release/transcoding chain is not authenticated by a primary manifest",
            "frame PTS denotes presentation time; release metadata does not expose reconstruction-window center",
            "motion/audio proxy peaks are low and multimodal because acoustic energy is not a direct motion measurement",
            "HuBERT boundary MRI frames 0-1 lack a centered feature and must be excluded",
        ],
    }
    sweep = {
        "status": "completed_no_go",
        "candidate_feature_time_offset_seconds": 0.0,
        "selected_feature_time_offset_seconds": None,
        "selection_basis": "zero is a container/acquisition candidate, not accepted because audio-to-MRI drift remains unbounded",
        "range_seconds": [-1.0, 1.0],
        "step_seconds": 0.005,
        "sign_convention": "audio/proxy query time = MRI PTS + feature_time_offset_seconds",
        "normalization": "per-offset z-normalization over temporal overlap only; no zero padding",
        "peaks": peaks,
        "landscape": landscape,
    }
    dump(args.output / "alignment_inventory.json", inventory)
    dump(args.output / "alignment_diagnostics.json", diagnostics)
    dump(args.output / "offset_sweep.json", sweep)

    plt.figure(figsize=(10, 6))
    for name in (
        "global_motion__rms",
        "mouth_roi_motion__rms",
        "mouth_roi_motion__spectral_flux",
        "dense_contour_displacement__spectral_flux",
        "textgrid_speech__audio_log_rms",
    ):
        plt.plot(offsets, metric_values[name], label=name)
    plt.axvline(0, color="black", linewidth=1, linestyle="--", label="container offset 0")
    plt.xlabel("audio query offset from MRI PTS (s)")
    plt.ylabel("overlap-only normalized correlation")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(args.output / "offset_sweep.png", dpi=150)
    plt.close()

    report = f"""# Phase 2A actual audio–MRI alignment report

## Source and provenance

The mounted file follows the USC 75-Speaker corpus layout and has SHA256
`{source_hash}`. Primary corpus/acquisition publications document synchronized
75-Speaker audio/rtMRI and USC shared-clock acquisition. No primary manifest was
found that authenticates this local MP4 or its AAC/H.264 transcoding chain, so
those publications are acquisition context—not proof that this derivative
preserved relative timing.

## Container and decoder timeline

Video has {len(video):,} frames at exact rate `{video_stream['avg_frame_rate']}`
with max affine PTS residual {np.max(np.abs(residual)):.3e} s. AAC begins with
{skip_samples} skip samples and ends with {discard_padding} discard-padding
samples. ffmpeg emits {len(audio):,} samples in {len(audio_frames)} contiguous
decoded frames from sample PTS 0; decoded frame `nb_samples` sum exactly matches
the emitted count. Tool versions and commands are recorded in the inventory.

## Offset and drift diagnostics

The corrected sign convention correlates `label(t)` with
`log_RMS(t + feature_time_offset)`, using overlap-only z-normalization. The
TextGrid/audio global peak is
{peaks['textgrid_speech__audio_log_rms']['best_offset_seconds']:+.3f} s
(r={peaks['textgrid_speech__audio_log_rms']['best_correlation']:.3f}; zero
r={peaks['textgrid_speech__audio_log_rms']['zero_offset_correlation']:.3f}).
Every sweep row stores its overlap count. Only {len(informative_anchors)} of four
local TextGrid/audio windows met the predeclared informativeness rule, so no
drift slope was fit. TextGrid anchors audio, not MRI. Global/ROI/contour motion
peaks remain multimodal and are not accepted as clocks. Consequently content
evidence cannot bound audio-to-MRI drift through the unverified transcode.

## Deterministic conditional mapping

Conditional on candidate offset 0, MRI frame `i` uses its exact presentation
timestamp `t_i`; decoded sample index is
`floor(t_i * {audio_rate} + 0.5)`. HuBERT is resampled to 16 kHz and feature
`j` is centered at `(j*320 + 199.5)/16000`. Supervised extraction now returns
and stores exact supported MRI frame indices so targets use the identical slice.
For this utterance frames 0–1 are outside centered support and frames
{supported_frames[0]}–{supported_frames[-1]} are retained. Audio-only inference
uses the same deterministic center/support rule.

## Gate 2: NO-GO

Decoder/PTS mapping, AAC priming/discard, and HuBERT boundaries are resolved.
Gate 2 nevertheless remains **NO-GO** because the mounted derivative's release
chain is unverified and no informative MRI-content anchors bound relative drift.
Offset 0 is a reproducible candidate, not an accepted alignment. Do not run the
tiny overfit or full AAI training until an authenticated source/manifest or an
independent audio–MRI synchronization anchor closes this gap.
"""
    (args.output / "alignment_report.md").write_text(report, encoding="utf-8")
    print(
        f"decoded video={len(video)} audio={len(audio)} gate=NO-GO "
        f"textgrid_offset={peaks['textgrid_speech__audio_log_rms']['best_offset_seconds']:+.3f}s"
    )


if __name__ == "__main__":
    main()
