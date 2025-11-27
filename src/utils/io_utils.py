"""
Project Sullivan - I/O Utilities

This module provides file I/O operations for various data formats.
"""

import json
from pathlib import Path
from typing import Any, Optional

import cv2
import h5py
import nibabel as nib
import numpy as np
import pydicom
import soundfile as sf


# ============================================================================
# MRI Loading
# ============================================================================


def load_mri_frame(path: str | Path, normalize: bool = True) -> np.ndarray:
    """
    Load a single MRI frame from various formats.

    Supports: PNG, JPG, DICOM (.dcm), NIfTI (.nii, .nii.gz)

    Args:
        path: Path to MRI frame file
        normalize: Whether to normalize to [0, 1] range

    Returns:
        np.ndarray: MRI frame as float32 array

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file format is not supported
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MRI frame not found: {path}")

    suffix = path.suffix.lower()

    # Load based on format
    if suffix in [".png", ".jpg", ".jpeg", ".bmp"]:
        frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise ValueError(f"Failed to load image: {path}")
        frame = frame.astype(np.float32)

    elif suffix == ".dcm":
        dcm = pydicom.dcmread(str(path))
        frame = dcm.pixel_array.astype(np.float32)

    elif suffix in [".nii", ".gz"]:
        nii = nib.load(str(path))
        frame = nii.get_fdata().astype(np.float32)
        # Handle 3D volume by taking middle slice
        if frame.ndim == 3:
            frame = frame[:, :, frame.shape[2] // 2]

    else:
        raise ValueError(f"Unsupported MRI format: {suffix}")

    # Normalize if requested
    if normalize:
        frame = normalize_array(frame)

    return frame


def load_mri_sequence(
    directory: str | Path,
    pattern: str = "*.png",
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Load a sequence of MRI frames from a directory.

    Args:
        directory: Directory containing MRI frames
        pattern: Glob pattern for frame files (default: "*.png")
        start_idx: Starting frame index
        end_idx: Ending frame index (None = all frames)
        normalize: Whether to normalize frames

    Returns:
        np.ndarray: MRI sequence (T, H, W) as float32 array

    Raises:
        FileNotFoundError: If directory does not exist
        ValueError: If no frames found
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Find all matching files
    frame_files = sorted(directory.glob(pattern))
    if not frame_files:
        raise ValueError(f"No frames found with pattern '{pattern}' in {directory}")

    # Slice if needed
    if end_idx is not None:
        frame_files = frame_files[start_idx:end_idx]
    else:
        frame_files = frame_files[start_idx:]

    # Load frames
    frames = []
    for frame_file in frame_files:
        frame = load_mri_frame(frame_file, normalize=normalize)
        frames.append(frame)

    return np.array(frames, dtype=np.float32)


def load_mri_from_video(
    video_path: str | Path, normalize: bool = True
) -> np.ndarray:
    """
    Load MRI frames from video file (MP4, AVI, etc.).

    Args:
        video_path: Path to video file
        normalize: Whether to normalize frames

    Returns:
        np.ndarray: MRI sequence (T, H, W) as float32 array

    Raises:
        FileNotFoundError: If video file does not exist
        ValueError: If video cannot be opened
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale if needed
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame = frame.astype(np.float32)
        if normalize:
            frame = normalize_array(frame)

        frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError(f"No frames extracted from video: {video_path}")

    return np.array(frames, dtype=np.float32)


# ============================================================================
# Audio Loading
# ============================================================================


def load_audio(
    path: str | Path, sr: Optional[int] = None
) -> tuple[np.ndarray, int]:
    """
    Load audio file with optional resampling.

    Args:
        path: Path to audio file (WAV, MP3, etc.)
        sr: Target sample rate (None = original rate)

    Returns:
        tuple: (audio waveform, sample rate)
            - audio: np.ndarray of shape (N,) as float32
            - sr: int, sample rate in Hz

    Raises:
        FileNotFoundError: If audio file does not exist
        ValueError: If audio cannot be loaded
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # Check if it's a video file (extract audio from video)
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    is_video = path.suffix.lower() in video_extensions

    if is_video:
        # Extract audio from video using opencv
        import cv2
        import tempfile
        import os

        # Use ffmpeg via subprocess to extract audio
        import subprocess

        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Extract audio using ffmpeg
            subprocess.run(
                ['ffmpeg', '-i', str(path), '-vn', '-acodec', 'pcm_s16le',
                 '-ar', '22050', '-ac', '1', tmp_path, '-y'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )

            # Load the extracted audio
            audio, original_sr = sf.read(tmp_path, dtype="float32")

        except Exception as e:
            raise ValueError(f"Failed to extract audio from video {path}: {e}")
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    else:
        # Load audio file directly
        try:
            audio, original_sr = sf.read(str(path), dtype="float32")
        except Exception as e:
            raise ValueError(f"Failed to load audio from {path}: {e}")

    # Convert stereo to mono if needed
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Resample if needed
    if sr is not None and sr != original_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=original_sr, target_sr=sr)
        original_sr = sr

    return audio.astype(np.float32), original_sr


def save_audio(
    path: str | Path, audio: np.ndarray, sr: int
) -> None:
    """
    Save audio to WAV file.

    Args:
        path: Output path
        audio: Audio waveform (N,) array
        sr: Sample rate in Hz
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sf.write(str(path), audio, sr)


# ============================================================================
# HDF5 Operations
# ============================================================================


def save_hdf5(
    data: dict[str, np.ndarray | Any],
    filepath: str | Path,
    compression: str = "gzip",
    compression_opts: int = 4,
) -> None:
    """
    Save data to HDF5 file with compression.

    Args:
        data: Dictionary of data to save
        filepath: Output HDF5 file path
        compression: Compression algorithm ('gzip', 'lzf', None)
        compression_opts: Compression level (0-9 for gzip)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filepath, "w") as f:
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                f.create_dataset(
                    key,
                    data=value,
                    compression=compression,
                    compression_opts=compression_opts,
                )
            elif isinstance(value, (int, float, str, bool)):
                f.attrs[key] = value
            else:
                # Convert to numpy array if possible
                try:
                    value_array = np.array(value)
                    f.create_dataset(
                        key,
                        data=value_array,
                        compression=compression,
                        compression_opts=compression_opts,
                    )
                except (ValueError, TypeError):
                    # Store as string if conversion fails
                    f.attrs[key] = str(value)


def load_hdf5(
    filepath: str | Path, keys: Optional[list[str]] = None
) -> dict[str, np.ndarray | Any]:
    """
    Load data from HDF5 file.

    Args:
        filepath: HDF5 file path
        keys: List of keys to load (None = load all)

    Returns:
        dict: Dictionary of loaded data

    Raises:
        FileNotFoundError: If file does not exist
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"HDF5 file not found: {filepath}")

    data = {}

    with h5py.File(filepath, "r") as f:
        if keys is None:
            # Load all datasets
            for key in f.keys():
                data[key] = f[key][:]
            # Load all attributes
            for key, value in f.attrs.items():
                data[key] = value
        else:
            # Load specific keys from both datasets and attributes
            for key in keys:
                if key in f:
                    data[key] = f[key][:]
                elif key in f.attrs:
                    data[key] = f.attrs[key]

    return data


# ============================================================================
# JSON Operations
# ============================================================================


def save_json(data: dict, filepath: str | Path, indent: int = 2) -> None:
    """
    Save dictionary to JSON file.

    Args:
        data: Dictionary to save
        filepath: Output JSON file path
        indent: Indentation level for pretty printing
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj

    data_converted = convert_numpy(data)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data_converted, f, indent=indent, ensure_ascii=False)


def load_json(filepath: str | Path) -> dict:
    """
    Load dictionary from JSON file.

    Args:
        filepath: JSON file path

    Returns:
        dict: Loaded data

    Raises:
        FileNotFoundError: If file does not exist
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# NumPy Operations
# ============================================================================


def save_numpy(data: np.ndarray, filepath: str | Path) -> None:
    """
    Save numpy array to .npy file.

    Args:
        data: Numpy array
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.save(filepath, data)


def load_numpy(filepath: str | Path) -> np.ndarray:
    """
    Load numpy array from .npy file.

    Args:
        filepath: .npy file path

    Returns:
        np.ndarray: Loaded array

    Raises:
        FileNotFoundError: If file does not exist
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Numpy file not found: {filepath}")
    return np.load(filepath)


# ============================================================================
# Utility Functions
# ============================================================================


def normalize_array(arr: np.ndarray, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
    """
    Normalize array to [min_val, max_val] range.

    Args:
        arr: Input array
        min_val: Minimum value of output range
        max_val: Maximum value of output range

    Returns:
        np.ndarray: Normalized array
    """
    arr_min, arr_max = arr.min(), arr.max()

    if arr_max == arr_min:
        # Constant array, return middle value
        return np.full_like(arr, (min_val + max_val) / 2, dtype=np.float32)

    # Normalize to [min_val, max_val]
    normalized = (arr - arr_min) / (arr_max - arr_min)
    normalized = normalized * (max_val - min_val) + min_val

    return normalized.astype(np.float32)


def ensure_directory(directory: str | Path) -> Path:
    """
    Ensure directory exists, create if not.

    Args:
        directory: Directory path

    Returns:
        Path: Directory path
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def list_files(directory: str | Path, pattern: str = "*", recursive: bool = False) -> list[Path]:
    """
    List files in directory matching pattern.

    Args:
        directory: Directory to search
        pattern: Glob pattern
        recursive: Whether to search recursively

    Returns:
        list[Path]: List of matching file paths (directories excluded)
    """
    directory = Path(directory)
    if not directory.exists():
        return []

    if recursive:
        files = [f for f in directory.rglob(pattern) if f.is_file()]
    else:
        files = [f for f in directory.glob(pattern) if f.is_file()]

    return sorted(files)
