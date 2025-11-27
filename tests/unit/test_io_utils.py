"""
Unit tests for src/utils/io_utils.py
"""

import pytest
import numpy as np
from pathlib import Path

from src.utils.io_utils import (
    load_mri_frame,
    save_audio,
    load_audio,
    save_hdf5,
    load_hdf5,
    save_json,
    load_json,
    save_numpy,
    load_numpy,
    normalize_array,
    ensure_directory,
    list_files,
)


class TestNormalizeArray:
    """Tests for normalize_array function."""

    def test_normalize_to_01(self):
        """Test normalization to [0, 1] range."""
        arr = np.array([0, 50, 100], dtype=np.float32)
        normalized = normalize_array(arr, 0.0, 1.0)

        assert normalized[0] == pytest.approx(0.0)
        assert normalized[1] == pytest.approx(0.5)
        assert normalized[2] == pytest.approx(1.0)

    def test_normalize_to_custom_range(self):
        """Test normalization to custom range."""
        arr = np.array([0, 50, 100], dtype=np.float32)
        normalized = normalize_array(arr, -1.0, 1.0)

        assert normalized[0] == pytest.approx(-1.0)
        assert normalized[1] == pytest.approx(0.0)
        assert normalized[2] == pytest.approx(1.0)

    def test_normalize_constant_array(self):
        """Test normalization of constant array."""
        arr = np.array([5, 5, 5], dtype=np.float32)
        normalized = normalize_array(arr, 0.0, 1.0)

        # Should return middle value (0.5)
        assert np.all(normalized == 0.5)

    def test_normalize_output_dtype(self):
        """Test that output is float32."""
        arr = np.array([0, 100], dtype=np.int32)
        normalized = normalize_array(arr)

        assert normalized.dtype == np.float32


class TestAudioOperations:
    """Tests for audio I/O operations."""

    def test_save_and_load_audio(self, tmp_path):
        """Test saving and loading audio."""
        audio_path = tmp_path / "test_audio.wav"

        # Create test audio
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        # Save
        save_audio(audio_path, audio, sr)
        assert audio_path.exists()

        # Load
        loaded_audio, loaded_sr = load_audio(audio_path)

        assert loaded_sr == sr
        assert loaded_audio.shape == audio.shape
        assert np.allclose(loaded_audio, audio, atol=1e-4)

    def test_load_nonexistent_audio(self, tmp_path):
        """Test loading nonexistent audio file."""
        audio_path = tmp_path / "nonexistent.wav"

        with pytest.raises(FileNotFoundError):
            load_audio(audio_path)

    def test_save_creates_parent_directory(self, tmp_path):
        """Test that save_audio creates parent directories."""
        audio_path = tmp_path / "nested" / "dir" / "audio.wav"

        audio = np.sin(np.linspace(0, 1, 16000)).astype(np.float32)
        save_audio(audio_path, audio, 16000)

        assert audio_path.exists()


class TestHDF5Operations:
    """Tests for HDF5 I/O operations."""

    def test_save_and_load_hdf5(self, tmp_path):
        """Test saving and loading HDF5."""
        hdf5_path = tmp_path / "test.h5"

        # Create test data
        data = {
            "array1": np.random.randn(100, 10).astype(np.float32),
            "array2": np.random.randn(50, 20).astype(np.float32),
            "scalar_int": 42,
            "scalar_float": 3.14,
            "scalar_str": "test",
        }

        # Save
        save_hdf5(data, hdf5_path)
        assert hdf5_path.exists()

        # Load
        loaded_data = load_hdf5(hdf5_path)

        # Arrays should be loaded normally
        assert np.allclose(loaded_data["array1"], data["array1"])
        assert np.allclose(loaded_data["array2"], data["array2"])
        # Scalars are stored as attributes and loaded back
        assert loaded_data["scalar_int"] == data["scalar_int"]
        assert loaded_data["scalar_float"] == pytest.approx(data["scalar_float"], abs=1e-5)
        assert loaded_data["scalar_str"] == data["scalar_str"]

    def test_load_specific_keys(self, tmp_path):
        """Test loading specific keys from HDF5."""
        hdf5_path = tmp_path / "test.h5"

        data = {
            "array1": np.random.randn(10).astype(np.float32),
            "array2": np.random.randn(20).astype(np.float32),
            "array3": np.random.randn(30).astype(np.float32),
        }

        save_hdf5(data, hdf5_path)

        # Load only specific keys
        loaded_data = load_hdf5(hdf5_path, keys=["array1", "array3"])

        assert "array1" in loaded_data
        assert "array3" in loaded_data
        assert "array2" not in loaded_data

    def test_load_nonexistent_hdf5(self, tmp_path):
        """Test loading nonexistent HDF5 file."""
        hdf5_path = tmp_path / "nonexistent.h5"

        with pytest.raises(FileNotFoundError):
            load_hdf5(hdf5_path)


class TestJSONOperations:
    """Tests for JSON I/O operations."""

    def test_save_and_load_json(self, tmp_path):
        """Test saving and loading JSON."""
        json_path = tmp_path / "test.json"

        data = {
            "name": "test",
            "value": 42,
            "nested": {"key": "value"},
            "list": [1, 2, 3],
        }

        # Save
        save_json(data, json_path)
        assert json_path.exists()

        # Load
        loaded_data = load_json(json_path)

        assert loaded_data == data

    def test_save_json_with_numpy_types(self, tmp_path):
        """Test saving JSON with numpy types."""
        json_path = tmp_path / "test_numpy.json"

        data = {
            "np_int": np.int32(42),
            "np_float": np.float32(3.14),
            "np_array": np.array([1, 2, 3]),
        }

        # Save (should convert numpy types)
        save_json(data, json_path)

        # Load and verify
        loaded_data = load_json(json_path)

        assert loaded_data["np_int"] == 42
        assert loaded_data["np_float"] == pytest.approx(3.14, abs=1e-5)
        assert loaded_data["np_array"] == [1, 2, 3]

    def test_load_nonexistent_json(self, tmp_path):
        """Test loading nonexistent JSON file."""
        json_path = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            load_json(json_path)


class TestNumpyOperations:
    """Tests for NumPy I/O operations."""

    def test_save_and_load_numpy(self, tmp_path):
        """Test saving and loading NumPy arrays."""
        npy_path = tmp_path / "test.npy"

        # Create test array
        arr = np.random.randn(100, 50).astype(np.float32)

        # Save
        save_numpy(arr, npy_path)
        assert npy_path.exists()

        # Load
        loaded_arr = load_numpy(npy_path)

        assert np.allclose(loaded_arr, arr)
        assert loaded_arr.dtype == arr.dtype

    def test_load_nonexistent_numpy(self, tmp_path):
        """Test loading nonexistent numpy file."""
        npy_path = tmp_path / "nonexistent.npy"

        with pytest.raises(FileNotFoundError):
            load_numpy(npy_path)


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_ensure_directory_creates(self, tmp_path):
        """Test that ensure_directory creates directory."""
        new_dir = tmp_path / "new" / "nested" / "dir"

        result = ensure_directory(new_dir)

        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == new_dir

    def test_ensure_directory_existing(self, tmp_path):
        """Test ensure_directory with existing directory."""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()

        result = ensure_directory(existing_dir)

        assert existing_dir.exists()
        assert result == existing_dir

    def test_list_files_non_recursive(self, tmp_path):
        """Test listing files non-recursively."""
        # Create test files
        (tmp_path / "file1.txt").touch()
        (tmp_path / "file2.txt").touch()
        (tmp_path / "file3.py").touch()
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "file4.txt").touch()

        # List all files
        files = list_files(tmp_path, "*", recursive=False)
        assert len(files) == 3

        # List only .txt files
        txt_files = list_files(tmp_path, "*.txt", recursive=False)
        assert len(txt_files) == 2

    def test_list_files_recursive(self, tmp_path):
        """Test listing files recursively."""
        # Create nested structure
        (tmp_path / "file1.txt").touch()
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "file2.txt").touch()
        subsubdir = subdir / "subsubdir"
        subsubdir.mkdir()
        (subsubdir / "file3.txt").touch()

        # List recursively
        files = list_files(tmp_path, "*.txt", recursive=True)
        assert len(files) == 3

    def test_list_files_nonexistent_dir(self, tmp_path):
        """Test listing files in nonexistent directory."""
        nonexistent = tmp_path / "nonexistent"

        files = list_files(nonexistent)
        assert len(files) == 0
