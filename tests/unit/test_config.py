"""
Unit tests for src/utils/config.py
"""

import pytest
from pathlib import Path
import tempfile

from src.utils.config import PreprocessConfig, load_config, create_default_config


class TestPreprocessConfig:
    """Tests for PreprocessConfig class."""

    def test_default_values(self):
        """Test that default configuration values are correct."""
        config = PreprocessConfig(
            raw_data_path="/test/raw",
            output_path="/test/output",
        )

        assert config.raw_data_path == "/test/raw"
        assert config.output_path == "/test/output"
        assert config.mri_target_size == (256, 256)
        assert config.mri_fps == 50.0
        assert config.audio_sr == 16000
        assert config.n_mfcc == 13
        assert config.n_mels == 80
        assert config.pca_components == 10
        assert config.train_ratio == 0.7
        assert config.val_ratio == 0.15
        assert config.test_ratio == 0.15

    def test_custom_values(self):
        """Test custom configuration values."""
        config = PreprocessConfig(
            raw_data_path="/custom/raw",
            output_path="/custom/output",
            mri_fps=60.0,
            audio_sr=22050,
            pca_components=20,
        )

        assert config.mri_fps == 60.0
        assert config.audio_sr == 22050
        assert config.pca_components == 20

    def test_validation_mri_size(self):
        """Test MRI size validation."""
        # Valid size
        config = PreprocessConfig(
            raw_data_path="/test", output_path="/test", mri_target_size=(512, 512)
        )
        assert config.mri_target_size == (512, 512)

        # Invalid size (negative)
        with pytest.raises(ValueError, match="must be positive"):
            PreprocessConfig(
                raw_data_path="/test", output_path="/test", mri_target_size=(-1, 256)
            )

    def test_validation_ratios(self):
        """Test dataset ratio validation."""
        # Valid ratios
        config = PreprocessConfig(
            raw_data_path="/test",
            output_path="/test",
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
        )
        assert config.train_ratio == 0.6

        # Invalid ratio (> 1)
        with pytest.raises(ValueError, match="between 0 and 1"):
            PreprocessConfig(
                raw_data_path="/test", output_path="/test", train_ratio=1.5
            )

        # Invalid ratio (negative)
        with pytest.raises(ValueError, match="between 0 and 1"):
            PreprocessConfig(
                raw_data_path="/test", output_path="/test", val_ratio=-0.1
            )

    def test_validation_positive_int(self):
        """Test positive integer validation."""
        # Invalid PCA components
        with pytest.raises(ValueError, match="must be positive"):
            PreprocessConfig(
                raw_data_path="/test", output_path="/test", pca_components=0
            )

        # Invalid autoencoder latent dim
        with pytest.raises(ValueError, match="must be positive"):
            PreprocessConfig(
                raw_data_path="/test", output_path="/test", autoencoder_latent_dim=-5
            )

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = PreprocessConfig(
            raw_data_path="/test/raw", output_path="/test/output", mri_fps=60.0
        )

        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["raw_data_path"] == "/test/raw"
        assert config_dict["output_path"] == "/test/output"
        assert config_dict["mri_fps"] == 60.0


class TestConfigFileOperations:
    """Tests for configuration file operations."""

    def test_save_and_load_yaml(self, tmp_path):
        """Test saving and loading YAML configuration."""
        config_path = tmp_path / "test_config.yaml"

        # Create and save config
        config = PreprocessConfig(
            raw_data_path="/test/raw",
            output_path="/test/output",
            mri_fps=60.0,
            pca_components=15,
        )
        config.to_yaml(config_path)

        # Check file exists
        assert config_path.exists()

        # Load and verify
        loaded_config = PreprocessConfig.from_yaml(config_path)
        assert loaded_config.raw_data_path == "/test/raw"
        assert loaded_config.output_path == "/test/output"
        assert loaded_config.mri_fps == 60.0
        assert loaded_config.pca_components == 15

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading from nonexistent file raises error."""
        config_path = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            PreprocessConfig.from_yaml(config_path)

    def test_load_config_function(self, tmp_path):
        """Test load_config convenience function."""
        config_path = tmp_path / "test_config.yaml"

        config = PreprocessConfig(
            raw_data_path="/test/raw", output_path="/test/output"
        )
        config.to_yaml(config_path)

        loaded = load_config(config_path)
        assert loaded.raw_data_path == "/test/raw"

    def test_create_default_config(self, tmp_path):
        """Test creating default configuration file."""
        config_path = tmp_path / "default_config.yaml"

        config = create_default_config(config_path)

        # Check file was created
        assert config_path.exists()

        # Check default values
        assert config.mri_fps == 50.0
        assert config.audio_sr == 16000
        assert config.pca_components == 10

    def test_save_creates_parent_directory(self, tmp_path):
        """Test that saving creates parent directories if needed."""
        nested_path = tmp_path / "nested" / "dir" / "config.yaml"

        config = PreprocessConfig(
            raw_data_path="/test/raw", output_path="/test/output"
        )
        config.to_yaml(nested_path)

        assert nested_path.exists()
        assert nested_path.parent.exists()
