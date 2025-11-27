"""
Project Sullivan - Configuration Management

This module provides configuration loading and validation using Pydantic.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class PreprocessConfig(BaseModel):
    """Configuration for preprocessing pipeline."""

    # Data paths
    raw_data_path: str = Field(..., description="Path to raw USC-TIMIT data")
    output_path: str = Field(..., description="Path to output processed data")

    # MRI settings
    mri_target_size: tuple[int, int] = Field(
        default=(256, 256), description="Target size for MRI frames (H, W)"
    )
    mri_fps: float = Field(default=50.0, description="MRI frame rate (frames per second)")
    denoise_method: str = Field(
        default="gaussian", description="MRI denoising method (gaussian, bilateral, nlm, bm3d)"
    )
    mri_denoise_method: str = Field(
        default="gaussian", description="MRI spatial denoising method (gaussian, bilateral, nlm)"
    )
    mri_denoise_sigma: float = Field(
        default=1.0, description="MRI spatial denoising sigma"
    )
    mri_temporal_window: int = Field(
        default=3, description="MRI temporal median filter window size (must be odd)"
    )

    # Audio settings
    audio_sr: int = Field(default=16000, description="Audio sample rate (Hz)")
    audio_sample_rate: int = Field(default=22050, description="Target audio sample rate (Hz)")
    audio_denoise: bool = Field(default=True, description="Whether to denoise audio")

    # Alignment
    alignment_method: str = Field(
        default="cross_correlation", description="Audio-MRI alignment method"
    )

    # Segmentation
    segmentation_model: str = Field(
        default="unet_resnet50", description="Segmentation model architecture"
    )
    segmentation_pretrained: bool = Field(
        default=True, description="Whether to use pre-trained weights"
    )
    num_classes: int = Field(default=5, description="Number of segmentation classes")
    segmentation_batch_size: int = Field(default=8, description="Batch size for segmentation")
    segmentation_device: str = Field(default="cuda", description="Device for segmentation (cuda/cpu)")

    # Parameterization
    parameterization_method: str = Field(
        default="pca", description="Parameterization method (pca, autoencoder, both)"
    )
    pca_components: int = Field(default=10, description="Number of PCA components")
    autoencoder_latent_dim: int = Field(
        default=10, description="Autoencoder latent dimension"
    )
    temporal_smoothing: bool = Field(
        default=True, description="Whether to apply temporal smoothing"
    )
    smoothing_method: str = Field(
        default="savgol", description="Smoothing method (savgol, gaussian, median)"
    )
    smoothing_window: int = Field(default=11, description="Smoothing window size")

    # Audio features
    n_mfcc: int = Field(default=13, description="Number of MFCC coefficients")
    n_mels: int = Field(default=80, description="Number of mel bands")
    hop_length: int = Field(default=160, description="Hop length for audio features (samples)")

    # Dataset building
    train_ratio: float = Field(default=0.7, description="Training set ratio")
    val_ratio: float = Field(default=0.15, description="Validation set ratio")
    test_ratio: float = Field(default=0.15, description="Test set ratio")
    speaker_level_split: bool = Field(
        default=True, description="Whether to split at speaker level"
    )
    compression: str = Field(default="gzip", description="HDF5 compression algorithm")
    compression_level: int = Field(default=4, description="HDF5 compression level (0-9)")

    # Validation
    run_validation: bool = Field(
        default=True, description="Whether to run validation after each stage"
    )
    generate_visualizations: bool = Field(
        default=True, description="Whether to generate visualizations"
    )

    @field_validator("mri_target_size")
    @classmethod
    def validate_mri_size(cls, v: tuple[int, int]) -> tuple[int, int]:
        """Validate MRI target size."""
        if len(v) != 2:
            raise ValueError("mri_target_size must be a tuple of 2 integers")
        if any(x <= 0 for x in v):
            raise ValueError("mri_target_size values must be positive")
        return v

    @field_validator("train_ratio", "val_ratio", "test_ratio")
    @classmethod
    def validate_ratio(cls, v: float) -> float:
        """Validate dataset split ratios."""
        if not 0 < v < 1:
            raise ValueError("Ratio must be between 0 and 1")
        return v

    @field_validator("pca_components", "autoencoder_latent_dim")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """Validate positive integer values."""
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "PreprocessConfig":
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            PreprocessConfig: Loaded and validated configuration

        Raises:
            FileNotFoundError: If YAML file does not exist
            yaml.YAMLError: If YAML file is invalid
            pydantic.ValidationError: If configuration is invalid
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def to_yaml(self, yaml_path: str | Path) -> None:
        """
        Save configuration to YAML file.

        Args:
            yaml_path: Path to save YAML configuration file
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and ensure tuples are converted to lists for YAML
        config_dict = self.model_dump()
        # Convert mri_target_size tuple to list for YAML compatibility
        if isinstance(config_dict.get("mri_target_size"), tuple):
            config_dict["mri_target_size"] = list(config_dict["mri_target_size"])

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            dict: Configuration as dictionary
        """
        return self.model_dump()


def load_config(config_path: str | Path) -> PreprocessConfig:
    """
    Load configuration from file.

    This is a convenience function that wraps PreprocessConfig.from_yaml().

    Args:
        config_path: Path to configuration file (YAML)

    Returns:
        PreprocessConfig: Loaded configuration

    Example:
        >>> config = load_config("configs/preprocess.yaml")
        >>> print(config.mri_fps)
        50.0
    """
    return PreprocessConfig.from_yaml(config_path)


def create_default_config(output_path: str | Path) -> PreprocessConfig:
    """
    Create a default configuration and save to file.

    Args:
        output_path: Path to save default configuration

    Returns:
        PreprocessConfig: Default configuration

    Example:
        >>> config = create_default_config("configs/preprocess.yaml")
        >>> # Edit the file and reload
        >>> config = load_config("configs/preprocess.yaml")
    """
    config = PreprocessConfig(
        raw_data_path="/path/to/data/raw/usc_timit",
        output_path="/path/to/data/processed",
    )
    config.to_yaml(output_path)
    return config
