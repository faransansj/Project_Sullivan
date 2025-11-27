"""
Project Sullivan - Pytest Configuration and Fixtures

This module provides common test fixtures and configuration for the test suite.
"""

import sys
from pathlib import Path

import pytest
import numpy as np

# Add src to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ============================================================================
# Path Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def data_root(project_root: Path) -> Path:
    """Return the data root directory."""
    return project_root / "data"


@pytest.fixture(scope="session")
def test_data_root(project_root: Path) -> Path:
    """Return the test data directory."""
    test_dir = project_root / "tests" / "fixtures" / "data"
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


@pytest.fixture(scope="session")
def test_output_root(project_root: Path) -> Path:
    """Return the test output directory."""
    output_dir = project_root / "tests" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ============================================================================
# Mock Data Fixtures
# ============================================================================


@pytest.fixture
def sample_mri_frame() -> np.ndarray:
    """
    Generate a synthetic MRI frame for testing.

    Returns:
        np.ndarray: Synthetic MRI frame (256, 256) with float32 dtype
    """
    # Create a simple synthetic MRI frame with some structure
    frame = np.zeros((256, 256), dtype=np.float32)

    # Add a circular structure (simulating head)
    y, x = np.ogrid[:256, :256]
    center_y, center_x = 128, 128
    radius = 100
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2
    frame[mask] = 0.5

    # Add some noise
    noise = np.random.normal(0, 0.05, (256, 256)).astype(np.float32)
    frame += noise
    frame = np.clip(frame, 0, 1)

    return frame


@pytest.fixture
def sample_mri_sequence(sample_mri_frame: np.ndarray) -> np.ndarray:
    """
    Generate a synthetic MRI sequence for testing.

    Args:
        sample_mri_frame: Single MRI frame fixture

    Returns:
        np.ndarray: Synthetic MRI sequence (T=100, H=256, W=256)
    """
    # Create sequence by adding temporal variation
    sequence = []
    for t in range(100):
        # Add temporal variation (simulating jaw movement)
        frame = sample_mri_frame.copy()
        variation = np.sin(2 * np.pi * t / 50) * 0.1
        frame = np.clip(frame + variation, 0, 1)
        sequence.append(frame)

    return np.array(sequence, dtype=np.float32)


@pytest.fixture
def sample_audio() -> tuple[np.ndarray, int]:
    """
    Generate a synthetic audio signal for testing.

    Returns:
        tuple: (audio waveform, sample rate)
            - audio: np.ndarray of shape (N,) with float32 dtype
            - sr: int, sample rate in Hz
    """
    sr = 16000
    duration = 2.0  # 2 seconds to match ~100 frames at 50fps
    t = np.linspace(0, duration, int(sr * duration))

    # Generate a simple audio signal with multiple frequencies
    audio = (
        0.3 * np.sin(2 * np.pi * 220 * t)  # A3
        + 0.2 * np.sin(2 * np.pi * 440 * t)  # A4
        + 0.1 * np.sin(2 * np.pi * 880 * t)  # A5
    )

    # Add some noise
    audio += np.random.normal(0, 0.05, len(audio))
    audio = np.clip(audio, -1, 1).astype(np.float32)

    return audio, sr


@pytest.fixture
def sample_segmentation_mask() -> np.ndarray:
    """
    Generate a synthetic segmentation mask for testing.

    Returns:
        np.ndarray: Segmentation mask (256, 256) with int32 dtype
                   0: background, 1: tongue, 2: jaw, 3: lips, 4: velum
    """
    mask = np.zeros((256, 256), dtype=np.int32)

    # Create simplified anatomical regions
    y, x = np.ogrid[:256, :256]

    # Tongue (simplified as ellipse)
    tongue_center_y, tongue_center_x = 160, 128
    tongue_mask = ((y - tongue_center_y) / 40) ** 2 + ((x - tongue_center_x) / 60) ** 2 <= 1
    mask[tongue_mask] = 1

    # Jaw (lower region)
    jaw_mask = y > 200
    mask[jaw_mask] = 2

    # Lips (front region)
    lips_mask = (x > 180) & (y > 120) & (y < 180)
    mask[lips_mask] = 3

    # Velum (upper back region)
    velum_mask = (x < 100) & (y < 100)
    mask[velum_mask] = 4

    return mask


@pytest.fixture
def sample_parameters() -> np.ndarray:
    """
    Generate synthetic articulatory parameters for testing.

    Returns:
        np.ndarray: Parameters (T=100, D=10) with float32 dtype
    """
    T, D = 100, 10

    # Generate smooth trajectories
    t = np.linspace(0, 2 * np.pi, T)
    parameters = np.zeros((T, D), dtype=np.float32)

    for d in range(D):
        # Each parameter has different frequency and phase
        freq = 0.5 + d * 0.1
        phase = d * np.pi / D
        parameters[:, d] = np.sin(freq * t + phase)

    # Add small noise
    parameters += np.random.normal(0, 0.05, (T, D)).astype(np.float32)

    return parameters


# ============================================================================
# Configuration Fixtures
# ============================================================================


@pytest.fixture
def sample_config_dict() -> dict:
    """
    Generate a sample configuration dictionary for testing.

    Returns:
        dict: Sample configuration
    """
    return {
        "raw_data_path": "/path/to/data/raw",
        "output_path": "/path/to/data/processed",
        "mri_target_size": [256, 256],
        "mri_fps": 50.0,
        "audio_sr": 16000,
        "n_mfcc": 13,
        "n_mels": 80,
        "pca_components": 10,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
    }


# ============================================================================
# Markers and Test Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow-running tests")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "data: Tests requiring downloaded data")


def pytest_collection_modifyitems(config, items):
    """
    Modify test items during collection.

    This function automatically marks tests based on their location:
    - tests/unit/* -> marked as 'unit'
    - tests/integration/* -> marked as 'integration'
    """
    for item in items:
        # Get the test file path relative to project root
        rel_path = Path(item.fspath).relative_to(PROJECT_ROOT)

        # Auto-mark based on directory
        if "unit" in str(rel_path):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(rel_path):
            item.add_marker(pytest.mark.integration)
