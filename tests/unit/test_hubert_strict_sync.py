import numpy as np
import pytest
import torch

from scripts.extract_hubert_features import sync_to_mri_frames as script_sync
from src.audio_features.hubert_extractor import (
    HUBERT_FEATURE_ORIGIN_SECONDS,
    HuBERTExtractor,
)
from src.inference.engine import InferenceEngine


def test_hubert_extractors_use_receptive_center_and_return_exact_pts_indices():
    features = np.arange(4, dtype=np.float32)[:, None]
    timestamps = np.arange(8, dtype=np.float64) * 781 / 65040
    extractor = HuBERTExtractor.__new__(HuBERTExtractor)

    class_features, class_indices = extractor._sync_to_mri_frames(
        features,
        len(timestamps),
        65040 / 781,
        mri_timestamps=timestamps,
    )
    script_features, script_indices = script_sync(
        features,
        len(timestamps),
        65040 / 781,
        mri_timestamps=timestamps,
    )

    np.testing.assert_array_equal(class_indices, [2, 3, 4, 5, 6])
    np.testing.assert_array_equal(script_indices, class_indices)
    np.testing.assert_array_equal(script_features, class_features)
    expected_first = (timestamps[2] - HUBERT_FEATURE_ORIGIN_SECONDS) / 0.02
    assert class_features[0, 0] == pytest.approx(expected_first)


def test_supervised_extract_returns_and_records_target_slice_indices():
    class FakeModel:
        def extract_features(self, waveform):
            return [torch.arange(4, dtype=torch.float32).reshape(1, 4, 1)], None

    extractor = HuBERTExtractor.__new__(HuBERTExtractor)
    extractor.device = "cpu"
    extractor.layer_index = 0
    extractor.model = FakeModel()
    features, indices = extractor.extract(
        np.zeros(1600, dtype=np.float32),
        8,
        65040 / 781,
        mri_timestamps=np.arange(8) * 781 / 65040,
        return_frame_indices=True,
    )

    np.testing.assert_array_equal(indices, [2, 3, 4, 5, 6])
    np.testing.assert_array_equal(extractor.last_supported_mri_frame_indices, indices)
    assert features.shape == (5, 1)

    with pytest.raises(ValueError, match="return_frame_indices=True"):
        extractor.extract(
            np.zeros(1600, dtype=np.float32),
            8,
            65040 / 781,
            mri_timestamps=np.arange(8) * 781 / 65040,
        )


def test_audio_only_policy_is_deterministic_and_excludes_unsupported_start():
    features = np.arange(4, dtype=np.float32)[:, None]
    extractor = HuBERTExtractor.__new__(HuBERTExtractor)

    first = extractor._sync_to_mri_frames(
        features,
        None,
        100.0,
        timeline_policy="truncate_to_hubert_support",
    )
    second = extractor._sync_to_mri_frames(
        features,
        None,
        100.0,
        timeline_policy="truncate_to_hubert_support",
    )

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[1], [2, 3, 4, 5, 6, 7])


def test_audio_only_inference_caller_explicitly_requests_truncation_without_model():
    calls = []

    class FakeExtractor:
        def extract(self, *args, **kwargs):
            calls.append((args, kwargs))
            return np.zeros((3, 1024), dtype=np.float32)

    engine = InferenceEngine.__new__(InferenceEngine)
    engine._hubert_extractor = FakeExtractor()
    result = engine._extract_hubert(np.zeros(16000, dtype=np.float32), 16000)

    assert result.shape == (3, 1024)
    assert calls[0][0][1] is None
    assert calls[0][1] == {
        "mri_fps": 83.3,
        "feature_time_offset_seconds": 0.0,
        "timeline_policy": "truncate_to_hubert_support",
    }


def test_hubert_extractors_reject_timeline_with_no_center_support():
    features = np.arange(4, dtype=np.float32)[:, None]
    extractor = HuBERTExtractor.__new__(HuBERTExtractor)

    with pytest.raises(ValueError, match="no MRI timestamps"):
        extractor._sync_to_mri_frames(features, 2, 100.0)
    with pytest.raises(ValueError, match="no MRI timestamps"):
        script_sync(features, 2, 100.0)
