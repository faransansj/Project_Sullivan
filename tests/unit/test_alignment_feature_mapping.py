import numpy as np
import pytest

from scripts.diagnose_audio_mri_alignment import normalized_overlap_correlation
from src.research.alignment_diagnostic import (
    audio_sample_indices,
    features_for_mri_frames,
    frame_timestamp,
    hubert_feature_center_timestamps,
    interpolate_feature_timestamps,
)


def test_frame_timestamp_is_zero_based():
    assert frame_timestamp(0, 83.28) == 0.0
    assert frame_timestamp(529, 83.28) == pytest.approx(529 / 83.28)
    with pytest.raises(ValueError):
        frame_timestamp(-1, 83.28)


def test_linear_feature_mapping_is_deterministic():
    features = np.array([[0.0, 10.0], [2.0, 14.0], [4.0, 18.0]], dtype=np.float32)
    timestamps = np.array([0.0, 0.5, 1.0])
    kwargs = {"feature_stride_seconds": 1.0, "feature_time_offset_seconds": 0.0}

    first = interpolate_feature_timestamps(features, timestamps, **kwargs)
    second = interpolate_feature_timestamps(features, timestamps, **kwargs)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first, [[0, 10], [1, 12], [2, 14]])


def test_feature_mapping_rejects_both_out_of_range_boundaries():
    features = np.arange(6, dtype=np.float32).reshape(3, 2)
    for timestamps, feature_time_offset in (
        (np.array([-0.01]), 0.0),
        (np.array([2.01]), 0.0),
        (np.array([0.0]), -0.01),
    ):
        with pytest.raises(ValueError, match="outside"):
            interpolate_feature_timestamps(
                features,
                timestamps,
                feature_stride_seconds=1.0,
                feature_time_offset_seconds=feature_time_offset,
            )
    # Exact endpoints remain valid; nothing is silently clamped.
    result = interpolate_feature_timestamps(
        features,
        np.array([0.0, 2.0]),
        feature_stride_seconds=1.0,
        feature_time_offset_seconds=0.0,
    )
    np.testing.assert_array_equal(result, features[[0, 2]])


def test_overlap_correlation_sign_queries_right_signal_at_t_plus_offset():
    times = np.arange(4, dtype=np.float64)
    labels = np.array([0.0, 1.0, 0.0, 0.0])
    delayed_audio = np.array([0.0, 0.0, 1.0, 0.0])

    correlation, overlap_count = normalized_overlap_correlation(
        times, labels, times, delayed_audio, 1.0
    )

    assert correlation == pytest.approx(1.0)
    assert overlap_count == 3


def test_exact_pts_maps_to_nearest_decoded_audio_sample():
    timestamps = np.array([0.0, 781 / 65040, 2744 * 781 / 65040])
    np.testing.assert_array_equal(audio_sample_indices(timestamps, 22050), [0, 265, 726546])


def test_hubert_receptive_field_centers_and_boundary_are_explicit():
    centers = hubert_feature_center_timestamps(3)
    np.testing.assert_allclose(centers, [0.01246875, 0.03246875, 0.05246875])
    features = np.arange(3, dtype=np.float32)[:, None]
    sampled = interpolate_feature_timestamps(
        features,
        centers[[0, 2]],
        feature_stride_seconds=0.02,
        feature_time_offset_seconds=0.0,
        feature_origin_seconds=centers[0],
    )
    np.testing.assert_array_equal(sampled[:, 0], [0, 2])
    with pytest.raises(ValueError, match="outside"):
        interpolate_feature_timestamps(
            features,
            np.array([0.0]),
            feature_stride_seconds=0.02,
            feature_time_offset_seconds=0.0,
            feature_origin_seconds=centers[0],
        )


def test_feature_time_offset_sign_selects_later_feature_for_positive_offset():
    features = np.arange(6, dtype=np.float32)[:, None]
    positive = features_for_mri_frames(
        features,
        np.array([2]),
        mri_frame_rate=2.0,
        feature_stride_seconds=0.5,
        feature_time_offset_seconds=0.5,
    )
    negative = features_for_mri_frames(
        features,
        np.array([2]),
        mri_frame_rate=2.0,
        feature_stride_seconds=0.5,
        feature_time_offset_seconds=-0.5,
    )
    np.testing.assert_array_equal(positive[:, 0], [3.0])
    np.testing.assert_array_equal(negative[:, 0], [1.0])
