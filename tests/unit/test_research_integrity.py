import json

import numpy as np
import pytest

from src.research.alignment_diagnostic import best_lag, lag_sweep
from src.research.contours import (
    ArticulatorContour,
    JsonContourLoader,
    flatten_contours,
    masked_point_loss,
    masked_velocity_loss,
    reshape_contours,
    resample_ordered_contour,
)
from src.research.metrics import (
    articulator_rmse,
    dimension_mean_pcc,
    global_pcc,
    hausdorff_distance,
    mae,
    point_euclidean_rmse,
    rmse,
    speaker_macro_contour_rmse,
    symmetric_chamfer_distance,
    utterance_mean_pcc,
)
from src.research.normalization import NormalizationArtifact
from src.research.reproducibility import ReproducibilityMetadata
from src.research.split_manifest import SampleAssignment, build_manifest


def test_split_manifest_rejects_duplicates_and_speaker_overlap(tmp_path):
    common = dict(
        dataset_version="synthetic-v1", seed=42, strategy="speaker_disjoint", git_sha="abc"
    )
    with pytest.raises(ValueError, match="Duplicate"):
        build_manifest(
            [
                SampleAssignment("same", "s1", "u1", "train"),
                SampleAssignment("same", "s2", "u2", "validation"),
                SampleAssignment("test", "s3", "u3", "test"),
            ],
            **common,
        )
    with pytest.raises(ValueError, match="Speaker overlap"):
        build_manifest(
            [
                SampleAssignment("a", "s1", "u1", "train"),
                SampleAssignment("b", "s1", "u2", "test"),
                SampleAssignment("c", "s2", "u3", "validation"),
            ],
            **common,
        )

    manifest = build_manifest(
        [
            SampleAssignment("a", "s1", "u1", "train"),
            SampleAssignment("b", "s2", "u2", "validation"),
            SampleAssignment("c", "s3", "u3", "test"),
        ],
        **common,
    )
    path = tmp_path / "manifest.json"
    manifest.save(path)
    assert manifest.load(path).sha256 == manifest.sha256


def test_normalization_is_train_only_and_round_trips(tmp_path):
    values = np.array([[[1.0, 2.0], [3.0, 6.0]], [[100.0, 100.0], [5.0, 10.0]]])
    mask = np.array([[True, True], [False, True]])
    with pytest.raises(ValueError, match="train"):
        NormalizationArtifact.fit(
            values,
            feature_name="contour",
            fit_split="validation",
            config={},
            manifest_hash="manifest",
            dataset_version="synthetic-v1",
        )
    artifact = NormalizationArtifact.fit(
        values,
        feature_name="contour",
        fit_split="train",
        config={"seed": 42},
        manifest_hash="manifest",
        dataset_version="synthetic-v1",
        mask=mask,
    )
    transformed = artifact.transform(values)
    assert np.allclose(artifact.inverse_transform(transformed), values)
    assert artifact.frame_count == 3
    path = tmp_path / "normalization.json"
    artifact.save(path)
    assert NormalizationArtifact.load(path).sha256 == artifact.sha256


def test_masked_scalar_metrics_have_distinct_aggregation():
    target = np.array([[[0.0, 0.0], [1.0, 2.0]], [[0.0, 2.0], [1.0, 0.0]]])
    prediction = target.copy()
    prediction[1, 1] = 99
    mask = np.array([[True, True], [True, False]])
    utterances = np.array([["u1", "u1"], ["u2", "u2"]])
    assert rmse(prediction, target, mask) == 0
    assert mae(prediction, target, mask) == 0
    assert global_pcc(prediction, target, mask) == pytest.approx(1.0)
    assert dimension_mean_pcc(prediction, target, mask) == pytest.approx(1.0)
    assert utterance_mean_pcc(prediction, target, utterances, mask) == pytest.approx(1.0)


def test_contour_metrics_use_mm_only_with_spacing():
    target = np.array([[0.0, 0.0], [1.0, 0.0]])
    prediction = target + np.array([1.0, 0.0])
    valid = np.array([True, True])
    assert point_euclidean_rmse(prediction, target, valid) == pytest.approx(1.0)
    assert point_euclidean_rmse(prediction, target, valid, (2.0, 3.0)) == pytest.approx(2.0)
    assert symmetric_chamfer_distance(target, target, valid) == 0
    assert hausdorff_distance(target, target, valid) == 0
    assert articulator_rmse({"tongue": prediction}, {"tongue": target}, {"tongue": valid}) == {
        "tongue": pytest.approx(1.0)
    }


def test_contour_contract_loader_and_resampling(tmp_path):
    document = {
        "samples": [
            {
                "sample_id": "sample-1",
                "speaker_id": "speaker-1",
                "utterance_id": "utt-1",
                "frame_index": 0,
                "timestamp": 0.0,
                "audio_path": "audio.wav",
                "mri_path": "frame.png",
                "articulators": {
                    "tongue": {
                        "coordinates": [[0, 0], [1, 0], [2, 0]],
                        "valid_mask": [True, False, True],
                    }
                },
                "pixel_spacing": {"x_mm_per_pixel": 1.0, "y_mm_per_pixel": 1.0},
            }
        ]
    }
    path = tmp_path / "contours.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    sample = JsonContourLoader(path).load("sample-1")
    result = resample_ordered_contour(sample.articulators["tongue"], 5)
    assert result.coordinates.shape == (5, 2)
    assert not result.valid_mask.any()
    assert result.coordinates[[0, -1], 0].tolist() == [0.0, 2.0]


def test_contour_losses_and_flatten_round_trip_respect_masks():
    target = np.zeros((2, 3, 1, 2, 2), dtype=float)
    prediction = target.copy()
    prediction[0, 1, 0, 0, 0] = 1.0
    valid = np.ones((2, 3, 1, 2), dtype=bool)
    valid[0, 1, 0, 0] = False
    assert masked_point_loss(prediction, target, valid) == 0.0
    assert masked_velocity_loss(target, target, valid) == 0.0
    assert np.array_equal(reshape_contours(flatten_contours(target), 1, 2), target)


def test_speaker_macro_contour_rmse_weights_speakers_equally():
    target = np.zeros((2, 2, 1, 1, 2))
    prediction = target.copy()
    prediction[1] = 2.0
    valid = np.ones((2, 2, 1, 1), dtype=bool)
    speakers = np.array([["s1", "s1"], ["s2", "s2"]])
    assert speaker_macro_contour_rmse(prediction, target, valid, speakers) == pytest.approx(
        np.sqrt(8) / 2
    )


def test_lag_sweep_is_validation_only_and_finds_known_lag():
    target = np.arange(10, dtype=float).reshape(1, 10, 1)
    prediction = np.pad(target[:, :-1], ((0, 0), (1, 0), (0, 0)))
    with pytest.raises(ValueError, match="validation-only"):
        lag_sweep(prediction, target, frame_rate=10, split="test")
    results = lag_sweep(
        prediction,
        target,
        frame_rate=10,
        split="validation",
        min_lag_ms=-200,
        max_lag_ms=200,
        step_ms=100,
    )
    assert best_lag(results)["lag_ms"] == 100


def test_reproducibility_metadata_writes_required_fields(tmp_path):
    metadata = ReproducibilityMetadata(
        git_sha="abc",
        config_hash="cfg",
        split_manifest_hash="split",
        normalization_artifact_hash="norm",
        dataset_version="synthetic-v1",
        target_representation="direct_contour",
        model_parameter_count=123,
        seed=42,
        checkpoint_selection_rule="min validation contour RMSE",
        status="planned",
        git_dirty=True,
        resolved_config={"seed": 42},
    )
    path = tmp_path / "run.json"
    metadata.save(path)
    assert json.loads(path.read_text())["status"] == "planned"
