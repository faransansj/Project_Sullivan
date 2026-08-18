import numpy as np
import pytest

from src.modeling.dataset import ArticulatoryDataset


def _files(tmp_path, indices):
    utterance = "sub061_2drt_17_topic1"
    audio_dir = tmp_path / "audio"
    parameter_dir = tmp_path / "parameters"
    (audio_dir / "hubert").mkdir(parents=True)
    (parameter_dir / "geometric").mkdir(parents=True)
    np.save(
        audio_dir / "hubert" / f"{utterance}_hubert.npy",
        np.array([[100.0], [101.0], [102.0]], dtype=np.float32),
    )
    np.save(
        audio_dir / "hubert" / f"{utterance}_hubert_mri_frame_indices.npy",
        np.asarray(indices),
    )
    np.save(
        parameter_dir / "geometric" / f"{utterance}_params.npy",
        np.arange(6, dtype=np.float32)[:, None] * 10,
    )
    return utterance, audio_dir, parameter_dir


@pytest.mark.parametrize("streaming", [False, True])
def test_hubert_sidecar_slices_original_target_frames_before_reconciliation(tmp_path, streaming):
    utterance, audio_dir, parameter_dir = _files(tmp_path, [2, 4, 5])
    dataset = ArticulatoryDataset(
        [utterance],
        audio_dir,
        parameter_dir,
        audio_feature_type="hubert",
        normalize_params=False,
        streaming=streaming,
    )

    features, targets, _ = dataset[0]

    assert features[:, 0].tolist() == [100.0, 101.0, 102.0]
    assert targets[:, 0].tolist() == [20.0, 40.0, 50.0]
    assert dataset.data[0]["alignment_contract"] == "exact_hubert_frame_indices"


@pytest.mark.parametrize(
    "indices, error",
    [
        ([2.0, 4.0, 5.0], "1-D integers"),
        ([[2, 4, 5]], "1-D integers"),
        ([2, 2, 5], "unique, strictly increasing, and in bounds"),
        ([4, 2, 5], "unique, strictly increasing, and in bounds"),
        ([-1, 2, 5], "unique, strictly increasing, and in bounds"),
        ([2, 4, 6], "unique, strictly increasing, and in bounds"),
        ([2, 4], "index count 2 != feature rows 3"),
    ],
)
def test_hubert_sidecar_rejects_malformed_indices(tmp_path, indices, error):
    utterance, audio_dir, parameter_dir = _files(tmp_path, indices)

    with pytest.raises(ValueError, match=error):
        ArticulatoryDataset(
            [utterance],
            audio_dir,
            parameter_dir,
            audio_feature_type="hubert",
            normalize_params=False,
        )


def test_hubert_without_sidecar_is_explicitly_legacy(tmp_path):
    utterance, audio_dir, parameter_dir = _files(tmp_path, [2, 4, 5])
    (audio_dir / "hubert" / f"{utterance}_hubert_mri_frame_indices.npy").unlink()

    with pytest.warns(RuntimeWarning, match="alignment is not exact"):
        dataset = ArticulatoryDataset(
            [utterance],
            audio_dir,
            parameter_dir,
            audio_feature_type="hubert",
            normalize_params=False,
        )

    assert dataset.data[0]["alignment_contract"] == "legacy_hubert_interpolation_no_sidecar"
    assert dataset[0][0].shape[0] == 6
