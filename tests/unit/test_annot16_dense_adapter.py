import numpy as np
from scipy.io import savemat
import pytest

from src.research.annot16 import Annot16DenseAdapter, DENSE_ARTICULATOR_MAPPING


def _write_track(tmp_path, frame_numbers=(2, 4), *, include_upper_lip=True):
    root = tmp_path / "75SpeakerAnnot16"
    track_dir = root / "sub061" / "track"
    track_dir.mkdir(parents=True)
    path = track_dir / "sub061_2drt_17_topic1_track.mat"
    records = np.empty((1, len(frame_numbers)), dtype=object)
    for position, frame_number in enumerate(frame_numbers):
        segments = np.empty(3, dtype=object)
        segments[0] = {
            "v": np.array(
                [[-40, 40], [-30, 30], [-20, 20], [-10, 10], [0, 0], [1, 1]], dtype=float
            ),
            "i": np.array([1, 2, 4, 5, 2, 3]),
            "mu": 0.1,
        }
        segments[1] = {
            "v": np.array([[2, 3], [4, 5], [6, 7]], dtype=float),
            "i": np.array([1, 2, 3]),
            "mu": 0.2,
        }
        segments[2] = {
            "v": np.array([[8, 9], [10, 11], [12, 13], [14, 15]], dtype=float),
            "i": np.array([1, 2, 5 if include_upper_lip else 4, 4]),
            "mu": 0.3,
        }
        records[0, position] = {
            "frameNo": frame_number,
            "template": 7,
            "contours": {"segment": segments},
        }
    savemat(path, {"trackdata": records})
    return root, path


def test_dense_adapter_uses_frame_number_equality_and_centered_transform(tmp_path):
    root, path = _write_track(tmp_path)
    sample = Annot16DenseAdapter(root, frame_rate=83.28).load(path, 2)

    assert sample.frame_index == 1
    assert sample.timestamp == pytest.approx(1 / 83.28)
    np.testing.assert_array_equal(sample.articulators["epiglottis"].coordinates, [[2, 2]])
    np.testing.assert_array_equal(sample.articulators["tongue"].coordinates, [[12, 12], [42, 42]])
    assert list(sample.articulators) == [item[2] for item in DENSE_ARTICULATOR_MAPPING]


def test_dense_adapter_is_deterministic_and_preserves_provenance(tmp_path):
    root, path = _write_track(tmp_path)
    adapter = Annot16DenseAdapter(root)
    first = adapter.load(path, 4)
    second = adapter.load(path, 4)

    assert list(first.articulators) == list(second.articulators)
    for name in first.articulators:
        np.testing.assert_array_equal(
            first.articulators[name].coordinates, second.articulators[name].coordinates
        )
    assert first.source_provenance["template"] == 7
    assert first.source_provenance["source_frame_number"] == 4
    assert first.source_provenance["dense_mapping"][0] == {
        "segment_index": 0,
        "identity": 1,
        "articulator": "epiglottis",
    }
    assert first.source_provenance["point_order"].startswith("source segment v row order")
    assert first.source_provenance["semantic_mapping_status"] == "PARTIAL"
    assert "not corpus-validated" in first.source_provenance["validation_scope"]


def test_dense_adapter_preserves_missing_contour_as_empty(tmp_path):
    root, path = _write_track(tmp_path, include_upper_lip=False)
    contour = Annot16DenseAdapter(root).load(path, 2).articulators["upper_lip"]
    assert contour.coordinates.shape == (0, 2)
    assert contour.valid_mask.shape == (0,)


def test_dense_adapter_rejects_missing_and_duplicate_frame_numbers(tmp_path):
    root, path = _write_track(tmp_path)
    adapter = Annot16DenseAdapter(root)
    with pytest.raises(KeyError, match="frameNo not found"):
        adapter.load(path, 3)

    root, path = _write_track(tmp_path / "duplicate", frame_numbers=(2, 2))
    with pytest.raises(ValueError, match="duplicate frameNo"):
        Annot16DenseAdapter(root).load(path, 2)


@pytest.mark.parametrize("frame_rate", [float("nan"), float("inf"), -float("inf")])
def test_dense_adapter_rejects_nonfinite_frame_rate(tmp_path, frame_rate):
    with pytest.raises(ValueError, match="finite and positive"):
        Annot16DenseAdapter(tmp_path, frame_rate=frame_rate)


def test_dense_adapter_rejects_malformed_mat(tmp_path):
    root = tmp_path / "75SpeakerAnnot16"
    path = root / "sub061" / "track" / "sub061_2drt_17_topic1_track.mat"
    path.parent.mkdir(parents=True)
    savemat(path, {"wrong": np.array([1])})
    with pytest.raises(ValueError, match="Malformed"):
        Annot16DenseAdapter(root).load(path, 1)
