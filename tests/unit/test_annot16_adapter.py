import json

from PIL import Image
import pytest

from src.research.annot16 import Annot16GroundTruthAdapter


def _fixture(tmp_path, coordinates=None):
    root = tmp_path / "75SpeakerAnnot16"
    annotations = root / "hand_ground_truth" / "ground_truth_json"
    images = root / "hand_ground_truth" / "extracted_frames_jpg"
    annotations.mkdir(parents=True)
    images.mkdir(parents=True)
    annotation = annotations / "sub061_2drt_17_topic1_track_frame-530.json"
    annotation.write_text(
        json.dumps({"tongue": coordinates or [[10, 20], [11, 19], [12, 20]]}),
        encoding="utf-8",
    )
    Image.new("L", (84, 84)).save(images / "sub061_2drt_17_topic1_video.mp4_frame-530.jpg")
    return root, annotation


def test_annot16_ground_truth_adapter_preserves_identifiers_and_coordinates(tmp_path):
    root, annotation = _fixture(tmp_path)
    sample = Annot16GroundTruthAdapter(root, frame_rate=83.28, pixel_spacing=(2.4, 2.4)).load(
        annotation
    )

    assert sample.speaker_id == "sub061"
    assert sample.utterance_id == "sub061_2drt_17_topic1"
    assert sample.frame_index == 529
    assert sample.timestamp == pytest.approx(529 / 83.28)
    assert sample.coordinate_convention == "image_xy_origin_top_left_x_right_y_down"
    assert sample.articulators["tongue"].coordinates.tolist() == [
        [10.0, 20.0],
        [11.0, 19.0],
        [12.0, 20.0],
    ]
    assert sample.articulators["tongue"].valid_mask.all()
    assert sample.pixel_spacing.x_mm_per_pixel == 2.4
    assert sample.source_provenance["source_frame_number"] == 530


def test_annot16_adapter_does_not_invent_timestamp_or_spacing(tmp_path):
    root, annotation = _fixture(tmp_path)
    sample = Annot16GroundTruthAdapter(root).load(annotation)
    assert sample.timestamp is None
    assert sample.pixel_spacing is None


@pytest.mark.parametrize("frame_rate", [float("nan"), float("inf"), -float("inf")])
def test_annot16_adapter_rejects_nonfinite_frame_rate(tmp_path, frame_rate):
    with pytest.raises(ValueError, match="finite and positive"):
        Annot16GroundTruthAdapter(tmp_path, frame_rate=frame_rate)


def test_annot16_adapter_rejects_unknown_articulator_and_missing_frame(tmp_path):
    root, annotation = _fixture(tmp_path)
    annotation.write_text(json.dumps({"unknown": [[1, 2], [2, 3]]}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown Annot-16 articulators"):
        Annot16GroundTruthAdapter(root).load(annotation)

    annotation.write_text(json.dumps({"tongue": [[1, 2], [2, 3]]}), encoding="utf-8")
    next((root / "hand_ground_truth" / "extracted_frames_jpg").glob("*.jpg")).unlink()
    with pytest.raises(FileNotFoundError):
        Annot16GroundTruthAdapter(root).load(annotation)
