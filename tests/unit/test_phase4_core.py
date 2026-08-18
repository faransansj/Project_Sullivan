import json

import numpy as np
import pytest
import torch

from src.inference.engine import InferenceEngine
from src.modeling.conformer_model import ConformerInversionModel
from src.modeling.dataset import create_dataloaders
from src.modeling.transformer import TransformerModel


def _write_sample(root, name, parameters):
    audio_dir = root / "audio"
    parameter_dir = root / "parameters"
    audio_dir.mkdir(exist_ok=True)
    parameter_dir.mkdir(exist_ok=True)
    np.savez(audio_dir / f"{name}_audio.npz", mel_spectrogram=np.zeros((len(parameters), 2)))
    np.savez(parameter_dir / f"{name}_params.npz", geometric_features=parameters)


def test_all_splits_use_training_normalization_stats(tmp_path):
    splits = tmp_path / "splits"
    splits.mkdir()
    for split, name in (("train", "train_sample"), ("val", "val_sample"), ("test", "test_sample")):
        (splits / f"{split}.json").write_text(json.dumps([name]), encoding="utf-8")

    _write_sample(tmp_path, "train_sample", np.array([[0.0], [10.0]]))
    _write_sample(tmp_path, "val_sample", np.array([[100.0], [200.0]]))
    _write_sample(tmp_path, "test_sample", np.array([[-10.0], [20.0]]))

    loaders = create_dataloaders(
        splits,
        tmp_path / "audio",
        tmp_path / "parameters",
        batch_size=1,
        num_workers=0,
    )

    assert loaders["val"].dataset[0][1].squeeze().tolist() == [10.0, 20.0]
    assert loaders["test"].dataset[0][1].squeeze().tolist() == [-1.0, 2.0]
    assert loaders["val"].dataset.param_min.tolist() == [0.0]
    assert loaders["val"].dataset.param_max.tolist() == [10.0]


@pytest.mark.parametrize("model_class", [ConformerInversionModel, TransformerModel])
def test_masked_mean_averages_frames_and_dimensions(model_class):
    error = torch.ones(1, 2, 24)
    mask = torch.tensor([[[1.0], [0.0]]])
    assert model_class._masked_mean(error, mask).item() == 1.0


@pytest.mark.parametrize("length", [1, 2])
def test_conformer_temporal_loss_is_finite_for_short_sequences(length):
    model = ConformerInversionModel.__new__(ConformerInversionModel)
    torch.nn.Module.__init__(model)
    model.criterion = torch.nn.MSELoss(reduction="none")
    values = torch.zeros(1, length, 3)
    mask = torch.ones(1, length, 1)

    losses = model._compute_temporal_loss(values, values, mask)

    assert torch.isfinite(losses["velocity_loss"])
    assert torch.isfinite(losses["acceleration_loss"])


@pytest.mark.parametrize(
    ("stats", "expected"),
    [
        ({"normalization_type": "minmax", "min": np.array([2.0]), "max": np.array([6.0])}, 4.0),
        ({"normalization_type": "zscore", "mean": np.array([2.0]), "std": np.array([4.0])}, 4.0),
    ],
)
def test_inference_denormalizes_supported_normalization_types(stats, expected):
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.stats = stats
    assert engine._denormalize(np.array([[0.5]])).item() == expected
