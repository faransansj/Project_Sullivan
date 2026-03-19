# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Project Sullivan** is a research project developing AI models that infer articulatory parameters (tongue position, jaw opening, lip shape, etc.) from audio signals alone using the USC-TIMIT Speech MRI Dataset.

**Current Status**: Phase 4 (정확도 개선) Active, Phase 5 (인프라) Planning
- Phase 2-A (Baseline LSTM): Test RMSE 1.011, PCC 0.105
- Phase 3 (Transformer) Complete: Global PCC 0.1982, 21.5M params, 24-dim output
- Phase 4 Active: HuBERT features + Conformer architecture + Inference Engine
- M2 Target: RMSE < 0.15, PCC > 0.50

**Key Technologies**: PyTorch, PyTorch Lightning, rtMRI processing, HuBERT, Conformer

---

## Essential Commands

### Environment Setup
```bash
source venv_sullivan/bin/activate
pip install -r requirements.txt
pytest                       # All tests
pytest tests/unit            # Unit tests only
pytest -m "not slow"         # Skip slow tests
pytest --cov=src             # With coverage
```

### Training Models

**Transformer (Phase 3 - Complete)**
```bash
python scripts/train_transformer.py --config configs/transformer_config.yaml
python scripts/train_transformer.py --config configs/transformer_quick_test.yaml  # CPU quick test
python scripts/train_transformer.py --config configs/transformer_a100.yaml        # A100 GPU
```

**Conformer (Phase 4 - Current)**
```bash
python scripts/train_conformer.py --config configs/conformer_a100_config.yaml     # A100 (default)
python scripts/train_conformer.py --config configs/conformer_medium_config.yaml   # d_model=512, 12L
python scripts/train_conformer.py --config configs/conformer_large_config.yaml    # d_model=1024, 18L
python scripts/train_conformer.py --config configs/conformer_hubert_config.yaml   # + HuBERT features
```

**Baseline LSTM (Phase 2-A - Complete)**
```bash
python scripts/train_baseline.py --config configs/baseline_config.yaml
```

### Feature Extraction (Phase 4)
```bash
# Extract HuBERT features (1024-dim, 50Hz) — Phase 4
python scripts/extract_hubert_features.py

# Extract mel-spectrogram features (80-dim)
python scripts/extract_audio_features.py

# Extract articulatory parameters (geometric + PCA)
python scripts/extract_articulatory_params.py
```

### Monitoring Training
```bash
bash scripts/start_tensorboard.sh           # http://localhost:6006
bash scripts/monitor_training_simple.sh
bash scripts/check_training_status.sh
```

### Data Processing Pipeline (Phase 1 - Complete)
```bash
python scripts/batch_preprocess.py --config configs/preprocess.yaml

python scripts/segment_subset.py \
  --data-root data/raw/usc_timit_full \
  --subjects sub013,sub014,sub015 \
  --output-dir data/processed/segmentations \
  --checkpoint models/segmentation/unet_best.pth

python scripts/create_dataset_splits.py
```

### Google Colab Training
```bash
bash scripts/prepare_data_for_colab.sh
# See docs/guides/COLAB_TRAINING_GUIDE.md
# Notebook: notebooks/Project_Sullivan_Transformer_Training.ipynb
```

### Code Quality
```bash
black src/ scripts/ tests/    # Format (line-length=100)
flake8 src/ scripts/ tests/   # Lint
mypy src/                      # Type check
```

---

## Architecture & Code Structure

### Data Flow Pipeline

**Phase 1: Data Preprocessing (Complete)**
```
Raw MRI + Audio
    ↓ [1] Alignment & Denoising (src/preprocessing/)
    ↓ HDF5 → data/processed/aligned/
    ↓ [2] U-Net Segmentation (src/segmentation/)
    ↓ Masks → data/processed/segmentations/
    ↓ [3] Parameter Extraction (src/parameter_extraction/)
    ↓ 14 geometric + 10 PCA features → data/processed/parameters/
    ↓ [4] Audio Feature Extraction (src/audio_features/)
    ↓ Mel (80-dim) or HuBERT (1024-dim) → data/processed/audio_features/
    ↓ [5] Dataset Splits (70/15/15) → data/processed/splits/
```

**Phase 2–4: Model Training**
```
Audio Features + Articulatory Parameters
    ↓ ArticulatoryDataset (src/modeling/dataset.py)
    ↓ Model (PyTorch Lightning)
      - BaselineLSTM       — 613K params, 80-dim mel → 14-dim geometric
      - TransformerModel   — 21.5M params, Phase 3 (complete)
      - ConformerInversionModel — 60M+ params, Phase 4 (current)
    ↓ Trained Models → models/
    ↓ Inference Engine → src/inference/engine.py
```

### Key Modules

**src/audio_features/**
- `mel_spectrogram.py` — 80-dim mel-spectrogram extraction
- `mfcc.py` — 13-dim MFCC
- `hubert_extractor.py` — **Phase 4**: Loads HuBERT-Large, extracts 1024-dim features at 50Hz, syncs to MRI frame rate via interpolation

**src/modeling/**
- `dataset.py` — `ArticulatoryDataset`: handles variable-length sequences with padding/masking, normalizes params to [0,1], supports streaming from zip archives
- `baseline_lstm.py` — 2-layer Bi-LSTM, 128 hidden units (Phase 2-A)
- `transformer.py` — 4-layer Transformer encoder, 8 heads, d_model=256→512, d_ff=1024 (Phase 3)
- `conformer_model.py` — **Phase 4**: `ConformerInversionModel` using torchaudio Conformer blocks, supports both Mel and HuBERT input dims, configurable depth (12–18 layers)
- `losses.py` — Custom loss functions including masked MSE, PCC-based losses
- `model_utils.py` — Shared masking/padding utilities

**src/inference/**
- `engine.py` — **Phase 4**: Unified inference interface for Transformer and Conformer, handles both Mel and HuBERT feature pipelines

**src/segmentation/**
- `unet.py` — 5-layer U-Net encoder-decoder (81.8% Dice score)
- Pre-trained: `models/segmentation/unet_best.pth`

**src/parameter_extraction/**
- `geometric_features.py` — 14 geometric features (tongue position, jaw opening, etc.)
- `pca_features.py` — 10 PCA components from segmentation masks

### Model Implementation Notes

All models inherit from `pl.LightningModule` with standard methods: `forward`, `training_step`, `validation_step`, `test_step`, `configure_optimizers`, `on_*_epoch_end`.

**Variable-Length Sequence Handling**: Dataset returns `(features, params, mask)` where mask=1 for valid frames. Models compute MSE only on valid frames:
```python
loss_mask = create_loss_mask(lengths, max_len)
loss = ((predictions - targets) ** 2 * loss_mask).sum() / loss_mask.sum()
```

**Config Interpolation**: YAML configs use `${section.key}` syntax resolved in training scripts. See `scripts/train_transformer.py` for the `load_config()` implementation.

**Precision Settings**: Conformer large uses `bf16-mixed` to prevent FP16 NaN overflow. Transformer configs use `fp32` or `16-mixed`.

### Config Variants

| Config | Model | Features | Notes |
|--------|-------|----------|-------|
| `conformer_a100_config.yaml` | Conformer | Mel 80-dim | Default Phase 4 |
| `conformer_medium_config.yaml` | Conformer | Mel 80-dim | d_model=512, 12L |
| `conformer_large_config.yaml` | Conformer | Mel 80-dim | d_model=1024, 18L, bf16 |
| `conformer_hubert_config.yaml` | Conformer | HuBERT 1024-dim | Phase 4 combo |
| `transformer_a100.yaml` | Transformer | Mel 80-dim | A100 optimized |

### Dataset Format

```
data/processed/
├── aligned/          # MRI+Audio aligned (HDF5): sub001/, sub007/
├── segmentations/    # Vocal tract masks (NPZ)
├── parameters/       # Articulatory params (NPZ)
├── audio_features/
│   └── hubert/       # HuBERT features (Phase 4 NEW)
└── splits/           # Train/val/test manifests (JSON)
    ├── train.json    # Format: ["sub001/uw001_01", ...]
    ├── val.json
    └── test.json
```

---

## Important Implementation Details

### Path Resolution
- Scripts use `Path(__file__).parent.parent` for project root
- Always add `sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))`
- Config paths are relative to project root

### Configuration Pattern
```python
config = load_config(args.config)    # Load + resolve ${} interpolation
model = create_model(config)
loaders = create_dataloaders(config)
trainer = pl.Trainer(
    max_epochs=config['training']['num_epochs'],
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    precision=config['training']['precision']  # 'fp32', '16-mixed', 'bf16-mixed'
)
trainer.fit(model, loaders['train'], loaders['val'])
trainer.test(model, loaders['test'])
```

### Logging
- TensorBoard: `logs/training/{experiment_name}/`
- Checkpoints: `models/{model_name}/`
- Checkpoint naming: `{model_name}-{epoch:02d}-{val_loss:.4f}.ckpt`

### HuBERT Feature Extraction (Phase 4)
- Model: `facebook/hubert-large-ls960-ft` loaded via `transformers`
- Output: 1024-dim at 50Hz, interpolated to match MRI frame rate
- Stored as HDF5 in `data/processed/audio_features/hubert/`
- Input dim in Conformer config must be set to 1024 when using HuBERT

---

## Testing

```
tests/
├── conftest.py      # Synthetic fixtures: MRI frames, audio, masks, params
├── unit/            # Fast isolated tests (alignment, config, denoising, I/O, logger)
└── integration/     # Multi-component tests (data loading pipeline)
```

Fixtures: `sample_mri_frame/sequence` (256×256), `sample_audio` (2s @ 16kHz), `sample_segmentation_mask` (5-class), `sample_parameters` (100 frames, 10 dims)

---

## Project Milestones

| Phase | Status | Key Outcome |
|-------|--------|-------------|
| Phase 1: Data Pipeline | ✅ Complete | U-Net 81.8% Dice, 468 utterances |
| Phase 2-A: Baseline LSTM | ✅ Complete | RMSE 1.011, PCC 0.105 |
| Phase 3: Core Goal | ✅ Complete | Global PCC 0.1982, 21.5M Transformer |
| Phase 4: 정확도 개선 | 🔄 Active | HuBERT + Conformer + A100 (target PCC > 0.4) |
| Phase 5-1: GPU 서버 | ⬜ Planning | A100/A6000 + UV pipeline |
| Phase 5-2: NAS 데이터 | ⬜ Planning | 600GB+ streaming DataLoader |
| Phase 5-3: 웹 데모 | ⬜ Planning | Dataset viewer, training dashboard |

---

## Key Documentation

- `docs/guides/COLAB_TRAINING_GUIDE.md` — Google Colab GPU training
- `docs/guides/PHASE4_ACCURACY_GUIDE.md` — Phase 4 accuracy pipeline
- `docs/guides/PHASE5_GPU_QUICKSTART.md` — GPU server setup
- `docs/reports/FINAL_PHASE3_COMPLETION_REPORT.md` — Phase 3 results
- `docs/reports/PHASE4_FINAL_REPORT.md` — Phase 4 progress
- `researcher_manual.md` — Detailed research protocol (Korean)

---

## Git Workflow

```
Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```
Large files excluded via `.gitignore`: `data/`, `models/`, `logs/`, `venv_sullivan/`
