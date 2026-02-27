# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Project Sullivan** is a research project developing AI models that infer articulatory parameters (tongue position, jaw opening, lip shape, etc.) from audio signals alone using the USC-TIMIT Speech MRI Dataset.

**Current Status**: Phase 5/6 Active, Phase 7 Planning
- Phase 2-A (Baseline LSTM) complete: Test RMSE 1.011, PCC 0.105
- Phase 2-B (Advanced architectures) in progress: Transformer implemented, training next
- M2 Target: RMSE < 0.15, PCC > 0.50

**Key Technologies**: PyTorch, PyTorch Lightning, rtMRI processing, audio feature extraction

---

## Essential Commands

### Environment Setup
```bash
# Activate virtual environment
source venv_sullivan/bin/activate

# Install dependencies (first time)
pip install -r requirements.txt

# Run tests
pytest                                    # All tests
pytest tests/unit                         # Unit tests only
pytest -m "not slow"                      # Skip slow tests
pytest --cov=src                          # With coverage
```

### Training Models

**Baseline LSTM (Phase 2-A - Complete)**
```bash
# Full training
python scripts/train_baseline.py --config configs/baseline_config.yaml

# Quick test
python scripts/train_baseline.py --config configs/baseline_quick_test.yaml
```

**Transformer (Phase 2-B - Current)**
```bash
# Full training (GPU recommended)
python scripts/train_transformer.py --config configs/transformer_config.yaml

# Quick test (CPU)
python scripts/train_transformer.py --config configs/transformer_quick_test.yaml

# With GPU specification
python scripts/train_transformer.py --config configs/transformer_config.yaml --gpus 1
```

### Monitoring Training
```bash
# Start TensorBoard
bash scripts/start_tensorboard.sh
# Access at http://localhost:6006

# Monitor training progress (simple)
bash scripts/monitor_training_simple.sh

# Check training status
bash scripts/check_training_status.sh
```

### Data Processing Pipeline (Phase 1 - Complete)

**Preprocessing (MRI/Audio alignment)**
```bash
python scripts/batch_preprocess.py --config configs/preprocess.yaml
```

**Segmentation (U-Net vocal tract segmentation)**
```bash
# Segment specific subjects
python scripts/segment_subset.py \
  --data-root data/raw/usc_timit_full \
  --subjects sub013,sub014,sub015 \
  --output-dir data/processed/segmentations \
  --checkpoint models/segmentation/unet_best.pth

# Full dataset segmentation
python scripts/segment_full_dataset.py
```

**Feature Extraction**
```bash
# Extract articulatory parameters (geometric + PCA features)
python scripts/extract_articulatory_params.py

# Extract audio features (mel-spectrogram + MFCC)
python scripts/extract_audio_features.py

# Create train/val/test splits
python scripts/create_dataset_splits.py
```

### Google Colab Training (Recommended for GPU)
```bash
# Prepare data archives for Colab
bash scripts/prepare_data_for_colab.sh

# See docs/COLAB_TRAINING_GUIDE.md for full instructions
# Use notebook: notebooks/Project_Sullivan_Transformer_Training.ipynb
```

### Code Quality
```bash
# Format code
black src/ scripts/ tests/

# Lint
flake8 src/ scripts/ tests/

# Type checking
mypy src/
```

---

## Architecture & Code Structure

### Data Flow Pipeline

**Phase 1: Data Preprocessing (Complete)**
```
Raw MRI + Audio
    ↓
[1] Alignment & Denoising (src/preprocessing/)
    ↓ HDF5 files → data/processed/aligned/
[2] U-Net Segmentation (src/segmentation/)
    ↓ Masks → data/processed/segmentations/
[3] Parameter Extraction (src/parameter_extraction/)
    ↓ 14 geometric + 10 PCA features → data/processed/parameters/
[4] Audio Feature Extraction (src/audio_features/)
    ↓ Mel-spectrogram (80-dim) + MFCC (13-dim) → data/processed/audio_features/
[5] Dataset Splits (70/15/15)
    ↓ JSON manifests → data/processed/splits/
```

**Phase 2: Model Training (Current)**
```
Audio Features + Articulatory Parameters
    ↓
Dataset (src/modeling/dataset.py)
    ↓
Model Training (PyTorch Lightning)
    - BaselineLSTM (src/modeling/baseline_lstm.py) - 613K params
    - TransformerModel (src/modeling/transformer.py) - 5M params
    ↓
Trained Models → models/
Logs → logs/training/
```

### Key Modules

**src/preprocessing/** - Phase 1: Data preprocessing
- `data_loader.py` - Load USC-TIMIT rtMRI and audio data
- `alignment.py` - MRI/audio temporal alignment using cross-correlation
- `denoising.py` - Gaussian and median filtering for MRI frames

**src/segmentation/** - U-Net vocal tract segmentation
- `unet.py` - U-Net architecture (5-layer encoder-decoder, 81.8% Dice score)
- `dataset.py` - Segmentation dataset loader
- Pre-trained model: `models/segmentation/unet_best.pth`

**src/parameter_extraction/** - Extract articulatory features from segmented MRI
- `geometric_features.py` - 14 geometric features (tongue position, jaw opening, etc.)
- `pca_features.py` - 10 PCA components from segmentation masks

**src/audio_features/** - Audio feature extraction
- `mel_spectrogram.py` - 80-dimensional mel-spectrogram (primary features)
- `mfcc.py` - 13-dimensional MFCC (alternative features)

**src/modeling/** - Phase 2: Neural network models
- `dataset.py` - PyTorch Dataset for audio-articulatory pairs
  - Handles variable-length sequences with padding/masking
  - Supports streaming from zip archives for large datasets
  - Normalizes parameters to [0, 1] range
- `baseline_lstm.py` - Bidirectional LSTM baseline (Phase 2-A)
  - 2-layer Bi-LSTM, 128 hidden units, 613K params
  - Input: 80-dim mel / Output: 14-dim geometric params
- `transformer.py` - Transformer encoder model (Phase 2-B)
  - 4 layers, 8 heads, d_model=256, d_ff=1024, 5M params
  - Learnable or sinusoidal positional encoding
  - Input projection: audio_dim → d_model
  - Output projection: d_model → param_dim
- `positional_encoding.py` - Positional encoding implementations
- `model_utils.py` - Shared utilities (padding masks, loss masks, activation functions)

**src/utils/** - Shared utilities
- `config.py` - YAML configuration loading
- `io_utils.py` - File I/O helpers
- `logger.py` - Logging setup

### Model Implementation Notes

**PyTorch Lightning Structure**:
All models inherit from `pl.LightningModule` with standard methods:
- `__init__()` - Model architecture definition
- `forward()` - Forward pass
- `training_step()` - Training logic with loss calculation
- `validation_step()` - Validation logic
- `test_step()` - Test evaluation
- `configure_optimizers()` - Optimizer and LR scheduler setup
- `on_train_epoch_end()` / `on_validation_epoch_end()` - Epoch-level metrics

**Variable-Length Sequence Handling**:
- Dataset returns sequences with padding mask
- Models use mask to ignore padded frames in loss calculation
- MSE loss computed only on valid (non-padded) frames

**Training Configuration**:
- All hyperparameters defined in YAML configs (`configs/`)
- Supports multiple config variants (full, quick_test, cpu_test)
- Config interpolation: `${data.audio_feature_type}` resolves to actual values

### Dataset Format

**USC-TIMIT Data Structure**:
```
data/raw/usc_timit_full/
├── sub001/
│   ├── uw001_01/
│   │   ├── uw001_01_rtMRI.mat       # Real-time MRI video
│   │   └── uw001_01_audio.wav       # Synchronized audio
│   └── ...
└── ...
```

**Processed Data Structure**:
```
data/processed/
├── aligned/          # MRI+Audio aligned (HDF5)
├── segmentations/    # Vocal tract masks (NPZ)
├── parameters/       # Articulatory params (NPZ)
├── audio_features/   # Mel/MFCC features (NPZ)
└── splits/          # Train/val/test manifests (JSON)
    ├── train.json
    ├── val.json
    └── test.json
```

**Split Manifest Format** (JSON):
```json
[
  "sub001/uw001_01",
  "sub001/uw001_02",
  ...
]
```

---

## Important Implementation Details

### 1. Sequence Padding and Masking

The dataset uses variable-length sequences. Models must handle padding correctly:

```python
# Dataset returns: (features, params, mask)
# mask = 1 for valid frames, 0 for padded frames

# In model training_step:
loss_mask = create_loss_mask(lengths, max_len)
loss = ((predictions - targets) ** 2 * loss_mask).sum() / loss_mask.sum()
```

### 2. Configuration Resolution

Transformer config uses string interpolation that must be resolved:

```python
# In train_transformer.py
def load_config(config_path: str) -> dict:
    # Load YAML and resolve ${variable} references
    # See scripts/train_transformer.py for full implementation
```

### 3. U-Net Segmentation Model

The U-Net expects single-channel input (grayscale MRI frames):
- Input shape: `(B, 1, H, W)` - batch, channels, height, width
- Output shape: `(B, n_classes, H, W)` - class probabilities per pixel
- Pre-trained checkpoint must be loaded for parameter extraction pipeline

### 4. Feature Normalization

Articulatory parameters are normalized per-feature:
```python
# In dataset._compute_normalization_stats()
# Computes min/max across all training data
# Normalizes to [0, 1] range for stable training
```

### 5. PyTorch Lightning Training Loop

Standard training pattern:
```python
trainer = pl.Trainer(
    max_epochs=config['training']['num_epochs'],
    callbacks=[early_stopping, checkpoint],
    logger=tensorboard_logger,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    precision=config['training']['precision']
)
trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)
```

### 6. Streaming Dataset for Large Data

For large datasets, use streaming mode to avoid loading all data into memory:
```python
dataset = ArticulatoryDataset(
    ...,
    streaming=True,
    zip_file_path='path/to/data.zip'
)
```

---

## Testing Strategy

**Test Structure**:
```
tests/
├── unit/          # Fast, isolated tests (< 1s each)
├── integration/   # Multi-component tests
└── conftest.py    # Shared fixtures (synthetic data)
```

**Key Fixtures** (from `conftest.py`):
- `sample_mri_frame/sequence` - Synthetic MRI data (256x256)
- `sample_audio` - Synthetic audio (2s @ 16kHz)
- `sample_segmentation_mask` - Synthetic 5-class mask
- `sample_parameters` - Synthetic articulatory params (100 frames, 10 dims)

**Test Markers**:
- `@pytest.mark.unit` - Unit tests (fast)
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow tests (> 5s)
- `@pytest.mark.gpu` - Requires GPU
- `@pytest.mark.data` - Requires downloaded data

---

## Common Development Workflows

### Adding a New Model Architecture

1. Create model file: `src/modeling/new_model.py`
   - Inherit from `pl.LightningModule`
   - Implement required methods (forward, training_step, etc.)
   - Use `model_utils.py` helper functions

2. Create config: `configs/new_model_config.yaml`
   - Follow structure of existing configs
   - Define model hyperparameters, training settings

3. Create training script: `scripts/train_new_model.py`
   - Copy structure from `train_transformer.py`
   - Add model-specific logic

4. Test with quick config:
   ```bash
   python scripts/train_new_model.py --config configs/new_model_quick_test.yaml
   ```

### Processing Additional Data

To add more subjects for training:

1. Ensure raw data exists: `data/raw/usc_timit_full/subXXX/`
2. Run preprocessing: `scripts/batch_preprocess.py`
3. Run segmentation: `scripts/segment_subset.py --subjects subXXX,subYYY`
4. Extract parameters: `scripts/extract_articulatory_params.py`
5. Extract audio features: `scripts/extract_audio_features.py`
6. Regenerate splits: `scripts/create_dataset_splits.py`

### Evaluating a Trained Model

```python
# Load checkpoint
model = TransformerModel.load_from_checkpoint('path/to/checkpoint.ckpt')

# Create test dataloader
test_loader = create_dataloaders(config, splits=['test'])['test']

# Test with trainer
trainer = pl.Trainer(accelerator='gpu')
results = trainer.test(model, test_loader)

# Results include: test_loss, test_rmse, test_mae, test_pearson
```

---

## Project Milestones and Targets

| Phase | Status | Key Outcome |
|-------|--------|-------------|
| **Phase 1: Data Pipeline** | ✅ Complete | MRI segmentation (81.8% Dice), 468 utterances |
| **Phase 2: Baseline Model** | ✅ Complete | Bi-LSTM: RMSE 1.011, PCC 0.105 |
| **Phase 3: Full-Scale Training** | ✅ Complete | Scaled dataset, RMSE optimization |
| **Phase 4: Shape Recovery** | ✅ Complete | **Global PCC 0.1982**, 21.5M param Transformer |
| **Phase 5: Inference Engine** | 🔄 Active | Web demo & real-time optimization |
| **Phase 6: A100 Training** | 🔄 Active | HuBERT features, Conformer upgrade |
| **Phase 7-1: GPU 서버 환경** | ⬜ Planning | A100/A6000 + UV pipeline |
| **Phase 7-2: NAS 데이터 연계** | ⬜ Planning | 600GB+ streaming DataLoader |
| **Phase 7-3: 웹 데모 & 모니터링** | ⬜ Planning | Dataset viewer, training dashboard |

**Current Focus**: Phase 5/6 completion, Phase 7 infrastructure planning.

---

## Key Documentation Files

- `README.md` - Project overview, quick start, milestones
- `researcher_manual.md` - Detailed research protocol (Korean)
- `docs/BASELINE_PERFORMANCE_REPORT.md` - Phase 2-A analysis
- `docs/M1_COMPLETION_REPORT.md` - Phase 1 completion status
- `docs/COLAB_TRAINING_GUIDE.md` - Google Colab GPU training
- `docs/NEXT_MILESTONES.md` - M2, M3, M4 roadmap
- `DATASET_USAGE_GUIDE.md` - How to use full USC-TIMIT dataset

---

## Critical Paths and Patterns

### Path Resolution
- Use absolute paths in configs: `data/processed/splits` (relative to project root)
- Scripts use `Path(__file__).parent.parent` to find project root
- Add `src/` to sys.path: `sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))`

### Configuration Pattern
All training scripts follow this pattern:
```python
config = load_config(args.config)  # Load YAML
model = create_model(config)       # Create model from config
loaders = create_dataloaders(config)  # Create train/val/test loaders
trainer = create_trainer(config)   # Create PyTorch Lightning trainer
trainer.fit(model, loaders['train'], loaders['val'])
trainer.test(model, loaders['test'])
```

### Logging Pattern
- TensorBoard logs: `logs/training/{experiment_name}/`
- Model checkpoints: `models/{model_name}/`
- Experiment logs follow naming: `{model_name}-{epoch:02d}-{val_loss:.4f}.ckpt`

### Git Workflow
- Main branch: `main`
- Commit with co-author tag:
  ```
  Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
  ```
- Large files excluded via `.gitignore`: `data/`, `models/`, `logs/`, `.venv/`

---

## Performance Optimization Notes

**Current Performance** (from M1 completion):
- U-Net segmentation: 22.8 fps (CPU)
- Selective segmentation: 75 utterances in 2.3 hours
- Training speed (baseline): ~30s/epoch on CPU, ~10s/epoch on GPU

**Optimization Strategies**:
1. Use GPU for training (set `device: cuda` in config)
2. Increase batch size if VRAM allows
3. Use mixed precision training (`precision: 16`)
4. Enable gradient accumulation if OOM (`accumulate_grad_batches`)
5. Use streaming dataset mode for large data (>10GB)

---

## Dependencies and Environment

**Python Version**: 3.9+

**Key Dependencies**:
- PyTorch 2.0+ (with torchvision, torchaudio)
- PyTorch Lightning 2.0+ (training framework)
- librosa (audio processing)
- opencv-python, scikit-image (image processing)
- nibabel, pydicom, SimpleITK (medical imaging)
- h5py (HDF5 file format)
- segmentation-models-pytorch (U-Net implementation)

**Code Quality Tools**:
- black (formatter, line-length=100)
- flake8 (linter)
- mypy (type checker)
- pytest (testing, with coverage)

**Virtual Environment**:
- Name: `venv_sullivan`
- Activate: `source venv_sullivan/bin/activate`

---

## Notes for Future Development

1. **Model Extensions**: Consider Conformer architecture (combines Transformer + Convolution) for improved temporal modeling

2. **Feature Engineering**: Current features are mel-spectrogram (80-dim). Could experiment with:
   - Combined mel + MFCC features
   - Raw waveform input with learned features
   - Prosodic features (pitch, energy, duration)

3. **Data Augmentation**: Currently not implemented. Could add:
   - Time stretching/shifting
   - Pitch shifting
   - Noise injection
   - SpecAugment for mel-spectrograms

4. **Multi-Task Learning**: Could add auxiliary tasks:
   - Phoneme recognition
   - Speaker identification
   - Speech/silence detection

5. **Attention Visualization**: Transformer attention weights could provide insights into which audio features drive specific articulatory movements

6. **Phase 7 Infrastructure** (Active Planning):
   - **외부 GPU 서버**: A100/A6000에서 UV 기반 학습 환경 구축
   - **NAS 데이터 연계**: 600GB+ 데이터셋 streaming 전략 (NAS 780M → GPU 서버)
   - **웹 데모 & 모니터링**: 데이터셋 뷰어, 학습 대시보드, 추론 데모
