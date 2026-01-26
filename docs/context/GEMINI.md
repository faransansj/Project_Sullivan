# Project Sullivan Context for Gemini

## 1. Project Overview
**Project Sullivan** is a research initiative to develop an **Acoustic-to-Articulatory Inversion** system.
**Goal:** Infer articulatory parameters (tongue position, jaw opening, etc.) directly from speech audio, using real-time MRI (rtMRI) data as ground truth.

**Current Status (Jan 2026):**
- **Phase:** Phase 1 (Preprocessing) is ~85% complete; Phase 2 (Modeling) has started.
- **Key Achievement:** U-Net segmentation model achieved 81.8% Dice score.
- **Immediate Bottleneck:** Processed data (HDF5, segments) is missing in the current local environment (`/home/Project_Sullivan/data`), requiring either fresh preprocessing or data download.

## 2. Technical Architecture

### Tech Stack
- **Language:** Python 3.9+
- **Deep Learning:** PyTorch 2.0+, PyTorch Lightning (training loop), Segmentation Models PyTorch (U-Net)
- **Audio/Data:** Librosa, OpenCV, h5py, Scipy
- **Config:** YAML (custom loader with interpolation)
- **Logging:** TensorBoard

### Data Pipeline
1.  **Raw Data:** USC-TIMIT dataset (Audio + rtMRI video).
2.  **Preprocessing (`src/preprocessing`):** Audio denoising, MRI/Audio alignment, HDF5 conversion.
3.  **Segmentation (`src/segmentation`):** U-Net extracts vocal tract masks from MRI frames.
4.  **Feature Extraction:**
    *   **Input:** Mel-spectrograms or MFCCs from Audio (`src/audio_features`).
    *   **Target:** Geometric parameters (14-dim) or PCA components (10-dim) from MRI masks (`src/parameter_extraction`).
5.  **Modeling (`src/modeling`):**
    *   **Baseline:** Bi-LSTM (`baseline_lstm.py`).
    *   **Main:** Transformer Encoder (`transformer.py`) predicting parameters from audio features.

## 3. Key Files & Directories

- **`src/`**: Core library code.
    - `modeling/transformer.py`: Main Transformer architecture (Encoder-only, temporal loss).
    - `modeling/dataset.py`: PyTorch Dataset implementation.
- **`scripts/`**: Executable entry points.
    - `train_transformer.py`: Main training script for Phase 2-B.
    - `batch_preprocess.py`: Driver for raw -> aligned HDF5 conversion.
    - `segment_subset.py`: Runs U-Net inference on aligned data.
- **`configs/`**: Configuration files (YAML).
    - `transformer_config.yaml`: Production config for Transformer training.
    - `preprocess.yaml`: Configuration for data preprocessing.
- **`data/`**: Data storage (Git-ignored).
    - `raw/`: Raw USC-TIMIT data (read-only).
    - `processed/`: Aligned HDF5, segmentations, extracted features (currently empty/incomplete).

## 4. Operational Commands

### Environment Setup
```bash
source venv_sullivan/bin/activate
pip install -r requirements.txt
```

### Data Preprocessing (Phase 1)
**Step 1: Alignment (Raw -> HDF5)**
```bash
python scripts/batch_preprocess.py --config configs/preprocess.yaml
```

**Step 2: Segmentation (HDF5 -> Masks)**
```bash
python scripts/segment_subset.py --data-root data/processed/aligned --output-dir data/processed/segmentations --checkpoint models/segmentation/unet_best.pth
```

### Training (Phase 2)
**Train Transformer (Main Model)**
```bash
python scripts/train_transformer.py --config configs/transformer_config.yaml --gpus 1
```

**Train Baseline (Quick Test)**
```bash
python scripts/train_baseline.py --config configs/baseline_quick_test.yaml --fast-dev-run
```

**Monitor Training**
```bash
tensorboard --logdir logs/training
```

## 5. Development Conventions

- **Configuration:** Use YAML files in `configs/`. Do not hardcode hyperparameters in scripts.
- **Style:** Adhere to `black` formatting and `flake8` linting.
- **Typing:** Use Python type hints (`typing` module) for function signatures.
- **Testing:** Run `pytest` for unit/integration tests.
- **Safety:** Always check data paths before running heavy processing scripts (`ls -lh` checks).

## 6. Known Issues / Context
- **Missing Data:** The local `data/processed` folder is likely empty. Before training, check if processed data exists. If not, suggest running the preprocessing pipeline on a small subset (e.g., `sub011`) for verification.
- **Compute:** Transformer training requires GPU (or very slow CPU). Preprocessing is CPU-intensive.
