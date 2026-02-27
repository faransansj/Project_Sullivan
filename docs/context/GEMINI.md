# Project Sullivan Context for Gemini

## 1. Project Overview
**Project Sullivan** is a research initiative to develop an **Acoustic-to-Articulatory Inversion** system.
**Goal:** Infer articulatory parameters (tongue position, jaw opening, etc.) directly from speech audio, using real-time MRI (rtMRI) data as ground truth.

**Current Status (Feb 2026):**
- **Phase 1–4:** Complete (Data Pipeline, Baseline LSTM, Full-Scale Training, High-Res Shape Recovery)
- **Phase 5/6:** Active (Inference Engine, A100 HuBERT training, Gradio UI)
- **Phase 7:** Planning (외부 GPU 서버 환경, NAS 데이터 연계, 웹 데모)
- **Key Achievement:** Phase 4 — Global PCC 0.1982, 21.5M param Transformer, 24-dim output
- **Immediate Focus:** Phase 7 infrastructure setup

## 2. Technical Architecture

### Tech Stack
- **Language:** Python 3.9+
- **Deep Learning:** PyTorch 2.0+, PyTorch Lightning, Segmentation Models PyTorch (U-Net)
- **Audio/Data:** Librosa, OpenCV, h5py, Scipy
- **Config:** YAML (custom loader with interpolation)
- **Package Manager:** UV (`uv sync`, `uv run`)
- **Logging:** TensorBoard

### Data Pipeline
1.  **Raw Data:** USC-TIMIT + HDDB datasets (Audio + rtMRI video).
2.  **Preprocessing (`src/preprocessing`):** Audio denoising, MRI/Audio alignment, HDF5 conversion.
3.  **Segmentation (`src/segmentation`):** U-Net extracts vocal tract masks from MRI frames.
4.  **Feature Extraction:**
    *   **Input:** Mel-spectrograms or MFCCs from Audio (`src/audio_features`).
    *   **Target:** Geometric parameters (14-dim) + PCA components (10-dim) from MRI masks (`src/parameter_extraction`).
5.  **Modeling (`src/modeling`):**
    *   **Baseline:** Bi-LSTM (`baseline_lstm.py`).
    *   **Main:** Transformer Encoder (`transformer.py`) — 6 layers, 8 heads, d_model=512.

## 3. Key Files & Directories

- **`src/`**: Core library code.
    - `modeling/transformer.py`: Main Transformer architecture (Encoder-only, temporal loss).
    - `modeling/dataset.py`: PyTorch Dataset implementation.
- **`scripts/`**: Executable entry points.
    - `train_transformer.py`: Main training script.
    - `batch_preprocess.py`: Raw → aligned HDF5 conversion.
    - `segment_subset.py`: Runs U-Net inference on aligned data.
    - `app.py`: Gradio web demo.
- **`configs/`**: Configuration files (YAML).
    - `transformer_config.yaml`: Production config for Transformer training.
    - `preprocess.yaml`: Configuration for data preprocessing.
- **`data/`**: Data storage (Git-ignored).
    - `raw/`: Raw USC-TIMIT data (read-only). 600GB+ on NAS.
    - `processed/`: Aligned HDF5, segmentations, extracted features.

## 4. Operational Commands

### Environment Setup (UV)
```bash
uv sync                     # Install all dependencies from uv.lock
uv run python scripts/...   # Run scripts in UV environment
```

### Training
```bash
uv run python scripts/train_transformer.py --config configs/transformer_config.yaml --gpus 1
```

### Monitor Training
```bash
tensorboard --logdir logs/training
```

## 5. Development Conventions

- **Configuration:** Use YAML files in `configs/`. Do not hardcode hyperparameters in scripts.
- **Style:** Adhere to `black` formatting and `flake8` linting.
- **Typing:** Use Python type hints (`typing` module) for function signatures.
- **Testing:** Run `pytest` for unit/integration tests.
- **Package Management:** Use `uv` (not pip/conda). Add packages with `uv add <pkg>`.

## 6. Infrastructure Context

### NAS Server (Storage Only)
- **Data:** 600GB+ USC-TIMIT dataset
- **Path:** `/mnt/HDDB/dataset/my_dataset/dataset/`
- **Compute:** 780M GPU — **insufficient for training**, storage only

### External GPU Servers (Phase 7 Target)
- **A100 / A6000** servers for training
- UV-based pipeline for reproducible environments
- Data transfer strategy: NAS → GPU server (rsync / NFS / streaming)

## 7. Phase 7 Roadmap

### 7-1: 외부 GPU 서버 환경 (A100/A6000)
- UV 기반 재현 가능한 학습 환경
- SSH 원격 학습 워크플로우

### 7-2: 대용량 데이터 학습 (NAS 600GB+)
- Streaming DataLoader 구현
- NAS ↔ GPU 서버 데이터 연계

### 7-3: 웹 기반 데모 & 모니터링
- 데이터셋 품질 검증 뷰어
- 학습 진행 모니터링 대시보드
- 추론 데모 페이지
