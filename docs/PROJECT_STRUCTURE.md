# 📂 Project Sullivan: Repository Structure

This document outlines the organized directory structure for Project Sullivan, optimized for Phase 4 and Phase 5 development.

## 📁 Root Directory (`/`)
- `README.md`: Entry point with project overview and phase roadmap.
- `pyproject.toml` / `uv.lock`: Dependency management (UV-based).
- `configs/`: Configuration YAML files.
- `scripts/`: Executable entry points for training, data processing, and utilities.
- `src/`: Core library code (modeling, inference, data handling).
- `docs/`: Comprehensive project documentation.
- `legacy/`: Archive of old test scripts and previous phase code.

---

## 💻 `src/` - Core Library
Contains reusable modules for the entire pipeline.
- `src/audio_features/`: Featrue extractors (Mel-spectrogram, HuBERT).
- `src/inference/`: Production `engine.py` for model predictions.
- `src/modeling/`: PyTorch models (`conformer_model.py`, `transformer.py`, datasets).
- `src/parameter_extraction/`: Extracting geometric and PCA parameters from masks.
- `src/preprocessing/`: Data loading, denoising, and MRI-audio alignment.
- `src/segmentation/`: U-Net based vocal tract mask generation.
- `src/utils/`: Shared utilities (logging, config parsing).

---

## 🚀 `scripts/` - Execution Entry Points
Executable scripts categorized by their purpose.
- **Training**:
  - `train_conformer.py`: Phase 4 Conformer training.
  - `train_transformer.py`: Phase 3 Master Model training.
- **Preprocessing**:
  - `batch_preprocess.py`: Runs full preprocessing pipeline (denoise, align).
  - `segment_subset.py`: Segments aligned dataset.
  - `extract_audio_features.py`: Extracts model input features.
  - `extract_articulatory_params.py`: Extracts target parameters.
- **Infrastructure (`scripts/infra/`)**:
  - `full_preprocess_pipeline.sh`: Automated NAS preprocessing workflow.
  - `setup_remote_env.sh`: Initializes remote A100 GPU servers.
  - `remote_train.sh`: Rsyncs code and runs background training on GPUs.
  - `transfer_to_gpu.sh`: Transfers processed features to GPU servers.
- **Demo (`scripts/web/` - *Planned*)**:
  - `app.py`: Gradio web interface.

---

## ⚙️ `configs/` - Configurations
YAML config files driving the core scripts.
- `conformer_a100_config.yaml`: Heavy training on A100 GPUs.
- `preprocess.yaml`: Standard preprocessing pipeline.
- `preprocess_nas.yaml`: CPU-optimized NAS processing config.
- `infra/ssh_config_template`: Example SSH setup for connecting to remote GPUs.

---

## 📚 `docs/` - Documentation
Project history, reports, and how-to guides.
- `docs/reports/`: Executive summaries and phase completion reports (e.g., `PHASE4_FINAL_REPORT.md`).
- `docs/plans/`: Future implementation plans (e.g., `PHASE5_1_GPU_SERVER.md`).
- `docs/guides/`: Actionable HOW-TO guides for environment setup, training, and deployment.
- `docs/context/`: Core architecture discussions and AI specific instructions like `GEMINI.md`.

---

## 📦 Data & Models (Ignored by Git)
- `data/raw/`: Original USC-TIMIT / HDDB downloads.
- `data/processed/`: Aligned HDF5, segmentations, extracted features.
- `models/`: Saved `.ckpt` files for U-Net, Transformer, Conformer.
- `logs/`: TensorBoard logs and script outputs.
