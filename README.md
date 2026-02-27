# 🗣️ Project Sullivan: Acoustic-to-Articulatory Inversion

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**Project Sullivan** is an advanced research initiative to develop a deep learning system capable of inferring high-resolution articulatory parameters (tongue position, jaw opening, lip shape) directly from speech audio. 

Using the **HDDB (Haskins IEEE-DB)** and **USC-TIMIT** datasets, the system reconstructs the vocal tract shape with high fidelity, enabling potential applications in speech therapy, silent speech interfaces, and linguistic research.

<div align="center">
  <img src="results/final_deliverables/master_animation.gif" alt="Master Animation" width="600"/>
  <p><i>Left - Ground Truth PCA Reconstruction, Right - Predicted Shape from Audio</i></p>
</div>

---

## 🚀 Key Achievements

We have successfully progressed from initial infrastructure to full **High-Resolution Shape Recovery** and are currently enhancing accuracy.

- **Phase 4 Accuracy Pipeline (Current)**: High-performance Conformer architecture (12 layers, d_model=512) with HuBERT self-supervised features and full hybrid loss. Optimized for A100 training.
- **Phase 3 Master Model (Legacy)**: A 21.5M parameter Transformer Encoder.
- **Performance (Phase 3)**:
    - **Global PCC**: **0.1982** (7.6x improvement over Phase 2 baseline).
    - **High-Fidelity Tracking**: Successfully recovers critical articulatory gestures like Jaw Opening (PCC 0.50) and Tongue Fronting (PCC 0.46).

---

## 📚 Documentation Index

Detailed documentation is organized by research phase and operational needs into subdirectories for cleaner navigation.

### 📖 Operational Guides (`docs/guides/`)
Everything you need to set up, run, and understand the project at a technical level.
- 🚀 **[Phase 5 GPU Quick Start](docs/guides/PHASE5_GPU_QUICKSTART.md)** - Training on external GPU servers (A100/A6000) with UV.
- 🎯 **[Phase 4 Accuracy Guide](docs/guides/PHASE4_ACCURACY_GUIDE.md)** - Using the Conformer model and HuBERT features.
- 💻 **[Environment Setup Required](docs/guides/ENVIRONMENT_SETUP_REQUIRED.md)** - Local setup instructions.
- 📊 **[Dataset Usage Guide](docs/guides/DATASET_USAGE_GUIDE.md)** - Handling HDDB and USC-TIMIT features.

### 📊 Reports & Analysis (`docs/reports/`)
Historical records, milestone completions, and status updates.
- 🏆 **[Phase 4 Final Report](docs/reports/PHASE4_FINAL_REPORT.md)** - High-Resolution Shape Recovery and Joint Tuning.
- 📈 **[Phase 3 Completion Report](docs/reports/FINAL_PHASE3_COMPLETION_REPORT.md)** - Scaling to full dataset and RMSE optimization.
- 🛠️ **[Current Status](docs/reports/CURRENT_STATUS.md)** - Live tracking of milestones and active tasks.
- 🧠 **[Segmentation Pipeline Methodology](docs/reports/METHODOLOGY_SEGMENTATION_PIPELINE.md)** - Core approach for MRI segmentation.

---

## 🛠️ Tech Stack

### Core Technologies
- **Package Manager**: UV (`uv run`, `uv sync`)
- **Framework**: PyTorch, PyTorch Lightning
- **Architecture**: Conformer (Phase 4), Transformer Encoder (Phase 3)

### Data Pipeline
- **Input**: HuBERT-Large (1024-dim, Phase 4), 80-band Mel-spectrograms (Phase 3)
- **Target**: 14 Geometric Features (OpenCV) + 10 PCA Components (Scikit-Learn)
- **Optimization**: AdamW, OneCycleLR, Hybrid Loss (MSE + PCC + Temporal)

---

## 🚦 Quick Start

### 1. Environment Setup (UV)
```bash
git clone https://github.com/faransansj/Project_Sullivan.git
cd Project_Sullivan
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra gpu
```

### 2. Train Conformer Model (Phase 4)
```bash
uv run python scripts/train_conformer.py --config configs/conformer_a100_config.yaml --gpus 1
```

### 3. Remote GPU Server Execution (Phase 5-1)
Deploy code, start training in background, and view logs across SSH.
```bash
./scripts/infra/remote_train.sh user@gpu-server configs/conformer_a100_config.yaml train_conformer.py
```

---

## 📅 Milestone History

| Milestone | Target | Status | Key Outcome |
| :--- | :--- | :--- | :--- |
| **M1: Data Pipeline** | Phase 1 | ✅ Complete | MRI segmentation complete (81.8% Dice). |
| **M2: Baseline Model** | Phase 2 | ✅ Complete | Bi-LSTM infrastructure & initial mapping. |
| **M3: Core Goal & Shape Recovery** | Phase 3 | ✅ Complete | **Global PCC 0.198**, PCA Reconstruction. |
| **M4: 정확도 개선 (Conformer)** | Phase 4 | ✅ Complete | HuBERT, Conformer 코드/가이드 완료. |
| **M5: GPU 서버 구축/UV** | Phase 5-1 | ✅ Complete | 원격 학습 스크립트, UV 환경 초기화 완. |
| **M6: NAS 데이터 연계** | Phase 5-2 | ⬜ Planned | 600GB+ 하이브리드 전송 전략 (rsync+streaming). |
| **M7: 웹 모니터링 / 데모** | Phase 5-3 | ⬜ Planned | Gradio 기반 대시보드 구조 및 데모 기획. |

---

<div align="center">
  <b>Project Lead</b>: Midori (AI Agent) • <b>Last Update</b>: Feb 2026
</div>