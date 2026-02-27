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

We have successfully progressed from initial infrastructure to full **High-Resolution Shape Recovery**.

- **Phase 4 Master Model**: A 21.5M parameter Transformer Encoder.
- **Output**: 24 Dimensions (14 Geometric Features + 10 PCA Components).
- **Performance**:
    - **Global PCC**: **0.1982** (7.6x improvement over Phase 2 baseline).
    - **High-Fidelity Tracking**: Successfully recovers critical articulatory gestures like Jaw Opening (PCC 0.50) and Tongue Fronting (PCC 0.46).
- **Phase 2 Baseline (Legacy)**: Bi-LSTM architecture established the feasibility of the mapping (RMSE 1.011).

---

## 📚 Documentation Index

Detailed documentation is organized by research phase and operational needs into subdirectories for cleaner navigation.

### 📖 Operational Guides (`docs/guides/`)
Everything you need to set up, run, and understand the project at a technical level.
- 🚀 **[Google Colab Quick Start](docs/guides/COLAB_QUICK_START.md)** - Training in cloud environments.
- 💻 **[Environment Setup Required](docs/guides/ENVIRONMENT_SETUP_REQUIRED.md)** - Local setup instructions.
- 📊 **[Dataset Usage Guide](docs/guides/DATASET_USAGE_GUIDE.md)** - Handling HDDB and USC-TIMIT features.
- 🌐 **[Web Demo Guide](docs/guides/QUICK_START_DEMO.md)** - Setup and usage for the interactive Phase 5 demo.
- 🧑‍🔬 **[Researcher Manual](docs/guides/researcher_manual.md)** - Comprehensive technical manual for the pipeline.

### 📊 Reports & Analysis (`docs/reports/`)
Historical records, milestone completions, and status updates.
- 🏆 **[Phase 4 Final Report](docs/reports/PHASE4_FINAL_REPORT.md)** - High-Resolution Shape Recovery and Joint Tuning.
- 📈 **[Phase 3 Completion Report](docs/reports/FINAL_PHASE3_COMPLETION_REPORT.md)** - Scaling to full dataset and RMSE optimization.
- 🛠️ **[Current Status](docs/reports/CURRENT_STATUS.md)** - Live tracking of milestones and active tasks.
- 🧠 **[Segmentation Pipeline Methodology](docs/reports/METHODOLOGY_SEGMENTATION_PIPELINE.md)** - Core approach for MRI segmentation.

---

## 🛠️ Tech Stack

### Core Technologies
- **Framework**: PyTorch, PyTorch Lightning, Gradio (Phase 5)
- **Architecture**: Transformer Encoder (6 layers, 8 heads, d_model=512)

### Data Pipeline
- **Input**: 80-band Mel-spectrograms (Librosa)
- **Target**: 14 Geometric Features (OpenCV) + 10 PCA Components (Scikit-Learn)
- **Optimization**: AdamW, Staged Curriculum Learning, Hybrid Loss (MSE + PCC)

---

## 🚦 Quick Start

### 1. Environment Setup
```bash
git clone https://github.com/faransansj/Project_Sullivan.git
cd Project_Sullivan
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Interactive Demo (Phase 5)
Experience the system in real-time using your microphone:
```bash
python scripts/app.py
```
*See **[QUICK_START_DEMO.md](docs/guides/QUICK_START_DEMO.md)** for more details.*

### 3. Evaluate Master Model
```bash
python scripts/evaluate_phase4d.py
```

---

## 📅 Milestone History

| Milestone | Target | Status | Key Outcome |
| :--- | :--- | :--- | :--- |
| **M1: Data Pipeline** | Phase 1 | ✅ Complete | MRI segmentation complete (81.8% Dice). |
| **M2: Baseline Model** | Phase 2 | ✅ Complete | Bi-LSTM infrastructure & initial mapping. |
| **M3: Core Goal** | Phase 3/4 | ✅ Complete | **Global PCC 0.198**, PCA Reconstruction. |
| **M4: Application** | Phase 5/6 | 🚧 Active | Web Demo & A100 High Performance. |
| **M5: GPU 서버 환경** | Phase 7-1 | ⬜ Planned | A100/A6000 + UV pipeline. |
| **M6: NAS 데이터 연계** | Phase 7-2 | ⬜ Planned | 600GB+ streaming DataLoader. |
| **M7: 웹 모니터링** | Phase 7-3 | ⬜ Planned | Dataset viewer & training dashboard. |

---

<div align="center">
  <b>Project Lead</b>: Midori (AI Agent) • <b>Last Update</b>: Feb 2026
</div>