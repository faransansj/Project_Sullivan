# Project Sullivan: Acoustic-to-Articulatory Inversion

**Project Sullivan** is an advanced research initiative to develop a deep learning system capable of inferring high-resolution articulatory parameters (tongue position, jaw opening, lip shape) directly from speech audio.

Using the **HDDB (Haskins IEEE-DB)** and **USC-TIMIT** datasets, the system reconstructs the vocal tract shape with high fidelity, enabling potential applications in speech therapy, silent speech interfaces, and linguistic research.

![Master Animation](results/final_deliverables/master_animation.gif)
*(Figure: Left - Ground Truth PCA Reconstruction, Right - Predicted Shape from Audio)*

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

Detailed documentation is organized by research phase and operational needs.

### 📊 Reports & Analysis
- **[Phase 4 Final Report](docs/reports/PHASE4_FINAL_REPORT.md)**: High-Resolution Shape Recovery and Joint Tuning.
- **[Phase 3 Completion Report](docs/archive/phase3/FINAL_PHASE3_COMPLETION_REPORT.md)**: Scaling to full dataset and RMSE optimization.
- **[Phase 2 Setup Report](docs/archive/phase1_2/PHASE2_SETUP_COMPLETE.md)**: Audio-to-parameter modeling infrastructure.
- **[Current Status](CURRENT_STATUS.md)**: Live tracking of milestones and active tasks.

### 📖 Operational Guides
- **[Web Demo Guide](docs/guides/WEB_DEMO_GUIDE.md)**: Setup and usage for the interactive Phase 5 demo.
- **[Dataset Usage Guide](docs/guides/DATASET_USAGE_GUIDE.md)**: Handling HDDB and USC-TIMIT features.
- **[Google Colab Quick Start](docs/guides/COLAB_QUICK_START.md)**: Training in cloud environments.

---

## 🛠️ Tech Stack

- **Framework**: PyTorch, PyTorch Lightning, Gradio (Phase 5)
- **Architecture**: Transformer Encoder (6 layers, 8 heads, d_model=512)
- **Data Pipeline**:
    - **Input**: 80-band Mel-spectrograms (Librosa)
    - **Target**: 14 Geometric Features (OpenCV) + 10 PCA Components (Scikit-Learn).
- **Optimization**: AdamW, Staged Curriculum Learning, Hybrid Loss (MSE + PCC).

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
*See **[QUICK_START_DEMO.md](QUICK_START_DEMO.md)** for more details.*

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
| **M4: Application** | Phase 5 | 🚧 Active | Web Demo & Real-time Optimization. |

---

**Project Lead**: Midori (AI Agent)
**Last Update**: Feb 2026