# Project Sullivan: Acoustic-to-Articulatory Inversion

**Project Sullivan** is an advanced research initiative to develop a deep learning system capable of inferring high-resolution articulatory parameters (tongue position, jaw opening, lip shape) directly from speech audio.

Using the **HDDB (Haskins IEEE-DB)** dataset, the system reconstructs the vocal tract shape with high fidelity, enabling potential applications in speech therapy, silent speech interfaces, and linguistic research.

![Master Animation](results/final_deliverables/master_animation.gif)
*(Figure: Left - Ground Truth PCA Reconstruction, Right - Predicted Shape from Audio)*

---

## 🚀 Key Achievements (Phase 4 Completed)

We have successfully moved beyond simple geometric tracking to full **High-Resolution Shape Recovery**.

- **Master Model**: A 21.5M parameter Transformer Encoder.
- **Output**: 24 Dimensions (14 Geometric Features + 10 PCA Components).
- **Performance**:
    - **Global PCC**: **0.1982** (7.6x improvement over initial baseline).
    - **High-Fidelity Tracking**: Successfully recovers critical articulatory gestures like Jaw Opening (PCC 0.50) and Tongue Fronting (PCC 0.46).
- **Methodology**: Implemented **Staged Curriculum Learning** to solve gradient dilution issues in multi-task learning.

---

## 🛠️ Tech Stack

- **Framework**: PyTorch, PyTorch Lightning
- **Architecture**: Transformer Encoder (6 layers, 8 heads, d_model=512)
- **Data Pipeline**:
    - **Input**: 80-band Mel-spectrograms (Librosa)
    - **Target**: 14 Geometric Features (OpenCV) + 10 PCA Components (Scikit-Learn) from rtMRI.
- **Optimization**: AdamW, Cosine Annealing, Hybrid Loss (MSE + PCC + Temporal).

---

## 📂 Project Structure

```
Project_Sullivan/
├── configs/                 # YAML configuration files
│   ├── transformer_phase4d_joint.yaml  # Final Master Model Config
│   └── ...
├── data/                    # Data storage (Git-ignored)
│   ├── processed/           # Audio features, Parameters, Splits
│   └── ...
├── docs/                    # Documentation & Reports
│   ├── archive/             # Archived progress reports
│   └── ...
├── models/                  # Saved Model Checkpoints
├── notebooks/               # Jupyter Notebooks for EDA
├── results/                 # Evaluation Results & Visualizations
│   └── final_deliverables/  # Master Animation & Key Metrics
├── scripts/                 # Executable Scripts
│   ├── train_phase4d_joint.py    # Joint Fine-Tuning Script
│   ├── evaluate_phase4d.py       # Final Evaluation Script
│   └── compare_reconstruction.py # Visualization Script
└── src/                     # Source Code
    ├── modeling/            # Transformer & Dataset Logic
    └── preprocessing/       # Feature Extraction
```

---

## 🚦 Quick Start

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/faransansj/Project_Sullivan.git
cd Project_Sullivan

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Inference (Evaluation)
To evaluate the pre-trained Master Model on the HDDB test set:
```bash
python scripts/evaluate_phase4d.py
```

### 3. Visualize Reconstruction
To generate the side-by-side animation of the vocal tract:
```bash
python scripts/compare_reconstruction.py
```

---

## 📅 Milestone History

| Phase | Goal | Status | Key Outcome |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Data Preprocessing (USC-TIMIT) | ✅ Complete | Data pipeline established. |
| **Phase 2** | Baseline Modeling (Bi-LSTM) | ✅ Complete | Initial feasibility proven. |
| **Phase 3** | Transformer Baseline | ✅ Complete | Solved mean-shape collapse (RMSE 0.05). |
| **Phase 4** | **HDDB High-Res Recovery** | ✅ **Complete** | **Global PCC 0.198**, PCA Shape Reconstruction. |
| **Phase 5** | Optimization & Clinical | 🚧 Next | Real-time inference & Dysarthria testing. |

---

## 📝 Key Reports

For detailed technical insights, please refer to the following reports:

- **[Phase 4 Final Report](PHASE4_FINAL_REPORT.md)**: Detailed analysis of the High-Resolution PCA Recovery, Gradient Dilution solution, and final metrics.
- **[Current Status](CURRENT_STATUS.md)**: Live tracking of the project's state.

---

**Project Lead**: Midori (AI Agent)
**Last Update**: Jan 2026