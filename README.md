# Project Sullivan

**Acoustic-to-Articulatory Parameter Inference from Speech MRI Data**

![Status](https://img.shields.io/badge/Status-Phase%201-yellow)
![Milestone](https://img.shields.io/badge/Milestone-M1-blue)
![Python](https://img.shields.io/badge/Python-3.9+-green)

---

## 🎯 Research Goal

**Primary Goal (Current Focus):** Develop an AI model that infers articulatory parameters (tongue position, jaw opening, lip shape, etc.) from audio signals alone.

**Input:** Audio Waveform
**Output:** Articulatory Parameters (10-dimensional vector)

**Success Criteria:** RMSE < 0.10, Pearson Correlation > 0.70

---

## 📊 Current Status

**Current Milestone:** M2 - Baseline Model Development (**Phase 2-B In Progress** 🟡)
**Current Phase:** Phase 2-B (Advanced Architecture - Transformer Implementation)

**Progress:**
- [x] Project structure initialized
- [x] Requirements defined
- [x] Git repository created
- [x] Data downloaded from figshare (468 utterances, 15 subjects)
- [x] EDA completed
- [x] Preprocessing pipeline built (alignment, denoising)
- [x] **U-Net Segmentation Pipeline Complete** (81.8% test Dice score, +16.9% above target)
- [x] **Segmentation infrastructure ready** (scripts tested and validated)
- [x] **Selective dataset segmentation** (75 utterances, 186,124 frames - COMPLETE ✅)
- [x] **Articulatory parameter extraction** (14 geometric + 10 PCA features - COMPLETE ✅)
- [x] **Audio feature extraction** (Mel-spectrogram + MFCC - COMPLETE ✅)
- [x] **Train/Val/Test dataset splits** (70/15/15 ratio, subject-level - COMPLETE ✅)
- [x] **Baseline LSTM model implementation** (Bi-LSTM, 613K params - COMPLETE ✅)
- [x] **Training pipeline setup** (PyTorch Lightning, TensorBoard - COMPLETE ✅)
- [x] **Baseline model training** (18 epochs, early stopped - COMPLETE ✅)
- [x] **Model evaluation and baseline report** (Test RMSE: 1.011, PCC: 0.105 - COMPLETE ✅)
- [x] **Transformer architecture implementation** (5M params, tested - COMPLETE ✅)
- [ ] **Transformer model training** (Phase 2-B, Next)
- [ ] **Conformer architecture implementation** (Phase 2-B, Pending)
- [ ] **M2 target achievement** (RMSE < 0.15, PCC > 0.50 - Phase 2-B Goal)

---

## 🗂️ Project Structure

```
Project_Sullivan/
├── data/                          # Data directory (not in git)
│   ├── raw/                       # Original USC-TIMIT data
│   ├── processed/                 # Preprocessed data
│   │   ├── segmentations/         # Vocal tract masks
│   │   ├── parameters/            # Articulatory parameters
│   │   ├── audio_features/        # Mel-spectrogram, MFCC
│   │   └── splits/                # Train/val/test splits
│   └── experiments/               # Experiment-specific data
├── src/                           # Source code
│   ├── preprocessing/             # Phase 1: Data preprocessing
│   ├── segmentation/              # U-Net segmentation
│   ├── parameter_extraction/      # Geometric & PCA features
│   ├── audio_features/            # Audio feature extraction
│   ├── modeling/                  # Phase 2: Model development
│   │   ├── dataset.py             # PyTorch Dataset
│   │   ├── baseline_lstm.py       # Bi-LSTM model (Phase 2-A)
│   │   ├── transformer.py         # Transformer model (Phase 2-B)
│   │   ├── positional_encoding.py # Positional encodings
│   │   └── model_utils.py         # Shared utilities
│   ├── evaluation/                # Evaluation metrics
│   └── utils/                     # Utilities
├── scripts/                       # Standalone scripts
│   ├── train_baseline.py          # Baseline training (Phase 2-A)
│   ├── train_transformer.py       # Transformer training (Phase 2-B)
│   ├── monitor_training_simple.sh # Training monitor
│   └── start_tensorboard.sh       # TensorBoard launcher
├── notebooks/                     # Jupyter notebooks
│   └── 01_EDA.ipynb              # Exploratory data analysis
├── configs/                       # Configuration files
│   ├── baseline_config.yaml       # Baseline LSTM config (Phase 2-A)
│   ├── baseline_quick_test.yaml   # Baseline quick test
│   ├── transformer_config.yaml    # Transformer config (Phase 2-B)
│   └── transformer_quick_test.yaml # Transformer quick test
├── logs/                          # Experiment logs (not in git)
├── models/                        # Trained models (not in git)
├── results/                       # Results & figures
├── docs/                          # Documentation
│   ├── researcher_manual.md       # Main research manual
│   ├── M1_COMPLETION_REPORT.md    # M1 completion report
│   └── DATA_DOWNLOAD_GUIDE.md     # Data download instructions
├── TRAINING_IN_PROGRESS.md        # Current training status 🟢
└── tests/                         # Unit tests
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
cd /home/midori/Develop/Project_Sullivan

# Create virtual environment
python3 -m venv venv_sullivan
source venv_sullivan/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

**Important:** The actual MRI data must be downloaded separately.

See detailed instructions: [`docs/DATA_DOWNLOAD_GUIDE.md`](docs/DATA_DOWNLOAD_GUIDE.md)

**Quick Download (for testing):**
- Visit: https://doi.org/10.6084/m9.figshare.13725546.v1
- Download 1-2 subject files
- Extract to `data/raw/`

### 3. Run EDA

```bash
jupyter notebook notebooks/01_EDA.ipynb
```

---

## 📋 Milestones

| Milestone | Target | Status | Completion Criteria | Progress |
|-----------|--------|--------|---------------------|----------|
| **M1: Data Pipeline** | Phase 1 | ✅ **100% Complete** | MRI-Audio paired dataset ready | Completed ✅ |
| **M2: Baseline Model** | Phase 2 | 🟡 **50% Complete** | RMSE < 0.15, PCC > 0.50 | Phase 2-A done, 2-B in progress |
| **M3: Core Goal** | Phase 2 | ⬜ Pending (Jan 2026) | RMSE < 0.10, PCC > 0.70 | - |
| **M4: Digital Twin** | Phase 3 | ⬜ Future (TBD) | 3D synthesis working | Phase 1-2 완료 후 착수 |

### Recent Achievements 🎉
- **Nov 30, 2025**: **Transformer Implementation Complete** 🚀 - 5M params, tested and validated
- **Nov 30, 2025**: **Phase 2-B Started** 🟡 - Advanced architecture development in progress
- **Nov 30, 2025**: **Phase 2-A COMPLETE** ✅ - Baseline LSTM trained (Test RMSE: 1.011, PCC: 0.105)
- **Nov 30, 2025**: **Performance analysis complete** 📊 - Comprehensive baseline report generated
- **Nov 30, 2025**: **Improvement strategy identified** 🎯 - Transformer/Conformer + feature engineering
- **Nov 30, 2025**: **M1 100% Complete** ✅ - All data pipeline tasks finished
- **Nov 29, 2025**: Selective segmentation finished (75 utterances, 186K frames, 2.3h, 22.8 fps)
- **Nov 27, 2025**: U-Net segmentation model trained with **81.8% test Dice score** (target: 70%, **+16.9% above**)

---

## 📖 Documentation

### Main Documents
- **[Researcher Manual](researcher_manual.md)** - Complete research protocol and guidelines
- **[Baseline Complete](BASELINE_COMPLETE.md)** - Phase 2-A completion summary ✅
- **[Baseline Performance Report](docs/BASELINE_PERFORMANCE_REPORT.md)** - Comprehensive analysis and next steps 📊
- **[M1 Completion Report](docs/M1_COMPLETION_REPORT.md)** - Milestone 1 completion status (100%)
- **[Next Milestones Plan](docs/NEXT_MILESTONES.md)** - Detailed roadmap for M2, M3, M4

### Technical Documentation
- **[Segmentation Complete](docs/PROJECT_SULLIVAN_SEGMENTATION_COMPLETE.md)** - Full segmentation pipeline report
- **[Methodology](docs/METHODOLOGY_SEGMENTATION_PIPELINE.md)** - Segmentation approach details
- **[U-Net Evaluation](docs/UNET_EVALUATION_RESULTS.md)** - Model performance analysis
- **[Segmentation Test](SEGMENTATION_TEST_REPORT.md)** - Pre-trained model test results

### Administrative
- **[Data Download Guide](docs/DATA_DOWNLOAD_GUIDE.md)** - How to obtain the dataset
- **[Meeting Notes](docs/meeting_notes/)** - Weekly meeting records
- **[Literature Review](docs/literature_review/)** - Paper reviews

---

## 🔬 Research Phases

### Phase 1: Data Preprocessing ⭐ (Current)

**Goal:** Extract articulatory parameters from MRI data

**Tasks:**
1. Data loading & exploration
2. MRI/Audio denoising & alignment
3. Vocal tract segmentation
4. Parameter extraction (PCA/Autoencoder)

**Output:** `data/processed/parameters/` (train/val/test splits)

### Phase 2: Audio-to-Parameter Model ⭐

**Goal:** Train AI model to predict articulatory parameters from audio

**Approaches:**
- Bi-LSTM baseline
- Transformer-based models
- Conformer architecture

**Target Performance:** RMSE < 0.10, PCC > 0.70

### Phase 3: Digital Twin (Future)

**Goal:** 3D vocal tract reconstruction & acoustic synthesis

*This phase starts after successful completion of Phase 1-2.*

---

## 📊 Dataset

**Name:** USC-TIMIT Speech MRI Dataset

**Citation:**
```bibtex
@article{lim2021multispeaker,
  title={A multispeaker dataset of raw and reconstructed speech production real-time MRI video and 3D volumetric images},
  author={Lim, Yongwan and Toutios, Asterios and others},
  journal={Scientific Data},
  volume={8},
  pages={187},
  year={2021}
}
```

**Details:**
- 75 speakers
- rtMRI videos (~50-80 fps)
- Synchronized audio (20 kHz)
- TIMIT sentences & phonetic tasks

---

## 🧪 Running Experiments

### Log Your Work

All experiments must be logged. See template in `researcher_manual.md`.

```bash
# Example experiment
python src/baseline/train.py --config configs/baseline_v1.yaml

# Log results
# Update logs/experiments/EXP-YYYYMMDD-NN.json
```

### Evaluation

```bash
python src/evaluation/evaluate.py --model models/baseline_v1.pth --split test
```

---

## 👥 Team Roles

| Role | Responsibilities | Priority |
|------|------------------|----------|
| Project Lead | Milestone tracking, coordination | ⭐⭐⭐ |
| Data Engineer | Phase 1 preprocessing pipeline | ⭐⭐⭐ |
| ML Engineer 1 | Phase 2 model development | ⭐⭐⭐ |
| ML Engineer 2 | Hyperparameter tuning | ⭐⭐ |
| Research Analyst | Literature review, metrics | ⭐⭐ |

---

## 📞 Contact & Support

**Project Lead:** [Name]
**Email:** [Email]

**Issues:** [GitHub Issues Link]

**Meetings:** Weekly [Day] [Time]

---

## 📄 License

Research use only. See USC-TIMIT dataset license for data usage terms.

---

## 🔗 Useful Links

- **Dataset (figshare):** https://doi.org/10.6084/m9.figshare.13725546.v1
- **Paper (arXiv):** https://arxiv.org/abs/2102.07896
- **USC SAIL Lab:** https://sail.usc.edu/

---

**Last Updated:** 2025-11-30
**Version:** 1.2
