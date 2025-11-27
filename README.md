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

**Current Milestone:** M2 - Baseline Model Training
**Current Phase:** Phase 2 - Audio-to-Parameter Modeling

**Phase 1 Progress (✅ COMPLETE):**
- [x] Project structure initialized
- [x] U-Net segmentation pipeline built
- [x] Model trained (81.8% Dice score)
- [x] Parameter extraction module implemented

**Phase 2 Progress (🚧 IN PROGRESS):**
- [x] Audio feature extraction module
- [x] Bi-LSTM baseline model implemented
- [x] Training pipeline complete
- [x] Evaluation metrics implemented
- [ ] Full dataset processing (Phase 1 → Phase 2)
- [ ] Model training on real data
- [ ] Baseline performance evaluation (RMSE < 0.15)

---

## 🗂️ Project Structure

```
Project_Sullivan/
├── data/                      # Data directory (not in git)
│   ├── raw/                   # Original USC-TIMIT data
│   ├── processed/             # Preprocessed data
│   └── experiments/           # Experiment-specific data
├── src/                       # Source code
│   ├── preprocessing/         # Phase 1: Data preprocessing
│   ├── modeling/              # Phase 2: Model development
│   ├── baseline/              # Baseline models
│   ├── evaluation/            # Evaluation metrics
│   └── utils/                 # Utilities
├── notebooks/                 # Jupyter notebooks
│   └── 01_EDA.ipynb          # Exploratory data analysis
├── configs/                   # Configuration files
├── logs/                      # Experiment logs
├── models/                    # Trained models
├── results/                   # Results & figures
├── docs/                      # Documentation
│   ├── researcher_manual.md   # Main research manual
│   └── DATA_DOWNLOAD_GUIDE.md # Data download instructions
└── tests/                     # Unit tests
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

| Milestone | Target | Status | Completion Criteria |
|-----------|--------|--------|---------------------|
| **M1: Data Pipeline** | Phase 1 | ✅ Complete | MRI segmentation complete (81.8% Dice) |
| **M2: Baseline Model** | Phase 2 | 🟡 In Progress | RMSE < 0.15, PCC > 0.50 |
| **M3: Core Goal** | Phase 2 | ⬜ Pending | RMSE < 0.10, PCC > 0.70 |
| **M4: Digital Twin** | Phase 3 | ⬜ Future | 3D synthesis working |

---

## 📖 Documentation

- **[Researcher Manual](docs/researcher_manual.md)** - Complete research protocol and guidelines
- **[Phase 1 Complete Report](docs/PROJECT_SULLIVAN_SEGMENTATION_COMPLETE.md)** - U-Net segmentation results
- **[Phase 2 Setup Report](docs/PHASE2_SETUP_COMPLETE.md)** - Audio-to-parameter infrastructure
- **[Data Download Guide](docs/DATA_DOWNLOAD_GUIDE.md)** - How to obtain the dataset

---

## 🔬 Research Phases

### Phase 1: Data Preprocessing ✅ (Complete)

**Goal:** Extract articulatory parameters from MRI data

**Completed Tasks:**
1. ✅ Data loading & exploration
2. ✅ MRI segmentation (U-Net: 81.8% Dice score)
3. ✅ Parameter extraction module (10-dimensional)
4. ⬜ Full dataset processing (468 utterances)

**Output:** `data/processed/segmentations/`, `data/processed/parameters/`

**Documentation:** [Phase 1 Complete Report](docs/PROJECT_SULLIVAN_SEGMENTATION_COMPLETE.md)

### Phase 2: Audio-to-Parameter Model ⭐ (Current)

**Goal:** Train AI model to predict articulatory parameters from audio

**Implemented Infrastructure:**
- ✅ Audio feature extraction (Mel-spectrogram, MFCC)
- ✅ PyTorch Dataset with variable-length sequences
- ✅ Bi-LSTM baseline model (256 hidden, 3 layers)
- ✅ Training pipeline with early stopping
- ✅ Evaluation metrics (RMSE, MAE, PCC)

**Next Steps:**
1. Process full dataset (Phase 1 → Phase 2 data)
2. Train baseline Bi-LSTM model
3. Evaluate and iterate to meet targets

**Target Performance:** RMSE < 0.10, PCC > 0.70

**Documentation:** [Phase 2 Setup Report](docs/PHASE2_SETUP_COMPLETE.md)

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

**Last Updated:** 2025-11-27
**Version:** 2.0 (Phase 2 Infrastructure Complete)
