# Project Sullivan - Next Milestones & Action Plan

**Date**: 2025-11-29
**Current Status**: M1 (95% Complete) → M2 (Next Up)
**Last Updated**: 2025-11-29

---

## 📊 Current Status Summary

### ✅ Completed Work

#### Phase 1: Data Preprocessing & Segmentation (95% Complete)

**1. Data Acquisition** ✅
- USC-TIMIT dataset downloaded (468 utterances, 15 subjects)
- Dataset structure analyzed and documented
- Data statistics collected

**2. Preprocessing Pipeline** ✅
- MRI/Audio alignment implemented (`src/preprocessing/alignment.py`)
- Denoising algorithms implemented (`src/preprocessing/denoising.py`)
- Data loader created (`src/preprocessing/data_loader.py`)
- Batch processing script available (`scripts/batch_preprocess.py`)

**3. Vocal Tract Segmentation** ✅ **[Major Achievement!]**
- **Method**: Hybrid approach (Traditional CV → U-Net from scratch)
- **Pseudo-labels**: 150 high-quality frames (15 subjects × 10 frames)
- **U-Net Model Performance**:
  - Test Dice Score: **81.8%** (target: 70%, **+16.9% above target**)
  - Tongue segmentation: **96.5%** (most critical articulator)
  - Jaw/Palate: 73.2%, Lips: 58.8%
  - Training time: 41 epochs (17 minutes on CPU)
- **Model Artifacts**:
  - Production model: `models/unet_scratch/unet_final.pth` (119 MB)
  - Best checkpoint: `models/unet_scratch/checkpoints/unet-epoch=041-val_dice_mean=0.8932.ckpt`
  - Evaluation results: `results/unet_evaluation/`
- **Documentation**:
  - Complete report: `docs/PROJECT_SULLIVAN_SEGMENTATION_COMPLETE.md`
  - Test report: `SEGMENTATION_TEST_REPORT.md`

---

## 🎯 Next Milestones (Detailed Action Plan)

### Milestone M1 Completion (5% Remaining) - Priority: 🔴 HIGH

**Estimated Time**: 2-3 weeks
**Blocking**: M2 cannot start until M1 is complete

#### Task 1.1: Full Dataset Segmentation
**Owner**: Data Engineer
**Deadline**: Week 1 (1 week)

**Objective**: Apply trained U-Net model to all 468 USC-TIMIT utterances

**Implementation**:
```python
# Script to create: scripts/segment_full_dataset.py

import torch
from src.segmentation.unet import UNet
from src.preprocessing.data_loader import load_all_utterances

# 1. Load trained model
model = UNet(n_classes=4)
model.load_state_dict(torch.load('models/unet_scratch/unet_final.pth'))
model.eval()

# 2. Load all MRI data
utterances = load_all_utterances('data/raw/usc_speech_mri/')

# 3. Batch inference
for utterance in utterances:
    for frame in utterance.frames:
        segmentation = model(frame)
        save_segmentation(segmentation, output_path)
```

**Output**:
```
data/processed/segmentations/
├── sub001_2drt_01_vcv1_r1_video/
│   ├── frame_0000_seg.npz
│   ├── frame_0001_seg.npz
│   └── ...
├── sub002_...
└── ...
```

**Acceptance Criteria**:
- [ ] All 468 utterances segmented
- [ ] Segmentation masks saved in NPZ format
- [ ] Visual quality check on 50 random frames
- [ ] Processing time logged (estimated: 2-3 hours on CPU)

---

#### Task 1.2: Articulatory Parameter Extraction
**Owner**: Data Engineer
**Deadline**: Week 2-3 (2 weeks)

**Objective**: Extract low-dimensional articulatory parameters from segmentation masks

**Method Options**:

**Option A: Geometric Features (Quick Start - Recommended)**
Extract hand-crafted features from segmentation masks:

```python
# Script to create: scripts/extract_articulatory_params.py

def extract_geometric_features(segmentation_mask):
    """
    Extract articulatory parameters from segmentation mask

    Returns:
        params: dict with following features (10-15 dimensions)
    """
    params = {}

    # Tongue features (5-7 dims)
    tongue_mask = (segmentation_mask == 1)
    params['tongue_area'] = np.sum(tongue_mask)
    params['tongue_centroid_x'], params['tongue_centroid_y'] = compute_centroid(tongue_mask)
    params['tongue_tip_position'] = find_tongue_tip(tongue_mask)
    params['tongue_dorsum_height'] = find_dorsum_height(tongue_mask)
    params['tongue_curvature'] = compute_curvature(tongue_mask)

    # Jaw features (2-3 dims)
    jaw_mask = (segmentation_mask == 2)
    params['jaw_opening'] = compute_jaw_opening(jaw_mask)
    params['jaw_position_y'] = compute_centroid(jaw_mask)[1]

    # Lip features (2-3 dims)
    lip_mask = (segmentation_mask == 3)
    params['lip_aperture'] = compute_lip_aperture(lip_mask)
    params['lip_protrusion'] = compute_lip_protrusion(lip_mask)

    # Constriction features (2-3 dims)
    params['constriction_degree'] = compute_constriction_degree(segmentation_mask)
    params['constriction_location'] = compute_constriction_location(segmentation_mask)

    return params
```

**Option B: PCA (Dimensionality Reduction)**
```python
from sklearn.decomposition import PCA

# Flatten all segmentation masks
masks_flattened = segmentations.reshape(num_frames, -1)

# Apply PCA
pca = PCA(n_components=10)
params = pca.fit_transform(masks_flattened)

# Save PCA model for inverse transform
save_pca_model(pca, 'models/parameter_extraction/pca_model.pkl')
```

**Option C: Autoencoder (More Flexible)**
```python
# Script to create: scripts/train_autoencoder.py

class ArticulatoryAutoencoder(nn.Module):
    def __init__(self, latent_dim=10):
        # Encoder: (84, 84) -> 10-dim latent
        # Decoder: 10-dim -> (84, 84)
        pass

# Train on segmentation masks
# Extract latent vectors as parameters
```

**Recommended Approach**: Start with **Option A (Geometric Features)** for interpretability, then compare with **Option B (PCA)** for performance.

**Output**:
```
data/processed/parameters/
├── geometric/
│   ├── sub001_2drt_01_vcv1_r1_video_params.npy  # (num_frames, 10-15)
│   └── ...
├── pca/
│   ├── sub001_2drt_01_vcv1_r1_video_params.npy  # (num_frames, 10)
│   └── ...
└── extraction_config.yaml
```

**Acceptance Criteria**:
- [ ] Parameter extraction method implemented
- [ ] Parameters extracted for all utterances
- [ ] Parameter statistics computed (mean, std, range)
- [ ] Visualization of parameter trajectories for 10 sample utterances
- [ ] Correlation analysis between parameters
- [ ] Documentation of parameter meanings

---

#### Task 1.3: Audio Feature Extraction
**Owner**: ML Engineer 1
**Deadline**: Week 2 (parallel with Task 1.2)

**Objective**: Extract audio features aligned with MRI frames

**Implementation**:
```python
# Script to create: scripts/extract_audio_features.py

import librosa

def extract_audio_features(audio_path, frame_timestamps, feature_type='mel'):
    """
    Extract audio features synchronized with MRI frames

    Args:
        audio_path: Path to audio file
        frame_timestamps: MRI frame timestamps
        feature_type: 'mel', 'mfcc', or 'spectrogram'

    Returns:
        features: (num_frames, feature_dim)
    """
    audio, sr = librosa.load(audio_path, sr=16000)

    if feature_type == 'mel':
        # Mel-spectrogram (80 bins recommended)
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr,
            n_mels=80,
            n_fft=512,
            hop_length=160
        )
        features = sync_to_mri_frames(mel_spec, frame_timestamps)

    elif feature_type == 'mfcc':
        # MFCC (13 coefficients)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features = sync_to_mri_frames(mfcc, frame_timestamps)

    return features
```

**Output**:
```
data/processed/audio_features/
├── mel_spectrogram/
│   ├── sub001_2drt_01_vcv1_r1_video_mel.npy  # (num_frames, 80)
│   └── ...
└── mfcc/
    ├── sub001_2drt_01_vcv1_r1_video_mfcc.npy  # (num_frames, 13)
    └── ...
```

**Acceptance Criteria**:
- [ ] Audio features extracted for all utterances
- [ ] Features synchronized with MRI frame timestamps
- [ ] Feature statistics documented
- [ ] Visualization of audio-parameter alignment for 10 samples

---

#### Task 1.4: Dataset Splitting (Train/Val/Test)
**Owner**: Data Engineer
**Deadline**: Week 3

**Objective**: Create subject-level splits to prevent data leakage

**Implementation**:
```python
# Script to create: scripts/create_dataset_splits.py

# Subject-level split (NOT frame-level!)
all_subjects = ['sub001', 'sub002', ..., 'sub075']

# Recommended split: 70% train, 15% val, 15% test
train_subjects = [...] # ~52 subjects
val_subjects = [...]   # ~11 subjects
test_subjects = [...]  # ~12 subjects

# Create split files
create_splits(
    train_subjects=train_subjects,
    val_subjects=val_subjects,
    test_subjects=test_subjects,
    output_dir='data/processed/splits/'
)
```

**Output**:
```
data/processed/splits/
├── train/
│   ├── audio_features/
│   ├── articulatory_params/
│   └── file_list.txt
├── val/
└── test/
```

**Acceptance Criteria**:
- [ ] Subject-level splits created (no subject appears in multiple splits)
- [ ] Split statistics computed (num utterances, total frames)
- [ ] Data balance verified across splits
- [ ] Split configuration saved (random seed=42 for reproducibility)

---

### ✅ Milestone M1 Completion Checklist

- [ ] Task 1.1: Full dataset segmentation complete
- [ ] Task 1.2: Articulatory parameters extracted
- [ ] Task 1.3: Audio features extracted
- [ ] Task 1.4: Dataset splits created
- [ ] **Final Output**: Audio-Parameter paired dataset ready for Phase 2
- [ ] **Documentation**: M1 completion report written

**Timeline**: 2-3 weeks from now → Target completion: **Dec 13-20, 2025**

---

## 🚀 Milestone M2: Baseline Audio-to-Parameter Model - Priority: 🟡 MEDIUM

**Start Date**: After M1 completion
**Estimated Duration**: 2-3 weeks
**Target Performance**: RMSE < 0.15, Pearson Correlation > 0.50

### Task 2.1: Baseline Model Implementation
**Owner**: ML Engineer 1

**Objective**: Implement simple Bi-LSTM model as baseline

**Architecture**:
```python
# File to create: src/modeling/baseline_lstm.py

class BaselineLSTMPredictor(nn.Module):
    def __init__(
        self,
        input_dim=80,     # Mel-spectrogram bins
        hidden_dim=128,   # Start small for baseline
        num_layers=2,
        output_dim=10,    # Articulatory parameters
        dropout=0.3
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # x: (batch, time, input_dim)
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output
```

**Training Configuration**:
```yaml
# File to create: configs/baseline_training.yaml

model:
  name: "BaselineLSTMPredictor"
  input_dim: 80
  hidden_dim: 128
  num_layers: 2
  output_dim: 10
  dropout: 0.3

training:
  batch_size: 16
  learning_rate: 0.001
  num_epochs: 50
  optimizer: "Adam"
  loss: "MSE"

data:
  train_path: "data/processed/splits/train"
  val_path: "data/processed/splits/val"
  test_path: "data/processed/splits/test"
```

### Task 2.2: Training & Evaluation
**Owner**: ML Engineer 1

**Training Script**:
```python
# File to create: scripts/train_baseline.py

# See researcher_manual.md section 3.2.3 for detailed implementation
```

**Evaluation Metrics**:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- Pearson Correlation per parameter
- Visualizations: predicted vs. ground truth trajectories

**Target Performance**:
- **RMSE**: < 0.15 (normalized parameters)
- **Pearson Correlation**: > 0.50 per parameter

### Task 2.3: Analysis & Documentation
**Owner**: Research Analyst

**Deliverables**:
- Training curves (loss, RMSE, correlation over epochs)
- Per-parameter performance analysis
- Failure case analysis
- Baseline report: `docs/BASELINE_MODEL_REPORT.md`

**Timeline**: 2-3 weeks → Target completion: **Jan 3-10, 2026**

---

## 🎯 Milestone M3: Core Goal Achievement - Priority: 🟢 CRITICAL

**Start Date**: After M2 completion
**Estimated Duration**: 8-12 weeks
**Target Performance**: RMSE < 0.10, Pearson Correlation > 0.70

### Key Strategies:

#### 3.1: Model Architecture Improvements
- **Transformer-based models**: Self-attention for long-range dependencies
- **Conformer**: Convolution + Transformer (SOTA for speech)
- **Multi-task learning**: Joint prediction of multiple objectives

#### 3.2: Data Augmentation
- Audio augmentation: time-stretching, pitch-shifting, noise injection
- Parameter augmentation: temporal perturbations

#### 3.3: Loss Function Engineering
- MSE + Smoothness loss (temporal consistency)
- Perceptual loss (articulatory trajectory realism)
- Multi-scale loss (different time resolutions)

#### 3.4: Hyperparameter Optimization
- Grid search / Bayesian optimization
- Learning rate scheduling
- Regularization tuning

**Target Completion**: **Feb-Mar 2026**

---

## 📊 Milestone M4: Digital Twin (Phase 3) - Priority: ⬜ FUTURE

**Prerequisites**:
- ✅ M3 completed (RMSE < 0.10, PCC > 0.70)
- ✅ Phase 2 results published/documented
- ✅ Project lead approval

**Estimated Start**: **After M3 completion**

**Components**:
1. 3D Mesh Generation from segmentations
2. Parametric vocal tract model
3. Physics-based acoustic simulation (VocalTractLab or FEM)
4. Neural vocoder (end-to-end learning alternative)

**Timeline**: TBD (Phase 1-2 focus first!)

---

## 📅 Overall Timeline

```
Nov 2025 (Current)
    │
    ├─ Week 1-3: M1 Completion (5% remaining)
    │   ├─ Full dataset segmentation
    │   ├─ Parameter extraction
    │   └─ Dataset splitting
    │
Dec 2025
    │
    ├─ Week 1-2: M2 Baseline Model
    │   ├─ LSTM implementation
    │   └─ Training & evaluation
    │
Jan 2026
    │
    ├─ Week 1-12: M3 Core Goal
    │   ├─ Model improvements
    │   ├─ Optimization
    │   └─ Target performance achievement
    │
Feb-Mar 2026
    │
    └─ M3 Completion & Documentation
        └─ Phase 2 results publication

Apr 2026+ (Future)
    │
    └─ M4 Digital Twin (if approved)
```

---

## 🎯 Success Criteria Summary

| Milestone | Metric | Target | Current Status |
|-----------|--------|--------|----------------|
| **M1** | Data pipeline | Audio-Parameter pairs ready | 95% complete |
| **M2** | RMSE | < 0.15 | Not started |
| **M2** | PCC | > 0.50 | Not started |
| **M3** | RMSE | < 0.10 | Not started |
| **M3** | PCC | > 0.70 | Not started |
| **M4** | 3D synthesis | Working | Future |

---

## 📝 Next Actions (This Week)

### Priority 1: Complete M1 (95% → 100%)
1. **Create full dataset segmentation script** (`scripts/segment_full_dataset.py`)
2. **Implement parameter extraction method** (Start with geometric features)
3. **Extract audio features** (Mel-spectrogram + MFCC)

### Priority 2: Prepare for M2
4. **Design baseline model architecture**
5. **Set up training infrastructure** (PyTorch Lightning modules)
6. **Create evaluation pipeline**

### Priority 3: Documentation
7. **Update researcher manual** with M1 completion details
8. **Write M1 completion report**
9. **Prepare M2 kickoff documentation**

---

## 📞 Resources & References

**Key Documents**:
- Researcher Manual: `researcher_manual.md`
- Segmentation Complete Report: `docs/PROJECT_SULLIVAN_SEGMENTATION_COMPLETE.md`
- Methodology: `docs/METHODOLOGY_SEGMENTATION_PIPELINE.md`
- U-Net Evaluation: `docs/UNET_EVALUATION_RESULTS.md`

**Code References**:
- Segmentation model: `src/segmentation/unet.py`
- Dataset utilities: `src/segmentation/dataset.py`
- Preprocessing: `src/preprocessing/`

**External Resources**:
- USC-TIMIT Dataset: https://doi.org/10.6084/m9.figshare.13725546.v1
- VocalTractLab: http://www.vocaltractlab.de/ (for Phase 3)

---

**Last Updated**: 2025-11-29
**Next Review**: After M1 completion (target: Dec 13-20, 2025)
