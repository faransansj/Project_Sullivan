# Project Sullivan: Project Overview & Status Report

**Project**: Acoustic-to-Articulatory Parameter Inference
**Current Status**: Phase 1 (100% Complete), Phase 2 (Baseline Complete, Transformer in Progress)
**Last Update**: 2025-12-23

---

## 1. Project Goal
Project Sullivan aims to synthesize low-dimensional articulatory parameters (tongue, jaw, lips) from speech audio by leveraging real-time MRI (rtMRI) data from the USC-TIMIT dataset.

## 2. Phase 1: Data Preprocessing & Segmentation (COMPLETE)
### 2.1 Preprocessing Pipeline
- **MRI/Audio Alignment**: Successfully synchronized audio with MRI frames.
- **Denoising**: Implemented spatial-temporal denoising for MRI and background noise reduction for audio.
- **Batch Processing**: All 15 recommended subjects (468 utterances) have been processed.

### 2.2 U-Net Segmentation
- **Hybrid Approach**: Used traditional CV for pseudo-labeling and trained U-Net from scratch.
- **Performance**: Achieved **81.8% Mean Dice Score** on the test set, with **96.5% for the tongue region**.
- **Result**: Production-ready segmentation model (`models/unet_scratch/unet_final.pth`).

---

## 3. Phase 2: Audio-to-Parameter Model (IN PROGRESS)
### 3.1 Feature Extraction
- **Articulatory Parameters**: Implemented extraction of 14-dimensional geometric features (tongue height, jaw opening, lip aperture, etc.).
- **Audio Features**: Implemented extraction of 80-dimensional Mel-spectrogram and MFCC features.

### 3.2 Baseline Performance (LSTM)
- **Model**: Bidirectional LSTM.
- **Performance**: RMSE 1.011, Pearson Correlation 0.105.
- **Status**: Completed as a reference baseline. Proved the task is learnable but needs more complex architecture.

### 3.3 Transformer Implementation
- **Status**: Transformer architecture has been added and training infrastructure is set up.
- **Improvements**: Includes positional encoding and self-attention for better temporal modeling.

---

## 4. Git Update Summary (Pulled Dec 23, 2025)
The following key updates were synchronized from the remote repository:
- **Phase 2 Infrastructure**: Added articulatory and audio feature extraction scripts.
- **Model Implementations**: Added `BaselineLSTM` and `Transformer` models in `src/modeling/`.
- **Colab Support**: Added Google Colab training infrastructure and setup guides.
- **Sequence Handling**: Optimized from fixed-length sequence splitting back to full utterance processing for better context.
- **Milestone Documentation**: New reports and plans added to `docs/`.

---

## 5. Next Milestones
- [ ] **M2 Target Achievement**: Reach RMSE < 0.15 and PCC > 0.50 using Transformer or Conformer models.
- [ ] **Data Expansion**: Utilize full dataset for training to improve generalization.
- [ ] **Feature Engineering**: Experiment with learnable audio features (wav2vec 2.0).
- [ ] **Digital Twin (Phase 3)**: Planning for 3D vocal tract synthesis.
