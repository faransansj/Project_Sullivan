# Phase 4 Accuracy Improvement: Loss Function Optimization

## Abstract

This document summarizes the experimental process and findings from Phase 4 of Project Sullivan, focused on improving the Pearson Correlation Coefficient (PCC) between predicted and ground-truth articulatory parameters. We identify a critical loss imbalance issue in the hybrid loss formulation and propose a reweighted loss configuration.

---

## 1. Background

**Task**: Acoustic-to-articulatory inversion — predicting 14-dimensional geometric articulatory parameters (tongue position, jaw opening, lip shape, etc.) from 80-dim mel-spectrogram features, using the USC-TIMIT Speech MRI Dataset.

**Model**: Conformer-based sequence-to-sequence model
- Architecture: 12 Conformer layers, d_model=512, 8 attention heads, FFN dim=2048, depthwise conv kernel=31
- Parameters: 72.9M
- Input: 80-dim mel-spectrogram at MRI frame rate
- Output: 14-dim geometric articulatory parameters

**Baseline (Phase 3)**: Transformer, 21.5M params → Global PCC 0.1982

---

## 2. Hybrid Loss Function

The Conformer model was trained with a composite loss:

$$\mathcal{L} = w_{\text{mse}} \cdot \mathcal{L}_{\text{mse}} + w_{\text{pcc}} \cdot \mathcal{L}_{\text{pcc}} + w_{\text{vel}} \cdot \mathcal{L}_{\text{vel}} + w_{\text{acc}} \cdot \mathcal{L}_{\text{acc}}$$

Where:
- $\mathcal{L}_{\text{mse}}$: Masked MSE over valid frames
- $\mathcal{L}_{\text{pcc}} = 1 - \text{PCC}$: Correlation loss
- $\mathcal{L}_{\text{vel}}$: First-order temporal smoothness loss
- $\mathcal{L}_{\text{acc}}$: Second-order temporal smoothness loss

---

## 3. Problem: Loss Imbalance

### 3.1 Observation

Across all training runs (version_3 through version_7), `val_loss` remained stuck in the range **2.10–2.26** despite RMSE converging to a reasonable value (~0.12).

| Version | Epochs | Precision | val_loss (best) | val_RMSE | val_PCC |
|---------|--------|-----------|-----------------|----------|---------|
| v3 | 5 | bf16-mixed | NaN | NaN | 0.000 |
| v7 (resumed from epoch=04) | ~25 | bf16-mixed | 2.130 | 0.120 | 0.148 |
| v7 (test) | — | — | 0.290 | 0.121 | 0.117 |

### 3.2 Root Cause Analysis

Decomposing `val_loss` with the original weights (`mse=0.8, pcc=2.0, vel=1.2, acc=0.5`):

| Loss term | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| $\mathcal{L}_{\text{mse}}$ | ~0.014 (RMSE≈0.12) | 0.8 | ~0.011 |
| $\mathcal{L}_{\text{pcc}}$ | ~0.87 (PCC≈0.13) | **2.0** | **~1.74** |
| $\mathcal{L}_{\text{vel}} + \mathcal{L}_{\text{acc}}$ | — | 1.2 + 0.5 | ~0.35 (est.) |
| **Total** | | | **~2.10** |

The PCC loss dominated at ~81% of the total loss. Since PCC improved very slowly from its initial value near 0.1, the gradient signal was overwhelmingly driven by a loss term that was not improving—effectively saturating the optimizer.

### 3.3 Additional Factors

1. **Resume from degraded checkpoint**: Version 7 resumed from `epoch=04, val_loss=2.1225`, which was already in a poor optimization basin. OneCycleLR's LR schedule was misaligned with the resumed epoch.
2. **Numerical instability**: bf16-mixed precision caused NaN loss in version_3 (5 epochs), indicating the loss landscape was sharp or the LR (5e-4) was too aggressive.
3. **Model scale vs. dataset size**: 72.9M parameters for a ~330 training utterances dataset may cause slow convergence without aggressive regularization.

---

## 4. Proposed Fix (v8 Configuration)

| Hyperparameter | Original | v8 |
|----------------|----------|----|
| `precision` | `bf16-mixed` | `32-true` |
| `learning_rate` | `5e-4` | `1e-4` |
| `num_epochs` | 100 | 150 |
| `mse_weight` | 0.8 | **1.0** |
| `pcc_weight` | **2.0** | **0.3** |
| `velocity_weight` | **1.2** | **0.2** |
| `acceleration_weight` | 0.5 | **0.1** |

**Rationale**:
- Reducing `pcc_weight` from 2.0 to 0.3 removes the gradient saturation from the PCC term, allowing MSE to drive early convergence
- `32-true` precision eliminates the NaN instability observed with bf16
- Lower LR (1e-4) with a fresh start avoids the OneCycleLR scheduler misalignment from resuming
- Longer training (150 epochs) compensates for the more conservative LR

**Expected loss decomposition (v8 target)**:

| Loss term | Weight | Expected contribution |
|-----------|--------|-----------------------|
| $\mathcal{L}_{\text{mse}}$ | 1.0 | ~0.014 |
| $\mathcal{L}_{\text{pcc}}$ | 0.3 | ~0.26 |
| $\mathcal{L}_{\text{vel}} + \mathcal{L}_{\text{acc}}$ | 0.2 + 0.1 | ~0.09 |
| **Total** | | **~0.36** |

This makes `val_loss` a more faithful proxy for model quality and allows the checkpoint selection (top-3 by `val_loss`) to actually track meaningful improvement.

---

## 5. Future Directions

### 5.1 HuBERT Features
Replacing 80-dim mel-spectrogram with 1024-dim HuBERT-Large (`facebook/hubert-large-ls960-ft`) features is expected to substantially improve PCC. HuBERT captures richer phonetic representations that correlate more directly with vocal tract configuration. Config: `conformer_hubert_config.yaml`.

### 5.2 Curriculum Loss Training
A staged loss introduction strategy:
- **Stage 1** (epochs 0–30): MSE only (`pcc_weight=0, vel_weight=0`)
- **Stage 2** (epochs 31–80): Add PCC loss gradually (`pcc_weight=0.1→0.3`)
- **Stage 3** (epochs 81+): Full hybrid loss

This prevents the PCC term from dominating before the model has learned basic position mapping.

### 5.3 Loss Normalization
Instead of fixed weights, normalizing each loss term to unit scale before weighting:

$$\mathcal{L}_{\text{pcc,norm}} = \frac{1 - \text{PCC}}{\mathbb{E}[1 - \text{PCC}]_{\text{init}}}$$

This makes weights interpretable as relative importance rather than absolute scale.

### 5.4 Data Augmentation
With only ~330 training utterances, the model likely underfits the articulatory space. Potential augmentations:
- SpecAugment on mel-spectrogram input
- Time-warping of audio-articulatory pairs
- Speaker-level normalization to reduce inter-speaker variance

---

## 6. Target Metrics (M2)

| Metric | Phase 2-A (LSTM) | Phase 3 (Transformer) | Phase 4 v7 | Phase 4 v8 (target) | M2 Goal |
|--------|------------------|-----------------------|------------|---------------------|---------|
| RMSE | 1.011 | — | **0.121** | < 0.12 | < 0.15 |
| PCC | 0.105 | 0.198 | 0.117 | > 0.30 | > 0.50 |

---

*Last updated: 2026-03-17*
*Experiment tracking: `logs/training/conformer_phase4_accuracy/`*
*Configs: `configs/conformer_a100_config.yaml`*
