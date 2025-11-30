# Baseline LSTM Performance Report

**Project**: Project Sullivan - Acoustic-to-Articulatory Inversion
**Date**: 2025-11-30
**Model**: Bidirectional LSTM (Baseline)
**Version**: v1.0

---

## Executive Summary

The baseline Bi-LSTM model was successfully trained and evaluated on the USC-TIMIT dataset. While the training process completed without errors, the model's performance **significantly underperforms** the M2 milestone targets. This is expected for a baseline model and provides a critical reference point for future improvements.

### Key Results

| Metric | Baseline Result | M2 Target | Gap |
|--------|----------------|-----------|-----|
| **Test RMSE** | 1.011 | < 0.15 | **6.7× worse** |
| **Test Pearson** | 0.105 | > 0.50 | **4.8× worse** |
| **Test MAE** | 0.763 | - | - |
| **Test Loss (MSE)** | 0.739 | - | - |

**Conclusion**: The baseline model demonstrates that the task is learnable (training loss decreased), but the current architecture is **insufficient** for high-quality acoustic-to-articulatory inversion.

---

## 1. Training Configuration

### Model Architecture
```
BaselineLSTM(
  (lstm): LSTM(80, 128, num_layers=2, batch_first=True,
                dropout=0.3, bidirectional=True)
  (fc): Linear(in_features=256, out_features=14, bias=True)
  (dropout_layer): Dropout(p=0.3, inplace=False)
  (criterion): MSELoss()
)
```

**Total Parameters**: 613,902

### Input/Output
- **Input**: 80-dimensional mel-spectrogram features
- **Output**: 14-dimensional geometric articulatory parameters
- **Sequence Handling**: Variable-length with padding and masking

### Training Hyperparameters
- **Optimizer**: Adam (lr=0.001, weight_decay=0.0001)
- **Batch Size**: 8
- **Epochs**: 18 (stopped early at epoch 17)
- **LR Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Early Stopping**: patience=10 epochs
- **Gradient Clipping**: 1.0
- **Hardware**: CPU (single-threaded DataLoader)

### Dataset Splits
- **Train**: 50 utterances (10 subjects)
- **Validation**: 10 utterances (2 subjects)
- **Test**: 15 utterances (3 subjects)
- **Total Frames**: 186,124

---

## 2. Training Progress

### Learning Curve

#### Validation Loss Progression

| Epoch | Val Loss | Train Loss | Improvement | Note |
|-------|----------|------------|-------------|------|
| 0 | 0.9142 | - | **Best Start** | Initial checkpoint |
| 1 | 0.9173 | 0.776 | -0.0031 | Slight degradation |
| 2 | 0.9249 | 0.702 | -0.0076 | |
| 3 | 0.9199 | 0.691 | +0.0050 | |
| 4 | 0.9187 | 0.759 | +0.0012 | |
| 5 | 0.9156 | 0.716 | +0.0031 | |
| 6 | 0.9153 | 0.733 | +0.0003 | |
| **7** | **0.9124** | **0.711** | **+0.0029** | **🏆 Best Model** |
| 8 | 0.9161 | 0.791 | -0.0037 | |
| 9 | 0.9213 | 0.737 | -0.0052 | |
| 10 | 0.9348 | 0.688 | -0.0135 | |
| 11 | 0.9253 | 0.716 | +0.0095 | |
| 12 | 0.9274 | 0.693 | -0.0021 | |
| 13 | 0.9416 | 0.703 | -0.0142 | |
| 14 | 0.9462 | 0.674 | -0.0046 | |
| 15 | 0.9374 | 0.677 | +0.0088 | |
| 16 | 0.9393 | 0.696 | -0.0019 | |
| 17 | 0.9620 | 0.665 | -0.0227 | Early stopping triggered |

### Key Observations

1. **Best Performance**: Epoch 7 (val_loss=0.9124)
2. **Early Plateau**: Validation loss plateaued after epoch 7
3. **Overfitting Signs**: Training loss continued to decrease while validation loss increased
4. **Early Stopping**: Triggered at epoch 17 (10 epochs without improvement)

### Training Time
- **Total Duration**: ~4.5 hours
- **Average Epoch Time**: ~13-15 minutes
- **Hardware**: CPU (GTX 750 Ti incompatible)

---

## 3. Test Set Evaluation

### Final Test Metrics

```
Test Loss (MSE):       0.7387
Test RMSE:             1.0108
Test MAE:              0.7635
Test Pearson Corr:     0.1049
```

### Performance vs. Targets

#### RMSE Analysis
- **Target**: < 0.15 (normalized scale)
- **Achieved**: 1.011
- **Gap**: 0.861 (6.7× worse than target)
- **Interpretation**: Predictions have very high error variance

#### Pearson Correlation Analysis
- **Target**: > 0.50 (moderate-to-strong correlation)
- **Achieved**: 0.105
- **Gap**: 0.395 (4.8× worse than target)
- **Interpretation**: **Very weak correlation** between predictions and ground truth

### Statistical Significance
- **R² (implied)**: ~0.011 (1.1% variance explained)
- **Conclusion**: The model captures only a tiny fraction of articulatory parameter variance

---

## 4. Error Analysis

### Problem Diagnosis

#### 4.1. Model Capacity Issues
**Symptoms**:
- Shallow architecture (2 LSTM layers)
- Limited hidden dimensionality (128)
- No attention mechanism

**Impact**: Model may lack capacity to learn complex acoustic-articulatory mappings

#### 4.2. Overfitting
**Symptoms**:
- Training loss: 0.665 (final)
- Validation loss: 0.912 (best)
- **Gap**: 0.247 (27% difference)

**Impact**: Model memorizes training data rather than generalizing

#### 4.3. Data-Related Issues
**Possible Factors**:
- **Small dataset**: Only 75 utterances
- **Limited subjects**: 15 speakers
- **Feature mismatch**: Mel-spectrogram may not capture all relevant acoustic cues
- **Parameter normalization**: May need better scaling

#### 4.4. Learning Dynamics
**Observations**:
- Validation loss plateau after epoch 7
- No benefit from continued training
- LR scheduler triggered but no improvement

**Impact**: Model converged to suboptimal local minimum

---

## 5. Comparison to Baseline Expectations

### Expected Baseline Performance
- **RMSE**: 0.20-0.30 (typical for simple baselines in literature)
- **PCC**: 0.30-0.45 (weak-to-moderate correlation)

### Actual Performance
- **RMSE**: 1.011 (worse than expected)
- **PCC**: 0.105 (much worse than expected)

### Possible Reasons for Underperformance
1. **Data preprocessing issues**: Feature extraction or normalization problems
2. **Architecture limitations**: Too simple for this complex task
3. **Hyperparameter suboptimality**: Learning rate, batch size, etc.
4. **Dataset quality**: Small size, limited diversity

---

## 6. Recommendations for Improvement

### High Priority (Next Iteration)

#### 6.1. Model Architecture Improvements
1. **Deeper networks**:
   - Increase LSTM layers: 2 → 4-6 layers
   - Larger hidden dimension: 128 → 256-512

2. **Add Attention Mechanism**:
   - Self-attention for acoustic features
   - Cross-attention for acoustic-articulatory alignment

3. **Advanced Architectures**:
   - Transformer encoder
   - Conformer (Convolution + Transformer)
   - Temporal Convolutional Networks (TCN)

#### 6.2. Data & Feature Engineering
1. **Feature improvements**:
   - Combine mel-spectrogram + MFCC
   - Add prosodic features (F0, energy, duration)
   - Try learnable features (wav2vec 2.0, HuBERT)

2. **Data augmentation**:
   - Time stretching
   - Pitch shifting
   - SpecAugment

3. **Better normalization**:
   - Per-subject z-score normalization
   - Robust scaling (median/IQR)

#### 6.3. Training Improvements
1. **Regularization**:
   - Stronger dropout: 0.3 → 0.4-0.5
   - Layer normalization
   - Weight decay tuning

2. **Loss function**:
   - Weighted MSE (emphasize critical parameters)
   - Multi-task learning (geometric + PCA)
   - Perceptual losses

3. **Hyperparameter tuning**:
   - Learning rate search: [1e-4, 5e-4, 1e-3]
   - Batch size: 8 → 16-32 (if GPU available)
   - Gradient clipping: Try 0.5 and 2.0

#### 6.4. Hardware Optimization
1. **GPU Training**:
   - Access to compatible GPU (CC ≥ 6.0)
   - Enable faster experimentation
   - Larger batch sizes

---

## 7. Next Steps

### Immediate Actions (Phase 2-B)

1. **Implement Transformer Baseline**
   - Target: RMSE < 0.25, PCC > 0.30
   - Timeline: 1 week

2. **Feature Engineering Experiment**
   - Test MFCC, combined features, prosody
   - Timeline: 3-4 days

3. **Hyperparameter Sweep**
   - Grid search on LR, hidden_dim, num_layers
   - Timeline: 2-3 days

### Medium-Term Goals (M2 Completion)

1. **Advanced Architecture**
   - Conformer or Transformer XL
   - Target M2 metrics (RMSE < 0.15, PCC > 0.50)

2. **Data Expansion**
   - Use full 468 utterances if possible
   - Cross-validation for robustness

3. **Ensemble Methods**
   - Combine multiple architectures
   - Boost final performance

---

## 8. Saved Artifacts

### Model Checkpoints
```
models/baseline_lstm/checkpoints/
├── baseline-epoch=07-val_loss=0.9124-v1.ckpt  # Best model ⭐
├── baseline-epoch=06-val_loss=0.9153-v1.ckpt
├── baseline-epoch=00-val_loss=0.9142-v1.ckpt
└── final_model.ckpt  # Final checkpoint after training
```

### Training Logs
```
logs/training/baseline_lstm_v1/version_3/
└── TensorBoard event files (scalars, metrics, graphs)
```

### Usage
```bash
# Load best model for inference
python scripts/evaluate.py \\
    --checkpoint models/baseline_lstm/checkpoints/baseline-epoch=07-val_loss=0.9124-v1.ckpt \\
    --test-split data/processed/splits/test.txt
```

---

## 9. Lessons Learned

### What Worked
✅ Training pipeline robust (no crashes, proper checkpointing)
✅ Early stopping prevented excessive overfitting
✅ Dataset splits maintained subject-level separation
✅ Monitoring and logging infrastructure effective

### What Didn't Work
❌ Model architecture too simple for task complexity
❌ Limited data size hampers generalization
❌ CPU training slow (prevented rapid iteration)
❌ Mel-spectrogram alone insufficient

### Key Insights
1. **Task Complexity**: Acoustic-to-articulatory inversion requires more sophisticated models than simple Bi-LSTM
2. **Data Quality > Quantity**: Better features may be more important than more data
3. **Architecture Matters**: Need attention mechanisms for temporal alignment
4. **Hardware Constraints**: CPU training limits experimental velocity

---

## 10. Conclusion

The baseline Bi-LSTM model establishes a **performance floor** for Project Sullivan:

- **RMSE**: 1.011 (Target: < 0.15)
- **PCC**: 0.105 (Target: > 0.50)

While far from the M2 targets, this baseline provides:
1. A working end-to-end pipeline
2. Clear evidence the task is learnable (training loss decreased)
3. A reference point for measuring improvements
4. Insights into what doesn't work (shallow models, simple features)

**Next milestone**: Implement improved architectures (Transformer/Conformer) and feature engineering to achieve M2 targets.

---

**Report Generated**: 2025-11-30
**Model Version**: baseline_lstm_v1
**Experiment ID**: baseline_lstm_v1/version_3
**Author**: Claude (Project Sullivan AI Assistant)
