# Training Status Report

**Date**: 2025-11-30
**Status**: 🟢 Training in Progress

---

## Current Training Status

### Baseline LSTM Model Training (Phase 2-A)

- **Model**: Bidirectional LSTM (613,902 parameters)
  - Input: 80-dim mel-spectrogram
  - Hidden: 128 units × 2 layers
  - Output: 14-dim geometric parameters

- **Training Progress**:
  - Epochs Completed: 7+/50
  - Current Validation Loss: 0.9124 (improving)
  - Training Method: CPU (single-threaded DataLoader)

- **Hardware Note**:
  - GPU (GTX 750 Ti) incompatible with PyTorch 2.9 (requires CC 6.0+, has CC 5.0)
  - Training on CPU with `num_workers=0` to avoid worker crashes

---

## Dataset Statistics

- **Total Utterances**: 75 (from 15 subjects)
- **Total Frames**: 186,124
- **Splits**:
  - Train: 50 utterances (10 subjects)
  - Validation: 10 utterances (2 subjects)
  - Test: 15 utterances (3 subjects)

---

## Completed Tasks (Phase 2-A)

### ✅ Data Pipeline (M1 - 100%)
- [x] Vocal tract segmentation (75 utterances)
- [x] Articulatory parameter extraction (geometric + PCA)
- [x] Audio feature extraction (mel-spectrogram + MFCC)
- [x] Dataset splitting (subject-level)

### ✅ Model Implementation
- [x] PyTorch Dataset for audio-parameter pairs
- [x] Bidirectional LSTM architecture
- [x] Training configuration and hyperparameters
- [x] PyTorch Lightning training loop
- [x] Model checkpointing and early stopping

### ✅ Training Infrastructure
- [x] Training scripts with error handling
- [x] TensorBoard logging
- [x] Training monitoring scripts
- [x] Checkpoint-based progress tracking

---

## Monitoring

### Current Training Session

**Process**: Running (PID varies, check with `pgrep -f train_baseline.py`)

**Monitor Training**:
```bash
# Simple status check
bash scripts/monitor_training_simple.sh

# Real-time monitoring (updates every 30 seconds)
watch -n 30 bash scripts/monitor_training_simple.sh

# TensorBoard visualization
bash scripts/start_tensorboard.sh
# Access at http://localhost:6006
```

**Check Checkpoints**:
```bash
ls -lht models/baseline_lstm/checkpoints/
```

---

## Latest Checkpoints

```
- last-v1.ckpt (Latest)
- baseline-epoch=07-val_loss=0.9124.ckpt
- baseline-epoch=06-val_loss=0.9153.ckpt
- baseline-epoch=00-val_loss=0.9142.ckpt
```

Validation loss trend: **Improving** (0.9153 → 0.9124)

---

## Next Steps

### Pending Tasks

1. **Complete Training** (In Progress)
   - Wait for 50 epochs to complete
   - Monitor for convergence and early stopping

2. **Model Evaluation** (Pending)
   - Evaluate on test split
   - Generate prediction visualizations
   - Compute final metrics (RMSE, MAE, PCC)

3. **Baseline Report** (Pending)
   - Compare against M2 targets (RMSE < 0.15, PCC > 0.50)
   - Document model performance
   - Identify areas for improvement

---

## Configuration

**Training Config**: `configs/baseline_config.yaml`
- Batch size: 8
- Learning rate: 0.001 (Adam optimizer)
- LR scheduler: ReduceLROnPlateau
- Gradient clipping: 1.0
- Early stopping patience: 10 epochs
- Number of workers: 0 (single-threaded)

**Quick Test Config**: `configs/baseline_quick_test.yaml` (10 epochs for rapid testing)

---

## Files Structure

```
Project_Sullivan/
├── configs/
│   ├── baseline_config.yaml          # Main training config
│   └── baseline_quick_test.yaml      # Quick test config (10 epochs)
├── scripts/
│   ├── train_baseline.py             # Training script
│   ├── monitor_training_simple.sh    # Status monitor
│   ├── start_tensorboard.sh          # TensorBoard launcher
│   └── check_training_status.sh      # Legacy monitor
├── src/
│   ├── modeling/
│   │   ├── dataset.py                # PyTorch Dataset
│   │   └── baseline_lstm.py          # Model architecture
│   ├── audio_features/               # Audio feature extraction
│   └── parameter_extraction/         # Articulatory parameters
├── models/baseline_lstm/
│   └── checkpoints/                  # Model checkpoints (gitignored)
└── logs/training/                    # Training logs (gitignored)
```

---

## Known Issues & Solutions

### ✅ Resolved Issues

1. **GPU Compatibility Error**
   - Issue: GTX 750 Ti (CC 5.0) incompatible with PyTorch 2.9
   - Solution: Training on CPU

2. **DataLoader Worker Crashes**
   - Issue: `RuntimeError: DataLoader worker killed by signal: Terminated`
   - Solution: Set `num_workers: 0` in config

3. **PyTorch Lightning Configuration Errors**
   - Issue: `devices` parameter with CPU, `verbose` in ReduceLROnPlateau
   - Solution: Fixed trainer initialization and removed unsupported parameters

4. **Metrics Computation IndexError**
   - Issue: Array dimension mismatch in masked metric calculation
   - Solution: Fixed mask squeezing in `baseline_lstm.py:231`

---

## Estimated Timeline

- **Training Duration**: ~12-15 hours total (CPU)
- **Started**: 2025-11-30 04:00 (approx)
- **Expected Completion**: 2025-11-30 16:00-19:00 (approx)

---

## Contact & Notes

**Project**: Project Sullivan - Acoustic-to-Articulatory Inversion
**Dataset**: USC-TIMIT (75 utterances, 15 subjects)
**Objective**: Baseline LSTM model for vocal tract parameter estimation

**Last Updated**: 2025-11-30 07:00
