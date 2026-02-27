# HDDB Data Processing Pipeline - Complete Implementation

**Date**: 2026-01-12
**Status**: ✅ **All Pipeline Scripts Implemented and Tested**
**Next Step**: Full 27-subject processing (in progress)

---

## 🎉 Summary

Successfully implemented and tested the complete HDDB data processing pipeline:

1. ✅ **MRI Segmentation** - U-Net inference on HDDB dataset
2. ✅ **Articulatory Parameter Extraction** - 14 geometric + 10 PCA features
3. ✅ **Audio Feature Extraction** - 80-dim Mel + 13-dim MFCC
4. ✅ **Dataset Split Generation** - Train/Val/Test (70/15/15)

All scripts tested successfully on sub010 (1 subject, 32 utterances).

---

## 📊 Test Results (sub010)

### Segmentation
- **Script**: `scripts/segment_hddb_dataset.py`
- **Input**: HDDB recon H5 files from `/mnt/HDDB/dataset/my_dataset/dataset/sub010/2drt/recon/`
- **Output**: Binary masks (0=background, 1=airway) in `data/processed/segmentations/`
- **Performance**: ~53 frames/second on CPU
- **Results**:
  - 32 utterances processed
  - 76,204 total frames
  - Processing time: 24.5 minutes
  - Output format: NPZ files with shape (num_frames, 84, 84)
  - Class distribution: ~72% background, ~28% airway (consistent)

### Parameter Extraction
- **Script**: `scripts/extract_articulatory_params_hddb.py`
- **Input**: Segmentation masks from previous step
- **Output**: Articulatory parameters in `data/processed/parameters/`
- **Features**:
  - 14 geometric features: area, centroid (x,y), bounding box (top/bottom/left/right), aspect ratio, solidity, extent, perimeter, circularity, ellipse ratio, ellipse angle
  - 10 PCA components from flattened masks
  - Total: 24 features per frame
- **Performance**: ~45 seconds for 76K frames
- **Results**:
  - 32 utterances processed
  - 76,204 frames
  - PCA explained variance: 33.85%
  - Output format: NPZ files with shape (num_frames, 24)

### Audio Feature Extraction
- **Script**: `scripts/extract_audio_features_hddb.py`
- **Input**: WAV files from `/mnt/HDDB/dataset/my_dataset/dataset/sub010/2drt/audio/`
- **Output**: Audio features in `data/processed/audio_features/`
- **Features**:
  - 80-dimensional log mel-spectrogram (primary features)
  - 13-dimensional MFCC coefficients (alternative features)
- **Parameters**:
  - Sample rate: 16000 Hz (resampled from 20000 Hz)
  - FFT size: 512
  - Hop length: 160 samples (10ms frames)
  - Frequency range: 0 - 8000 Hz
- **Performance**: ~3 seconds for 32 utterances
- **Results**:
  - 32 utterances processed
  - 15.8 minutes total audio duration
  - Output format: NPZ files with mel (T, 80) and mfcc (T, 13)

### Dataset Splitting
- **Script**: `scripts/create_dataset_splits_hddb.py`
- **Input**: All three processed datasets
- **Output**: JSON manifests in `data/processed/splits/`
- **Strategy**: Subject-level splitting (all utterances from same subject in same split)
- **Ratios**: Train 70%, Val 15%, Test 15%
- **Results** (1 subject):
  - Complete utterances found: 32
  - All assigned to test set (expected with single subject)
  - Generated: train.json, val.json, test.json, split_summary.json

---

## 🗂️ Output Directory Structure

```
data/processed/
├── segmentations/
│   └── {utterance_name}/
│       └── {utterance_name}_segmentations.npz
│           - segmentations: (T, 84, 84) uint8
│           - class_distributions: (T, 2) float32
│           - metadata: utterance_name, num_frames, class_names
│
├── parameters/
│   ├── {utterance_name}_params.npz
│   │   - parameters: (T, 24) float32 [14 geometric + 10 PCA]
│   │   - geometric_features: (T, 14) float32
│   │   - pca_features: (T, 10) float32
│   │   - feature_names: list of 14 feature names
│   └── pca_model.npz
│       - PCA model components and statistics
│
├── audio_features/
│   └── {utterance_name}_audio.npz
│       - mel_spectrogram: (T, 80) float32
│       - mfcc: (T, 13) float32
│       - metadata: sample_rate, hop_length, n_fft
│
└── splits/
    ├── train.json
    ├── val.json
    ├── test.json
    └── split_summary.json
```

**Note**: `{utterance_name}` format: `sub010_2drt_01_vcv1_r1_recon`

---

## 🚀 Full Dataset Processing (In Progress)

### Status
- ✅ All 4 pipeline scripts implemented and tested
- ✅ sub010 (1 subject) processing complete
- 🔄 **Full 27-subject segmentation running in background**
  - Command: `nohup python scripts/segment_hddb_dataset.py --subjects all --device cpu --output-dir data/processed/segmentations > /tmp/full_segmentation.log 2>&1 &`
  - PID: 10995
  - Monitor: `tail -f /tmp/full_segmentation.log`
  - Estimated time: 9-14 hours (27 subjects × ~25 min/subject)

### Next Steps (After Segmentation Completes)

1. **Run Parameter Extraction on All 27 Subjects**
   ```bash
   python scripts/extract_articulatory_params_hddb.py \
     --segmentation-dir data/processed/segmentations \
     --output-dir data/processed/parameters
   ```

2. **Run Audio Feature Extraction on All 27 Subjects**
   ```bash
   python scripts/extract_audio_features_hddb.py \
     --subjects all \
     --data-root /mnt/HDDB/dataset/my_dataset/dataset \
     --output-dir data/processed/audio_features
   ```

3. **Generate Final Dataset Splits**
   ```bash
   python scripts/create_dataset_splits_hddb.py \
     --segmentation-dir data/processed/segmentations \
     --parameter-dir data/processed/parameters \
     --audio-dir data/processed/audio_features \
     --output-dir data/processed/splits
   ```

4. **Verify Dataset Integrity**
   - Check split_summary.json for expected distribution
   - Verify frame counts match across all three data types
   - Test PyTorch DataLoader with sample batches

5. **Start Transformer Training**
   - Use existing training pipeline from `scripts/train_transformer.py`
   - May need to adapt DataLoader for HDDB format
   - Expected training time: 2-3 hours on GPU (Google Colab T4)

---

## 📋 Script Usage

### 1. Segmentation
```bash
python scripts/segment_hddb_dataset.py \
  --subjects sub010,sub011,sub012 \  # or "all" for all subjects
  --data-root /mnt/HDDB/dataset/my_dataset/dataset \
  --output-dir data/processed/segmentations \
  --model models/unet_scratch/unet_best.pth \
  --device cpu  # or "cuda"
```

### 2. Parameter Extraction
```bash
python scripts/extract_articulatory_params_hddb.py \
  --segmentation-dir data/processed/segmentations \
  --output-dir data/processed/parameters \
  --no-pca  # Skip PCA if only geometric features needed
```

### 3. Audio Feature Extraction
```bash
python scripts/extract_audio_features_hddb.py \
  --subjects sub010,sub011,sub012 \  # or "all"
  --data-root /mnt/HDDB/dataset/my_dataset/dataset \
  --output-dir data/processed/audio_features \
  --no-mel  # Skip mel-spectrogram if not needed
  --no-mfcc  # Skip MFCC if not needed
```

### 4. Dataset Splitting
```bash
python scripts/create_dataset_splits_hddb.py \
  --segmentation-dir data/processed/segmentations \
  --parameter-dir data/processed/parameters \
  --audio-dir data/processed/audio_features \
  --output-dir data/processed/splits \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 42
```

---

## 🔧 Technical Details

### Frame Rate Alignment
- **MRI**: Variable frame rate (~50-80 fps depending on sequence)
- **Audio**: 16 kHz sample rate, 10ms hop → 100 fps
- **Solution**: Need temporal alignment in DataLoader
  - Option 1: Resample audio features to match MRI frame count
  - Option 2: Interpolate MRI parameters to match audio frame rate
  - Option 3: Use attention mechanism to learn alignment (preferred for Transformer)

### Data Naming Convention
- **MRI/Segmentation/Parameters**: `{subject}_{modality}_{sequence}_{task}_{repetition}_recon`
- **Audio**: WAV files have `_audio.wav` suffix, converted to `_recon` in feature files
- **Example**: `sub010_2drt_01_vcv1_r1_recon`

### U-Net Model
- **Architecture**: 5-layer encoder-decoder with skip connections
- **Input**: (B, 1, 96, 96) - grayscale MRI (padded from 84×84)
- **Output**: (B, 1, 96, 96) - binary segmentation logits
- **Post-processing**: Sigmoid + threshold at 0.5, unpad to 84×84
- **Performance**: 81.8% Dice score on USC-TIMIT test set

### PCA Model
- **Input**: Flattened 84×84 = 7056-dim binary masks
- **Output**: 10 principal components
- **Explained Variance**: 33.85% (sub010 only)
- **Note**: PCA is fitted on all training data and saved for inference

---

## 📈 Expected Full Dataset Statistics

Based on HDDB metadata and sub010 results:

- **Total subjects**: 27
- **Estimated utterances**: ~800 (varies by subject)
- **Estimated total frames**: ~1.9M MRI frames
- **Total audio duration**: ~8-10 hours
- **Dataset size**:
  - Segmentations: ~3.8 GB (compressed NPZ)
  - Parameters: ~180 MB (compressed NPZ)
  - Audio features: ~1.5 GB (compressed NPZ)
  - **Total**: ~5.5 GB

**Split Distribution** (70/15/15 by subjects):
- Train: ~18-19 subjects, ~560 utterances
- Val: ~4 subjects, ~120 utterances
- Test: ~4 subjects, ~120 utterances

---

## ✅ Pipeline Validation Checklist

- [x] Segmentation script works on HDDB recon H5 files
- [x] Parameter extraction successfully extracts geometric + PCA features
- [x] Audio extraction loads WAV files and computes mel/MFCC
- [x] Dataset splitter finds complete utterances and creates manifests
- [x] All output file formats verified (NPZ structure, shapes, dtypes)
- [x] Naming convention consistent across all pipeline stages
- [x] Full 27-subject segmentation launched in background
- [ ] Full dataset parameter extraction (pending segmentation)
- [ ] Full dataset audio extraction (pending segmentation)
- [ ] Final dataset splits generation (pending above)
- [ ] PyTorch DataLoader adaptation for HDDB format
- [ ] Transformer model training on HDDB dataset

---

## 🎯 Milestone Progress Update

### M1: Data Pipeline Completion
**Previous Status**: 85% (USC-TIMIT only)
**Current Status**: 95% (HDDB scripts complete, full processing in progress)

**Completed**:
- ✅ MRI segmentation pipeline (HDDB compatible)
- ✅ Articulatory parameter extraction (14 geometric + 10 PCA)
- ✅ Audio feature extraction (Mel + MFCC)
- ✅ Dataset split generation (subject-level)
- ✅ All scripts tested on sub010

**In Progress**:
- 🔄 Full 27-subject processing (est. 9-14 hours)

**Remaining**:
- ⏳ Dataset integrity verification
- ⏳ DataLoader adaptation for HDDB

**Estimated Completion**: Within 24 hours (pending full segmentation)

### M2: Baseline Model Training
**Status**: Ready to proceed after M1 completion

**Plan**:
- Use existing Transformer implementation
- Adapt DataLoader for HDDB format
- Train on Google Colab T4 GPU (2-3 hours)
- Target: RMSE < 0.15, PCC > 0.50

---

## 🐛 Issues Resolved

1. ✅ **H5 File Structure**: HDDB recon files use 'recon' key (not complex nested structure)
2. ✅ **Binary Segmentation**: Model outputs 1 class (airway), not multi-class
3. ✅ **PyTorch Lightning Checkpoint**: Handle 'model.' prefix in state_dict keys
4. ✅ **Naming Convention**: Audio files now use `_recon` suffix to match MRI naming
5. ✅ **Audio Source**: WAV files in separate `audio/` directory, not in H5 files

---

## 📝 Notes

- Sub010 has 32 utterances with ~76K MRI frames (15.8 min audio)
- Segmentation speed: ~53 fps on CPU (Intel Xeon)
- Full dataset processing is CPU-bound, can be parallelized across subjects
- Frame rate mismatch (MRI vs audio) will be handled in DataLoader
- PCA model needs to be refitted on full training set after splits are final

---

**Report Generated**: 2026-01-12 03:50 UTC
**Author**: Claude & Research Team
**Next Review**: After full 27-subject processing completes
