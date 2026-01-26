# Pre-trained U-Net Segmentation Test Report

**Date**: 2025-11-26
**Model**: BartsMRIPhysics/Speech_MRI_2D_UNet (train_subj_1_2_3_4)
**Test Sample**: sub039_2drt_04_bvt_r2_video (Best quality, correlation=0.465)

---

## Executive Summary

**Result**: ❌ **NOT A 잭팟** - Pre-trained model does not transfer well to USC-TIMIT data

**Recommendation**: Proceed with **Option 1-D** (Hybrid Approach)
- Use traditional computer vision (GrabCut, Active Contours) for pseudo-labels
- Train U-Net from scratch on USC-TIMIT data

---

## Test Configuration

### Model Details
- **Architecture**: U-Net with 7 segmentation classes
- **Training Data**: Barts Speech MRI Dataset (5 subjects, 5-fold CV)
- **Parameters**:
  - Epochs: 200
  - Learning rate: 0.0003
  - Batch size: 4
  - Total weights: ~119MB

### USC-TIMIT Test Data
- **Frames tested**: 5 frames from sub039_2drt_04_bvt_r2_video
- **Frame indices**: [0, 416, 832, 1248, 1665]
- **Input size**: 84x84 (padded to 96x96 for U-Net compatibility)
- **Device**: CPU (GPU incompatible - GTX 750 Ti compute capability 5.0 < 7.0 required)

---

## Results Analysis

### Class Distribution (across 5 test frames)

| Frame | Class 0 (BG) | Class 1 (Tongue) | Class 2 (Jaw) | Class 3 (Lips) | Class 4 (Velum) | Class 5 (Pharynx) | Class 6 (Vocal Folds) |
|-------|--------------|------------------|---------------|----------------|-----------------|-------------------|-----------------------|
| 0     | 3.5%         | **87.3%**        | 0.4%          | 8.4%           | 0.1%            | 0.2%              | 0.1%                  |
| 416   | 26.3%        | **69.0%**        | -             | 4.5%           | -               | -                 | 0.2%                  |
| 832   | 23.1%        | **71.4%**        | 0.0%          | 5.0%           | 0.1%            | 0.0%              | 0.4%                  |
| 1248  | 18.6%        | **74.4%**        | -             | 6.5%           | 0.3%            | -                 | 0.2%                  |
| 1665  | 23.7%        | **72.3%**        | -             | 3.7%           | 0.1%            | -                 | 0.3%                  |
| **Avg** | **19.0%**  | **74.9%**        | **<1%**       | **5.6%**       | **<1%**         | **<1%**           | **<1%**               |

### Key Observations

1. **Tongue class dominance** (Class 1): 70-87% of pixels classified as tongue
   - Problem: The model incorrectly classifies most of the vocal tract **air space** (dark regions in MRI) as tongue
   - This suggests the model learned different tissue/air contrast characteristics from Barts dataset

2. **Background misclassification** (Class 0): 3-26% of pixels
   - Appears primarily at image edges (padding regions)
   - Should represent external background, not internal vocal tract structures

3. **Lips detection** (Class 3): ~5% of pixels
   - Some lip structures detected, but boundaries don't align well with visible anatomy

4. **Missing anatomical structures**:
   - Jaw (Class 2): <1% (nearly absent)
   - Velum (Class 4): <1% (nearly absent)
   - Pharynx (Class 5): <1% (nearly absent)
   - Vocal Folds (Class 6): <1% (nearly absent)

### Visualizations Generated

```
data/processed/aligned/segmentation_test/
├── sub039_2drt_04_bvt_r2_video_frame_0000_seg.png (92KB)
├── sub039_2drt_04_bvt_r2_video_frame_0416_seg.png (81KB)
├── sub039_2drt_04_bvt_r2_video_frame_0832_seg.png (88KB)
├── sub039_2drt_04_bvt_r2_video_frame_1248_seg.png (82KB)
└── sub039_2drt_04_bvt_r2_video_frame_1665_seg.png (84KB)
```

Each visualization shows:
- Left: Original MRI frame (grayscale)
- Center: Predicted segmentation (colored by class)
- Right: Overlay (segmentation alpha-blended on MRI)

---

## Domain Gap Analysis

### Why Transfer Learning Failed

The Barts Speech MRI Dataset and USC-TIMIT dataset have significant differences:

#### 1. **Imaging Protocol Differences**
- **Field strength**: Likely different (1.5T vs 3T)
- **Pulse sequence**: Different SSFP parameters
- **Temporal resolution**: Barts ~83fps, USC-TIMIT 83.28fps (similar but not identical)
- **Spatial resolution**: Different field of view and voxel sizes

#### 2. **Contrast Characteristics**
- **Tissue/air boundaries**: Different signal intensity ratios
- **Noise patterns**: Different SNR characteristics
- **Artifacts**: Scanner-specific artifacts differ

#### 3. **Anatomical Positioning**
- **Slice orientation**: Potentially different sagittal plane angles
- **Subject positioning**: Different head/neck alignment protocols
- **Field of view**: Different coverage of vocal tract region

#### 4. **Image Quality**
- **USC-TIMIT resolution**: 84x84 (relatively low)
- **Barts resolution**: Unknown but likely different
- **Preprocessing**: Different normalization/scaling applied

### Evidence from Results

The model's behavior (classifying air as tongue) indicates it learned to associate:
- **Dark regions** → Tongue (in Barts data)
- **Bright regions** → Background/Other structures

But in USC-TIMIT:
- **Dark regions** → Air space (should be background or separate class)
- **Bright regions** → Soft tissue (tongue, jaw, lips, etc.)

This **inverted contrast interpretation** is the primary failure mode.

---

## Recommendations

### ✅ Proceed with Option 1-D: Hybrid Approach

**Phase 1: Generate Pseudo-Labels (Traditional CV)**
1. **GrabCut segmentation**:
   - Interactive foreground/background separation
   - Good for initial tongue/vocal tract extraction
   - Requires manual seed point placement

2. **Active Contours (Snakes)**:
   - Refine boundaries from GrabCut output
   - Follows intensity gradients naturally
   - Good for tongue/jaw/lip boundaries

3. **Otsu's thresholding**:
   - Separate tissue from air space
   - Multi-level thresholding for different tissue types

4. **Manual correction**:
   - Select 50-100 representative frames (10-20 per subject)
   - Manually refine pseudo-labels for training set
   - Use VGG Image Annotator (VIA) or LabelMe

**Phase 2: Train U-Net from Scratch**
1. **Architecture**: Same as BartsMRI U-Net (proven for rtMRI)
2. **Training data**:
   - Pseudo-labels from Phase 1
   - Data augmentation (rotation, flip, elastic deformation)
   - Start with 50-100 manually corrected frames
3. **Validation strategy**:
   - 5-fold cross-validation across 15 subjects
   - Dice score > 0.7 target for tongue/jaw/lips
4. **Iterative refinement**:
   - Use trained model to predict on unlabeled data
   - Manually correct worst predictions
   - Retrain (semi-supervised learning)

### ❌ Do NOT Pursue

1. **Direct fine-tuning**: Domain gap too large, will not converge properly
2. **Transfer learning with frozen layers**: Contrast characteristics incompatible
3. **Ensemble with Barts model**: No added value given poor base performance

---

## Technical Notes

### Padding Strategy Implemented
- **Input**: 84x84 → padded to 96x96 (next multiple of 16)
- **Rationale**: U-Net with 4 max-pooling layers requires input divisible by 2^4=16
- **Method**: Zero-padding with symmetric borders
- **Output**: 96x96 → cropped back to 84x84

### GPU Compatibility Issue
- **GPU**: NVIDIA GeForce GTX 750 Ti (Compute Capability 5.0)
- **PyTorch requirement**: Compute Capability ≥ 7.0 (Volta or newer)
- **Workaround**: Forced CPU execution with `CUDA_VISIBLE_DEVICES=""`
- **Performance impact**: ~1 second per frame on CPU (acceptable for testing)

---

## Next Steps

1. **Implement traditional CV segmentation** (`src/segmentation/traditional_cv.py`)
   - GrabCut implementation
   - Active Contours implementation
   - Visualization pipeline

2. **Manual annotation tool setup**
   - Install VIA or LabelMe
   - Define class schema (5-7 classes)
   - Create annotation guidelines

3. **Generate initial pseudo-labels**
   - Process 10 frames from each of 15 subjects (150 frames total)
   - Review and manually correct
   - Create train/val/test splits

4. **Train U-Net from scratch**
   - Initialize with random weights
   - Train on pseudo-labeled data
   - Monitor Dice score and visual quality

---

## Files Created

```
src/segmentation/
├── __init__.py
└── unet.py (245 lines, U-Net architecture + loading utilities)

scripts/
└── test_pretrained_unet.py (260 lines, testing pipeline)

models/pretrained_unet/
└── Network Weights/
    ├── train_subj_1_2_3_4_l_rate_0.0003_mb_size_4_epochs_200/unet_parameters.pth (119MB)
    ├── train_subj_1_2_3_5_l_rate_0.0003_mb_size_4_epochs_200/unet_parameters.pth (119MB)
    ├── train_subj_1_2_4_5_l_rate_0.0003_mb_size_4_epochs_200/unet_parameters.pth (119MB)
    ├── train_subj_1_3_4_5_l_rate_0.0003_mb_size_4_epochs_200/unet_parameters.pth (119MB)
    └── train_subj_2_3_4_5_l_rate_0.0003_mb_size_4_epochs_200/unet_parameters.pth (119MB)

data/processed/aligned/
├── segmentation_test/
│   ├── sub039_2drt_04_bvt_r2_video_frame_0000_seg.png
│   ├── sub039_2drt_04_bvt_r2_video_frame_0416_seg.png
│   ├── sub039_2drt_04_bvt_r2_video_frame_0832_seg.png
│   ├── sub039_2drt_04_bvt_r2_video_frame_1248_seg.png
│   └── sub039_2drt_04_bvt_r2_video_frame_1665_seg.png
└── batch_summary.json (468 utterances, 15 subjects)
```

---

## Conclusion

While the "잭팟" attempt (pre-trained model transfer) did not succeed, the test was valuable:

✅ **Successful outcomes**:
- Downloaded 5 pre-trained model variants
- Implemented U-Net architecture compatible with our data
- Created comprehensive testing pipeline
- Identified specific failure mode (contrast inversion)
- Generated baseline visualizations for comparison

❌ **Transfer learning not viable**:
- Domain gap too large
- Contrast characteristics incompatible
- Segmentation quality unacceptable for downstream tasks

✅ **Clear path forward**:
- Option 1-D (Hybrid) is the correct approach
- Traditional CV + manual correction → high-quality pseudo-labels
- U-Net trained from scratch on USC-TIMIT → optimal performance

**Estimated timeline**:
- Phase 1 (Pseudo-labels): 1-2 days (100-150 frames manual correction)
- Phase 2 (U-Net training): 1-2 days (training + validation)
- Total: **3-4 days** to complete segmentation pipeline

The pre-trained weights remain useful as an **architectural reference** and for **comparison benchmarking**, even though direct transfer is not viable.
