# Project Sullivan: Dataset Download & Preprocessing Guide

This guide explains how to properly download the USC-TIMIT dataset and run the full data preprocessing pipeline to extract features and articulatory parameters needed for Project Sullivan.

## 1. Dataset Download

Project Sullivan uses the **USC-TIMIT** database (rtMRI video and audio).

1. **Download Link**: [USC-TIMIT Dataset on Figshare](https://doi.org/10.6084/m9.figshare.13725546.v1)
2. **Placement**: Once downloaded, extract and place all raw data in the `data/raw/` directory.
   - Example path: `data/raw/USC-TIMIT/`

⚠️ **Note**: The full dataset can be quite large (600GB+). Ensure you have sufficient storage space before downloading. Typically, NAS (Network Attached Storage) is used to store this dataset.

## 2. Preprocessing Pipeline

The preprocessing pipeline takes raw audio and MRI frames and systematically converts them into ready-to-use inputs and targets for the AI models. 

The process is fully automated for batch-processing via the `full_preprocess_pipeline.sh` bash script. We use `uv` for script execution.

### Running the Full Pipeline

To run the entire pipeline for all subjects:
```bash
bash scripts/infra/full_preprocess_pipeline.sh --all
```

To run the pipeline for specific subjects:
```bash
bash scripts/infra/full_preprocess_pipeline.sh --subjects sub011 sub012
```

To resume the pipeline from a specific step (e.g., Step 3):
```bash
bash scripts/infra/full_preprocess_pipeline.sh --all --skip-to 3
```

You can also run a dry-run (preview execution plan) by adding `--dry-run`.

### Pipeline Steps Explained

The pipeline consists of 5 main steps:

#### **Step 1: Batch Preprocess (Raw → Aligned HDF5)**
- **Script**: `scripts/batch_preprocess.py`
- **What it does**: Cleans up audio (denoising), aligns exact timings between MRI and Audio using cross-correlation or synchronization signals, and packages them into HDF5 formats under `data/processed/aligned/`.

#### **Step 2: Segmentation (Aligned HDF5 → Masks)**
- **Script**: `scripts/segment_subset.py`
- **What it does**: Runs U-Net inference on the aligned MRI frames to generate binary/categorical semantic masks of the vocal tract.
- **Output**: Masks saved under `data/processed/segmentations/`.

#### **Step 3: Extract Audio Features (HDF5 → Mel/MFCC)**
- **Script**: `scripts/extract_audio_features.py`
- **What it does**: Takes the denoised audio from HDF5 files and processes it into Mel-spectrograms or MFCCs, ready to be ingested by Transformer/Conformer encoders.
- **Output**: Arrays saved under `data/processed/audio_features/`.

#### **Step 4: Extract Articulatory Parameters (Masks → Geometric + PCA)**
- **Script**: `scripts/extract_articulatory_params.py`
- **What it does**: Extracts geometric constraints (14-dim output) and PCA-based articulatory parameters (10-dim representation) from the masked MRI frames.
- **Output**: Array targets saved under `data/processed/parameters/`.

#### **Step 5: Package Results (Optional for External Server)**
- **What it does**: Validates the output folders, counts sizes, and compresses the final targets (`audio_features`, `parameters`, `splits`) into `data/transfer_archives/` for easy moving to external A100/A6000 GPU computation nodes.

## 3. Preprocessed Output Structure

When the pipeline is complete, your `data/processed/` directory will look similar to this:

```
data/processed/
├── aligned/            # Intermediate step: Aligned multi-modal files
├── segmentations/      # Extracted U-Net masks
├── audio_features/     # Final Models Inputs
└── parameters/         # Final Models Targets (Ground Truth)
```

## 4. Configuration

The preprocessing steps use YAML files in `configs/` to configure the specifics.
- **Default Pipeline Config**: `configs/preprocess_nas.yaml` (Optimized for running CPU-heavy processing safely on the NAS)
- **Settings**: You can adjust audio sample rates, max frames, filtering methods, and device selections from this configuration.
