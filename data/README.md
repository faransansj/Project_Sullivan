# Data Directory

This directory contains all data for Project Sullivan.

**⚠️ Data files are NOT tracked by Git due to size.**

## Structure

```
data/
├── raw/                   # Original USC-TIMIT data (download required)
├── processed/             # Preprocessed data
│   ├── aligned/          # Denoised & aligned MRI/audio
│   ├── segmented/        # Segmented MRI frames
│   └── parameters/       # Extracted articulatory parameters
│       ├── train/
│       ├── val/
│       └── test/
└── experiments/          # Experiment-specific data
```

## Data Setup & Preprocessing

See: [`DATA_PREPROCESSING_GUIDE.md`](./DATA_PREPROCESSING_GUIDE.md)

This comprehensive guide explains how to:
1. Download the raw USC-TIMIT database
2. Run the full automated pipeline to generate aligned features and target parameters.

**Quick Dataset Link:** https://doi.org/10.6084/m9.figshare.13725546.v1
