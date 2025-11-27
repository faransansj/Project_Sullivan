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

## Download Data

See: `../docs/DATA_DOWNLOAD_GUIDE.md`

**Quick Link:** https://doi.org/10.6084/m9.figshare.13725546.v1
