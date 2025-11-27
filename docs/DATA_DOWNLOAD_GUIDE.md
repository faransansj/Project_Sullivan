# USC Speech MRI Dataset Download Guide

**Project Sullivan - Milestone M1**
**Updated:** 2025-11-25

---

## ⚠️ Important Notice

The `usc_speech_mri-master.zip` file contains **only the reconstruction code**, not the actual MRI data.

**You must download the actual dataset separately from figshare.**

---

## 📦 Dataset Information

### Official Dataset Location

**Figshare Repository:** https://doi.org/10.6084/m9.figshare.13725546.v1

### Dataset Details

- **Total Size:** ~500 GB (raw data) + ~100 GB (reconstructed data)
- **Format:** HDF5 (.h5), MATLAB (.mat), Video (.mp4)
- **Subjects:** 75 speakers
- **Content:**
  - Raw k-space MRI data
  - Reconstructed rtMRI videos
  - Synchronized audio recordings
  - 3D volumetric images

---

## 📥 Download Instructions

### Option 1: Download Specific Files (Recommended for Initial Testing)

**For Phase 1 development, you only need a subset of the data.**

1. Visit the figshare link: https://doi.org/10.6084/m9.figshare.13725546.v1

2. Download **one or two subject files** to start:
   - Look for files named: `subject_XX.tar.gz` or similar
   - Recommended: Start with subject 01 (native English speaker)

3. Extract to the project:
   ```bash
   cd /home/midori/Develop/Project_Sullivan/data/raw/
   tar -xzf ~/Downloads/subject_01.tar.gz
   ```

### Option 2: Download Full Dataset (For Production)

**Warning: This will download ~600 GB of data!**

```bash
# Use wget or curl with the figshare API
# (Specific commands will be provided once you access the figshare page)

cd /home/midori/Develop/Project_Sullivan/data/raw/
# Download script here
```

---

## 📂 Expected Data Structure

After downloading and extracting, your directory should look like:

```
data/raw/
├── usc_speech_mri-master/          # Reconstruction code (already present)
└── usc_timit_data/                 # Actual data (to be downloaded)
    ├── subject_01/
    │   ├── raw/
    │   │   ├── kspace_data.h5      # Raw MRI k-space data
    │   │   └── trajectory.mat      # Sampling trajectory
    │   ├── reconstructed/
    │   │   ├── mri_video.mp4       # Reconstructed MRI video
    │   │   ├── frames/             # Individual frames
    │   │   │   ├── frame_0001.png
    │   │   │   ├── frame_0002.png
    │   │   │   └── ...
    │   │   └── metadata.json       # Frame timestamps, etc.
    │   ├── audio/
    │   │   ├── synchronized_audio.wav
    │   │   └── audio_metadata.json
    │   └── volumetric/
    │       └── 3d_volume.nii.gz    # 3D volumetric data
    ├── subject_02/
    └── ...
```

---

## 🔍 Dataset Contents by Subject

Each subject folder typically contains:

### 1. Raw Data (`raw/`)
- **k-space data**: Raw MRI measurements (HDF5 format)
- **Trajectory**: Spiral sampling trajectory (MATLAB format)
- **Coil sensitivity maps**: For multi-coil reconstruction

### 2. Reconstructed Data (`reconstructed/`)
- **MRI videos**: Mid-sagittal view of vocal tract during speech
- **Frame rate**: ~50-80 fps
- **Resolution**: 68×68 to 84×84 pixels
- **Format**: MP4 video or individual PNG frames

### 3. Audio Data (`audio/`)
- **Synchronized audio**: Speech recorded during MRI acquisition
- **Sample rate**: 20 kHz (typical)
- **Format**: WAV
- **Synchronization info**: Timestamps for audio-video alignment

### 4. Utterances
- **TIMIT sentences**: Standard phonetically balanced sentences
- **Sustained vowels**: /a/, /i/, /u/
- **Nonsense words**: For specific phonetic analysis

---

## 📋 Recommended Download for Phase 1 (Milestone M1)

For initial development and testing:

**Minimum Dataset:**
- 1-2 subjects with reconstructed data
- ~5-10 GB per subject

**Files to prioritize:**
1. Reconstructed MRI frames (`.png` or `.mp4`)
2. Synchronized audio (`.wav`)
3. Metadata (`.json` or `.mat`)

**Skip for now:**
- Raw k-space data (only needed if you want to re-reconstruct)
- 3D volumetric data (for Phase 3)

---

## 🛠️ Data Verification Checklist

After downloading, verify your data:

```bash
cd /home/midori/Develop/Project_Sullivan

# Check if MRI frames exist
find data/raw -name "*.png" -o -name "*.mp4" | head -5

# Check if audio files exist
find data/raw -name "*.wav" | head -5

# Check file sizes
du -sh data/raw/*/
```

Expected output:
```
data/raw/usc_speech_mri-master/: ~10 MB (code only)
data/raw/subject_01/: ~5-10 GB (with data)
```

---

## 📞 Need Help?

### Issues with Download
- Check figshare status: https://status.figshare.com/
- Alternative mirrors may be available in the paper

### Dataset Questions
- **Paper (arXiv):** https://arxiv.org/abs/2102.07896
- **GitHub Issues:** https://github.com/yongwanlim/USC_Speech_MRI

### Project-Specific Questions
- Contact Project Lead: [이름]
- GitHub Issues: [Project Repository]

---

## 📚 Citation

If you use this dataset, please cite:

```bibtex
@article{lim2021multispeaker,
  title={A multispeaker dataset of raw and reconstructed speech production real-time MRI video and 3D volumetric images},
  author={Lim, Yongwan and Toutios, Asterios and Bliesener, Yannick and others},
  journal={Scientific Data},
  volume={8},
  number={1},
  pages={187},
  year={2021},
  publisher={Nature Publishing Group}
}
```

---

## ✅ Next Steps After Download

1. **Update researcher_manual.md** with actual data paths
2. **Run EDA** (notebooks/01_EDA.ipynb)
3. **Document data statistics** (docs/data_statistics.md)
4. **Begin Phase 1 preprocessing**

---

*Last updated: 2025-11-25*
*For: Milestone M1 - Data Pipeline Construction*
