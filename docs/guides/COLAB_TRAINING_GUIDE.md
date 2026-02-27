# Project Sullivan - Google Colab Training Guide

**Quick Start Guide for Training on Google Colab**

---

## 📋 Overview

This guide walks you through training the Transformer model on Google Colab with free GPU access.

**Time Required**: ~15 minutes setup + 2-3 hours training

---

## 🚀 Step-by-Step Instructions

### Step 1: Prepare Data Archives (Local Machine)

Data archives are already created in `colab_data_archives/`:

```bash
colab_data_archives/
├── processed_data_all.tar.gz      (78M) - ⭐ Recommended
├── audio_features.tar.gz          (48M)
├── parameters.tar.gz              (11M)
├── segmentations.tar.gz           (19M)
└── splits.tar.gz                  (4K)
```

**Recommendation**: Use `processed_data_all.tar.gz` (all data in one file)

---

### Step 2: Upload to Google Drive

1. **Go to Google Drive**: https://drive.google.com

2. **Create a folder** (optional but recommended):
   - Name: `Project_Sullivan_Data`

3. **Upload archive**:
   - Drag and drop `processed_data_all.tar.gz` into the folder
   - Wait for upload to complete (~2-3 minutes depending on connection)

---

### Step 3: Get Shareable Link & File ID

1. **Right-click** on the uploaded file → **Share**

2. **Change access**:
   - Click "Restricted" → Select "Anyone with the link"
   - Role: Viewer
   - Click "Copy link"

3. **Extract File ID** from the URL:
   ```
   https://drive.google.com/file/d/1a2B3c4D5e6F7g8H9i0J_EXAMPLE_ID/view?usp=sharing
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^
                                    This is your FILE_ID
   ```

4. **Save the FILE_ID** - you'll need it in Step 5

**Example File ID**: `1a2B3c4D5e6F7g8H9i0J_EXAMPLE_ID`

---

### Step 4: Push Code to GitHub

If you haven't already:

```bash
# Create GitHub repo (if not exists)
gh repo create Project_Sullivan --public --source=. --remote=origin

# Push code
git push -u origin main
```

**Alternative**: Upload via GitHub web interface
- Go to https://github.com/new
- Create repository named `Project_Sullivan`
- Follow instructions to push existing repository

---

### Step 5: Open Colab Notebook

1. **Upload notebook to Google Drive**:
   - Upload `notebooks/Project_Sullivan_Transformer_Training.ipynb` to Drive

2. **Open with Google Colab**:
   - Right-click notebook → Open with → Google Colaboratory

3. **Or create new notebook** and copy cells from the provided notebook

---

### Step 6: Configure Notebook

In the **Configuration** cell, update these variables:

```python
# Google Drive File ID
GDRIVE_FILE_ID_ALL = '1a2B3c4D5e6F7g8H9i0J_EXAMPLE_ID'  # ← Your actual file ID
USE_COMBINED_ARCHIVE = True

# GitHub Repository
GITHUB_REPO = 'YOUR_USERNAME/Project_Sullivan'  # ← Your GitHub username
BRANCH = 'main'

# Training mode
QUICK_TEST = False  # True for 10-epoch test, False for full 50-epoch training
```

---

### Step 7: Change Runtime to GPU

**CRITICAL STEP** - Don't skip this!

1. **Menu**: Runtime → Change runtime type
2. **Hardware accelerator**: GPU
3. **GPU type**: T4 (free tier)
4. Click **Save**

Verify GPU is available:
```python
import torch
print(torch.cuda.is_available())  # Should print: True
```

---

### Step 8: Run Training

**Execute cells in order** (Shift+Enter to run each cell):

1. ✅ Check GPU availability
2. ✅ Clone GitHub repository
3. ✅ Install dependencies (~2 minutes)
4. ✅ Download & extract data (~3-5 minutes)
5. ✅ Start training (~2-3 hours)

**Training Progress**:
- Watch the progress bar in the output
- Or open TensorBoard for live metrics

---

### Step 9: Monitor Training

**Option A: In-cell Output**
- Logs appear directly in the training cell
- Shows epoch, loss, metrics

**Option B: TensorBoard**
```python
%load_ext tensorboard
%tensorboard --logdir logs/training/
```

**Expected Metrics** (by end of training):
- Train Loss: ~0.15-0.25
- Val Loss: ~0.20-0.30
- RMSE: 0.20-0.30 (Target: 3-5× better than baseline 1.011)
- PCC: 0.30-0.45 (Target: 3-4× better than baseline 0.105)

---

### Step 10: Download Results

**Option A: Direct Download**
```python
# Run the "Download Results" cell
# Downloads zip file to your computer
```

**Option B: Save to Google Drive**
```python
# Run the "Save to Google Drive" cell
# Results saved to: MyDrive/Project_Sullivan_Results/
```

**What's included**:
- Best model checkpoint (`.ckpt`)
- Training logs
- TensorBoard events
- Metrics CSV

---

## 📊 Expected Training Timeline

| Phase | Duration | GPU | Notes |
|-------|----------|-----|-------|
| **Setup** | 5-10 min | - | Clone repo, install deps |
| **Data Download** | 3-5 min | - | Download from Drive |
| **Quick Test (10 epochs)** | 20-30 min | T4 | Validation only |
| **Full Training (50 epochs)** | 2-3 hours | T4 | Production model |

**Colab Free Tier Limits**:
- Maximum session: 12 hours
- May disconnect after 90 minutes idle
- Keep browser tab open during training

---

## ⚠️ Troubleshooting

### GPU Not Available
**Problem**: `torch.cuda.is_available()` returns `False`

**Solution**:
1. Runtime → Change runtime type → GPU
2. Wait ~1 minute for GPU allocation
3. May need to try again later if GPUs are busy

---

### Download Fails
**Problem**: `gdown` fails to download data

**Solution**:
1. Verify file ID is correct
2. Check sharing is "Anyone with link"
3. Try browser download + manual upload to Colab:
   ```python
   from google.colab import files
   uploaded = files.upload()  # Select file from computer
   ```

---

### Out of Memory
**Problem**: CUDA out of memory error

**Solution**:
1. Reduce batch size in config:
   ```yaml
   training:
     batch_size: 8  # Reduce from 16
   ```
2. Use gradient accumulation:
   ```yaml
   training:
     accumulate_grad_batches: 2
   ```

---

### Session Timeout
**Problem**: Colab disconnected during training

**Solution**:
1. Checkpoints are saved every epoch
2. Re-run training - will resume from last checkpoint
3. Or enable Drive mounting to auto-save checkpoints:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   # Update checkpoint path in config
   ```

---

## 🎯 After Training

### Evaluate Results

1. **Check metrics**:
   - Open TensorBoard
   - Look at validation RMSE and PCC
   - Compare to baseline (RMSE: 1.011, PCC: 0.105)

2. **Success criteria**:
   - ✅ RMSE < 0.30 (minimum success)
   - ✅ PCC > 0.30 (minimum success)
   - 🎯 RMSE < 0.20, PCC > 0.40 (target success)

3. **If performance is poor**:
   - Check training curves for overfitting
   - Try longer training (increase epochs)
   - Adjust learning rate or model size

### Next Steps

**If training succeeds**:
1. Download best checkpoint
2. Update project documentation
3. Proceed to Conformer implementation (Phase 2-B next)

**If training fails**:
1. Review training logs
2. Check data loading (verify splits are correct)
3. Try quick test first (10 epochs)
4. Ask for help with error messages

---

## 📁 File Locations in Colab

After running notebook:

```
/content/Project_Sullivan/
├── data/
│   └── processed/           # Extracted data
│       ├── audio_features/
│       ├── parameters/
│       ├── segmentations/
│       └── splits/
├── logs/
│   └── training/            # TensorBoard logs
│       └── transformer_*/
│           ├── checkpoints/
│           │   ├── best.ckpt           # Best model
│           │   └── epoch=*.ckpt        # Epoch checkpoints
│           ├── events.*                # TensorBoard events
│           └── metrics.csv             # Training metrics
└── scripts/
    └── train_transformer.py
```

---

## 💡 Tips & Best Practices

1. **Start with Quick Test**:
   - Set `QUICK_TEST = True` first
   - Verify everything works (10 epochs, ~30 min)
   - Then run full training

2. **Keep Browser Open**:
   - Colab may disconnect if inactive
   - Move mouse occasionally or enable auto-clicker

3. **Monitor Progress**:
   - Check TensorBoard regularly
   - Loss should decrease steadily
   - If loss plateaus early, may need tuning

4. **Save Checkpoints to Drive**:
   - Mount Drive before training
   - Update checkpoint path to Drive folder
   - Prevents loss if session disconnects

5. **Use Multiple Sessions**:
   - Run quick test in one session
   - Run full training in another
   - Compare different hyperparameters

---

## 📞 Getting Help

**Common Issues**:
- Check [Troubleshooting](#-troubleshooting) section above
- Review error messages carefully
- Google Colab has extensive documentation

**Project Issues**:
- Open issue on GitHub repository
- Include error messages and logs
- Attach screenshots if helpful

---

**Last Updated**: 2025-12-01
**Guide Version**: 1.0
**Tested On**: Google Colab (Free Tier, T4 GPU)
