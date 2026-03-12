#!/usr/bin/env bash
# =============================================================================
# Full pipeline for USC-TIMIT 2D RT video dataset
# =============================================================================
#
# Usage:
#   bash scripts/run_pipeline_mp4.sh                 # full run (all subjects)
#   bash scripts/run_pipeline_mp4.sh sub001,sub002   # specific subjects only
#
# Steps:
#   1. extract_frames_from_mp4  → data/processed/aligned/
#   2. segment_mp4_dataset      → data/processed/segmentations/
#   3. extract_articulatory_params  → data/processed/parameters/
#   4. extract_audio_features       → data/processed/audio_features/
#   5. create_dataset_splits        → data/processed/splits/
# =============================================================================

set -euo pipefail

SUBJECTS="${1:-}"          # optional: "sub001,sub002"
WORKERS="${2:-4}"          # parallel workers for frame extraction
DEVICE="${3:-cpu}"         # cpu or cuda for segmentation
MODEL="models/unet_scratch/unet_best.pth"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Use uv run if available, otherwise fall back to venv or system python
if command -v uv &>/dev/null; then
    PYTHON="uv run python"
elif [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif [ -f "venv_sullivan/bin/python" ]; then
    source venv_sullivan/bin/activate
    PYTHON="python"
else
    PYTHON="python"
fi

echo "============================================================"
echo " Project Sullivan — MP4 Pipeline"
echo " Subjects : ${SUBJECTS:-all}"
echo " Workers  : $WORKERS"
echo " Device   : $DEVICE"
echo " Root     : $PROJECT_ROOT"
echo "============================================================"

mkdir -p logs

# Build optional subject flag
SUBJ_FLAG=""
if [ -n "$SUBJECTS" ]; then
    SUBJ_FLAG="--subjects $SUBJECTS"
fi

# ------------------------------------------------------------
# Step 1: Extract frames + audio from MP4 → HDF5
# ------------------------------------------------------------
echo ""
echo "[1/5] Extracting frames and audio from MP4..."
$PYTHON scripts/extract_frames_from_mp4.py \
    --data-root dl_data/dataset_2drt_video_only \
    --output-dir data/processed/aligned \
    --workers "$WORKERS" \
    --skip-existing \
    $SUBJ_FLAG

# ------------------------------------------------------------
# Step 2: Segment vocal tract (U-Net) → NPZ
# ------------------------------------------------------------
echo ""
echo "[2/5] Segmenting vocal tract..."
$PYTHON scripts/segment_mp4_dataset.py \
    --aligned-dir data/processed/aligned \
    --output-dir data/processed/segmentations \
    --model "$MODEL" \
    --device "$DEVICE" \
    --skip-existing \
    $SUBJ_FLAG

# ------------------------------------------------------------
# Step 3: Extract articulatory parameters → NPY
# ------------------------------------------------------------
echo ""
echo "[3/5] Extracting articulatory parameters..."
$PYTHON scripts/extract_articulatory_params.py \
    --segmentation-dir data/processed/segmentations \
    --output-dir data/processed/parameters \
    --method geometric

# ------------------------------------------------------------
# Step 4: Extract audio features → NPY
# ------------------------------------------------------------
echo ""
echo "[4/5] Extracting audio features..."
$PYTHON scripts/extract_audio_features.py \
    --aligned-dir data/processed/aligned \
    --segmentation-dir data/processed/segmentations \
    --output-dir data/processed/audio_features \
    --features mel

# ------------------------------------------------------------
# Step 5: Create train/val/test splits → JSON
# ------------------------------------------------------------
echo ""
echo "[5/5] Creating dataset splits..."
$PYTHON scripts/create_dataset_splits.py \
    --segmentation-dir data/processed/segmentations \
    --parameter-dir data/processed/parameters \
    --audio-feature-dir data/processed/audio_features \
    --output-dir data/processed/splits \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15

echo ""
echo "============================================================"
echo " Pipeline complete!"
echo " Processed data: data/processed/"
echo " Splits:         data/processed/splits/"
echo "============================================================"
