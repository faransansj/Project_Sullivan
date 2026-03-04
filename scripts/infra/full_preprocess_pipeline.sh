#!/bin/bash
# ============================================
# Project Sullivan: Full Preprocessing Pipeline
# ============================================
# NAS에서 600GB+ raw 데이터를 전처리하여 학습용 feature로 변환
#
# Usage:
#   bash scripts/infra/full_preprocess_pipeline.sh [OPTIONS]
#
# Options:
#   --config PATH        Config file (default: configs/preprocess_nas.yaml)
#   --subjects SUB...    특정 피험자만 처리
#   --all                전체 피험자 처리
#   --skip-to STEP       이미 완료된 단계 건너뛰기 (1-5)
#   --dry-run            실행 없이 계획만 출력
#   --features TYPE      오디오 피처 타입: mel|mfcc|both (default: mel)
#   --params METHOD      파라미터 추출: geometric|pca|both (default: both)
#
# Example:
#   # 전체 파이프라인 (전체 피험자)
#   bash scripts/infra/full_preprocess_pipeline.sh --all
#
#   # 특정 피험자만, Step 3부터 재개
#   bash scripts/infra/full_preprocess_pipeline.sh --subjects sub011 sub012 --skip-to 3
#
#   # Dry-run (미리보기)
#   bash scripts/infra/full_preprocess_pipeline.sh --all --dry-run
# ============================================

set -euo pipefail

# ── Defaults ──
CONFIG="configs/preprocess_nas.yaml"
SUBJECT_ARGS=""
SKIP_TO=1
DRY_RUN=false
FEATURES="mel"
PARAMS="both"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/preprocessing"
LOG_FILE="${LOG_DIR}/full_pipeline_${TIMESTAMP}.log"

# ── Colors for output ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ── Parse Arguments ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"; shift 2 ;;
        --subjects)
            shift
            SUBJECTS=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SUBJECTS+=("$1"); shift
            done
            SUBJECT_ARGS="--subjects ${SUBJECTS[*]}"
            ;;
        --all)
            SUBJECT_ARGS="--all"; shift ;;
        --skip-to)
            SKIP_TO="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        --features)
            FEATURES="$2"; shift 2 ;;
        --params)
            PARAMS="$2"; shift 2 ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"; exit 1 ;;
    esac
done

# ── Validate ──
if [ -z "${SUBJECT_ARGS}" ]; then
    echo -e "${RED}ERROR: Must specify --subjects or --all${NC}"
    echo "Usage: bash scripts/infra/full_preprocess_pipeline.sh --all"
    echo "       bash scripts/infra/full_preprocess_pipeline.sh --subjects sub011 sub012"
    exit 1
fi

if [ ! -f "${CONFIG}" ]; then
    echo -e "${RED}ERROR: Config not found: ${CONFIG}${NC}"
    exit 1
fi

# ── Setup logging ──
mkdir -p "${LOG_DIR}"

log() {
    echo -e "$1" | tee -a "${LOG_FILE}"
}

step_header() {
    local step=$1
    local title=$2
    log ""
    log "${BLUE}════════════════════════════════════════════════════════${NC}"
    log "${BLUE}  Step ${step}: ${title}${NC}"
    log "${BLUE}════════════════════════════════════════════════════════${NC}"
}

step_skip() {
    local step=$1
    local title=$2
    log "${YELLOW}  ⏭️  Step ${step}: ${title} — SKIPPED (skip-to=${SKIP_TO})${NC}"
}

step_done() {
    local step=$1
    log "${GREEN}  ✅ Step ${step} complete${NC}"
}

# ── Header ──
log ""
log "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
log "${GREEN}║  Project Sullivan — Full Preprocessing Pipeline        ║${NC}"
log "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
log ""
log "  Config:     ${CONFIG}"
log "  Subjects:   ${SUBJECT_ARGS}"
log "  Features:   ${FEATURES}"
log "  Params:     ${PARAMS}"
log "  Skip to:    Step ${SKIP_TO}"
log "  Dry run:    ${DRY_RUN}"
log "  Log file:   ${LOG_FILE}"
log "  Started:    $(date)"
log ""

if [ "${DRY_RUN}" = true ]; then
    log "${YELLOW}══ DRY RUN MODE — Commands will be printed but not executed ══${NC}"
    log ""
fi

# ── Step 1: Batch Preprocess (Raw → Aligned HDF5) ──
if [ "${SKIP_TO}" -le 1 ]; then
    step_header 1 "Batch Preprocess (Raw → Aligned HDF5)"
    CMD="uv run python scripts/batch_preprocess.py --config ${CONFIG} ${SUBJECT_ARGS}"
    log "  Command: ${CMD}"

    if [ "${DRY_RUN}" = false ]; then
        eval "${CMD}" 2>&1 | tee -a "${LOG_FILE}"
    fi
    step_done 1
else
    step_skip 1 "Batch Preprocess"
fi

# ── Step 2: Segmentation (HDF5 → Masks) ──
if [ "${SKIP_TO}" -le 2 ]; then
    step_header 2 "Segmentation (Aligned HDF5 → Masks)"
    CMD="uv run python scripts/segment_subset.py"
    log "  Command: ${CMD}"
    log "  Note: Uses CPU on NAS (segmentation_device: cpu in config)"

    if [ "${DRY_RUN}" = false ]; then
        eval "${CMD}" 2>&1 | tee -a "${LOG_FILE}"
    fi
    step_done 2
else
    step_skip 2 "Segmentation"
fi

# ── Step 3: Extract Audio Features (HDF5 → Mel/MFCC npy) ──
if [ "${SKIP_TO}" -le 3 ]; then
    step_header 3 "Extract Audio Features (→ ${FEATURES})"
    CMD="uv run python scripts/extract_audio_features.py --features ${FEATURES}"
    log "  Command: ${CMD}"

    if [ "${DRY_RUN}" = false ]; then
        eval "${CMD}" 2>&1 | tee -a "${LOG_FILE}"
    fi
    step_done 3
else
    step_skip 3 "Audio Feature Extraction"
fi

# ── Step 4: Extract Articulatory Parameters (Masks → Geometric+PCA npy) ──
if [ "${SKIP_TO}" -le 4 ]; then
    step_header 4 "Extract Articulatory Parameters (→ ${PARAMS})"
    CMD="uv run python scripts/extract_articulatory_params.py --method ${PARAMS}"
    log "  Command: ${CMD}"

    if [ "${DRY_RUN}" = false ]; then
        eval "${CMD}" 2>&1 | tee -a "${LOG_FILE}"
    fi
    step_done 4
else
    step_skip 4 "Parameter Extraction"
fi

# ── Step 5: Package Results ──
if [ "${SKIP_TO}" -le 5 ]; then
    step_header 5 "Package & Verify Results"

    PROCESSED_DIR="data/processed"
    ARCHIVE_DIR="data/transfer_archives"
    mkdir -p "${ARCHIVE_DIR}"

    if [ "${DRY_RUN}" = false ]; then
        # Report sizes
        log ""
        log "  📊 Processed Data Summary:"
        log "  ─────────────────────────────────────────"

        for subdir in aligned segmentations audio_features parameters splits; do
            if [ -d "${PROCESSED_DIR}/${subdir}" ]; then
                SIZE=$(du -sh "${PROCESSED_DIR}/${subdir}" 2>/dev/null | cut -f1)
                COUNT=$(find "${PROCESSED_DIR}/${subdir}" -type f 2>/dev/null | wc -l | tr -d ' ')
                log "  ${subdir}/: ${SIZE} (${COUNT} files)"
            else
                log "  ${subdir}/: ${YELLOW}NOT FOUND${NC}"
            fi
        done

        TOTAL_SIZE=$(du -sh "${PROCESSED_DIR}" 2>/dev/null | cut -f1)
        log "  ─────────────────────────────────────────"
        log "  Total: ${TOTAL_SIZE}"
        log ""

        # Create tar archives for transfer
        log "  📦 Creating transfer archives..."

        for subdir in audio_features parameters splits; do
            if [ -d "${PROCESSED_DIR}/${subdir}" ]; then
                ARCHIVE="${ARCHIVE_DIR}/${subdir}.tar.gz"
                log "  Compressing ${subdir} → ${ARCHIVE}"
                tar -czf "${ARCHIVE}" -C "${PROCESSED_DIR}" "${subdir}/"
                ARCHIVE_SIZE=$(du -sh "${ARCHIVE}" | cut -f1)
                log "    → ${ARCHIVE_SIZE}"
            fi
        done

        # Total archive size
        ARCHIVE_TOTAL=$(du -sh "${ARCHIVE_DIR}" 2>/dev/null | cut -f1)
        log ""
        log "  📦 Transfer archive total: ${ARCHIVE_TOTAL}"
    else
        log "  Would create tar.gz archives in ${ARCHIVE_DIR}/"
        log "  Contents: audio_features/ parameters/ splits/"
    fi

    step_done 5
fi

# ── Summary ──
log ""
log "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
log "${GREEN}║  ✅ Pipeline Complete!                                  ║${NC}"
log "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
log ""
log "  Log:      ${LOG_FILE}"
log "  Archives: data/transfer_archives/"
log "  Finished: $(date)"
log ""
log "  🚀 Next step: Transfer to A100 server"
log "     bash scripts/infra/transfer_to_gpu.sh --server user@snu-server"
log ""
