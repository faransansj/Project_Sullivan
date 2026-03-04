#!/bin/bash
# ============================================
# Project Sullivan: Transfer Preprocessed Data to GPU Server
# ============================================
# NAS에서 전처리된 feature를 A100/A6000 GPU 서버로 전송
#
# Usage:
#   bash scripts/infra/transfer_to_gpu.sh --server user@snu-server
#   bash scripts/infra/transfer_to_gpu.sh --server user@snu-server --method scp
#   bash scripts/infra/transfer_to_gpu.sh --server user@snu-server --verify
#   bash scripts/infra/transfer_to_gpu.sh --server user@snu-server --archives-only
#
# Options:
#   --server HOST        SSH 목적지 (필수, 예: user@123.45.67.89)
#   --method METHOD      전송 방식: rsync|scp (default: rsync)
#   --remote-dir PATH    원격 프로젝트 경로 (default: ~/Project_Sullivan)
#   --verify             전송 후 체크섬 검증
#   --archives-only      tar.gz 아카이브만 전송 (더 빠름)
#   --dry-run            전송 없이 계획만 출력
# ============================================

set -euo pipefail

# ── Defaults ──
SERVER=""
METHOD="rsync"
REMOTE_DIR="~/Project_Sullivan"
VERIFY=false
ARCHIVES_ONLY=false
DRY_RUN=false
LOCAL_PROCESSED="data/processed"
LOCAL_ARCHIVES="data/transfer_archives"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/transfer_${TIMESTAMP}.log"

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ── Parse Arguments ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --server)
            SERVER="$2"; shift 2 ;;
        --method)
            METHOD="$2"; shift 2 ;;
        --remote-dir)
            REMOTE_DIR="$2"; shift 2 ;;
        --verify)
            VERIFY=true; shift ;;
        --archives-only)
            ARCHIVES_ONLY=true; shift ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"; exit 1 ;;
    esac
done

# ── Validate ──
if [ -z "${SERVER}" ]; then
    echo -e "${RED}ERROR: Must specify --server${NC}"
    echo "Usage: bash scripts/infra/transfer_to_gpu.sh --server user@gpu-server"
    exit 1
fi

mkdir -p "$(dirname "${LOG_FILE}")"

log() {
    echo -e "$1" | tee -a "${LOG_FILE}"
}

# ── Header ──
log ""
log "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
log "${GREEN}║  Project Sullivan — Transfer to GPU Server              ║${NC}"
log "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
log ""
log "  Server:       ${SERVER}"
log "  Method:       ${METHOD}"
log "  Remote dir:   ${REMOTE_DIR}"
log "  Archives:     ${ARCHIVES_ONLY}"
log "  Verify:       ${VERIFY}"
log "  Dry run:      ${DRY_RUN}"
log "  Started:      $(date)"
log ""

# ── Step 1: Pre-flight Check ──
log "${BLUE}[1/4] Pre-flight Check${NC}"

# Check local data
if [ "${ARCHIVES_ONLY}" = true ]; then
    if [ ! -d "${LOCAL_ARCHIVES}" ]; then
        log "${RED}ERROR: Archive directory not found: ${LOCAL_ARCHIVES}${NC}"
        log "Run full_preprocess_pipeline.sh first to create archives."
        exit 1
    fi
    LOCAL_SIZE=$(du -sh "${LOCAL_ARCHIVES}" | cut -f1)
    LOCAL_FILES=$(find "${LOCAL_ARCHIVES}" -type f | wc -l | tr -d ' ')
    log "  Local archives: ${LOCAL_SIZE} (${LOCAL_FILES} files)"
else
    if [ ! -d "${LOCAL_PROCESSED}" ]; then
        log "${RED}ERROR: Processed data not found: ${LOCAL_PROCESSED}${NC}"
        exit 1
    fi
    LOCAL_SIZE=$(du -sh "${LOCAL_PROCESSED}" | cut -f1)
    log "  Local processed: ${LOCAL_SIZE}"
fi

# Check SSH connectivity
log "  Testing SSH connection..."
if [ "${DRY_RUN}" = false ]; then
    if ssh -o ConnectTimeout=10 -o BatchMode=yes "${SERVER}" "echo 'SSH OK'" 2>/dev/null; then
        log "  ${GREEN}SSH: Connected ✅${NC}"
    else
        log "  ${RED}SSH: Connection failed ❌${NC}"
        log "  Check your SSH config and credentials."
        exit 1
    fi

    # Check remote disk space
    log "  Checking remote disk space..."
    REMOTE_FREE=$(ssh "${SERVER}" "df -h ${REMOTE_DIR} 2>/dev/null | tail -1 | awk '{print \$4}'" 2>/dev/null || echo "unknown")
    log "  Remote free space: ${REMOTE_FREE}"
else
    log "  ${YELLOW}SSH check skipped (dry-run)${NC}"
fi

# ── Step 2: Prepare Remote Directory ──
log ""
log "${BLUE}[2/4] Prepare Remote Directory${NC}"

if [ "${DRY_RUN}" = false ]; then
    ssh "${SERVER}" "mkdir -p ${REMOTE_DIR}/data/processed ${REMOTE_DIR}/data/transfer_archives"
    log "  Created remote directories ✅"
else
    log "  Would create: ${REMOTE_DIR}/data/processed/"
fi

# ── Step 3: Transfer Data ──
log ""
log "${BLUE}[3/4] Transfer Data (${METHOD})${NC}"

transfer_rsync() {
    local src=$1
    local dst=$2
    log "  rsync: ${src} → ${SERVER}:${dst}"
    if [ "${DRY_RUN}" = false ]; then
        rsync -avz --progress --partial --timeout=300 \
            "${src}" "${SERVER}:${dst}" 2>&1 | tail -5 | tee -a "${LOG_FILE}"
    fi
}

transfer_scp() {
    local src=$1
    local dst=$2
    log "  scp: ${src} → ${SERVER}:${dst}"
    if [ "${DRY_RUN}" = false ]; then
        scp -r "${src}" "${SERVER}:${dst}" 2>&1 | tee -a "${LOG_FILE}"
    fi
}

TRANSFER_START=$(date +%s)

if [ "${ARCHIVES_ONLY}" = true ]; then
    # Transfer compressed archives
    REMOTE_ARCHIVE_DIR="${REMOTE_DIR}/data/transfer_archives"

    for archive in "${LOCAL_ARCHIVES}"/*.tar.gz; do
        if [ -f "${archive}" ]; then
            BASENAME=$(basename "${archive}")
            SIZE=$(du -sh "${archive}" | cut -f1)
            log "  📦 ${BASENAME} (${SIZE})"

            if [ "${METHOD}" = "rsync" ]; then
                transfer_rsync "${archive}" "${REMOTE_ARCHIVE_DIR}/"
            else
                transfer_scp "${archive}" "${REMOTE_ARCHIVE_DIR}/"
            fi
        fi
    done

    # Decompress on remote
    if [ "${DRY_RUN}" = false ]; then
        log ""
        log "  📂 Decompressing archives on remote server..."
        ssh "${SERVER}" "
            cd ${REMOTE_DIR}/data/transfer_archives
            for f in *.tar.gz; do
                echo \"  Extracting \${f}...\"
                tar -xzf \"\${f}\" -C ${REMOTE_DIR}/data/processed/
            done
            echo '  Done.'
        " 2>&1 | tee -a "${LOG_FILE}"
    fi
else
    # Transfer processed directories directly
    for subdir in audio_features parameters splits; do
        SRC="${LOCAL_PROCESSED}/${subdir}/"
        DST="${REMOTE_DIR}/data/processed/"

        if [ -d "${LOCAL_PROCESSED}/${subdir}" ]; then
            SIZE=$(du -sh "${LOCAL_PROCESSED}/${subdir}" | cut -f1)
            log "  📂 ${subdir}/ (${SIZE})"

            if [ "${METHOD}" = "rsync" ]; then
                transfer_rsync "${SRC}" "${DST}"
            else
                transfer_scp "${LOCAL_PROCESSED}/${subdir}" "${DST}"
            fi
        else
            log "  ${YELLOW}⚠️  ${subdir}/ not found, skipping${NC}"
        fi
    done
fi

TRANSFER_END=$(date +%s)
TRANSFER_DURATION=$((TRANSFER_END - TRANSFER_START))
log ""
log "  Transfer time: ${TRANSFER_DURATION} seconds"

# ── Step 4: Verify ──
log ""
log "${BLUE}[4/4] Verify Transfer${NC}"

if [ "${VERIFY}" = true ] && [ "${DRY_RUN}" = false ]; then
    log "  Comparing local and remote file counts..."

    for subdir in audio_features parameters splits; do
        LOCAL_COUNT=$(find "${LOCAL_PROCESSED}/${subdir}" -type f 2>/dev/null | wc -l | tr -d ' ')
        REMOTE_COUNT=$(ssh "${SERVER}" "find ${REMOTE_DIR}/data/processed/${subdir} -type f 2>/dev/null | wc -l" | tr -d ' ')

        if [ "${LOCAL_COUNT}" = "${REMOTE_COUNT}" ]; then
            log "  ${GREEN}${subdir}/: ${LOCAL_COUNT} files ✅${NC}"
        else
            log "  ${RED}${subdir}/: LOCAL=${LOCAL_COUNT} REMOTE=${REMOTE_COUNT} ❌${NC}"
        fi
    done

    # Check remote sizes
    log ""
    log "  Remote data sizes:"
    ssh "${SERVER}" "
        cd ${REMOTE_DIR}/data/processed
        for d in audio_features parameters splits; do
            if [ -d \"\${d}\" ]; then
                echo \"    \${d}/: \$(du -sh \"\${d}\" | cut -f1)\"
            fi
        done
    " 2>&1 | tee -a "${LOG_FILE}"
else
    if [ "${VERIFY}" = true ]; then
        log "  ${YELLOW}Verification skipped (dry-run)${NC}"
    else
        log "  Verification skipped (use --verify to enable)"
    fi
fi

# ── Summary ──
log ""
log "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
log "${GREEN}║  ✅ Transfer Complete!                                  ║${NC}"
log "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
log ""
log "  Server:    ${SERVER}"
log "  Remote:    ${REMOTE_DIR}/data/processed/"
log "  Duration:  ${TRANSFER_DURATION} seconds"
log "  Log:       ${LOG_FILE}"
log ""
log "  🚀 Next steps on GPU server:"
log "     ssh ${SERVER}"
log "     cd ${REMOTE_DIR}"
log "     uv run python scripts/train_conformer.py --config configs/conformer_a100_config.yaml --gpus 1"
log ""
