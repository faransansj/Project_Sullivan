#!/bin/bash
# Project Sullivan - HDDB Dataset Processing Pipeline
# 전체 데이터 처리 파이프라인 자동화 스크립트

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT="/home/Project_Sullivan"
DATA_ROOT="/mnt/HDDB/dataset/my_dataset/dataset"
OUTPUT_ROOT="${PROJECT_ROOT}/data/processed_hddb"
VENV_PATH="${PROJECT_ROOT}/venv_sullivan"

# 처리할 피험자 리스트 (단계별 확장)
STAGE1_SUBJECTS="sub010,sub011,sub012,sub013,sub014"  # 5명
STAGE2_SUBJECTS="sub015,sub016,sub017,sub018,sub019,sub030,sub031,sub032,sub033,sub034"  # 추가 10명
STAGE3_SUBJECTS="sub035,sub036,sub037,sub038,sub039"  # 나머지 5명 (선택)

# GPU 사용 여부
USE_GPU=true  # false로 설정하면 CPU 사용

# ============================================================================
# Helper Functions
# ============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

error() {
    echo "[ERROR] $1" >&2
    exit 1
}

check_requirements() {
    log "환경 확인 중..."

    # 가상환경 확인
    if [ ! -d "${VENV_PATH}" ]; then
        error "가상환경을 찾을 수 없습니다: ${VENV_PATH}"
    fi

    # 데이터 경로 확인
    if [ ! -d "${DATA_ROOT}" ]; then
        error "데이터셋을 찾을 수 없습니다: ${DATA_ROOT}"
    fi

    # 디스크 공간 확인 (최소 500GB)
    AVAILABLE_SPACE=$(df -BG "${PROJECT_ROOT}" | tail -1 | awk '{print $4}' | sed 's/G//')
    if [ "${AVAILABLE_SPACE}" -lt 500 ]; then
        log "⚠️  경고: 디스크 공간 부족 (${AVAILABLE_SPACE}GB available, 500GB+ recommended)"
    fi

    log "✅ 환경 확인 완료"
}

activate_venv() {
    log "가상환경 활성화..."
    source "${VENV_PATH}/bin/activate"
}

# ============================================================================
# Phase 0: 데이터 분석
# ============================================================================

phase0_analyze_dataset() {
    log "=================================================="
    log "Phase 0: 데이터셋 분석"
    log "=================================================="

    activate_venv

    log "데이터셋 통계 수집 중..."
    python scripts/collect_dataset_stats.py \
        --data-root "${DATA_ROOT}" \
        --output-file data/hddb_dataset_stats.json

    log "✅ Phase 0 완료"
}

# ============================================================================
# Phase 1: 데이터 전처리
# ============================================================================

phase1_preprocess() {
    local SUBJECTS=$1
    local STAGE=$2

    log "=================================================="
    log "Phase 1.${STAGE}: 데이터 전처리 (${SUBJECTS})"
    log "=================================================="

    activate_venv

    # 1.1 MRI/오디오 정렬
    log "Step 1.1: MRI/오디오 정렬 중..."
    python scripts/batch_preprocess_hddb.py \
        --data-root "${DATA_ROOT}" \
        --subjects "${SUBJECTS}" \
        --output-dir "${OUTPUT_ROOT}/aligned" \
        --config configs/preprocess_hddb.yaml

    # 1.2 세그멘테이션
    log "Step 1.2: U-Net 세그멘테이션 중..."
    if [ "${USE_GPU}" = true ]; then
        GPU_FLAG="--device cuda"
    else
        GPU_FLAG="--device cpu"
    fi

    python scripts/segment_subset.py \
        --data-root "${OUTPUT_ROOT}/aligned" \
        --subjects "${SUBJECTS}" \
        --output-dir "${OUTPUT_ROOT}/segmentations" \
        --checkpoint models/segmentation/unet_best.pth \
        --batch-size 32 \
        ${GPU_FLAG}

    # 1.3 파라미터 추출
    log "Step 1.3: Articulatory 파라미터 추출 중..."
    python scripts/extract_articulatory_params.py \
        --segmentation-dir "${OUTPUT_ROOT}/segmentations" \
        --output-dir "${OUTPUT_ROOT}/parameters" \
        --subjects "${SUBJECTS}" \
        --param-type both

    # 1.4 오디오 특징 추출
    log "Step 1.4: 오디오 특징 추출 중..."
    python scripts/extract_audio_features.py \
        --audio-dir "${OUTPUT_ROOT}/aligned" \
        --output-dir "${OUTPUT_ROOT}/audio_features" \
        --subjects "${SUBJECTS}" \
        --feature-type mel \
        --n-mels 80

    log "✅ Phase 1.${STAGE} 완료"
}

phase1_create_splits() {
    log "=================================================="
    log "Phase 1: 데이터셋 분할 (Train/Val/Test)"
    log "=================================================="

    activate_venv

    python scripts/create_dataset_splits.py \
        --data-root "${OUTPUT_ROOT}" \
        --output-dir "${OUTPUT_ROOT}/splits" \
        --split-ratios 0.7 0.15 0.15 \
        --split-by subject \
        --seed 42

    log "✅ 데이터셋 분할 완료"
}

# ============================================================================
# Phase 2: 모델 훈련
# ============================================================================

phase2_train_baseline() {
    log "=================================================="
    log "Phase 2.1: Baseline LSTM 훈련"
    log "=================================================="

    activate_venv

    # Quick test 먼저 실행
    log "Quick test 실행 중..."
    python scripts/train_baseline.py \
        --config configs/baseline_quick_test.yaml \
        --fast-dev-run

    # 전체 훈련
    log "전체 훈련 시작..."
    if [ "${USE_GPU}" = true ]; then
        GPU_FLAG="--gpus 1"
    else
        GPU_FLAG=""
    fi

    python scripts/train_baseline.py \
        --config configs/baseline_config_hddb.yaml \
        ${GPU_FLAG}

    log "✅ Baseline 훈련 완료"
}

phase2_train_transformer() {
    log "=================================================="
    log "Phase 2.2: Transformer 훈련"
    log "=================================================="

    activate_venv

    if [ "${USE_GPU}" != true ]; then
        error "Transformer 훈련에는 GPU가 필요합니다. USE_GPU=true로 설정하세요."
    fi

    python scripts/train_transformer.py \
        --config configs/transformer_config_hddb.yaml \
        --gpus 1

    log "✅ Transformer 훈련 완료"
}

phase2_evaluate() {
    log "=================================================="
    log "Phase 2.3: 모델 평가"
    log "=================================================="

    activate_venv

    # Baseline 평가
    log "Baseline 모델 평가 중..."
    python scripts/evaluate_model.py \
        --checkpoint models/baseline_lstm_hddb/best.ckpt \
        --config configs/baseline_config_hddb.yaml \
        --split test \
        --output results/baseline_hddb_evaluation.json

    # Transformer 평가
    log "Transformer 모델 평가 중..."
    python scripts/evaluate_model.py \
        --checkpoint models/transformer_hddb/best.ckpt \
        --config configs/transformer_config_hddb.yaml \
        --split test \
        --output results/transformer_hddb_evaluation.json

    log "✅ 모델 평가 완료"
}

# ============================================================================
# Main Pipeline
# ============================================================================

show_usage() {
    cat << EOF
Usage: $0 [command] [options]

Commands:
    all-stage1      - 전체 파이프라인 실행 (Stage 1: 5명)
    all-stage2      - 전체 파이프라인 실행 (Stage 2: 15명 추가)
    all-stage3      - 전체 파이프라인 실행 (Stage 3: 나머지 5명)

    phase0          - Phase 0: 데이터셋 분석
    phase1-s1       - Phase 1: 전처리 (Stage 1: 5명)
    phase1-s2       - Phase 1: 전처리 (Stage 2: 추가 10명)
    phase1-s3       - Phase 1: 전처리 (Stage 3: 나머지 5명)
    phase1-split    - Phase 1: 데이터셋 분할
    phase2-baseline - Phase 2: Baseline LSTM 훈련
    phase2-transformer - Phase 2: Transformer 훈련
    phase2-eval     - Phase 2: 모델 평가

    help            - 이 도움말 표시

Options:
    --gpu           - GPU 사용 (기본값: true)
    --cpu           - CPU 사용

Examples:
    # 전체 파이프라인 (Stage 1: 5명)
    $0 all-stage1

    # Phase 1만 실행
    $0 phase1-s1
    $0 phase1-split

    # Phase 2만 실행
    $0 phase2-transformer --gpu

    # CPU로 baseline 훈련
    $0 phase2-baseline --cpu
EOF
}

main() {
    local COMMAND=${1:-help}

    # Parse options
    shift
    while [[ $# -gt 0 ]]; do
        case $1 in
            --gpu)
                USE_GPU=true
                shift
                ;;
            --cpu)
                USE_GPU=false
                shift
                ;;
            *)
                log "알 수 없는 옵션: $1"
                shift
                ;;
        esac
    done

    # Check requirements
    check_requirements

    # Execute command
    case $COMMAND in
        all-stage1)
            log "🚀 전체 파이프라인 시작 (Stage 1: 5명)"
            phase0_analyze_dataset
            phase1_preprocess "${STAGE1_SUBJECTS}" 1
            phase1_create_splits
            phase2_train_baseline
            phase2_train_transformer
            phase2_evaluate
            log "🎉 전체 파이프라인 완료!"
            ;;

        all-stage2)
            log "🚀 전체 파이프라인 시작 (Stage 2: 추가 10명)"
            phase1_preprocess "${STAGE2_SUBJECTS}" 2
            phase1_create_splits
            phase2_train_transformer
            phase2_evaluate
            log "🎉 Stage 2 완료!"
            ;;

        all-stage3)
            log "🚀 전체 파이프라인 시작 (Stage 3: 나머지 5명)"
            phase1_preprocess "${STAGE3_SUBJECTS}" 3
            phase1_create_splits
            phase2_train_transformer
            phase2_evaluate
            log "🎉 Stage 3 완료!"
            ;;

        phase0)
            phase0_analyze_dataset
            ;;

        phase1-s1)
            phase1_preprocess "${STAGE1_SUBJECTS}" 1
            ;;

        phase1-s2)
            phase1_preprocess "${STAGE2_SUBJECTS}" 2
            ;;

        phase1-s3)
            phase1_preprocess "${STAGE3_SUBJECTS}" 3
            ;;

        phase1-split)
            phase1_create_splits
            ;;

        phase2-baseline)
            phase2_train_baseline
            ;;

        phase2-transformer)
            phase2_train_transformer
            ;;

        phase2-eval)
            phase2_evaluate
            ;;

        help|--help|-h)
            show_usage
            ;;

        *)
            log "알 수 없는 명령어: $COMMAND"
            show_usage
            exit 1
            ;;
    esac
}

# Run main
main "$@"
