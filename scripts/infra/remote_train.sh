#!/bin/bash
# ============================================
# Project Sullivan: Remote Training Launcher
# ============================================
# Phase 5-1: 로컬 → GPU 서버 코드 동기화 후 학습 실행
#
# Usage:
#   ./scripts/infra/remote_train.sh <server> [config] [script] [extra_args...]
#
# Examples:
#   # Transformer 학습
#   ./scripts/infra/remote_train.sh user@a100-server
#
#   # Conformer A100 학습
#   ./scripts/infra/remote_train.sh user@a100-server configs/conformer_a100_config.yaml train_conformer.py
#
#   # Conformer + auto-resume
#   ./scripts/infra/remote_train.sh user@a100-server configs/conformer_a100_config.yaml train_conformer.py --auto-resume
# ============================================

set -euo pipefail

# -----------------------------------------------
# Arguments
# -----------------------------------------------
SERVER="${1:?Usage: $0 <user@server> [config] [script] [extra_args...]}"
CONFIG="${2:-configs/conformer_a100_config.yaml}"
SCRIPT="${3:-train_conformer.py}"
shift 3 2>/dev/null || true
EXTRA_ARGS="$*"

REMOTE_DIR="~/Project_Sullivan"
EXPERIMENT_ID="$(date +%Y%m%d_%H%M%S)"
LOG_NAME="train_${EXPERIMENT_ID}.log"

echo "============================================"
echo "  Remote Training Launcher"
echo "============================================"
echo "  Server:     ${SERVER}"
echo "  Config:     ${CONFIG}"
echo "  Script:     scripts/${SCRIPT}"
echo "  Experiment: ${EXPERIMENT_ID}"
echo "  Extra args: ${EXTRA_ARGS:-none}"
echo ""

# -----------------------------------------------
# Step 1: Sync code to remote (exclude heavy dirs)
# -----------------------------------------------
echo "=== [1/3] Syncing code to ${SERVER} ==="
rsync -avz --progress \
    --exclude '.venv' \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'data/' \
    --exclude 'models/' \
    --exclude 'logs/' \
    --exclude '.mypy_cache' \
    --exclude '.pytest_cache' \
    --exclude 'node_modules' \
    ./ "${SERVER}:${REMOTE_DIR}/"

echo "  ✅ Code synced."
echo ""

# -----------------------------------------------
# Step 2: Start training in background via nohup
# -----------------------------------------------
echo "=== [2/3] Starting training on ${SERVER} ==="
ssh "${SERVER}" "
    cd ${REMOTE_DIR} && \
    mkdir -p logs && \
    export PATH=\"\${HOME}/.local/bin:\${PATH}\" && \
    echo 'Starting: uv run python scripts/${SCRIPT} --config ${CONFIG} --gpus 1 ${EXTRA_ARGS}' && \
    nohup uv run python scripts/${SCRIPT} \
        --config ${CONFIG} \
        --gpus 1 \
        ${EXTRA_ARGS} \
        > logs/${LOG_NAME} 2>&1 &
    echo \"PID: \$!\"
"

echo "  ✅ Training launched in background."
echo ""

# -----------------------------------------------
# Step 3: Print monitoring instructions
# -----------------------------------------------
echo "=== [3/3] Monitoring ==="
echo ""
echo "  📋 View training log:"
echo "     ssh ${SERVER} 'tail -f ${REMOTE_DIR}/logs/${LOG_NAME}'"
echo ""
echo "  📊 TensorBoard (run from local):"
echo "     ssh -L 6006:localhost:6006 ${SERVER} 'cd ${REMOTE_DIR} && tensorboard --logdir logs/training'"
echo "     → Then open http://localhost:6006"
echo ""
echo "  🔍 GPU status:"
echo "     ./scripts/infra/check_gpu_status.sh ${SERVER}"
echo ""
echo "  🛑 Stop training:"
echo "     ssh ${SERVER} 'pkill -f ${SCRIPT}'"
echo ""
echo "  📥 Download results (after training):"
echo "     rsync -avz ${SERVER}:${REMOTE_DIR}/models/ ./models/"
echo "     rsync -avz ${SERVER}:${REMOTE_DIR}/logs/ ./logs/"
echo ""
