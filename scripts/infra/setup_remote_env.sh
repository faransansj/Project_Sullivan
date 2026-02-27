#!/bin/bash
# ============================================
# Project Sullivan: Remote GPU Server Setup
# ============================================
# Phase 5-1: UV 기반 원격 GPU 환경 초기화
#
# Usage (from local machine):
#   ssh user@gpu-server 'bash -s' < scripts/infra/setup_remote_env.sh
#
# Or copy and run:
#   scp scripts/infra/setup_remote_env.sh user@gpu-server:~/
#   ssh user@gpu-server 'bash setup_remote_env.sh'
# ============================================

set -euo pipefail

REPO_URL="https://github.com/faransansj/Project_Sullivan.git"
PROJECT_DIR="${HOME}/Project_Sullivan"
LOG_FILE="${HOME}/sullivan_setup_$(date +%Y%m%d_%H%M%S).log"

# Tee output to both console and log
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "============================================"
echo "  Project Sullivan — GPU Server Setup"
echo "  $(date)"
echo "============================================"

# -----------------------------------------------
# [1/6] System Check
# -----------------------------------------------
echo ""
echo "=== [1/6] System Check ==="
echo "  OS: $(uname -s) $(uname -r)"
echo "  CPU: $(nproc) cores"
echo "  RAM: $(free -h 2>/dev/null | awk '/^Mem:/{print $2}' || echo 'N/A')"

if command -v nvidia-smi &>/dev/null; then
    echo ""
    echo "  GPU detected:"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
else
    echo "  ⚠️  WARNING: nvidia-smi not found. GPU may not be available."
    echo "  Continuing with CPU-only setup..."
fi

# -----------------------------------------------
# [2/6] Install UV
# -----------------------------------------------
echo ""
echo "=== [2/6] Install UV Package Manager ==="
if command -v uv &>/dev/null; then
    echo "  UV already installed: $(uv --version)"
else
    echo "  Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
    echo "  UV installed: $(uv --version)"
fi

# Ensure UV is in PATH for subsequent commands
export PATH="${HOME}/.local/bin:${PATH}"

# -----------------------------------------------
# [3/6] Clone / Update Repository
# -----------------------------------------------
echo ""
echo "=== [3/6] Clone / Update Repository ==="
if [ -d "${PROJECT_DIR}" ]; then
    echo "  Repository exists. Pulling latest changes..."
    cd "${PROJECT_DIR}"
    git pull origin main
else
    echo "  Cloning repository..."
    git clone "${REPO_URL}" "${PROJECT_DIR}"
    cd "${PROJECT_DIR}"
fi

echo "  Current branch: $(git branch --show-current)"
echo "  Latest commit: $(git log -1 --format='%h %s')"

# -----------------------------------------------
# [4/6] UV Sync (GPU extras)
# -----------------------------------------------
echo ""
echo "=== [4/6] UV Sync with GPU Extras ==="
uv sync --extra gpu
echo "  Dependencies synchronized."

# -----------------------------------------------
# [5/6] Verify PyTorch + CUDA
# -----------------------------------------------
echo ""
echo "=== [5/6] Verify PyTorch & CUDA ==="
uv run python3 -c "
import torch
import sys

print(f'  Python:    {sys.version.split()[0]}')
print(f'  PyTorch:   {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'  CUDA version:   {torch.version.cuda}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name}')
        print(f'    VRAM: {props.total_mem / 1e9:.1f} GB')
        print(f'    Compute capability: {props.major}.{props.minor}')
else:
    print('  ⚠️  CUDA not available — CPU-only mode.')

# Quick tensor test
x = torch.randn(2, 2)
if torch.cuda.is_available():
    x = x.cuda()
    print(f'  GPU tensor test: PASSED ✅')
else:
    print(f'  CPU tensor test: PASSED ✅')
"

# -----------------------------------------------
# [6/6] Create Working Directories
# -----------------------------------------------
echo ""
echo "=== [6/6] Create Working Directories ==="
mkdir -p "${PROJECT_DIR}/data/processed"
mkdir -p "${PROJECT_DIR}/models"
mkdir -p "${PROJECT_DIR}/logs/training"
echo "  data/processed/  — ready"
echo "  models/          — ready"
echo "  logs/training/   — ready"

# -----------------------------------------------
# Summary
# -----------------------------------------------
echo ""
echo "============================================"
echo "  ✅ Setup Complete!"
echo "============================================"
echo "  Project dir:  ${PROJECT_DIR}"
echo "  Setup log:    ${LOG_FILE}"
echo ""
echo "  Next steps:"
echo "    1. Transfer data:"
echo "       rsync -avz nas:/path/to/data/ ${PROJECT_DIR}/data/processed/"
echo ""
echo "    2. Start training:"
echo "       cd ${PROJECT_DIR}"
echo "       uv run python scripts/train_conformer.py --config configs/conformer_a100_config.yaml --gpus 1"
echo ""
echo "    3. Monitor with TensorBoard (from local machine):"
echo "       ssh -L 6006:localhost:6006 $(whoami)@$(hostname) 'cd ${PROJECT_DIR} && tensorboard --logdir logs/training'"
echo ""
