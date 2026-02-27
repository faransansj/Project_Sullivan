#!/bin/bash
# ============================================
# Project Sullivan: GPU Server Health Check
# ============================================
# Phase 5-1: 원격 GPU 서버 상태 확인
#
# Usage:
#   ./scripts/infra/check_gpu_status.sh [server]
#
# Examples:
#   ./scripts/infra/check_gpu_status.sh user@a100-server
#   ./scripts/infra/check_gpu_status.sh sullivan-gpu    # SSH config alias
# ============================================

set -euo pipefail

SERVER="${1:?Usage: $0 <user@server>}"

echo "============================================"
echo "  GPU Server Status: ${SERVER}"
echo "  $(date)"
echo "============================================"

ssh "${SERVER}" '
echo ""
echo "=== 🖥️  GPU Status ==="
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
        --format=csv,noheader,nounits | \
    while IFS="," read -r idx name mem_used mem_total util temp; do
        echo "  GPU ${idx}: ${name}"
        echo "    VRAM: ${mem_used}MB / ${mem_total}MB ($(echo "scale=0; ${mem_used}*100/${mem_total}" | bc)% used)"
        echo "    Utilization: ${util}%"
        echo "    Temperature: ${temp}°C"
    done
else
    echo "  ⚠️  nvidia-smi not found"
fi

echo ""
echo "=== 💾 Disk Usage ==="
if [ -d ~/Project_Sullivan ]; then
    df -h ~/Project_Sullivan/ 2>/dev/null | tail -1 | awk "{printf \"  Disk: %s / %s (%s used)\n\", \$3, \$2, \$5}"
    echo "  Project size: $(du -sh ~/Project_Sullivan/ 2>/dev/null | cut -f1)"
    if [ -d ~/Project_Sullivan/data ]; then
        echo "  Data size:    $(du -sh ~/Project_Sullivan/data/ 2>/dev/null | cut -f1)"
    fi
    if [ -d ~/Project_Sullivan/models ]; then
        echo "  Models size:  $(du -sh ~/Project_Sullivan/models/ 2>/dev/null | cut -f1)"
    fi
else
    echo "  ⚠️  Project directory not found"
fi

echo ""
echo "=== 🏃 Active Training Processes ==="
PROCS=$(ps aux | grep -E "train_(transformer|conformer)" | grep -v grep || true)
if [ -n "${PROCS}" ]; then
    echo "${PROCS}" | awk "{printf \"  PID: %s  CPU: %s%%  MEM: %s%%  CMD: \", \$2, \$3, \$4; for(i=11;i<=NF;i++) printf \"%s \", \$i; print \"\"}"
else
    echo "  No active training processes."
fi

echo ""
echo "=== 📄 Recent Training Logs ==="
if ls ~/Project_Sullivan/logs/train_*.log 1>/dev/null 2>&1; then
    ls -lt ~/Project_Sullivan/logs/train_*.log | head -5 | awk "{print \"  \" \$NF \" (\" \$6 \" \" \$7 \" \" \$8 \")\"}"
    echo ""
    echo "  Latest log tail (last 5 lines):"
    LATEST=$(ls -t ~/Project_Sullivan/logs/train_*.log | head -1)
    tail -5 "${LATEST}" | sed "s/^/    /"
else
    echo "  No training logs found."
fi

echo ""
echo "=== 🔧 UV Environment ==="
if command -v uv &>/dev/null; then
    echo "  UV: $(uv --version 2>/dev/null || echo "installed but version unknown")"
else
    echo "  ⚠️  UV not installed"
fi
'

echo ""
echo "============================================"
echo "  Status check complete."
echo "============================================"
