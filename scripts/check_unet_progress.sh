#!/bin/bash
# Quick script to monitor U-Net training progress

echo "======================================"
echo "U-Net 훈련 진행 상황"
echo "======================================"
echo ""

# Check if training is running
if pgrep -f "train_unet.py" > /dev/null; then
    echo "✅ 훈련 진행 중"
    echo ""
else
    echo "⚠️  훈련 프로세스가 실행 중이 아닙니다"
    echo ""
fi

# Show latest checkpoints
echo "📁 최근 체크포인트:"
ls -lht models/unet_scratch/checkpoints/*.ckpt 2>/dev/null | head -5 | awk '{print "  ", $9, "("$5")"}'
echo ""

# Show training output (last 30 lines)
echo "📊 최근 훈련 로그:"
tail -30 /tmp/claude/-home-Project-Sullivan/tasks/bfbd08c.output 2>/dev/null | grep -E "Epoch [0-9]+:" | tail -5
echo ""

echo "======================================"
echo "TensorBoard 실행: bash scripts/start_tensorboard.sh"
echo "======================================"
