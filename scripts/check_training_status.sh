#!/bin/bash
# Quick Training Status Check

echo "╔════════════════════════════════════════════════════════════╗"
echo "║          Project Sullivan - Training Status               ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if training is running
if pgrep -f "train_baseline.py" > /dev/null; then
    echo "✅ Status: TRAINING IN PROGRESS"
    echo "📍 Process ID: $(pgrep -f train_baseline.py | head -1)"
    echo ""

    # Show current epoch/batch
    echo "📊 Current Progress:"
    grep -E "Epoch [0-9]+:" logs/training/training_output.log | tail -1
    echo ""

    # Show latest metrics
    echo "📈 Latest Metrics:"
    grep -E "(train_loss|val_loss|val_rmse)" logs/training/training_output.log | tail -5
    echo ""

    # Estimate completion
    EPOCHS_DONE=$(grep -c "Epoch.*100%" logs/training/training_output.log)
    echo "🔢 Epochs Completed: $EPOCHS_DONE / 50"
    echo ""

else
    echo "⏹️  Status: NOT RUNNING"
    echo ""

    # Check if completed
    if grep -q "EXPERIMENT COMPLETE" logs/training/training_output.log; then
        echo "✅ Training COMPLETED"
        echo ""
        echo "📊 Final Test Results:"
        grep -A 10 "Test metric" logs/training/training_output.log | tail -15
    else
        echo "⚠️  Training may have stopped unexpectedly"
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Commands:"
echo "  📋 View logs:        tail -f logs/training/training_output.log"
echo "  📊 TensorBoard:      tensorboard --logdir=logs/training"
echo "  🔍 Check checkpoints: ls -lh models/baseline_lstm/checkpoints/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
