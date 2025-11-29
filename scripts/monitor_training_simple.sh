#!/bin/bash
# Simple Training Monitor (Checkpoint-based)

echo "╔════════════════════════════════════════════════════════════╗"
echo "║       Project Sullivan - Training Monitor (Simple)        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if training is running
PID=$(pgrep -f "train_baseline.py" | head -1)
if [ -n "$PID" ]; then
    echo "✅ Status: TRAINING IN PROGRESS"
    echo "📍 Process ID: $PID"

    # CPU usage
    CPU=$(ps aux | grep $PID | grep -v grep | awk '{print $3}')
    echo "💻 CPU Usage: ${CPU}%"
    echo ""
else
    echo "⏹️  Status: NOT RUNNING"
    echo ""
fi

# Check checkpoints
echo "📁 Recent Checkpoints:"
if [ -d "models/baseline_lstm/checkpoints" ]; then
    ls -lht models/baseline_lstm/checkpoints/*.ckpt 2>/dev/null | head -5 | while read line; do
        filename=$(echo $line | awk '{print $NF}')
        timestamp=$(echo $line | awk '{print $6, $7, $8}')

        # Extract epoch and val_loss from filename
        if [[ $filename =~ epoch=([0-9]+)-val_loss=([0-9]+\.[0-9]+) ]]; then
            epoch="${BASH_REMATCH[1]}"
            val_loss="${BASH_REMATCH[2]}"
            echo "   Epoch $epoch | val_loss=$val_loss | $timestamp"
        elif [[ $filename == *"last"* ]]; then
            echo "   Latest checkpoint | $timestamp"
        fi
    done

    # Calculate progress
    LATEST_EPOCH=$(ls -t models/baseline_lstm/checkpoints/*.ckpt 2>/dev/null | head -1 | grep -oP 'epoch=\K[0-9]+' | head -1)
    if [ -n "$LATEST_EPOCH" ]; then
        PROGRESS=$((($LATEST_EPOCH + 1) * 100 / 50))
        echo ""
        echo "📊 Progress: Epoch $((LATEST_EPOCH + 1))/50 (${PROGRESS}%)"
    fi
else
    echo "   No checkpoints found"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "💡 Tips:"
echo "   • tensorboard --logdir=logs/training/baseline_lstm_v1"
echo "   • watch -n 30 bash scripts/monitor_training_simple.sh"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
