#!/bin/bash
# Start TensorBoard for Project Sullivan

echo "Starting TensorBoard..."
echo ""

# Check if already running
if pgrep -f "tensorboard" > /dev/null; then
    echo "⚠️  TensorBoard is already running"
    echo ""
    echo "Access at: http://localhost:6006"
    echo ""
    echo "To stop: pkill -f tensorboard"
    exit 0
fi

# Start TensorBoard
VIRTUAL_ENV=/home/midori/Develop/Project_Sullivan/venv_sullivan \
/home/midori/Develop/Project_Sullivan/venv_sullivan/bin/python \
-m tensorboard.main \
--logdir=logs/training/baseline_lstm_v1 \
--port=6006 \
--bind_all \
> /dev/null 2>&1 &

sleep 2

if pgrep -f "tensorboard" > /dev/null; then
    echo "✅ TensorBoard started successfully!"
    echo ""
    echo "📊 Access at: http://localhost:6006"
    echo ""
    echo "To stop: pkill -f tensorboard"
else
    echo "❌ Failed to start TensorBoard"
    exit 1
fi
