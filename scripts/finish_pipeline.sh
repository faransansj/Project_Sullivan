#!/bin/bash
PID=$1
LOG_FILE="/tmp/pipeline_completion.log"

echo "Starting pipeline completion script for PID $PID" > $LOG_FILE
echo "Monitoring process $PID..."

# Wait for PID
while kill -0 $PID 2> /dev/null; do
    # sleep 30
    sleep 30
done

echo "Process $PID finished at $(date). Proceeding with next steps." >> $LOG_FILE

# Generate splits
echo "Generating dataset splits..." >> $LOG_FILE
/home/Project_Sullivan/.venv/bin/python scripts/create_dataset_splits_hddb.py --output-dir data/processed/splits >> $LOG_FILE 2>&1

if [ $? -ne 0 ]; then
    echo "Error generating splits. Aborting." >> $LOG_FILE
    exit 1
fi

echo "Splits generated successfully." >> $LOG_FILE

# Start Training
echo "Starting Transformer training..." >> $LOG_FILE
/home/Project_Sullivan/.venv/bin/python scripts/train_transformer.py --config configs/transformer_config.yaml --gpus 0 >> $LOG_FILE 2>&1

if [ $? -ne 0 ]; then
    echo "Training failed. Check log." >> $LOG_FILE
    exit 1
fi

echo "Pipeline complete at $(date)." >> $LOG_FILE
