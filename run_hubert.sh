#!/bin/bash
cd ~/Project_Sullivan
export CUDA_VISIBLE_DEVICES=0
source .venv/bin/activate
python scripts/train_conformer.py --config configs/conformer_hubert_small_curriculum_config.yaml --gpus 1 > logs/conformer_hubert_small_curriculum.log 2>&1
