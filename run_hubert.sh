#!/bin/bash
cd ~/Project_Sullivan
export CUDA_VISIBLE_DEVICES=GPU-39a72a91-24ce-0e6a-30e5-1f53e62ce2d8
source .venv/bin/activate
python scripts/train_conformer.py --config configs/conformer_hubert_small_curriculum_config.yaml --gpus 1 > logs/conformer_hubert_small_curriculum.log 2>&1
