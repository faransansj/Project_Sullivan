#!/usr/bin/env python3
"""
Train Transformer Model - Phase 4-D (Joint Fine-Tuning)

Step 4: Load Phase 4-C Model (24 dims), Unfreeze All, Train with Low LR.
"""

import argparse
import sys
import yaml
import torch
import pytorch_lightning as pl
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from modeling.dataset import create_dataloaders
from modeling.transformer import TransformerModel
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/transformer_phase4d_joint.yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to Phase 4-C checkpoint')
    parser.add_argument('--gpus', type=int, default=0)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    pl.seed_everything(config['seed'])
    
    # 1. Load Phase 4-C Model
    print(f"Loading Phase 4-C checkpoint from {args.checkpoint}")
    
    # We use the standard TransformerModel class now, as we want standard training_step (all losses)
    model = TransformerModel.load_from_checkpoint(
        args.checkpoint,
        strict=False, # Relax strictness if minor attribute mismatches (e.g. 'phase')
        # Override hparams with Phase 4-D config
        learning_rate=config['optimization']['learning_rate'],
        mse_weight=config['model']['mse_weight'],
        pcc_weight=config['model']['pcc_weight'],
        velocity_weight=config['model']['velocity_weight'],
        acceleration_weight=config['model']['acceleration_weight']
    )
    
    # Ensure everything is unfrozen (default, but explicit check)
    print("Unfreezing all parameters...")
    for param in model.parameters():
        param.requires_grad = True
        
    # 2. Data Loaders
    print("Creating dataloaders...")
    dataloaders = create_dataloaders(
        Path(config['data']['splits_dir']),
        Path(config['data']['audio_feature_dir']),
        Path(config['data']['parameter_dir']),
        parameter_type='combined', # 24 dims
        sequence_length=config['data']['sequence_length'],
        streaming=True,
        batch_size=config['training']['batch_size'],
        num_workers=0
    )
    
    # 3. Trainer
    logger = TensorBoardLogger(
        save_dir=config['logging']['log_dir'],
        name=config['logging']['experiment_name']
    )
    
    callbacks = [
        ModelCheckpoint(**config['callbacks']['model_checkpoint']),
        EarlyStopping(**config['callbacks']['early_stopping'])
    ]
    
    trainer = pl.Trainer(
        max_epochs=config['training']['num_epochs'],
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        gradient_clip_val=config['training']['gradient_clip_val'],
        accelerator='cpu' if args.gpus == 0 else 'gpu',
        devices=1 if args.gpus == 0 else args.gpus
    )
    
    print("\nSTARTING PHASE 4-D: JOINT FINE-TUNING")
    
    trainer.fit(
        model,
        train_dataloaders=dataloaders['train'],
        val_dataloaders=dataloaders['val']
    )

if __name__ == "__main__":
    main()
