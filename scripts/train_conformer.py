"""
Train Conformer Model for High-Performance Articulatory Inversion (Phase 6)
"""

import argparse
import sys
from pathlib import Path
import yaml
import re
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from modeling.dataset import create_dataloaders
from modeling.conformer_model import ConformerInversionModel
# Use the same config loading logic
from train_transformer import load_config

def main():
    parser = argparse.ArgumentParser(description='Train Conformer for Articulatory Inversion')
    parser.add_argument('--config', type=str, default='configs/transformer_a100.yaml', help='Path to config file')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    args = parser.parse_args()

    config = load_config(args.config)
    pl.seed_everything(config['seed'])

    # Data
    dataloaders = create_dataloaders(
        splits_dir=Path(config['data']['splits_dir']),
        audio_feature_dir=Path(config['data']['audio_feature_dir']),
        parameter_dir=Path(config['data']['parameter_dir']),
        audio_feature_type=config['data']['audio_feature_type'],
        parameter_type=config['data']['parameter_type'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        sequence_length=config['data']['sequence_length'],
        streaming=config['training'].get('streaming', True)
    )

    # Model
    model = ConformerInversionModel(
        input_dim=config['model']['input_dim'],
        d_model=config['model']['d_model'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        ffn_dim=config['model']['d_ff'],
        output_dim=config['model']['output_dim'],
        dropout=config['model']['dropout'],
        learning_rate=config['optimization']['learning_rate']
    )

    # Logger & Trainer
    logger = TensorBoardLogger(save_dir=config['logging']['log_dir'], name="conformer_a100")
    
    trainer = pl.Trainer(
        max_epochs=config['training']['num_epochs'],
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else 1,
        precision=config['training']['precision'],
        gradient_clip_val=config['training']['gradient_clip_val'],
        logger=logger,
        callbacks=[
            ModelCheckpoint(monitor='val_loss', save_top_k=3, mode='min', filename='conformer-{epoch:02d}-{val_loss:.4f}'),
            EarlyStopping(monitor='val_loss', patience=15)
        ]
    )

    print("\n" + "="*60)
    print("🚀 CONFORMER A100 RAID START")
    print("="*60)
    trainer.fit(model, dataloaders['train'], dataloaders['val'])

if __name__ == "__main__":
    main()
