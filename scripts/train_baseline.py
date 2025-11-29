#!/usr/bin/env python3
"""
Train Baseline Bi-LSTM Model

This script trains the baseline model for acoustic-to-articulatory inversion.

Usage:
    python scripts/train_baseline.py --config configs/baseline_config.yaml
    python scripts/train_baseline.py --config configs/baseline_config.yaml --fast-dev-run
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from modeling.dataset import create_dataloaders
from modeling.baseline_lstm import BaselineLSTM


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict) -> BaselineLSTM:
    """Create model from configuration."""
    model_config = config['model']

    model = BaselineLSTM(
        input_dim=model_config['input_dim'],
        hidden_dim=model_config['hidden_dim'],
        num_layers=model_config['num_layers'],
        output_dim=model_config['output_dim'],
        dropout=model_config['dropout'],
        learning_rate=model_config['learning_rate']
    )

    return model


def create_callbacks(config: dict) -> list:
    """Create training callbacks."""
    callbacks = []

    # Model checkpoint
    checkpoint_config = config['callbacks']['model_checkpoint']
    save_dir = Path(config['logging']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir / 'checkpoints',
        monitor=checkpoint_config['monitor'],
        mode=checkpoint_config['mode'],
        save_top_k=checkpoint_config['save_top_k'],
        save_last=checkpoint_config['save_last'],
        filename=checkpoint_config['filename'],
        verbose=True
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    early_stop_config = config['callbacks']['early_stopping']
    early_stop_callback = EarlyStopping(
        monitor=early_stop_config['monitor'],
        patience=early_stop_config['patience'],
        mode=early_stop_config['mode'],
        verbose=early_stop_config['verbose']
    )
    callbacks.append(early_stop_callback)

    return callbacks


def main():
    parser = argparse.ArgumentParser(description="Train baseline Bi-LSTM model")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/baseline_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--fast-dev-run',
        action='store_true',
        help='Run a fast development run (1 batch per epoch)'
    )
    parser.add_argument(
        '--gpus',
        type=int,
        default=0,
        help='Number of GPUs to use (0 for CPU)'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print("Configuration loaded:")
    print(yaml.dump(config, default_flow_style=False))

    # Set seed for reproducibility
    pl.seed_everything(config['seed'])

    # Create dataloaders
    print("\nCreating dataloaders...")
    dataloaders = create_dataloaders(
        splits_dir=Path(config['data']['splits_dir']),
        audio_feature_dir=Path(config['data']['audio_feature_dir']),
        parameter_dir=Path(config['data']['parameter_dir']),
        audio_feature_type=config['data']['audio_feature_type'],
        parameter_type=config['data']['parameter_type'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        sequence_length=config['data']['sequence_length']
    )

    # Create model
    print("\nCreating model...")
    model = create_model(config)
    print(f"Model created: {model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create callbacks
    callbacks = create_callbacks(config)

    # Create logger
    logger = TensorBoardLogger(
        save_dir=config['logging']['log_dir'],
        name=config['logging']['experiment_name']
    )

    # Create trainer
    if args.gpus == 0:
        # CPU training
        trainer = pl.Trainer(
            max_epochs=config['training']['num_epochs'],
            callbacks=callbacks,
            logger=logger,
            accelerator='cpu',
            devices=1,
            precision=config['training']['precision'],
            gradient_clip_val=config['training']['gradient_clip_val'],
            accumulate_grad_batches=config['training']['accumulate_grad_batches'],
            log_every_n_steps=config['logging']['log_every_n_steps'],
            fast_dev_run=args.fast_dev_run,
            enable_progress_bar=True,
            enable_model_summary=True
        )
    else:
        # GPU training
        trainer = pl.Trainer(
            max_epochs=config['training']['num_epochs'],
            callbacks=callbacks,
            logger=logger,
            accelerator='gpu',
            devices=args.gpus,
            precision=config['training']['precision'],
            gradient_clip_val=config['training']['gradient_clip_val'],
            accumulate_grad_batches=config['training']['accumulate_grad_batches'],
            log_every_n_steps=config['logging']['log_every_n_steps'],
            fast_dev_run=args.fast_dev_run,
            enable_progress_bar=True,
            enable_model_summary=True
        )

    # Train
    print("\n" + "=" * 60)
    print("TRAINING START")
    print("=" * 60)

    trainer.fit(
        model,
        train_dataloaders=dataloaders['train'],
        val_dataloaders=dataloaders['val']
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    # Test
    print("\nRunning test evaluation...")
    trainer.test(model, dataloaders=dataloaders['test'])

    # Save final model
    final_model_path = Path(config['logging']['save_dir']) / 'final_model.ckpt'
    trainer.save_checkpoint(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Logs saved to: {logger.log_dir}")
    print(f"Checkpoints saved to: {Path(config['logging']['save_dir']) / 'checkpoints'}")


if __name__ == '__main__':
    main()
