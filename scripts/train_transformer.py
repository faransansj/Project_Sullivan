#!/usr/bin/env python3
"""
Train Transformer Model for Acoustic-to-Articulatory Inversion

This script trains the Transformer model for Phase 2-B.

Usage:
    python scripts/train_transformer.py --config configs/transformer_config.yaml
    python scripts/train_transformer.py --config configs/transformer_quick_test.yaml
    python scripts/train_transformer.py --config configs/transformer_config.yaml --gpus 1
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
from modeling.transformer import TransformerModel


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file and resolve interpolations."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Flatten config to help with resolution
    flat_config = {}
    def flatten(d, prefix=''):
        for k, v in d.items():
            if isinstance(v, dict):
                flatten(v, prefix + k + '.')
                flat_config[k] = v # Also store section roots
            flat_config[prefix + k] = v
            flat_config[k] = v # Store key without prefix for simple placeholders
            
    flatten(config)
    
    # Resolver function
    def resolve_str(s):
        if not isinstance(s, str): return s
        pattern = re.compile(r'\${([^}]+)}')
        while True:
            match = pattern.search(s)
            if not match: break
            key = match.group(1)
            val = flat_config.get(key, f"${{{key}}}")
            if isinstance(val, dict):
                # If it's a dict, we can't easily interpolate into a string
                # unless we want to support something complex.
                # For now, let's assume it's a leaf value.
                break
            s = s.replace(f"${{{key}}}", str(val))
        return s

    def resolve_recursive(d):
        if isinstance(d, dict):
            return {k: resolve_recursive(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [resolve_recursive(v) for v in d]
        elif isinstance(d, str):
            return resolve_str(d)
        else:
            return d

    return resolve_recursive(config)


def create_model(config: dict) -> TransformerModel:
    """Create Transformer model from configuration."""
    model_config = config['model']

    model = TransformerModel(
        input_dim=model_config['input_dim'],
        d_model=model_config['d_model'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        d_ff=model_config['d_ff'],
        output_dim=model_config['output_dim'],
        dropout=model_config['dropout'],
        pos_encoding=model_config.get('pos_encoding', 'learnable'),
        activation=model_config.get('activation', 'gelu'),
        learning_rate=model_config['learning_rate'],
        weight_decay=model_config['weight_decay'],
        max_seq_len=model_config.get('max_seq_len', 5000)
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
    parser = argparse.ArgumentParser(description="Train Transformer model")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/transformer_config.yaml',
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
    parser.add_argument(
        '--overfit-batches',
        type=int,
        default=0,
        help='Overfit on N batches for testing (0 = disabled)'
    )
    parser.add_argument(
        '--streaming',
        action='store_true',
        help='Enable dataset streaming'
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
        num_workers=config['dataloader']['num_workers'],
        sequence_length=config['data']['sequence_length'],
        streaming=args.streaming or config['dataloader'].get('streaming', False),
        zip_file_path=config['data'].get('zip_file_path', None)
    )

    # Create model
    print("\nCreating Transformer model...")
    model = create_model(config)
    print(f"\nModel architecture:")
    print(model)

    # Create callbacks
    callbacks = create_callbacks(config)

    # Create logger
    logger = TensorBoardLogger(
        save_dir=config['logging']['log_dir'],
        name=config['logging']['experiment_name']
    )

    # Create trainer
    trainer_kwargs = {
        'max_epochs': config['training']['num_epochs'],
        'callbacks': callbacks,
        'logger': logger,
        'precision': config['training']['precision'],
        'gradient_clip_val': config['training']['gradient_clip_val'],
        'accumulate_grad_batches': config['training']['accumulate_grad_batches'],
        'log_every_n_steps': config['logging']['log_every_n_steps'],
        'fast_dev_run': args.fast_dev_run,
        'enable_progress_bar': False,  # Disabled for log file compatibility
        'enable_model_summary': True
    }

    # Add overfit_batches if specified
    if args.overfit_batches > 0:
        trainer_kwargs['overfit_batches'] = args.overfit_batches
        print(f"\n⚠️  OVERFITTING MODE: Training on {args.overfit_batches} batch(es)")

    # Set device
    if args.gpus == 0:
        trainer_kwargs['accelerator'] = 'cpu'
        trainer_kwargs['devices'] = 1
        print("\n🖥️  Training on CPU")
    else:
        trainer_kwargs['accelerator'] = 'gpu'
        trainer_kwargs['devices'] = args.gpus
        print(f"\n🚀 Training on {args.gpus} GPU(s)")

    trainer = pl.Trainer(**trainer_kwargs)

    # Train
    print("\n" + "=" * 60)
    print("TRANSFORMER TRAINING START")
    print("=" * 60)
    print(f"Experiment: {config['logging']['experiment_name']}")
    print(f"Model: {config['model']['name']}")
    print(f"d_model: {config['model']['d_model']}, Layers: {config['model']['num_layers']}, Heads: {config['model']['num_heads']}")
    print("=" * 60 + "\n")

    trainer.fit(
        model,
        train_dataloaders=dataloaders['train'],
        val_dataloaders=dataloaders['val']
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    # Test (skip if overfitting)
    if args.overfit_batches == 0:
        print("\nRunning test evaluation...")
        trainer.test(model, dataloaders=dataloaders['test'])

        # Save final model
        final_model_path = Path(config['logging']['save_dir']) / 'final_model.ckpt'
        trainer.save_checkpoint(final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")
    else:
        print("\n⚠️  Skipping test evaluation (overfitting mode)")

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Logs saved to: {logger.log_dir}")
    print(f"Checkpoints saved to: {Path(config['logging']['save_dir']) / 'checkpoints'}")


if __name__ == '__main__':
    main()
