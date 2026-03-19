#!/usr/bin/env python3
"""
Train Conformer Model for Acoustic-to-Articulatory Inversion — Phase 4

Supports both Mel-spectrogram and HuBERT features.
Reuses config loading logic from train_transformer.py.

Usage:
    # Mel features (default)
    uv run python scripts/train_conformer.py --config configs/conformer_a100_config.yaml --gpus 1

    # HuBERT features
    uv run python scripts/train_conformer.py --config configs/conformer_a100_config.yaml --gpus 1

    # Resume from checkpoint
    uv run python scripts/train_conformer.py --config configs/conformer_a100_config.yaml --gpus 1 --resume-from models/conformer/checkpoints/last.ckpt
"""

import argparse
import sys
from pathlib import Path

import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from modeling.dataset import create_dataloaders, AudioAugmentation
from modeling.conformer_model import ConformerInversionModel

# Reuse config loader from train_transformer
from train_transformer import load_config


def create_model(config: dict) -> ConformerInversionModel:
    """Create Conformer model from configuration."""
    mc = config['model']
    return ConformerInversionModel(
        input_dim=mc['input_dim'],
        d_model=mc['d_model'],
        num_layers=mc['num_layers'],
        num_heads=mc['num_heads'],
        ffn_dim=mc['d_ff'],
        depthwise_conv_kernel_size=mc.get('depthwise_conv_kernel_size', 31),
        output_dim=mc['output_dim'],
        dropout=mc['dropout'],
        learning_rate=mc['learning_rate'],
        weight_decay=mc['weight_decay'],
        mse_weight=mc.get('mse_weight', 1.0),
        pcc_weight=mc.get('pcc_weight', 2.0),
        velocity_weight=mc.get('velocity_weight', 1.0),
        acceleration_weight=mc.get('acceleration_weight', 0.5),
        curriculum_warmup_epochs=mc.get('curriculum_warmup_epochs', 0),
        curriculum_ramp_epochs=mc.get('curriculum_ramp_epochs', 0),
    )


def main():
    parser = argparse.ArgumentParser(
        description='Train Conformer for Articulatory Inversion (Phase 4)'
    )
    parser.add_argument(
        '--config', type=str,
        default='configs/conformer_a100_config.yaml',
        help='Path to config file',
    )
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument(
        '--fast-dev-run', action='store_true', help='Debug with 1 batch',
    )
    parser.add_argument(
        '--resume-from', type=str, default=None,
        help='Checkpoint path to resume training',
    )
    parser.add_argument(
        '--auto-resume', action='store_true',
        help='Auto-resume from latest checkpoint in save_dir',
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print("Configuration loaded:")
    print(yaml.dump(config, default_flow_style=False))

    pl.seed_everything(config['seed'])

    # Create augmentation (train only)
    train_augmentation = None
    aug_cfg = config.get('augmentation', {})
    if aug_cfg.get('enabled', False):
        train_augmentation = AudioAugmentation(
            time_mask_max_len=aug_cfg.get('time_mask_max_len', 30),
            time_mask_num=aug_cfg.get('time_mask_num', 2),
            freq_mask_max_len=aug_cfg.get('freq_mask_max_len', 64),
            freq_mask_num=aug_cfg.get('freq_mask_num', 2),
            noise_std=aug_cfg.get('noise_std', 0.01),
        )
        print(f"\n🎲 Augmentation enabled: time_mask×{aug_cfg.get('time_mask_num',2)}"
              f"(max {aug_cfg.get('time_mask_max_len',30)}), "
              f"freq_mask×{aug_cfg.get('freq_mask_num',2)}"
              f"(max {aug_cfg.get('freq_mask_max_len',64)}), "
              f"noise_std={aug_cfg.get('noise_std',0.01)}")

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
        sequence_length=config['data']['sequence_length'],
        streaming=config['data'].get('streaming', False),
        train_augmentation=train_augmentation,
    )

    # Create model
    print("\nCreating Conformer model...")
    model = create_model(config)

    # Callbacks
    save_dir = Path(config['logging']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt_config = config['callbacks']['model_checkpoint']
    callbacks = [
        ModelCheckpoint(
            dirpath=save_dir / 'checkpoints',
            monitor=ckpt_config['monitor'],
            mode=ckpt_config['mode'],
            save_top_k=ckpt_config['save_top_k'],
            save_last=ckpt_config['save_last'],
            filename=ckpt_config['filename'],
            verbose=True,
        ),
        EarlyStopping(
            monitor=config['callbacks']['early_stopping']['monitor'],
            patience=config['callbacks']['early_stopping']['patience'],
            mode=config['callbacks']['early_stopping']['mode'],
            verbose=True,
        ),
        LearningRateMonitor(logging_interval='step'),
    ]

    # Logger
    logger = TensorBoardLogger(
        save_dir=config['logging']['log_dir'],
        name=config['logging']['experiment_name'],
    )

    # Auto-resume
    ckpt_path = args.resume_from
    if args.auto_resume and ckpt_path is None:
        last_ckpt = save_dir / 'checkpoints' / 'last.ckpt'
        if last_ckpt.exists():
            ckpt_path = str(last_ckpt)
            print(f"\n🔄 Auto-resuming from: {ckpt_path}")

    # Trainer
    trainer_kwargs = {
        'max_epochs': config['training']['num_epochs'],
        'callbacks': callbacks,
        'logger': logger,
        'precision': config['training']['precision'],
        'gradient_clip_val': config['training']['gradient_clip_val'],
        'accumulate_grad_batches': config['training'].get('accumulate_grad_batches', 1),
        'log_every_n_steps': config['logging']['log_every_n_steps'],
        'fast_dev_run': args.fast_dev_run,
        'enable_progress_bar': True,
        'enable_model_summary': True,
    }

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
    print("🚀 CONFORMER TRAINING START — Phase 4 정확도 개선")
    print("=" * 60)
    print(f"Experiment: {config['logging']['experiment_name']}")
    mc = config['model']
    print(f"Model: {mc['name']} | d_model={mc['d_model']} | "
          f"layers={mc['num_layers']} | heads={mc['num_heads']}")
    print(f"Precision: {config['training']['precision']} | "
          f"Batch: {config['training']['batch_size']}")
    print("=" * 60 + "\n")

    trainer.fit(
        model,
        train_dataloaders=dataloaders['train'],
        val_dataloaders=dataloaders['val'],
        ckpt_path=ckpt_path,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    # Test
    if not args.fast_dev_run:
        print("\nRunning test evaluation...")
        trainer.test(model, dataloaders=dataloaders['test'])

        final_path = save_dir / 'final_model.ckpt'
        trainer.save_checkpoint(final_path)
        print(f"\nFinal model saved to: {final_path}")

    print(f"\nLogs: {logger.log_dir}")
    print(f"Checkpoints: {save_dir / 'checkpoints'}")


if __name__ == '__main__':
    main()
