#!/usr/bin/env python3
"""
Train Transformer Model - Phase 4-C (Staged Curriculum Learning)

Step 1: Load 14-dim Geometric Model
Step 2: Freeze Geometric Layers
Step 3: Train PCA Head (10 dims)
"""

import argparse
import sys
import yaml
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pathlib import Path
from copy import deepcopy

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from modeling.dataset import create_dataloaders
from modeling.transformer import TransformerModel
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

class Phase4CTransformer(TransformerModel):
    """
    Subclass for Phase 4-C Curriculum Learning.
    Overrides training step to focus on PCA loss only.
    """
    def __init__(self, **kwargs):
        self.phase = kwargs.pop('phase', 'pca_only')
        super().__init__(**kwargs)
        
    def training_step(self, batch, batch_idx):
        audio, params, lengths, _ = batch
        
        # DEBUG: Check dimensions
        if params.shape[-1] != 24:
            print(f"ERROR: Expected 24 dims, got {params.shape[-1]}")
            # If we are in sanity check, this might crash.
        
        # Forward pass
        pred_params = self(audio, lengths)

        # Create mask
        mask = self._create_mask(lengths, pred_params.shape[1]).to(pred_params.device)
        mask = mask.unsqueeze(-1)

        # Split prediction and target
        # Dims 0-13: Geometric (Ignored in loss)
        # Dims 14-23: PCA (Target)
        
        pred_pca = pred_params[:, :, 14:]
        target_pca = params[:, :, 14:]
        mask_expanded = mask # broadcastable
        
        # PCA Position Loss
        pca_mse = self.criterion(pred_pca, target_pca)
        pca_loss = (pca_mse * mask_expanded).sum() / mask_expanded.sum()
        
        # PCA Temporal Losses
        # We need to compute temporal loss only on PCA part
        temporal_losses = self._compute_temporal_loss(pred_params[:, :, 14:], params[:, :, 14:], mask)
        velocity_loss = temporal_losses['velocity_loss']
        acceleration_loss = temporal_losses['acceleration_loss']
        
        # PCA PCC Loss
        # We pass only the PCA slice to _compute_pcc_loss
        # Note: _compute_pcc_loss expects (batch, seq, feat)
        pcc_loss = self._compute_pcc_loss(pred_params[:, :, 14:], params[:, :, 14:], mask)
        
        # Total Loss (PCA Only)
        loss = (self.mse_weight * pca_loss) + \
               (self.pcc_weight * pcc_loss) + \
               (self.velocity_weight * velocity_loss) + \
               (self.acceleration_weight * acceleration_loss)
               
        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mse_pca', pca_loss, on_step=False, on_epoch=True)
        self.log('train_pcc_pca', pcc_loss, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        audio, params, lengths, _ = batch
        pred_params = self(audio, lengths)
        mask = self._create_mask(lengths, pred_params.shape[1]).to(pred_params.device)
        mask = mask.unsqueeze(-1)
        
        # Compute PCA Loss for validation monitoring
        pred_pca = pred_params[:, :, 14:]
        target_pca = params[:, :, 14:]
        
        pca_mse = self.criterion(pred_pca, target_pca)
        pca_loss = (pca_mse * mask).sum() / mask.sum()
        
        pcc_loss = self._compute_pcc_loss(pred_pca, target_pca, mask)
        
        loss = pca_loss + pcc_loss # Simple validation metric
        
        self.log('val_loss_pca', loss, on_epoch=True, prog_bar=True)
        self.log('val_mse_pca', pca_loss, on_epoch=True)
        self.log('val_pcc_pca', pcc_loss, on_epoch=True)
        
        return loss

def load_and_adapt_model(checkpoint_path, config):
    """
    Load 14-dim Geometric Model and adapt to 24-dim Phase 4-C Model.
    """
    print(f"Loading pretrained 14-dim model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    
    # Create new Phase 4-C model (24 dims)
    model = Phase4CTransformer(
        input_dim=config['model']['input_dim'],
        d_model=config['model']['d_model'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        d_ff=config['model']['d_ff'],
        output_dim=24, # Force 24 dims
        dropout=config['model']['dropout'],
        learning_rate=config['optimization']['learning_rate'],
        pcc_weight=config['model']['pcc_weight'],
        phase="pca_only"
    )
    
    # Transfer weights
    # Everything matches except 'output_projection.weight' and 'output_projection.bias'
    new_state_dict = model.state_dict()
    
    print("Transferring weights...")
    for key in state_dict:
        if 'output_projection' in key:
            continue # Handle manually
        if key in new_state_dict:
            new_state_dict[key] = state_dict[key]
            
    # Handle Output Projection
    # Old: (14, 512)
    # New: (24, 512)
    old_weight = state_dict['output_projection.weight']
    old_bias = state_dict['output_projection.bias']
    
    # Copy geometric weights (first 14 rows)
    new_state_dict['output_projection.weight'][:14, :] = old_weight
    new_state_dict['output_projection.bias'][:14] = old_bias
    
    # Initialize PCA weights (last 10 rows) randomly (already done by init, but explicit check)
    # We leave them as initialized by the constructor.
    
    # Load modified state dict
    model.load_state_dict(new_state_dict)
    
    # Freeze Geometric Layers
    print("Freezing Encoder and Geometric Projection...")
    
    # Freeze Encoder & Input Projection
    for param in model.input_projection.parameters():
        param.requires_grad = False
    for param in model.pos_encoding.parameters():
        param.requires_grad = False
    for param in model.transformer_encoder.parameters():
        param.requires_grad = False
        
    # Note: We CANNOT freeze partial Linear layer weights easily.
    # But since our training_step only computes loss on the PCA outputs, 
    # the gradients for the first 14 rows of output_projection will be zero.
    # So effectively, the Geometric head is frozen.
    
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/transformer_phase4c_curriculum.yaml')
    parser.add_argument('--gpus', type=int, default=0)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    pl.seed_everything(config['seed'])
    
    # 1. Load & Adapt Model
    pretrained_ckpt = "models/transformer/final_model.ckpt"
    model = load_and_adapt_model(pretrained_ckpt, config)
    
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
    
    print("\nSTARTING PHASE 4-C: PCA HEAD TRAINING")
    print("Encoder is frozen. Only PCA output weights will update.")
    
    trainer.fit(
        model,
        train_dataloaders=dataloaders['train'],
        val_dataloaders=dataloaders['val']
    )

if __name__ == "__main__":
    main()
