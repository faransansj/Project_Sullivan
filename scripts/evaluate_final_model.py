import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import torch
import yaml
from src.modeling.transformer import TransformerModel
from src.modeling.dataset import create_dataloaders
import pytorch_lightning as pl

def evaluate():
    with open('configs/transformer_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    model = TransformerModel.load_from_checkpoint('models/transformer/final_model.ckpt')
    model.eval()
    
    # Create dataloaders
    loaders = create_dataloaders(
        Path(config['data']['splits_dir']),
        Path(config['data']['audio_feature_dir']),
        Path(config['data']['parameter_dir']),
        sequence_length=config['data']['sequence_length'],
        streaming=config['data']['streaming']
    )
    
    trainer = pl.Trainer(accelerator='cpu', precision=32)
    results = trainer.test(model, loaders['test'])
    print(results)

if __name__ == "__main__":
    evaluate()
