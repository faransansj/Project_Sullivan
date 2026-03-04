#!/usr/bin/env python3
"""Simple Transformer test without PyTorch Lightning progress bars"""

import sys
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from modeling.dataset import create_dataloaders, collate_fn
from modeling.transformer import TransformerModel

def main():
    print("=" * 60)
    print("SIMPLE TRANSFORMER TEST")
    print("=" * 60)

    # Load config
    with open('configs/transformer_quick_test.yaml') as f:
        config = yaml.safe_load(f)

    print("\n1. Creating datasets...")
    dataloaders = create_dataloaders(
        splits_dir=Path(config['data']['splits_dir']),
        audio_feature_dir=Path(config['data']['audio_feature_dir']),
        parameter_dir=Path(config['data']['parameter_dir']),
        audio_feature_type=config['data']['audio_feature_type'],
        parameter_type=config['data']['parameter_type'],
        batch_size=config['training']['batch_size'],
        num_workers=0
    )

    train_loader = dataloaders['train']
    print(f"   Train samples: {len(train_loader.dataset)}")

    # Create model
    print("\n2. Creating Transformer model...")
    model_config = config['model']
    model = TransformerModel(
        input_dim=model_config['input_dim'],
        d_model=model_config['d_model'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        d_ff=model_config['d_ff'],
        output_dim=model_config['output_dim'],
        dropout=model_config['dropout'],
        learning_rate=model_config['learning_rate']
    )

    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    print("\n3. Testing forward pass...")
    batch = next(iter(train_loader))
    audio_features, targets, seq_lengths, utterance_names = batch

    print(f"   Batch shape: {audio_features.shape}")
    print(f"   Target shape: {targets.shape}")
    print(f"   Seq lengths: {seq_lengths}")

    with torch.no_grad():
        output = model(audio_features, seq_lengths)
        print(f"   Output shape: {output.shape}")

    # Try one training step
    print("\n4. Testing training step...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_config['learning_rate'])

    model.train()
    optimizer.zero_grad()

    # Compute loss manually
    outputs = model(audio_features, seq_lengths)
    loss = torch.nn.functional.mse_loss(outputs, targets)
    print(f"   Initial loss: {loss.item():.4f}")

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    print("\n5. Running 3 training epochs (manual loop)...")
    for epoch in range(3):
        model.train()
        total_loss = 0
        batch_count = 0

        for i, batch in enumerate(train_loader):
            audio, targets, lengths, _ = batch

            optimizer.zero_grad()
            outputs = model(audio, lengths)
            loss = torch.nn.functional.mse_loss(outputs, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            if (i + 1) % 3 == 0:
                print(f"   Epoch {epoch}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / batch_count
        print(f"   Epoch {epoch} complete - Avg Loss: {avg_loss:.4f}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE - Transformer is working!")
    print("=" * 60)

if __name__ == '__main__':
    main()
