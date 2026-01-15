#!/usr/bin/env python3
"""
Visualize Transformer Model Predictions vs Ground Truth

This script loads a trained model checkpoint and generates visualizations
comparing predicted vs actual articulatory parameters for test samples.

Usage:
    python scripts/visualize_predictions.py --checkpoint models/transformer_quick/checkpoints/transformer-quick-epoch=00-val_loss=1.0174.ckpt --num_samples 5
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from modeling.transformer import TransformerModel
from modeling.baseline_lstm import BaselineLSTM
from modeling.dataset import create_dataloaders, ArticulatoryDataset


# Geometric feature names for plotting
FEATURE_NAMES = [
    'Tongue Tip X',
    'Tongue Tip Y',
    'Tongue Dorsum X',
    'Tongue Dorsum Y',
    'Tongue Root X',
    'Tongue Root Y',
    'Jaw Opening',
    'Lip Aperture',
    'Lip Protrusion',
    'Velum Height',
    'Larynx Height',
    'Pharynx Width',
    'Tongue Length',
    'Vocal Tract Area'
]


def load_model(checkpoint_path: str):
    """Load trained model from checkpoint (Transformer or LSTM)."""
    print(f"Loading model from: {checkpoint_path}")

    # Try loading the checkpoint to inspect its structure
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict_keys = checkpoint['state_dict'].keys()

    # Detect model type based on state_dict keys
    if any('lstm' in key for key in state_dict_keys):
        print("Detected BaselineLSTM model")
        model = BaselineLSTM.load_from_checkpoint(checkpoint_path)
    else:
        print("Detected Transformer model")
        model = TransformerModel.load_from_checkpoint(checkpoint_path)

    model.eval()
    return model


def get_predictions(model: TransformerModel, dataloader: DataLoader, num_samples: int = 5):
    """
    Get predictions for a few test samples.

    Returns
    -------
    results : list of dict
        Each dict contains:
        - 'utterance_name': str
        - 'ground_truth': np.ndarray (seq_len, 14)
        - 'predicted': np.ndarray (seq_len, 14)
        - 'length': int (actual sequence length before padding)
    """
    results = []

    # Get dataset for denormalization
    dataset = dataloader.dataset

    with torch.no_grad():
        for i, (audio_features, parameters, lengths, utterance_names) in enumerate(dataloader):
            if i >= num_samples:
                break

            # Get predictions
            predictions = model(audio_features, lengths)

            # Denormalize both ground truth and predictions using dataset's method
            parameters = dataset.denormalize_parameters(parameters)
            predictions = dataset.denormalize_parameters(predictions)

            # Convert to numpy and take first sample in batch
            gt = parameters[0].cpu().numpy()
            pred = predictions[0].cpu().numpy()
            seq_len = lengths[0].item()

            # Trim to actual length (remove padding)
            gt = gt[:seq_len]
            pred = pred[:seq_len]

            results.append({
                'utterance_name': utterance_names[0],
                'ground_truth': gt,
                'predicted': pred,
                'length': seq_len
            })

            print(f"Processed {utterance_names[0]}: {seq_len} frames")

    return results


def plot_predictions(results: list, output_dir: Path):
    """
    Create visualization plots for predictions vs ground truth.

    Creates one figure per sample with subplots for each feature.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample_idx, result in enumerate(results):
        utterance_name = result['utterance_name']
        gt = result['ground_truth']
        pred = result['predicted']
        seq_len = result['length']

        # Create time axis
        time = np.arange(seq_len)

        # Create figure with subplots (7 rows x 2 cols for 14 features)
        fig, axes = plt.subplots(7, 2, figsize=(16, 20))
        fig.suptitle(f'Prediction vs Ground Truth: {utterance_name}', fontsize=16, fontweight='bold')

        axes = axes.flatten()

        for feat_idx in range(14):
            ax = axes[feat_idx]

            # Plot ground truth and prediction
            ax.plot(time, gt[:, feat_idx], 'b-', linewidth=2, label='Ground Truth', alpha=0.7)
            ax.plot(time, pred[:, feat_idx], 'r--', linewidth=2, label='Predicted', alpha=0.7)

            # Calculate error metrics for this feature
            mse = np.mean((gt[:, feat_idx] - pred[:, feat_idx]) ** 2)
            mae = np.mean(np.abs(gt[:, feat_idx] - pred[:, feat_idx]))
            correlation = np.corrcoef(gt[:, feat_idx], pred[:, feat_idx])[0, 1]

            ax.set_title(f'{FEATURE_NAMES[feat_idx]}\nMAE: {mae:.3f}, Corr: {correlation:.3f}',
                        fontsize=10, fontweight='bold')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Value')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        output_file = output_dir / f'prediction_{sample_idx+1}_{utterance_name.replace("/", "_")}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

    print(f"\nAll visualizations saved to: {output_dir}")


def plot_summary_statistics(results: list, output_dir: Path):
    """
    Create a summary plot showing correlation and MAE for each feature across all samples.
    """
    num_features = 14
    num_samples = len(results)

    # Collect metrics for each feature
    correlations = np.zeros((num_samples, num_features))
    maes = np.zeros((num_samples, num_features))

    for i, result in enumerate(results):
        gt = result['ground_truth']
        pred = result['predicted']

        for feat_idx in range(num_features):
            # Correlation
            corr = np.corrcoef(gt[:, feat_idx], pred[:, feat_idx])[0, 1]
            correlations[i, feat_idx] = corr

            # MAE
            mae = np.mean(np.abs(gt[:, feat_idx] - pred[:, feat_idx]))
            maes[i, feat_idx] = mae

    # Create summary figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Summary: Prediction Quality Across Features', fontsize=16, fontweight='bold')

    # Plot 1: Mean correlation per feature
    mean_corr = np.mean(correlations, axis=0)
    std_corr = np.std(correlations, axis=0)
    x = np.arange(num_features)

    ax1.bar(x, mean_corr, yerr=std_corr, capsize=5, alpha=0.7, color='steelblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels([name.replace(' ', '\n') for name in FEATURE_NAMES],
                        rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Pearson Correlation', fontsize=12)
    ax1.set_title('Mean Correlation by Feature', fontsize=14, fontweight='bold')
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Mean MAE per feature
    mean_mae = np.mean(maes, axis=0)
    std_mae = np.std(maes, axis=0)

    ax2.bar(x, mean_mae, yerr=std_mae, capsize=5, alpha=0.7, color='coral')
    ax2.set_xticks(x)
    ax2.set_xticklabels([name.replace(' ', '\n') for name in FEATURE_NAMES],
                        rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Mean Absolute Error', fontsize=12)
    ax2.set_title('Mean MAE by Feature', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save figure
    output_file = output_dir / 'summary_statistics.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved summary: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize model predictions vs ground truth")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='Number of test samples to visualize (default: 5)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='visualizations/predictions',
        help='Output directory for plots (default: visualizations/predictions)'
    )
    parser.add_argument(
        '--splits_dir',
        type=str,
        default='data/processed/splits',
        help='Directory containing dataset splits'
    )
    parser.add_argument(
        '--audio_feature_dir',
        type=str,
        default='data/processed/audio_features',
        help='Directory containing audio features'
    )
    parser.add_argument(
        '--parameter_dir',
        type=str,
        default='data/processed/parameters',
        help='Directory containing parameters'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for inference (default: 1)'
    )

    args = parser.parse_args()

    # Load model
    model = load_model(args.checkpoint)

    # Create test dataloader
    print("\nLoading test dataset...")
    dataloaders = create_dataloaders(
        splits_dir=Path(args.splits_dir),
        audio_feature_dir=Path(args.audio_feature_dir),
        parameter_dir=Path(args.parameter_dir),
        audio_feature_type='mel',
        parameter_type='geometric',
        batch_size=args.batch_size,
        num_workers=0,
        sequence_length=500,  # Match training config
        streaming=False,
        zip_file_path=None
    )

    test_loader = dataloaders['test']
    print(f"Test dataset: {len(test_loader.dataset)} samples")

    # Get predictions
    print(f"\nGenerating predictions for {args.num_samples} samples...")
    results = get_predictions(model, test_loader, num_samples=args.num_samples)

    # Create visualizations
    output_dir = Path(args.output_dir)
    print(f"\nCreating visualizations...")
    plot_predictions(results, output_dir)
    plot_summary_statistics(results, output_dir)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    all_correlations = []
    all_maes = []

    for result in results:
        gt = result['ground_truth']
        pred = result['predicted']

        for feat_idx in range(14):
            corr = np.corrcoef(gt[:, feat_idx], pred[:, feat_idx])[0, 1]
            mae = np.mean(np.abs(gt[:, feat_idx] - pred[:, feat_idx]))
            all_correlations.append(corr)
            all_maes.append(mae)

    print(f"Mean Correlation: {np.mean(all_correlations):.4f} ± {np.std(all_correlations):.4f}")
    print(f"Mean MAE: {np.mean(all_maes):.4f} ± {np.std(all_maes):.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
