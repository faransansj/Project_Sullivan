#!/usr/bin/env python3
"""
Evaluate trained U-Net on test set and generate visualizations.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.segmentation.dataset import SegmentationDatasetSplit, create_train_val_test_splits
from src.segmentation.unet import UNet
from src.utils.logger import get_logger, setup_logger
from src.utils.io_utils import ensure_directory

# Setup logging
setup_logger(level="INFO", console=True)
logger = get_logger(__name__)


def load_model(model_path: Path, device: str = 'cpu') -> UNet:
    """Load trained U-Net model."""
    model = UNet(n_classes=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    logger.info(f"Loaded model from {model_path}")
    return model


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> Dict:
    """Compute segmentation metrics."""
    metrics = {}

    # Per-class Dice scores
    dice_scores = []
    iou_scores = []

    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
        iou = (intersection + 1e-7) / (pred_c.sum() + target_c.sum() - intersection + 1e-7)

        dice_scores.append(dice.item())
        iou_scores.append(iou.item())

    metrics['dice_per_class'] = dice_scores
    metrics['iou_per_class'] = iou_scores
    metrics['dice_mean'] = np.mean(dice_scores)
    metrics['iou_mean'] = np.mean(iou_scores)

    # Pixel accuracy
    metrics['pixel_accuracy'] = (pred == target).float().mean().item()

    return metrics


def evaluate_dataset(model: UNet, dataloader: DataLoader, device: str = 'cpu') -> Dict:
    """Evaluate model on a dataset."""
    all_metrics = []

    with torch.no_grad():
        for images, masks, metadata in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            # Compute metrics for each sample
            for i in range(images.size(0)):
                metrics = compute_metrics(preds[i], masks[i], num_classes=4)
                metrics['utterance_name'] = metadata['utterance_name'][i]
                metrics['frame_index'] = metadata['frame_index'][i]
                all_metrics.append(metrics)

    # Aggregate metrics
    aggregated = {
        'dice_mean': float(np.mean([m['dice_mean'] for m in all_metrics])),
        'iou_mean': float(np.mean([m['iou_mean'] for m in all_metrics])),
        'pixel_accuracy': float(np.mean([m['pixel_accuracy'] for m in all_metrics])),
        'dice_per_class': np.mean([m['dice_per_class'] for m in all_metrics], axis=0).tolist(),
        'iou_per_class': np.mean([m['iou_per_class'] for m in all_metrics], axis=0).tolist(),
        # Note: per_sample_metrics omitted for JSON serialization
    }

    return aggregated


def visualize_predictions(
    model: UNet,
    dataset: SegmentationDatasetSplit,
    output_dir: Path,
    num_samples: int = 10,
    device: str = 'cpu'
):
    """Generate visualization of model predictions."""
    ensure_directory(output_dir)

    class_names = ['Background/Air', 'Tongue', 'Jaw/Palate', 'Lips']
    colors = plt.cm.tab10(np.linspace(0, 1, 10))[:4]

    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)

    for idx in indices:
        image, mask, metadata = dataset[idx]

        # Predict
        with torch.no_grad():
            image_input = image.unsqueeze(0).to(device)
            logits = model(image_input)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu()

        # Create visualization
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # Original image
        axes[0].imshow(image.squeeze(0), cmap='gray')
        axes[0].set_title('MRI Frame')
        axes[0].axis('off')

        # Ground truth
        axes[1].imshow(mask, cmap='tab10', vmin=0, vmax=9)
        axes[1].set_title('Ground Truth (Pseudo-label)')
        axes[1].axis('off')

        # Prediction
        axes[2].imshow(pred, cmap='tab10', vmin=0, vmax=9)
        axes[2].set_title('U-Net Prediction')
        axes[2].axis('off')

        # Overlay
        axes[3].imshow(image.squeeze(0), cmap='gray')
        axes[3].imshow(pred, cmap='tab10', alpha=0.5, vmin=0, vmax=9)
        axes[3].set_title('Overlay')
        axes[3].axis('off')

        # Add metrics
        metrics = compute_metrics(pred, mask, num_classes=4)
        fig.suptitle(
            f"{metadata['utterance_name']} - Frame {metadata['frame_index']}\n"
            f"Dice: {metrics['dice_mean']:.3f} | IoU: {metrics['iou_mean']:.3f} | "
            f"Acc: {metrics['pixel_accuracy']:.3f}",
            fontsize=12, fontweight='bold'
        )

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors[i], label=class_names[i])
            for i in range(4)
        ]
        axes[3].legend(handles=legend_elements, loc='center left',
                      bbox_to_anchor=(1, 0.5), fontsize=10)

        plt.tight_layout()

        save_path = output_dir / f"pred_{metadata['utterance_name']}_frame{metadata['frame_index']:04d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved visualization: {save_path.name}")


def plot_training_curves(metrics_csv: Path, output_dir: Path):
    """Plot training curves from metrics CSV."""
    import pandas as pd

    ensure_directory(output_dir)

    # Load metrics
    df = pd.read_csv(metrics_csv)

    # Filter out rows without epoch info
    df = df.dropna(subset=['epoch'])

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Loss
    ax = axes[0, 0]
    train_loss = df[df['train/loss'].notna()]
    val_loss = df[df['val/loss'].notna()]
    ax.plot(train_loss['epoch'], train_loss['train/loss'], label='Train', linewidth=2)
    ax.plot(val_loss['epoch'], val_loss['val/loss'], label='Validation', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 2: Dice Score
    ax = axes[0, 1]
    train_dice = df[df['train/dice_mean'].notna()]
    val_dice = df[df['val/dice_mean'].notna()]
    ax.plot(train_dice['epoch'], train_dice['train/dice_mean'], label='Train', linewidth=2)
    ax.plot(val_dice['epoch'], val_dice['val/dice_mean'], label='Validation', linewidth=2)
    ax.axhline(y=0.70, color='r', linestyle='--', label='Target (70%)', linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Dice Score', fontsize=12)
    ax.set_title('Dice Score Progression', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 3: IoU
    ax = axes[1, 0]
    val_iou = df[df['val/iou_mean'].notna()]
    ax.plot(val_iou['epoch'], val_iou['val/iou_mean'], linewidth=2, color='green')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('IoU Score', fontsize=12)
    ax.set_title('Validation IoU Score', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 4: Per-class Dice scores
    ax = axes[1, 1]
    class_names = ['Background', 'Tongue', 'Jaw', 'Lips']
    class_cols = ['val/dice_background', 'val/dice_tongue', 'val/dice_jaw', 'val/dice_lips']

    for i, (name, col) in enumerate(zip(class_names, class_cols)):
        class_data = df[df[col].notna()]
        if not class_data.empty:
            ax.plot(class_data['epoch'], class_data[col], label=name, linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Dice Score', fontsize=12)
    ax.set_title('Per-Class Dice Scores', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / 'training_curves.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved training curves: {save_path}")


def main():
    """Main evaluation function."""
    logger.info("=" * 70)
    logger.info("U-NET MODEL EVALUATION")
    logger.info("=" * 70)

    # Paths
    model_path = Path("models/unet_scratch/unet_final.pth")
    pseudo_labels_dir = Path("data/processed/pseudo_labels")
    output_dir = Path("results/unet_evaluation")
    metrics_csv = Path("models/unet_scratch/logs/unet_training/version_2/metrics.csv")

    ensure_directory(output_dir)

    # Load model
    logger.info("\nLoading trained model...")
    device = 'cpu'
    model = load_model(model_path, device=device)

    # Create datasets
    logger.info("\nCreating datasets...")
    train_paths, val_paths, test_paths = create_train_val_test_splits(
        pseudo_labels_dir, val_ratio=0.15, test_ratio=0.15, random_seed=42
    )

    test_dataset = SegmentationDatasetSplit(test_paths, augment=False)
    val_dataset = SegmentationDatasetSplit(val_paths, augment=False)

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    # Evaluate on test set
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATING ON TEST SET")
    logger.info("=" * 70)

    test_metrics = evaluate_dataset(model, test_loader, device=device)

    logger.info(f"\nTest Set Results:")
    logger.info(f"  Mean Dice Score: {test_metrics['dice_mean']:.4f}")
    logger.info(f"  Mean IoU Score: {test_metrics['iou_mean']:.4f}")
    logger.info(f"  Pixel Accuracy: {test_metrics['pixel_accuracy']:.4f}")
    logger.info(f"\nPer-Class Dice Scores:")
    class_names = ['Background/Air', 'Tongue', 'Jaw/Palate', 'Lips']
    for i, name in enumerate(class_names):
        logger.info(f"  {name}: {test_metrics['dice_per_class'][i]:.4f}")

    # Evaluate on validation set (for comparison)
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATING ON VALIDATION SET")
    logger.info("=" * 70)

    val_metrics = evaluate_dataset(model, val_loader, device=device)

    logger.info(f"\nValidation Set Results:")
    logger.info(f"  Mean Dice Score: {val_metrics['dice_mean']:.4f}")
    logger.info(f"  Mean IoU Score: {val_metrics['iou_mean']:.4f}")
    logger.info(f"  Pixel Accuracy: {val_metrics['pixel_accuracy']:.4f}")

    # Save metrics
    logger.info("\n" + "=" * 70)
    logger.info("SAVING EVALUATION RESULTS")
    logger.info("=" * 70)

    results = {
        'test_set': test_metrics,
        'validation_set': val_metrics,
        'model_path': str(model_path),
        'num_test_samples': len(test_dataset),
        'num_val_samples': len(val_dataset),
    }

    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")

    # Generate visualizations
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 70)

    logger.info("\nGenerating prediction visualizations...")
    vis_dir = output_dir / 'predictions'
    visualize_predictions(model, test_dataset, vis_dir, num_samples=10, device=device)

    logger.info("\nGenerating training curves...")
    if metrics_csv.exists():
        plot_training_curves(metrics_csv, output_dir)
    else:
        logger.warning(f"Metrics CSV not found: {metrics_csv}")

    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"  - Evaluation metrics: {results_path}")
    logger.info(f"  - Prediction visualizations: {vis_dir}")
    logger.info(f"  - Training curves: {output_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()
