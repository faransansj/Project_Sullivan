"""
Evaluation metrics and functions for Phase 2 models

Implements evaluation metrics for articulatory parameter prediction:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- Pearson Correlation Coefficient (PCC)
- Per-parameter metrics
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error


def compute_rmse(predictions: np.ndarray,
                targets: np.ndarray,
                mask: Optional[np.ndarray] = None) -> float:
    """
    Compute Root Mean Squared Error

    Args:
        predictions: (num_frames, 10) predicted parameters
        targets: (num_frames, 10) ground truth parameters
        mask: (num_frames,) boolean mask for valid frames

    Returns:
        RMSE value
    """
    if mask is not None:
        predictions = predictions[mask]
        targets = targets[mask]

    rmse = np.sqrt(mean_squared_error(targets, predictions))
    return rmse


def compute_mae(predictions: np.ndarray,
               targets: np.ndarray,
               mask: Optional[np.ndarray] = None) -> float:
    """
    Compute Mean Absolute Error

    Args:
        predictions: (num_frames, 10) predicted parameters
        targets: (num_frames, 10) ground truth parameters
        mask: (num_frames,) boolean mask

    Returns:
        MAE value
    """
    if mask is not None:
        predictions = predictions[mask]
        targets = targets[mask]

    mae = mean_absolute_error(targets, predictions)
    return mae


def compute_pearson_correlation(predictions: np.ndarray,
                                targets: np.ndarray,
                                mask: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
    """
    Compute Pearson Correlation Coefficient

    Args:
        predictions: (num_frames, 10) predicted parameters
        targets: (num_frames, 10) ground truth parameters
        mask: (num_frames,) boolean mask

    Returns:
        (mean_correlation, per_param_correlations) tuple
    """
    if mask is not None:
        predictions = predictions[mask]
        targets = targets[mask]

    num_params = predictions.shape[1]
    correlations = np.zeros(num_params)

    for i in range(num_params):
        pred_param = predictions[:, i].flatten()
        target_param = targets[:, i].flatten()

        # Compute Pearson correlation
        if len(pred_param) > 1 and np.std(pred_param) > 0 and np.std(target_param) > 0:
            corr, _ = pearsonr(pred_param, target_param)
            correlations[i] = corr
        else:
            correlations[i] = 0.0

    mean_corr = np.mean(correlations)
    return mean_corr, correlations


def compute_per_parameter_metrics(predictions: np.ndarray,
                                  targets: np.ndarray,
                                  mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    Compute metrics for each parameter dimension

    Args:
        predictions: (num_frames, 10) predictions
        targets: (num_frames, 10) targets
        mask: (num_frames,) mask

    Returns:
        Dictionary with per-parameter RMSE, MAE, and correlation
    """
    if mask is not None:
        predictions = predictions[mask]
        targets = targets[mask]

    num_params = predictions.shape[1]

    rmse_per_param = np.zeros(num_params)
    mae_per_param = np.zeros(num_params)
    corr_per_param = np.zeros(num_params)

    for i in range(num_params):
        pred = predictions[:, i]
        target = targets[:, i]

        # RMSE
        rmse_per_param[i] = np.sqrt(mean_squared_error(target, pred))

        # MAE
        mae_per_param[i] = mean_absolute_error(target, pred)

        # Correlation
        if len(pred) > 1 and np.std(pred) > 0 and np.std(target) > 0:
            corr, _ = pearsonr(pred, target)
            corr_per_param[i] = corr
        else:
            corr_per_param[i] = 0.0

    return {
        'rmse': rmse_per_param,
        'mae': mae_per_param,
        'correlation': corr_per_param
    }


def evaluate_model(model: torch.nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  device: torch.device,
                  denormalize_fn: Optional[callable] = None) -> Dict[str, any]:
    """
    Evaluate model on a dataset

    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        denormalize_fn: Function to denormalize predictions and targets

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()

    all_predictions = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for batch in dataloader:
            audio = batch['audio'].to(device)
            params = batch['params'].to(device)
            lengths = batch['lengths'].to(device)
            mask = batch['mask']

            # Forward pass
            predictions = model(audio, lengths)

            # Move to CPU and convert to numpy
            predictions = predictions.cpu().numpy()
            params = params.cpu().numpy()
            mask = mask.cpu().numpy()

            all_predictions.append(predictions)
            all_targets.append(params)
            all_masks.append(mask)

    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0)  # (num_samples, time, 10)
    targets = np.concatenate(all_targets, axis=0)
    masks = np.concatenate(all_masks, axis=0)  # (num_samples, time)

    # Flatten to (total_frames, 10)
    batch_size, max_time, param_dim = predictions.shape
    predictions_flat = predictions.reshape(-1, param_dim)
    targets_flat = targets.reshape(-1, param_dim)
    mask_flat = masks.reshape(-1)

    # Denormalize if function provided
    if denormalize_fn is not None:
        predictions_flat = denormalize_fn(predictions_flat)
        targets_flat = denormalize_fn(targets_flat)

    # Compute overall metrics
    rmse = compute_rmse(predictions_flat, targets_flat, mask_flat)
    mae = compute_mae(predictions_flat, targets_flat, mask_flat)
    mean_corr, per_param_corr = compute_pearson_correlation(
        predictions_flat, targets_flat, mask_flat
    )

    # Compute per-parameter metrics
    per_param_metrics = compute_per_parameter_metrics(
        predictions_flat, targets_flat, mask_flat
    )

    # Parameter names
    param_names = [
        'tongue_area', 'tongue_centroid_x', 'tongue_centroid_y',
        'tongue_tip_y', 'tongue_curvature',
        'jaw_height', 'jaw_angle',
        'lip_aperture', 'lip_protrusion',
        'constriction_degree'
    ]

    results = {
        'rmse': float(rmse),
        'mae': float(mae),
        'mean_correlation': float(mean_corr),
        'per_param_correlation': {
            param_names[i]: float(per_param_corr[i])
            for i in range(len(param_names))
        },
        'per_param_rmse': {
            param_names[i]: float(per_param_metrics['rmse'][i])
            for i in range(len(param_names))
        },
        'per_param_mae': {
            param_names[i]: float(per_param_metrics['mae'][i])
            for i in range(len(param_names))
        },
        'num_samples': len(predictions),
        'num_frames': int(mask_flat.sum())
    }

    return results


def print_evaluation_results(results: Dict[str, any], split_name: str = "Test"):
    """
    Print evaluation results in a formatted way

    Args:
        results: Results dictionary from evaluate_model()
        split_name: Name of dataset split
    """
    print("\n" + "=" * 80)
    print(f"{split_name} Set Evaluation Results")
    print("=" * 80)

    print(f"\nOverall Metrics:")
    print(f"  RMSE: {results['rmse']:.4f}")
    print(f"  MAE:  {results['mae']:.4f}")
    print(f"  Mean Pearson Correlation: {results['mean_correlation']:.4f}")

    print(f"\n  Samples: {results['num_samples']}")
    print(f"  Total Frames: {results['num_frames']}")

    print(f"\nPer-Parameter Metrics:")
    print(f"{'Parameter':<25} {'RMSE':<10} {'MAE':<10} {'Correlation':<12}")
    print("-" * 80)

    param_names = list(results['per_param_correlation'].keys())
    for param in param_names:
        rmse = results['per_param_rmse'][param]
        mae = results['per_param_mae'][param]
        corr = results['per_param_correlation'][param]
        print(f"{param:<25} {rmse:<10.4f} {mae:<10.4f} {corr:<12.4f}")

    print("=" * 80)

    # Check if goals are met
    print(f"\nGoal Achievement:")
    print(f"  RMSE < 0.10:        {'✓ PASS' if results['rmse'] < 0.10 else '✗ FAIL'} (Target: M3)")
    print(f"  RMSE < 0.15:        {'✓ PASS' if results['rmse'] < 0.15 else '✗ FAIL'} (Target: M2)")
    print(f"  PCC > 0.70:         {'✓ PASS' if results['mean_correlation'] > 0.70 else '✗ FAIL'} (Target: M3)")
    print(f"  PCC > 0.50:         {'✓ PASS' if results['mean_correlation'] > 0.50 else '✗ FAIL'} (Target: M2)")

    print("=" * 80 + "\n")


def save_evaluation_results(results: Dict[str, any], output_path: str):
    """Save evaluation results to JSON file"""
    import json

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved evaluation results to {output_path}")
