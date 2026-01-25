import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import yaml
from sklearn.decomposition import PCA

# Add project root to path
sys.path.insert(0, str(Path.cwd()))

from src.modeling.transformer import TransformerModel
from src.modeling.dataset import create_dataloaders

def load_pca_model(path):
    """Load PCA model components and mean"""
    data = np.load(path)
    return data['components'], data['mean']

def reconstruct_mask(pca_params, components, mean, shape=(84, 84)):
    """
    Reconstruct mask from PCA parameters.
    """
    flattened = np.dot(pca_params, components) + mean
    return flattened.reshape(shape)

def visualize_comparison(baseline_model_path, baseline_config_path, pca_path, output_dir):
    # Load PCA
    print(f"Loading PCA model from {pca_path}")
    components, mean = load_pca_model(pca_path)
    
    # Load Baseline Model (Phase 4 - 14 dims)
    print(f"Loading Baseline Transformer from {baseline_model_path}")
    device = torch.device('cpu')
    model = TransformerModel.load_from_checkpoint(baseline_model_path, map_location=device)
    model.eval()
    
    # Load Config for Dataset
    # We need a config that loads 'combined' parameters to get the GT PCA
    with open(baseline_config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Create Test Loader
    print("Creating dataloader...")
    loaders = create_dataloaders(
        Path(config['data']['splits_dir']),
        Path(config['data']['audio_feature_dir']),
        Path(config['data']['parameter_dir']),
        parameter_type='combined', # Load 24 dims (14 Geo + 10 PCA)
        sequence_length=None,
        streaming=False,
        batch_size=1,
        num_workers=0
    )
    test_loader = loaders['test']
    
    # Get a sample
    # We want a sample with clear tongue shape.
    # Let's pick sample #10
    sample_idx = 10
    dataset = test_loader.dataset
    audio, targets, utterance_name = dataset[sample_idx]
    
    print(f"Analyzing utterance: {utterance_name}")
    
    # Inference (Baseline Model)
    # Baseline expects audio input.
    audio_input = audio.unsqueeze(0).to(device)
    with torch.no_grad():
        preds_geo = model(audio_input) # Output: (1, T, 14)
        
    # Denormalize
    # Note: Dataset is configured for 'combined' (24 dims), so denormalize might expect 24 dims?
    # No, denormalize_parameters uses self.param_min/max/mean/std.
    # If dataset computed stats on 24 dims, these stats are 24 dims.
    # We need to be careful.
    
    # Check dataset stats shape
    if dataset.normalization_type == 'minmax':
        stats_dim = dataset.param_min.shape[1]
    else:
        stats_dim = dataset.param_mean.shape[1]
        
    print(f"Dataset stats dimension: {stats_dim}")
    
    # Helper to denormalize 14 dims using the first 14 dims of stats
    def denorm_geo(p):
        if dataset.normalization_type == 'minmax':
            min_v = torch.FloatTensor(dataset.param_min[:, :14])
            range_v = torch.FloatTensor(dataset.param_range[:, :14])
            return p * range_v + min_v
        else:
            mean_v = torch.FloatTensor(dataset.param_mean[:, :14])
            std_v = torch.FloatTensor(dataset.param_std[:, :14])
            return p * std_v + mean_v

    # Helper to denormalize 24 dims
    def denorm_combined(p):
        return dataset.denormalize_parameters(p)

    preds_geo_denorm = denorm_geo(preds_geo).squeeze(0).cpu().numpy()
    targets_combined_denorm = denorm_combined(targets.unsqueeze(0)).squeeze(0).cpu().numpy()
    
    # Extract GT PCA (last 10 dims)
    gt_pca = targets_combined_denorm[:, 14:]
    
    # Select a frame (middle)
    frame_idx = preds_geo_denorm.shape[0] // 2
    
    # 1. Reconstruct Mask from GT PCA (The "Target High-Res Shape")
    pca_mask = reconstruct_mask(gt_pca[frame_idx], components, mean)
    
    # 2. Visualize Baseline Prediction (Geometric)
    # Geo params: [area, cx, cy, bbox_top, bbox_bottom, bbox_left, bbox_right, ...]
    pred_g = preds_geo_denorm[frame_idx]
    cx, cy = pred_g[1], pred_g[2]
    # Denormalized coordinates are likely normalized to 0-1 (image relative)?
    # Wait, the extraction script normalized them by image size (W, H).
    # So we multiply by 84.
    
    h, w = 84, 84
    cx_px, cy_px = cx * w, cy * h
    
    bb_top, bb_bottom = pred_g[3] * h, pred_g[4] * h
    bb_left, bb_right = pred_g[5] * w, pred_g[6] * w
    bb_w = bb_right - bb_left
    bb_h = bb_bottom - bb_top
    
    # Plot Comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot 1: Baseline Geometric Prediction
    axes[0].set_xlim(0, 84)
    axes[0].set_ylim(84, 0) # Flip Y
    axes[0].set_aspect('equal')
    axes[0].set_title("Phase 4 Baseline Prediction\n(Geometric Bounding Box + Centroid)")
    
    # Draw BBox
    rect = patches.Rectangle((bb_left, bb_top), bb_w, bb_h, linewidth=2, edgecolor='r', facecolor='none', label='Predicted BBox')
    axes[0].add_patch(rect)
    # Draw Centroid
    axes[0].plot(cx_px, cy_px, 'ro', label='Predicted Centroid')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot 2: Phase 4-B Target (PCA Reconstruction)
    im = axes[1].imshow(pca_mask, cmap='gray')
    axes[1].set_title("Phase 4-B Target\n(PCA Reconstructed Shape)")
    axes[1].plot(cx_px, cy_px, 'rx', label='Baseline Centroid') # Overlay baseline centroid for ref
    axes[1].legend()
    plt.colorbar(im, ax=axes[1])
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(output_dir) / f'phase4b_target_preview_{utterance_name}.png'
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    baseline_model = "models/transformer/final_model.ckpt"
    baseline_config = "configs/transformer_config.yaml" # Use this but override parameter_type in loader
    pca_path = "data/processed/parameters/pca_model.npz"
    output_dir = "results/phase4b_preview"
    
    visualize_comparison(baseline_model, baseline_config, pca_path, output_dir)