import sys
import os
from pathlib import Path
import yaml
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Add project root to path
sys.path.insert(0, str(Path.cwd()))

from src.modeling.transformer import TransformerModel
from src.modeling.dataset import create_dataloaders

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def load_pca_model(path):
    data = np.load(path)
    return data['components'], data['mean']

def reconstruct_mask(pca_params, components, mean, shape=(84, 84)):
    flattened = np.dot(pca_params, components) + mean
    return flattened.reshape(shape)

def compute_metrics(predictions, targets):
    """
    Compute RMSE and PCC per feature.
    """
    num_features = predictions.shape[1]
    metrics = []
    
    for i in range(num_features):
        pred_feat = predictions[:, i]
        target_feat = targets[:, i]
        
        rmse = np.sqrt(mean_squared_error(target_feat, pred_feat))
        
        if np.std(pred_feat) < 1e-6 or np.std(target_feat) < 1e-6:
            pcc = 0.0
        else:
            pcc, _ = pearsonr(pred_feat, target_feat)
            
        metrics.append({'Index': i, 'RMSE': rmse, 'PCC': pcc})
        
    return pd.DataFrame(metrics)

def evaluate_phase4d():
    config_path = 'configs/transformer_phase4d_joint.yaml'
    # Use best checkpoint
    model_path = 'logs/training/transformer_phase4d_joint/version_0/checkpoints/transformer-stage4D-epoch=00-val_loss=12832097697792.0000.ckpt'
    output_dir = 'results/phase4d_evaluation'
    pca_path = 'data/processed/parameters/pca_model.npz'
    
    ensure_dir(output_dir)
    
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loading model from {model_path}")
    device = torch.device('cpu')
    model = TransformerModel.load_from_checkpoint(model_path, map_location=device)
    model.eval()
    
    # Load PCA for reconstruction (needed for animation)
    print(f"Loading PCA from {pca_path}")
    components, pca_mean = load_pca_model(pca_path)
    
    print("Creating dataloaders...")
    loaders = create_dataloaders(
        Path(config['data']['splits_dir']),
        Path(config['data']['audio_feature_dir']),
        Path(config['data']['parameter_dir']),
        parameter_type='combined', # 24 dims
        sequence_length=None,
        streaming=False,
        batch_size=1,
        num_workers=0
    )
    test_loader = loaders['test']
    
    all_preds = []
    all_targets = []
    
    samples_for_animation = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            audio, targets, _, utterance_names = batch
            audio = audio.to(device)
            targets = targets.to(device)
            
            preds = model(audio)
            
            # Denormalize
            preds_denorm = test_loader.dataset.denormalize_parameters(preds)
            targets_denorm = test_loader.dataset.denormalize_parameters(targets)
            
            preds_np = preds_denorm.squeeze(0).cpu().numpy()
            targets_np = targets_denorm.squeeze(0).cpu().numpy()
            
            all_preds.append(preds_np)
            all_targets.append(targets_np)
            
            # Save a few samples for animation selection
            if len(samples_for_animation) < 5:
                samples_for_animation.append({
                    'name': utterance_names[0],
                    'preds': preds_np,
                    'targets': targets_np
                })

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # --- Metrics ---
    metrics_df = compute_metrics(all_preds, all_targets)
    
    # Separate Geometric (0-13) and PCA (14-23)
    geo_metrics = metrics_df[metrics_df['Index'] < 14].copy()
    pca_metrics = metrics_df[metrics_df['Index'] >= 14].copy()
    pca_metrics['Component'] = pca_metrics['Index'] - 14
    
    print("\n" + "="*50)
    print("PHASE 4-D JOINT AUDIT REPORT")
    print("="*50)
    
    print("\nGeometric Features (0-13):")
    print(geo_metrics[['Index', 'PCC', 'RMSE']].to_string(index=False))
    print(f"Avg Geometric PCC: {geo_metrics['PCC'].mean():.4f}")
    
    print("\nPCA Components (14-23):")
    print(pca_metrics[['Component', 'PCC', 'RMSE']].to_string(index=False))
    print(f"Avg PCA PCC: {pca_metrics['PCC'].mean():.4f}")
    
    print("\nOverall Stats:")
    print(f"Global PCC (24 dims): {metrics_df['PCC'].mean():.4f}")
    
    metrics_df.to_csv(Path(output_dir) / 'phase4d_metrics.csv', index=False)
    
    # --- Animation ---
    # Pick the first sample
    sample = samples_for_animation[0]
    name = sample['name']
    p_pca = sample['preds'][:, 14:] # (T, 10)
    t_pca = sample['targets'][:, 14:] # (T, 10)
    
    print(f"\nGenerating Master Animation for {name} ({len(p_pca)} frames)...")
    
    # Limit to first 100 frames
    limit = 100
    p_pca = p_pca[:limit]
    t_pca = t_pca[:limit]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Pre-compute frames
    pred_frames = []
    target_frames = []
    for i in range(len(p_pca)):
        pred_frames.append(reconstruct_mask(p_pca[i], components, pca_mean))
        target_frames.append(reconstruct_mask(t_pca[i], components, pca_mean))
        
    im_target = axes[0].imshow(target_frames[0], cmap='gray', vmin=-0.5, vmax=1.5)
    axes[0].set_title("Ground Truth (PCA)")
    
    im_pred = axes[1].imshow(pred_frames[0], cmap='gray', vmin=-0.5, vmax=1.5)
    axes[1].set_title("Predicted (Phase 4-D Joint)")
    
    def update(frame_idx):
        im_target.set_data(target_frames[frame_idx])
        im_pred.set_data(pred_frames[frame_idx])
        return im_target, im_pred
        
    ani = animation.FuncAnimation(fig, update, frames=len(p_pca), blit=True, interval=50)
    
    gif_path = Path(output_dir) / f'master_animation_{name}.gif'
    ani.save(gif_path, writer='pillow', fps=20)
    print(f"Animation saved to {gif_path}")

if __name__ == "__main__":
    evaluate_phase4d()
