import sys
import os
from pathlib import Path
import json
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path.cwd()))

from src.modeling.transformer import TransformerModel
from src.modeling.dataset import create_dataloaders

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def compute_metrics(predictions, targets, feature_names):
    """
    Compute RMSE and PCC per feature.
    predictions: (N, 14)
    targets: (N, 14)
    """
    metrics = []
    
    for i, feature_name in enumerate(feature_names):
        pred_feat = predictions[:, i]
        target_feat = targets[:, i]
        
        # RMSE
        rmse = np.sqrt(mean_squared_error(target_feat, pred_feat))
        
        # Pearson Correlation
        if np.std(pred_feat) < 1e-6 or np.std(target_feat) < 1e-6:
            pcc = 0.0
        else:
            pcc, _ = pearsonr(pred_feat, target_feat)
            
        metrics.append({
            'Feature': feature_name,
            'RMSE': rmse,
            'PCC': pcc
        })
        
    return pd.DataFrame(metrics)

def evaluate_model(config_path, model_path, output_dir):
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    ensure_dir(output_dir)
    
    # Load Model
    print(f"Loading model from {model_path}")
    device = torch.device('cpu') # Evaluate on CPU for simplicity/stability
    try:
        model = TransformerModel.load_from_checkpoint(model_path, map_location=device)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Create Dataloaders
    print("Creating dataloaders...")
    loaders = create_dataloaders(
        Path(config['data']['splits_dir']),
        Path(config['data']['audio_feature_dir']),
        Path(config['data']['parameter_dir']),
        sequence_length=config['data']['sequence_length'],
        streaming=config['data']['streaming'],
        batch_size=1, # Process one by one for cleaner evaluation
        num_workers=0
    )
    
    test_loader = loaders['test']
    
    # Feature names (Geometric - Hardcoded based on GeometricFeatureExtractor)
    feature_names = [
        'tongue_area', 'tongue_centroid_x', 'tongue_centroid_y', 'tongue_tip_y', 
        'tongue_dorsum_height', 'tongue_width', 'jaw_area', 'jaw_centroid_y', 
        'jaw_opening', 'lip_area', 'lip_centroid_y', 'lip_aperture', 
        'constriction_degree', 'constriction_location_y'
    ]
    
    all_preds = []
    all_targets = []
    
    # Store 3 sample trajectories for visualization
    sample_plots = []
    samples_collected = 0
    
    print("Running inference on Test Set...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            audio, targets, lengths, utterance_names = batch
            audio = audio.to(device)
            targets = targets.to(device)
            
            # Forward pass
            preds = model(audio)
            
            # Denormalize
            # Access dataset from loader to get denormalization stats
            preds_denorm = test_loader.dataset.denormalize_parameters(preds)
            targets_denorm = test_loader.dataset.denormalize_parameters(targets)
            
            # Convert to numpy
            preds_np = preds_denorm.squeeze(0).cpu().numpy()
            targets_np = targets_denorm.squeeze(0).cpu().numpy()
            
            # Store for global metrics
            all_preds.append(preds_np)
            all_targets.append(targets_np)
            
            # Collect Samples for Visualization
            if samples_collected < 3:
                sample_data = {
                    'utterance': utterance_names[0],
                    'preds': preds_np,
                    'targets': targets_np
                }
                sample_plots.append(sample_data)
                samples_collected += 1
    
    # Concatenate all
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    print(f"Total frames evaluated: {all_preds.shape[0]}")
    
    # --- 1. Per-Parameter Analysis ---
    metrics_df = compute_metrics(all_preds, all_targets, feature_names)
    print("\nPer-Parameter Metrics:")
    print(metrics_df)
    metrics_df.to_csv(Path(output_dir) / 'per_parameter_metrics.csv', index=False)
    
    # --- 2. Global Metrics ---
    global_rmse = np.mean(metrics_df['RMSE'])
    global_pcc = np.mean(metrics_df['PCC'])
    
    summary = {
        'Global RMSE': global_rmse,
        'Global PCC': global_pcc,
        'Model Path': str(model_path)
    }
    with open(Path(output_dir) / 'summary_metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)
        
    print("\nGlobal Summary:")
    print(summary)
    
    # --- 3. Visual Synthesis ---
    print("\nGenerating Visualizations...")
    for i, sample in enumerate(sample_plots):
        utt_name = sample['utterance']
        p = sample['preds'][:100, :] # First 100 frames
        t = sample['targets'][:100, :]
        
        # Save raw CSV
        df_sample = pd.DataFrame(p, columns=[f"Pred_{n}" for n in feature_names])
        df_target_sample = pd.DataFrame(t, columns=[f"Target_{n}" for n in feature_names])
        df_combined = pd.concat([df_sample, df_target_sample], axis=1)
        df_combined.to_csv(Path(output_dir) / f'sample_{i}_{utt_name}.csv', index=False)
        
        # Plot - Select 4 key features to plot to keep it readable
        # Tongue Centroid Y (2), Tongue Dorsum Height (4), Constriction Location Y (13), Tongue Area (0)
        key_indices = [2, 4, 13, 0] 
        key_names = ['Tongue Centroid Y', 'Tongue Dorsum Height', 'Constriction Location Y', 'Tongue Area']
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        fig.suptitle(f"Prediction vs Ground Truth: {utt_name}", fontsize=16)
        
        for ax_idx, (feat_idx, feat_name) in enumerate(zip(key_indices, key_names)):
            ax = axes[ax_idx]
            ax.plot(t[:, feat_idx], label='Ground Truth', color='black', linewidth=2, alpha=0.7)
            ax.plot(p[:, feat_idx], label='Prediction', color='red', linestyle='--', linewidth=2)
            ax.set_ylabel(feat_name)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
        axes[-1].set_xlabel("Frame Index")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(Path(output_dir) / f'plot_{i}_{utt_name}.png')
        plt.close()
        
    print(f"Evaluation complete. Results saved to {output_dir}")

if __name__ == "__main__":
    # Hardcoded best model based on previous steps
    best_model = "Project_Sullivan_Final_Transformer.ckpt"
    config = "configs/transformer_config.yaml"
    output = "results/final_deliverable"
    
    evaluate_model(config, best_model, output)
