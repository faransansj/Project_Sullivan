import sys
import os
from pathlib import Path
import json
import yaml
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from collections import defaultdict

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

def audit_baseline():
    config_path = 'configs/transformer_config.yaml'
    model_path = 'models/transformer/final_model.ckpt'
    output_dir = 'results/hddb_audit'
    
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    ensure_dir(output_dir)
    
    # Load Model
    print(f"Loading model from {model_path}")
    device = torch.device('cpu')
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
        sequence_length=None, # Use full utterances for evaluation
        streaming=False, # Use standard loading for evaluation
        batch_size=1,
        num_workers=0
    )
    
    test_loader = loaders['test']
    
    # Feature names
    feature_names = [
        'area', 'cx', 'cy', 'bbox_top', 'bbox_bottom', 
        'bbox_left', 'bbox_right', 'aspect_ratio', 'solidity', 
        'extent', 'perimeter', 'circularity', 'ellipse_ratio', 
        'ellipse_angle'
    ]
    
    all_preds = []
    all_targets = []
    
    # For subject-wise analysis
    subject_metrics = defaultdict(lambda: {'preds': [], 'targets': []})
    
    print("Running inference on Test Set...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            audio, targets, lengths, utterance_names = batch
            audio = audio.to(device)
            targets = targets.to(device)
            
            # Forward pass
            preds = model(audio)
            
            # Denormalize
            preds_denorm = test_loader.dataset.denormalize_parameters(preds)
            targets_denorm = test_loader.dataset.denormalize_parameters(targets)
            
            # Convert to numpy
            preds_np = preds_denorm.squeeze(0).cpu().numpy()
            targets_np = targets_denorm.squeeze(0).cpu().numpy()
            
            all_preds.append(preds_np)
            all_targets.append(targets_np)
            
            # Store for subject analysis
            utt_name = utterance_names[0]
            subject_id = utt_name.split('_')[0]
            subject_metrics[subject_id]['preds'].append(preds_np)
            subject_metrics[subject_id]['targets'].append(targets_np)

    # --- 1. Global Metrics (Ranked Table) ---
    all_preds_global = np.concatenate(all_preds, axis=0)
    all_targets_global = np.concatenate(all_targets, axis=0)
    
    metrics_df = compute_metrics(all_preds_global, all_targets_global, feature_names)
    
    # Sort by PCC descending
    metrics_df = metrics_df.sort_values(by='PCC', ascending=False)
    
    print("\n" + "="*50)
    print("METRIC REPORT: Ranked by PCC")
    print("="*50)
    print(metrics_df.to_string(index=False, float_format="%.4f"))
    metrics_df.to_csv(Path(output_dir) / 'global_metrics_ranked.csv', index=False)
    
    # --- 2. Subject Performance ---
    print("\n" + "="*50)
    print("SUBJECT PERFORMANCE: Global PCC per Subject")
    print("="*50)
    
    subj_results = []
    for subj_id, data in subject_metrics.items():
        s_preds = np.concatenate(data['preds'], axis=0)
        s_targets = np.concatenate(data['targets'], axis=0)
        
        # Calculate mean PCC across all features for this subject
        s_metrics = compute_metrics(s_preds, s_targets, feature_names)
        mean_pcc = s_metrics['PCC'].mean()
        mean_rmse = s_metrics['RMSE'].mean()
        
        subj_results.append({
            'Subject': subj_id,
            'Global PCC': mean_pcc,
            'Global RMSE': mean_rmse,
            'Frames': s_preds.shape[0]
        })
        
    subj_df = pd.DataFrame(subj_results).sort_values(by='Global PCC', ascending=False)
    print(subj_df.to_string(index=False, float_format="%.4f"))
    subj_df.to_csv(Path(output_dir) / 'subject_metrics.csv', index=False)
    
    # --- 3. Phase 3 Comparison ---
    # USC-TIMIT Baseline (Phase 3)
    # Global RMSE: 0.051
    # Global PCC: 0.026
    
    current_global_rmse = metrics_df['RMSE'].mean()
    current_global_pcc = metrics_df['PCC'].mean()
    
    print("\n" + "="*50)
    print("COMPARISON WITH PHASE 3 (USC-TIMIT)")
    print("="*50)
    print(f"Phase 3 (USC-TIMIT): RMSE=0.0510, PCC=0.0260")
    print(f"Phase 4 (HDDB)     : RMSE={current_global_rmse:.4f}, PCC={current_global_pcc:.4f}")
    
    diff_pcc = current_global_pcc - 0.026
    diff_rmse = current_global_rmse - 0.051
    
    print(f"Improvement PCC    : {diff_pcc:+.4f}")
    print(f"Improvement RMSE   : {diff_rmse:+.4f} (Lower is better)")
    
    if diff_pcc > 0:
        print("\nCONCLUSION: HDDB training improved correlation!")
    else:
        print("\nCONCLUSION: HDDB training did NOT improve correlation.")

if __name__ == "__main__":
    audit_baseline()
