import json
import os
from pathlib import Path
import glob

def generate_split():
    audio_dir = Path("data/processed/audio_features/mel_spectrogram")
    param_dir = Path("data/processed/parameters/geometric")
    
    # Find all audio files
    audio_files = list(audio_dir.glob("*_mel.npy"))
    valid_utterances = []
    
    print(f"Found {len(audio_files)} audio files")
    
    for audio_path in audio_files:
        utterance_name = audio_path.stem.replace("_mel", "")
        
        # Check for corresponding parameter file
        param_path = param_dir / f"{utterance_name}_params.npy"
        
        if param_path.exists():
            valid_utterances.append({
                "utterance_name": utterance_name,
                "audio_feature_path": str(audio_path),
                "parameter_path": str(param_path)
            })
            
    print(f"Found {len(valid_utterances)} valid pairs")
    
    # Split into train/val
    # simple 80/20 split
    split_idx = int(len(valid_utterances) * 0.8)
    train_utts = valid_utterances[:split_idx]
    val_utts = valid_utterances[split_idx:]
    
    # Format for dataset.py (list of dicts or list of strings?)
    # Let's check dataset.py. Usually it expects a list of utterance IDs or dicts.
    # The read file of train.json showed a list of dicts with keys: utterance_name, segmentation, parameters, audio.
    # But wait, dataset.py might be flexible.
    
    # Let's match the train.json format exactly
    # "utterance_name": "sub010_...",
    # "segmentation": "...",
    # "parameters": "...",
    # "audio": "..." (this usually points to npz, but here we have npy splits)
    
    # Actually, the extract scripts created .npy files in subfolders.
    # The legacy train.json pointed to .npz files.
    # I need to check if dataset.py handles .npy in subfolders.
    
    train_data = {
        "split": "train",
        "num_subjects": 1,
        "num_utterances": len(train_utts),
        "subjects": ["sub011"],
        "utterances": train_utts
    }
    
    val_data = {
        "split": "val",
        "num_subjects": 1,
        "num_utterances": len(val_utts),
        "subjects": ["sub011"],
        "utterances": val_utts
    }
    
    os.makedirs("data/processed/splits", exist_ok=True)
    
    with open("data/processed/splits/train_sub011.json", "w") as f:
        json.dump(train_utts, f, indent=2) # Dump list or dict? 
        # train.json was a Dict with "utterances": [List]
        # But some dataloaders expect just the list. 
        # I will save the full dict structure similar to what I saw.
        
    with open("data/processed/splits/train_sub011_full.json", "w") as f:
        json.dump(train_data, f, indent=2)

    with open("data/processed/splits/val_sub011_full.json", "w") as f:
        json.dump(val_data, f, indent=2)
        
    print("Splits saved.")

if __name__ == "__main__":
    generate_split()
