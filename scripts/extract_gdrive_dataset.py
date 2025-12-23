#!/usr/bin/env python3
"""
Quick Start - Extract Dataset from Google Drive and Start Training

This script automatically:
1. Checks if Dataset.zip exists in Google Drive
2. Extracts to local Colab storage
3. Sets up paths for training
"""

import os
import zipfile
from pathlib import Path

def extract_dataset_from_gdrive():
    """Extract Dataset.zip from Google Drive Project_Sullivan folder."""
    
    # Paths
    gdrive_base = "/content/drive/MyDrive"
    source_dir = f"{gdrive_base}/Project_Sullivan"
    zip_file = f"{source_dir}/Dataset"
    extract_to = "/content/sullivan_data"
    
    print("=" * 60)
    print("📦 Dataset Extraction")
    print("=" * 60)
    
    # Mount Google Drive
    if os.path.exists('/content/drive/MyDrive'):
        print("\n1️⃣ Google Drive already mounted.")
    else:
        print("\n1️⃣ Mounting Google Drive...")
        from google.colab import drive
        drive.mount('/content/drive')
    
    # Check if zip exists
    if not os.path.exists(zip_file):
        print(f"\n❌ Dataset not found: {zip_file}")
        print("Please check Google Drive Project_Sullivan folder.")
        return None
    
    # If the file is too large (>50GB), avoid extraction and use Zip Streaming
    file_size_gb = os.path.getsize(zip_file) / (1024**3)
    print(f"✅ Found Dataset: {file_size_gb:.2f} GB")
    
    # Check for NESTED dataset.zip (common issue)
    with zipfile.ZipFile(zip_file, 'r') as z:
        all_files = z.namelist()
        if 'dataset.zip' in all_files:
            print("\n⚠️  Detected nested 'dataset.zip' inside the main file!")
            print("   The file you pointed to seems to be a collection of datasets.")
            
            inner_zip_info = z.getinfo('dataset.zip')
            inner_size_gb = inner_zip_info.file_size / (1024**3)
            print(f"   Target 'dataset.zip' size: {inner_size_gb:.2f} GB")
            
            target_ready_zip = f"{source_dir}/dataset_ready.zip"
            
            if os.path.exists(target_ready_zip):
                 if os.path.getsize(target_ready_zip) == inner_zip_info.file_size:
                    print(f"✅ Found already extracted inner zip: {target_ready_zip}")
                    zip_file = target_ready_zip
                 else:
                    print("⚠️  Existing target zip size mismatch. Re-extracting...")
                    zip_file = None # Trigger extraction
            else:
                 zip_file = None # Trigger extraction
                 
            if zip_file is None:
                print(f"⏳ Extracting inner 'dataset.zip' to Google Drive ({target_ready_zip})...")
                print("   This is a one-time process. Please wait...")
                
                # Stream copy to drive to avoid local disk fill
                with z.open('dataset.zip') as source, open(target_ready_zip, 'wb') as target:
                    import shutil
                    shutil.copyfileobj(source, target)
                
                print("✅ Extraction complete.")
                zip_file = target_ready_zip
                
    # Refresh file size for the new target
    file_size_gb = os.path.getsize(zip_file) / (1024**3)
    print(f"🎯 Using Training Dataset: {zip_file} ({file_size_gb:.2f} GB)")
    
    if file_size_gb > 50:
        print("\n⚠️  Dataset is too large for local extraction (>50GB).")
        print("🔄 Configuring for Zip Streaming Mode...")
        
        # We return a dict or special indicator. 
        # But this script is main entry.
        # We should update the config file to point to this zip!
        
        # Read config
        import yaml
        config_path = 'configs/colab_gdrive_config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Update config
        config['data']['zip_file_path'] = str(zip_file)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
            
        print(f"✅ Config updated with zip_file_path: {zip_file}")
        
        # EXTRACT ESSENTIALS: We need 'splits' to exist locally for the training script
        print("📂 Extracting essential metadata (splits)...")
        with zipfile.ZipFile(zip_file, 'r') as z:
            # Look for members starting with 'splits/'
            splits_members = [m for m in z.namelist() if m.startswith('splits/')]
            if not splits_members:
                # Try with 'Dataset/splits/' if it's nested
                splits_members = [m for m in z.namelist() if 'splits/' in m]
            
            if splits_members:
                z.extractall(extract_to, members=splits_members)
                print(f"✅ Essential metadata extracted to {extract_to}")
            else:
                print("⚠️  'splits/' folder not found in zip. Auto-generating from contents...")
                # List zip contents to find audio feature files
                all_files = z.namelist()
                
                # Look for patterns like 'audio_features/mel_spectrogram/XXX_mel.npy'
                # or 'XXX/audio_features/mel_spectrogram/XXX_mel.npy'
                import re
                utterance_names = set()
                for f in all_files:
                    # Match patterns ending in _mel.npy or _mfcc.npy
                    match = re.search(r'([^/]+)_mel\.npy$', f)
                    if match:
                        utterance_names.add(match.group(1))
                    match = re.search(r'([^/]+)_mfcc\.npy$', f)
                    if match:
                        utterance_names.add(match.group(1))
                
                if not utterance_names:
                    # Try another pattern: _params.npy
                    for f in all_files:
                        match = re.search(r'([^/]+)_params\.npy$', f)
                        if match:
                            utterance_names.add(match.group(1))
                
                if utterance_names:
                    print(f"   Found {len(utterance_names)} utterances. Creating splits...")
                    utterance_list = sorted(list(utterance_names))
                    
                    # 80/10/10 split
                    n = len(utterance_list)
                    train_end = int(n * 0.8)
                    val_end = int(n * 0.9)
                    
                    train_uttrs = utterance_list[:train_end]
                    val_uttrs = utterance_list[train_end:val_end]
                    test_uttrs = utterance_list[val_end:]
                    
                    # Create splits directory structure
                    splits_base = os.path.join(extract_to, 'splits')
                    for split_name, uttrs in [('train', train_uttrs), ('val', val_uttrs), ('test', test_uttrs)]:
                        split_dir = os.path.join(splits_base, split_name)
                        os.makedirs(split_dir, exist_ok=True)
                        with open(os.path.join(split_dir, 'utterance_list.txt'), 'w') as f:
                            f.write('\n'.join(uttrs))
                    
                    print(f"✅ Auto-generated splits: train={len(train_uttrs)}, val={len(val_uttrs)}, test={len(test_uttrs)}")
                else:
                    print("❌ Could not find any utterance files in the zip.")
                    print("   Please check the zip structure or create splits manually.")
                    print("\n🔍 Sample files in zip (first 30):")
                    for i, f in enumerate(all_files[:30]):
                        print(f"   {i+1}. {f}")
                    if len(all_files) > 30:
                        print(f"   ... and {len(all_files) - 30} more files")
                
        print("✅ You can now run training with --streaming")
        return "/content/drive/MyDrive/Project_Sullivan" # Return mount point
        
    print(f"\n2️⃣ Extracting to: {extract_to}")
    os.makedirs(extract_to, exist_ok=True)
    
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print("✅ Extraction complete!")
    
    # List extracted contents
    print(f"\n📁 Extracted contents:")
    for item in os.listdir(extract_to):
        item_path = os.path.join(extract_to, item)
        if os.path.isdir(item_path):
            num_files = len(list(Path(item_path).rglob('*')))
            print(f"   📂 {item}/ ({num_files} files)")
        else:
            size_mb = os.path.getsize(item_path) / (1024**2)
            print(f"   📄 {item} ({size_mb:.1f} MB)")
    
    print("\n" + "=" * 60)
    return extract_to

if __name__ == "__main__":
    data_dir = extract_dataset_from_gdrive()
    if data_dir:
        print(f"\n🎯 Data ready at: {data_dir}")
        print("\nNext: Run training script")
