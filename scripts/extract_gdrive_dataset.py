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
    
    # STRATEGY CHANGE: Use fuse-zip to mount the outer zip without copying
    # This solves the "Disk is almost full" issue caused by caching
    
    print("\n🚀 Strategy: Using fuse-zip to mount dataset (Zero Disk Usage)")
    
    # 1. Install fuse-zip
    print("🔧 Installing fuse-zip...")
    os.system("apt-get update -qq && apt-get install -y fuse-zip")
    
    # 2. Create mount point
    mount_point = "/content/sullivan_outer_zip_mount"
    if os.path.ismount(mount_point):
        print(f"   Unmounting existing {mount_point}...")
        os.system(f"fusermount -u {mount_point}")
    os.makedirs(mount_point, exist_ok=True)
    
    # 3. Mount the OUTER zip
    print(f"🔗 Mounting {zip_file} -> {mount_point}")
    # -r: read only, which is safer and faster
    # -o allow_other: required for access
    ret = os.system(f"fuse-zip -r '{zip_file}' {mount_point}")
    
    if ret != 0:
        print("❌ fuse-zip mount failed. Please check if the file is a valid zip.")
        return None
        
    print("✅ Mount successful!")
    
    # 4. Check contents
    try:
        mounted_files = os.listdir(mount_point)
    except OSError:
        print("❌ Cannot list mounted directory. Mount might have failed silently.")
        return None
        
    target_zip = None
    
    if 'dataset.zip' in mounted_files:
        target_zip = os.path.join(mount_point, 'dataset.zip')
        print(f"🎯 Found inner dataset: {target_zip}")
    else:
        # Fallback logic if names are different
        zips = [f for f in mounted_files if f.endswith('.zip') and 'dataset' in f]
        if zips:
            target_zip = os.path.join(mount_point, zips[0])
            print(f"🎯 Found candidate dataset: {target_zip}")
            
    if not target_zip:
        print("❌ Could not find 'dataset.zip' inside the mounted file.")
        print(f"   Contents: {mounted_files}")
        return None
        
    # 5. Configure Training to use this VIRTUAL path
    import yaml
    config_path = 'configs/colab_gdrive_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    config['data']['zip_file_path'] = str(target_zip)
    
    # Auto-generate splits from the INNER zip if needed
    print("📂 Checking metadata in inner zip...")
    extract_to = "/content/sullivan_data"
    os.makedirs(extract_to, exist_ok=True)
    
    try:
        # Access the inner zip file directly via the virtual mount
        with zipfile.ZipFile(target_zip, 'r') as z:
             # Look for splits folder or generate
             splits_members = [m for m in z.namelist() if m.startswith('splits/')]
             if splits_members:
                 z.extractall(extract_to, members=splits_members)
                 print("✅ Splits extracted.")
             else:
                 print("⚠️  Generating splits from inner zip contents...")
                 # List zip contents to find audio feature files
                 all_files = z.namelist()
                 
                 import re
                 utterance_names = set()
                 for f in all_files:
                    match = re.search(r'([^/]+)_mel\.npy$', f)
                    if match: utterance_names.add(match.group(1))
                    
                 if not utterance_names:
                     # Check params
                     for f in all_files:
                        match = re.search(r'([^/]+)_params\.npy$', f)
                        if match: utterance_names.add(match.group(1))

                 if utterance_names:
                     utterance_list = sorted(list(utterance_names))
                     n = len(utterance_list)
                     train_end = int(n * 0.8)
                     val_end = int(n * 0.9)
                     
                     splits_base = os.path.join(extract_to, 'splits')
                     for split_name, uttrs in [('train', utterance_list[:train_end]), 
                                              ('val', utterance_list[train_end:val_end]), 
                                              ('test', utterance_list[val_end:])]:
                         split_dir = os.path.join(splits_base, split_name)
                         os.makedirs(split_dir, exist_ok=True)
                         with open(os.path.join(split_dir, 'utterance_list.txt'), 'w') as f:
                             f.write('\n'.join(uttrs))
                     print(f"✅ Auto-generated splits from inner zip! ({len(utterance_names)} items)")
                 else:
                     print("❌ Still no utterances found in inner zip.")
                     print("Sample files:", all_files[:20])

    except Exception as e:
        print(f"⚠️ Error reading inner zip metadata: {e}") 
        print("Note: This might be due to fuse-zip latency.")

    with open(config_path, 'w') as f:
        yaml.dump(config, f)
        
    print(f"✅ Config updated to use virtual path: {target_zip}")
    print("✅ Ready for training (Zero Disk Usage Mode)")
    
    return "/content/drive/MyDrive/Project_Sullivan"

if __name__ == "__main__":
    data_dir = extract_dataset_from_gdrive()
    if data_dir:
        print(f"\n🎯 Data ready at: {data_dir}")
        print("\nNext: Run training script")
