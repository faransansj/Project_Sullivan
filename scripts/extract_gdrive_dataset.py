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
    print("\n1️⃣ Mounting Google Drive...")
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Check if zip exists
    if not os.path.exists(zip_file):
        print(f"\n❌ Dataset not found: {zip_file}")
        print("Please check Google Drive Project_Sullivan folder.")
        return None
    
    file_size = os.path.getsize(zip_file) / (1024**3)
    print(f"✅ Found Dataset: {file_size:.2f} GB")
    
    # Extract
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
