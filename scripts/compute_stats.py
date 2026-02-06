#!/usr/bin/env python3
"""
Compute Normalization Statistics for Geometric Parameters

This script calculates the min, max, mean, and std of all geometric parameter files
in data/processed/parameters/geometric/ and saves them to a JSON file.
"""

import os
import glob
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

def compute_stats(input_dir, output_file):
    print(f"Searching for .npy files in {input_dir}...")
    files = glob.glob(os.path.join(input_dir, "*.npy"))
    
    if not files:
        print("No .npy files found!")
        return

    print(f"Found {len(files)} files. Computing statistics...")

    # Initialize online statistics computation
    # We'll use a simple approach: accumulate all data (if memory allows) 
    # or iterate to compute min/max/mean/std incrementally.
    # Given the number of files, let's try to do it incrementally to be safe.
    
    all_min = None
    all_max = None
    total_sum = None
    total_sq_sum = None
    total_count = 0
    
    # First pass: Min, Max, Sum, SumSq
    for fpath in tqdm(files):
        data = np.load(fpath)
        # data shape: (T, 14)
        
        if all_min is None:
            all_min = np.min(data, axis=0)
            all_max = np.max(data, axis=0)
            total_sum = np.sum(data, axis=0)
            total_sq_sum = np.sum(data**2, axis=0)
            total_count = data.shape[0]
        else:
            all_min = np.minimum(all_min, np.min(data, axis=0))
            all_max = np.maximum(all_max, np.max(data, axis=0))
            total_sum += np.sum(data, axis=0)
            total_sq_sum += np.sum(data**2, axis=0)
            total_count += data.shape[0]

    # Calculate Mean and Std
    mean = total_sum / total_count
    # Variance = E[X^2] - (E[X])^2
    variance = (total_sq_sum / total_count) - (mean ** 2)
    std = np.sqrt(np.maximum(variance, 0)) # Clip to 0 to avoid negative due to precision

    stats = {
        "min": all_min.tolist(),
        "max": all_max.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "count": int(total_count)
    }

    print("Statistics computed.")
    print(f"Total frames: {total_count}")
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"Stats saved to {output_file}")

if __name__ == "__main__":
    INPUT_DIR = "data/processed/parameters/geometric"
    OUTPUT_FILE = "data/processed/stats_geometric.json"
    
    # Ensure directory exists
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    
    compute_stats(INPUT_DIR, OUTPUT_FILE)
