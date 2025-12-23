#!/usr/bin/env python3
"""
Dataset Sharding Script for Large Datasets

Splits a large dataset into smaller shards for incremental training on Colab.
Each shard can be trained independently and checkpoints merged later.

Usage:
    python scripts/shard_dataset.py /path/to/data 10 --output /path/to/shards
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import List
import numpy as np


def get_file_list(data_dir: Path, extensions: List[str] = None) -> List[Path]:
    """Get all data files in directory."""
    if extensions is None:
        extensions = ['.npy', '.npz', '.h5', '.hdf5']
    
    files = []
    for ext in extensions:
        files.extend(data_dir.rglob(f'*{ext}'))
    
    return sorted(files)


def create_shards(
    data_dir: Path,
    num_shards: int,
    output_dir: Path,
    copy_files: bool = False
) -> None:
    """
    Create data shards by splitting file list.
    
    Args:
        data_dir: Source data directory
        num_shards: Number of shards to create
        output_dir: Output directory for shard manifests
        copy_files: If True, copy files to shard directories (uses more space)
    """
    print(f"📦 Creating {num_shards} shards from: {data_dir}")
    
    # Get all data files
    files = get_file_list(data_dir)
    print(f"   Found {len(files)} files")
    
    if len(files) == 0:
        print("❌ No data files found!")
        return
    
    # Calculate shard size
    shard_size = len(files) // num_shards
    remainder = len(files) % num_shards
    
    print(f"   Shard size: ~{shard_size} files each")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create shards
    start_idx = 0
    shard_info = []
    
    for shard_id in range(num_shards):
        # Calculate shard range
        extra = 1 if shard_id < remainder else 0
        end_idx = start_idx + shard_size + extra
        
        shard_files = files[start_idx:end_idx]
        
        # Create shard manifest
        shard_manifest = {
            'shard_id': shard_id,
            'num_files': len(shard_files),
            'files': [str(f.relative_to(data_dir)) for f in shard_files]
        }
        
        # Save manifest
        manifest_path = output_dir / f'shard_{shard_id:03d}.json'
        with open(manifest_path, 'w') as f:
            json.dump(shard_manifest, f, indent=2)
        
        print(f"   Shard {shard_id}: {len(shard_files)} files -> {manifest_path}")
        
        # Optionally copy files
        if copy_files:
            shard_dir = output_dir / f'shard_{shard_id:03d}'
            shard_dir.mkdir(exist_ok=True)
            for src_file in shard_files:
                dst_file = shard_dir / src_file.relative_to(data_dir)
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
        
        shard_info.append({
            'shard_id': shard_id,
            'manifest': str(manifest_path),
            'num_files': len(shard_files)
        })
        
        start_idx = end_idx
    
    # Save overall summary
    summary = {
        'source_dir': str(data_dir),
        'num_shards': num_shards,
        'total_files': len(files),
        'shards': shard_info
    }
    
    summary_path = output_dir / 'sharding_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Sharding complete!")
    print(f"   Summary: {summary_path}")
    print(f"   Total files: {len(files)}")
    print(f"   Shards created: {num_shards}")


def main():
    parser = argparse.ArgumentParser(description='Shard large dataset for incremental training')
    parser.add_argument('data_dir', type=Path, help='Source data directory')
    parser.add_argument('num_shards', type=int, help='Number of shards to create')
    parser.add_argument('--output', type=Path, default=None, help='Output directory for shards')
    parser.add_argument('--copy', action='store_true', help='Copy files to shard directories')
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.data_dir.parent / f'{args.data_dir.name}_shards'
    
    create_shards(
        data_dir=args.data_dir,
        num_shards=args.num_shards,
        output_dir=args.output,
        copy_files=args.copy
    )


if __name__ == '__main__':
    main()
