#!/usr/bin/env python3
"""
Generate Pseudo-Labels for U-Net Training
Uses ROI + Adaptive Threshold to create high-quality segmentation masks.
"""

import sys
import os
import numpy as np
import cv2
import json
from pathlib import Path
from tqdm import tqdm
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing.hddb_data_loader import HDDBLoader

def extract_roi(frame, roi_params=None):
    """Extract Region of Interest to focus on vocal tract area."""
    h, w = frame.shape
    
    if roi_params is None:
        roi_params = {
            'top': 0.25,
            'bottom': 0.95,
            'left': 0.15,
            'right': 0.85
        }
    
    y1 = int(h * roi_params['top'])
    y2 = int(h * roi_params['bottom'])
    x1 = int(w * roi_params['left'])
    x2 = int(w * roi_params['right'])
    
    roi_frame = frame[y1:y2, x1:x2]
    roi_coords = (y1, y2, x1, x2)
    
    return roi_frame, roi_coords

def segment_vocal_tract_adaptive(frame):
    """
    Segment vocal tract using Adaptive Threshold.
    This was the best method from testing (33.6% airway ratio).
    """
    # Smooth
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    
    # Adaptive thresholding
    binary_mask = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=2
    )
    
    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    processed = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return processed

def assess_mask_quality(mask):
    """
    Assess segmentation mask quality.
    
    Returns:
        quality_score: 0-100 score (higher is better)
        metrics: Dict with detailed metrics
    """
    total_pixels = mask.size
    airway_pixels = np.sum(mask == 255)
    airway_ratio = airway_pixels / total_pixels
    
    # Connected components analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    num_components = num_labels - 1  # Exclude background
    
    # Find largest component
    if num_components > 0:
        component_sizes = stats[1:, cv2.CC_STAT_AREA]
        largest_component_size = np.max(component_sizes)
        largest_component_ratio = largest_component_size / airway_pixels if airway_pixels > 0 else 0
    else:
        largest_component_ratio = 0
    
    # Quality scoring
    score = 0
    
    # 1. Airway ratio should be in reasonable range (10-40%)
    if 0.10 <= airway_ratio <= 0.40:
        score += 50
    elif 0.05 <= airway_ratio <= 0.50:
        score += 30
    else:
        score += 0
    
    # 2. Number of components (fewer is better, should be < 10)
    if num_components < 5:
        score += 30
    elif num_components < 10:
        score += 20
    elif num_components < 20:
        score += 10
    
    # 3. Largest component dominance (should be > 50% of total airway)
    if largest_component_ratio > 0.7:
        score += 20
    elif largest_component_ratio > 0.5:
        score += 10
    
    metrics = {
        'airway_ratio': float(airway_ratio),
        'num_components': int(num_components),
        'largest_component_ratio': float(largest_component_ratio),
        'quality_score': score
    }
    
    return score, metrics

def select_frames(mri_data, num_frames, strategy='distributed'):
    """
    Select frames from MRI sequence.
    
    Args:
        mri_data: (T, H, W) MRI frames
        num_frames: Number of frames to select
        strategy: 'distributed', 'random', or 'middle'
    
    Returns:
        List of frame indices
    """
    total_frames = len(mri_data)
    
    if strategy == 'distributed':
        # Evenly distributed throughout sequence
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    elif strategy == 'random':
        # Random selection
        indices = np.random.choice(total_frames, num_frames, replace=False)
        indices = np.sort(indices)
    elif strategy == 'middle':
        # Focus on middle portion (skip beginning/end)
        start = total_frames // 4
        end = 3 * total_frames // 4
        indices = np.linspace(start, end, num_frames, dtype=int)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return indices.tolist()

def generate_pseudo_labels(
    data_root,
    output_dir,
    num_subjects=5,
    frames_per_subject=40,
    min_quality_score=50,
    frame_strategy='distributed'
):
    """
    Generate pseudo-labels for U-Net training.
    
    Args:
        data_root: Path to HDDB dataset
        output_dir: Output directory for pseudo-labels
        num_subjects: Number of subjects to use (default: 5 for ~200 samples)
        frames_per_subject: Frames to extract per subject
        min_quality_score: Minimum quality score to accept (0-100)
        frame_strategy: Frame selection strategy
    """
    print("=" * 70)
    print("🏷️  Pseudo-Label Generation for U-Net Training")
    print("=" * 70)
    
    # Initialize loader
    loader = HDDBLoader(data_root)
    subjects = loader.get_subject_list()[:num_subjects]
    
    print(f"\n📊 Configuration:")
    print(f"   - Subjects: {num_subjects}")
    print(f"   - Frames per subject: {frames_per_subject}")
    print(f"   - Target samples: {num_subjects * frames_per_subject}")
    print(f"   - Min quality score: {min_quality_score}")
    print(f"   - Frame strategy: {frame_strategy}")
    
    # Create output directories
    output_path = Path(output_dir)
    images_dir = output_path / 'images'
    masks_dir = output_path / 'masks'
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Metadata storage
    metadata = {
        'config': {
            'num_subjects': num_subjects,
            'frames_per_subject': frames_per_subject,
            'min_quality_score': min_quality_score,
            'frame_strategy': frame_strategy
        },
        'samples': []
    }
    
    # Process each subject
    total_generated = 0
    total_rejected = 0
    
    for subject_id in subjects:
        print(f"\n{'='*70}")
        print(f"Processing {subject_id}...")
        
        # Get utterances for this subject
        utterances = loader.get_utterance_list(subject_id)
        
        if not utterances:
            print(f"   ⚠️  No utterances found for {subject_id}, skipping")
            continue
        
        # Use first utterance (can be extended to use multiple)
        utterance_id = utterances[0]
        print(f"   Utterance: {utterance_id}")
        
        # Load MRI data
        data = loader.load_utterance(utterance_id, load_mri=True, load_audio=False)
        mri_data = data['mri_frames']
        
        print(f"   MRI shape: {mri_data.shape}")
        
        # Select frames
        frame_indices = select_frames(mri_data, frames_per_subject, frame_strategy)
        
        # Process each frame
        subject_generated = 0
        subject_rejected = 0
        
        for idx in tqdm(frame_indices, desc=f"   {subject_id}"):
            raw_frame = mri_data[idx]
            
            # Normalize to 0-255
            norm_frame = cv2.normalize(raw_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Extract ROI
            roi_frame, roi_coords = extract_roi(norm_frame)
            
            # Segment
            mask = segment_vocal_tract_adaptive(roi_frame)
            
            # Assess quality
            quality_score, metrics = assess_mask_quality(mask)
            
            # Check if meets quality threshold
            if quality_score < min_quality_score:
                subject_rejected += 1
                continue
            
            # Save image and mask
            sample_id = f"{subject_id}_frame{idx:04d}"
            
            # Save ROI image
            image_path = images_dir / f"{sample_id}.png"
            cv2.imwrite(str(image_path), roi_frame)
            
            # Save mask
            mask_path = masks_dir / f"{sample_id}.png"
            cv2.imwrite(str(mask_path), mask)
            
            # Record metadata
            metadata['samples'].append({
                'sample_id': sample_id,
                'subject': subject_id,
                'utterance': utterance_id,
                'frame_idx': int(idx),
                'roi_coords': [int(x) for x in roi_coords],
                'image_path': str(image_path.relative_to(output_path)),
                'mask_path': str(mask_path.relative_to(output_path)),
                'quality_score': quality_score,
                'metrics': metrics
            })
            
            subject_generated += 1
        
        total_generated += subject_generated
        total_rejected += subject_rejected
        
        print(f"   ✅ Generated: {subject_generated}, Rejected: {subject_rejected}")
    
    # Save metadata
    metadata['summary'] = {
        'total_generated': total_generated,
        'total_rejected': total_rejected,
        'acceptance_rate': total_generated / (total_generated + total_rejected) if (total_generated + total_rejected) > 0 else 0
    }
    
    metadata_path = output_path / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Summary
    print(f"\n{'='*70}")
    print("📋 Summary")
    print(f"{'='*70}")
    print(f"✅ Total samples generated: {total_generated}")
    print(f"❌ Total samples rejected: {total_rejected}")
    print(f"📊 Acceptance rate: {metadata['summary']['acceptance_rate']*100:.1f}%")
    print(f"\n📁 Output:")
    print(f"   - Images: {images_dir}")
    print(f"   - Masks: {masks_dir}")
    print(f"   - Metadata: {metadata_path}")
    print(f"{'='*70}\n")
    
    return metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate pseudo-labels for U-Net training')
    parser.add_argument('--data-root', type=str, 
                       default='/mnt/HDDB/dataset/my_dataset/dataset',
                       help='Path to HDDB dataset')
    parser.add_argument('--output-dir', type=str,
                       default='data/pseudo_labels',
                       help='Output directory for pseudo-labels')
    parser.add_argument('--num-subjects', type=int, default=5,
                       help='Number of subjects to use')
    parser.add_argument('--frames-per-subject', type=int, default=40,
                       help='Frames to extract per subject')
    parser.add_argument('--min-quality-score', type=int, default=50,
                       help='Minimum quality score (0-100)')
    parser.add_argument('--frame-strategy', type=str, default='distributed',
                       choices=['distributed', 'random', 'middle'],
                       help='Frame selection strategy')
    
    args = parser.parse_args()
    
    generate_pseudo_labels(
        data_root=args.data_root,
        output_dir=args.output_dir,
        num_subjects=args.num_subjects,
        frames_per_subject=args.frames_per_subject,
        min_quality_score=args.min_quality_score,
        frame_strategy=args.frame_strategy
    )
