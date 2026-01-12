#!/usr/bin/env python3
"""
Improved Vocal Tract Segmentation with ROI Focusing
Addresses the background contamination issue by focusing on vocal tract region.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from skimage.filters import threshold_otsu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing.hddb_data_loader import HDDBLoader

def extract_roi(frame, roi_params=None):
    """
    Extract Region of Interest (ROI) focusing on vocal tract area.

    Args:
        frame: MRI frame (H, W)
        roi_params: Dict with 'top', 'bottom', 'left', 'right' as fractions (0-1)
                   Default focuses on center-bottom area

    Returns:
        roi_frame: Extracted ROI
        roi_coords: (y1, y2, x1, x2) coordinates
    """
    h, w = frame.shape

    if roi_params is None:
        # Default: Focus on center-bottom (where vocal tract typically is)
        # Remove edges to eliminate background
        roi_params = {
            'top': 0.25,      # Start from 25% down (skip top background)
            'bottom': 0.95,   # End at 95% (skip bottom edge)
            'left': 0.15,     # Start from 15% right (skip left edge)
            'right': 0.85     # End at 85% (skip right edge)
        }

    y1 = int(h * roi_params['top'])
    y2 = int(h * roi_params['bottom'])
    x1 = int(w * roi_params['left'])
    x2 = int(w * roi_params['right'])

    roi_frame = frame[y1:y2, x1:x2]
    roi_coords = (y1, y2, x1, x2)

    return roi_frame, roi_coords

def segment_vocal_tract(frame, method='manual', threshold_val=100):
    """
    Segment vocal tract from MRI frame.

    Args:
        frame: MRI frame (normalized to 0-255)
        method: 'otsu', 'manual', or 'adaptive'
        threshold_val: Threshold value for manual method

    Returns:
        binary_mask: Binary segmentation mask
        thresh_val: Threshold value used
    """
    # Smooth
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    if method == 'otsu':
        # Otsu thresholding
        thresh_val = threshold_otsu(blurred)
        binary_mask = (blurred < thresh_val).astype(np.uint8) * 255

    elif method == 'manual':
        # Manual threshold
        binary_mask = (blurred < threshold_val).astype(np.uint8) * 255
        thresh_val = threshold_val

    elif method == 'adaptive':
        # Adaptive thresholding
        binary_mask = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11,
            C=2
        )
        thresh_val = 'adaptive'

    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    processed = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel, iterations=1)

    return processed, thresh_val

def test_roi_segmentation():
    """
    Test segmentation with ROI focusing and multiple threshold methods.
    """
    print("=" * 70)
    print("🔬 Improved Vocal Tract Segmentation (ROI-Focused)")
    print("=" * 70)

    # 1. Load Data
    print("\n📦 Loading data...")
    data_root = "/mnt/HDDB/dataset/my_dataset/dataset"
    loader = HDDBLoader(data_root)

    if not loader.subjects:
        print("❌ No subjects found.")
        return

    subject_id = loader.subjects[0]
    utterance_list = loader.get_utterance_list(subject_id)
    utterance_id = utterance_list[0]

    print(f"✅ Subject: {subject_id}")
    print(f"✅ Utterance: {utterance_id}")

    # Load MRI
    data = loader.load_utterance(utterance_id, load_mri=True, load_audio=False)
    mri_data = data['mri_frames']

    print(f"✅ MRI shape: {mri_data.shape}")

    # 2. Select test frame (middle)
    frame_idx = len(mri_data) // 2
    raw_frame = mri_data[frame_idx]

    # Normalize to 0-255
    norm_frame = cv2.normalize(raw_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    print(f"\n{'='*70}")
    print(f"📍 Processing frame {frame_idx}/{len(mri_data)}")
    print(f"{'='*70}")

    # 3. Extract ROI
    print("\n🎯 Step 1: Extract ROI (Remove Background)")
    roi_frame, roi_coords = extract_roi(norm_frame)
    y1, y2, x1, x2 = roi_coords

    print(f"   - Original size: {norm_frame.shape}")
    print(f"   - ROI size: {roi_frame.shape}")
    print(f"   - ROI coordinates: ({y1}:{y2}, {x1}:{x2})")

    # 4. Test multiple segmentation methods
    print(f"\n{'='*70}")
    print("🔧 Step 2: Test Multiple Segmentation Methods")
    print(f"{'='*70}")

    methods = [
        ('otsu', None, 'Otsu Auto'),
        ('manual', 80, 'Manual (thresh=80)'),
        ('manual', 100, 'Manual (thresh=100)'),
        ('manual', 120, 'Manual (thresh=120)'),
        ('adaptive', None, 'Adaptive Threshold')
    ]

    results = []

    for method, threshold_val, label in methods:
        print(f"\n   Testing: {label}")

        # Segment ROI
        if method == 'adaptive':
            mask, thresh = segment_vocal_tract(roi_frame, method='adaptive')
        elif method == 'otsu':
            mask, thresh = segment_vocal_tract(roi_frame, method='otsu')
        else:
            mask, thresh = segment_vocal_tract(roi_frame, method='manual', threshold_val=threshold_val)

        # Calculate metrics
        airway_pixels = np.sum(mask == 255)
        total_pixels = mask.size
        airway_ratio = airway_pixels / total_pixels

        print(f"      - Threshold: {thresh}")
        print(f"      - Airway pixels: {airway_pixels:,} ({airway_ratio*100:.1f}%)")

        results.append({
            'label': label,
            'method': method,
            'threshold': thresh,
            'mask': mask,
            'airway_ratio': airway_ratio
        })

    # 5. Find best result (airway ratio in reasonable range: 10-35%)
    print(f"\n{'='*70}")
    print("📊 Evaluation")
    print(f"{'='*70}")

    best_result = None
    target_range = (0.10, 0.35)

    for result in results:
        ratio = result['airway_ratio']
        in_range = target_range[0] <= ratio <= target_range[1]
        status = "✅ GOOD" if in_range else "⚠️"

        print(f"{status} {result['label']}: {ratio*100:.1f}%")

        if in_range and (best_result is None or
                        abs(ratio - 0.225) < abs(best_result['airway_ratio'] - 0.225)):
            best_result = result

    if best_result:
        print(f"\n🎯 Best method: {best_result['label']} ({best_result['airway_ratio']*100:.1f}%)")
    else:
        print(f"\n⚠️ No method in ideal range (10-35%), using closest")
        best_result = min(results, key=lambda x: abs(x['airway_ratio'] - 0.225))
        print(f"   Selected: {best_result['label']} ({best_result['airway_ratio']*100:.1f}%)")

    # 6. Visualization
    print(f"\n{'='*70}")
    print("📊 Creating visualization...")
    print(f"{'='*70}")

    fig = plt.figure(figsize=(18, 12))

    # Row 1: Original and ROI extraction
    ax1 = plt.subplot(3, 3, 1)
    ax1.imshow(norm_frame, cmap='gray')
    ax1.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                fill=False, edgecolor='red', linewidth=2))
    ax1.set_title('Original Frame\n(Red: ROI)', fontsize=10)
    ax1.axis('off')

    ax2 = plt.subplot(3, 3, 2)
    ax2.imshow(roi_frame, cmap='gray')
    ax2.set_title(f'Extracted ROI\n{roi_frame.shape}', fontsize=10)
    ax2.axis('off')

    ax3 = plt.subplot(3, 3, 3)
    ax3.imshow(best_result['mask'], cmap='gray')
    ax3.set_title(f'Best Segmentation\n{best_result["label"]}', fontsize=10, color='green')
    ax3.axis('off')

    # Row 2-3: All methods comparison
    for i, result in enumerate(results):
        ax = plt.subplot(3, 3, 4 + i)
        ax.imshow(result['mask'], cmap='gray')

        # Color code title based on quality
        ratio = result['airway_ratio']
        if 0.10 <= ratio <= 0.35:
            color = 'green'
        elif 0.05 <= ratio <= 0.45:
            color = 'orange'
        else:
            color = 'red'

        ax.set_title(f'{result["label"]}\n{ratio*100:.1f}% airway',
                    fontsize=9, color=color)
        ax.axis('off')

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'segmentation_roi_test.png')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved: {save_path}")

    # 7. Summary
    print(f"\n{'='*70}")
    print("📋 Summary")
    print(f"{'='*70}")
    print(f"Subject: {subject_id}")
    print(f"Utterance: {utterance_id}")
    print(f"Frame: {frame_idx}/{len(mri_data)}")
    print(f"\nROI Extraction:")
    print(f"  - Original: {norm_frame.shape}")
    print(f"  - ROI: {roi_frame.shape}")
    print(f"  - Removed background: {100*(1 - roi_frame.size/norm_frame.size):.1f}%")
    print(f"\nBest Segmentation:")
    print(f"  - Method: {best_result['label']}")
    print(f"  - Threshold: {best_result['threshold']}")
    print(f"  - Airway ratio: {best_result['airway_ratio']*100:.1f}%")

    print(f"\n{'='*70}")
    print("🎯 Next Steps:")
    if 0.10 <= best_result['airway_ratio'] <= 0.35:
        print("✅ EXCELLENT: Segmentation quality is good!")
        print("   → Ready to generate pseudo-labels for U-Net training")
        print(f"   → Use method: {best_result['label']}")
    else:
        print("⚠️ Need fine-tuning:")
        print("   → Adjust ROI parameters")
        print("   → Or try different threshold values")

    print(f"{'='*70}\n")

    return best_result

if __name__ == "__main__":
    best_result = test_roi_segmentation()
