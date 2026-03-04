#!/usr/bin/env python3
"""
Test Basic Segmentation with Thresholding
Tests if vocal tract (airway) can be separated using simple image processing.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import cv2
from skimage.filters import threshold_otsu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing.hddb_data_loader import HDDBLoader

def test_initial_segmentation():
    """
    Test basic vocal tract segmentation using thresholding.
    """
    print("=" * 70)
    print("🔬 Vocal Tract Segmentation Test")
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

    if not utterance_list:
        print(f"❌ No utterances found for {subject_id}.")
        return

    utterance_id = utterance_list[0]

    print(f"✅ Testing segmentation on:")
    print(f"   - Subject: {subject_id}")
    print(f"   - Utterance: {utterance_id}")

    # Load MRI data
    data = loader.load_utterance(utterance_id, load_mri=True, load_audio=False)
    mri_data = data['mri_frames']  # (Time, H, W)

    print(f"   - MRI shape: {mri_data.shape}")

    # 2. Select frames to analyze
    # Choose multiple frames: beginning, middle, end
    frame_indices = [
        len(mri_data) // 4,      # 25%
        len(mri_data) // 2,      # 50% (middle)
        3 * len(mri_data) // 4   # 75%
    ]

    frames_to_process = []
    for idx in frame_indices:
        raw_frame = mri_data[idx]
        frames_to_process.append((idx, raw_frame))

    print(f"\n🖼️ Processing {len(frames_to_process)} sample frames...")

    # 3. Process each frame
    results = []

    for frame_idx, raw_frame in frames_to_process:
        print(f"\n{'='*70}")
        print(f"Processing frame {frame_idx}/{len(mri_data)}...")

        # Normalize to 0-255 uint8
        norm_frame = cv2.normalize(raw_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # A. Gaussian Blur (reduce noise)
        blurred = cv2.GaussianBlur(norm_frame, (5, 5), 0)

        # B. Otsu Thresholding
        # In MRI: Air (vocal tract) is DARK, Tissue is BRIGHT
        # We want to segment the airway (dark regions)
        thresh_val = threshold_otsu(blurred)
        print(f"   - Otsu threshold value: {thresh_val}")

        # Create binary mask: pixels BELOW threshold are airway (set to 255)
        binary_mask = (blurred < thresh_val).astype(np.uint8) * 255

        # C. Morphological Operations (Noise Removal)
        kernel = np.ones((3, 3), np.uint8)

        # Opening: remove small white noise
        opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Closing: fill small black holes
        processed_mask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

        # D. Additional processing: Extract ROI (Region of Interest)
        # Focus on lower center area (where vocal tract is typically visible)
        h, w = norm_frame.shape
        roi_y1, roi_y2 = h // 3, h  # Bottom 2/3
        roi_x1, roi_x2 = w // 4, 3 * w // 4  # Middle 50%

        roi_frame = norm_frame[roi_y1:roi_y2, roi_x1:roi_x2]
        roi_mask = processed_mask[roi_y1:roi_y2, roi_x1:roi_x2]

        # Calculate segmentation quality metrics
        airway_pixels = np.sum(processed_mask == 255)
        total_pixels = processed_mask.size
        airway_ratio = airway_pixels / total_pixels

        print(f"   ✅ Processing complete:")
        print(f"      - Airway pixels: {airway_pixels:,} ({airway_ratio*100:.1f}%)")
        print(f"      - ROI size: {roi_frame.shape}")

        results.append({
            'frame_idx': frame_idx,
            'original': norm_frame,
            'blurred': blurred,
            'binary': binary_mask,
            'processed': processed_mask,
            'roi_frame': roi_frame,
            'roi_mask': roi_mask,
            'thresh_val': thresh_val,
            'airway_ratio': airway_ratio
        })

    # 4. Visualization
    print(f"\n{'='*70}")
    print("📊 Creating visualization...")

    n_frames = len(results)
    fig, axes = plt.subplots(n_frames, 4, figsize=(16, 4 * n_frames))

    if n_frames == 1:
        axes = axes.reshape(1, -1)

    for i, result in enumerate(results):
        # Column 1: Original
        axes[i, 0].imshow(result['original'], cmap='gray')
        axes[i, 0].set_title(f"Frame {result['frame_idx']}\nOriginal")
        axes[i, 0].axis('off')

        # Column 2: Otsu Binary
        axes[i, 1].imshow(result['binary'], cmap='gray')
        axes[i, 1].set_title(f"Otsu Threshold\n(val={result['thresh_val']:.1f})")
        axes[i, 1].axis('off')

        # Column 3: Processed (Morphology)
        axes[i, 2].imshow(result['processed'], cmap='gray')
        axes[i, 2].set_title(f"Morphology\n({result['airway_ratio']*100:.1f}% airway)")
        axes[i, 2].axis('off')

        # Column 4: ROI
        axes[i, 3].imshow(result['roi_mask'], cmap='gray')
        axes[i, 3].set_title(f"ROI (Vocal Tract Area)")
        axes[i, 3].axis('off')

    plt.tight_layout()

    # Save result
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'segmentation_test.png')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Segmentation test image saved to: {save_path}")

    # 5. Summary and Analysis
    print(f"\n{'='*70}")
    print("📋 Segmentation Test Summary")
    print(f"{'='*70}")
    print(f"Subject: {subject_id}")
    print(f"Utterance: {utterance_id}")
    print(f"Total MRI frames: {len(mri_data)}")
    print(f"Frames analyzed: {len(results)}")
    print(f"\nPer-frame analysis:")

    for i, result in enumerate(results):
        print(f"\n  Frame {result['frame_idx']}:")
        print(f"    - Otsu threshold: {result['thresh_val']:.1f}")
        print(f"    - Airway ratio: {result['airway_ratio']*100:.1f}%")

    avg_airway = np.mean([r['airway_ratio'] for r in results])
    print(f"\n  Average airway ratio: {avg_airway*100:.1f}%")

    print(f"\n{'='*70}")
    print("🔍 Evaluation:")

    if 5 < avg_airway * 100 < 40:
        print("✅ GOOD: Airway ratio in reasonable range (5-40%)")
        print("   → Thresholding successfully separates vocal tract")
        print("   → Ready for pseudo-label generation")
    elif avg_airway * 100 <= 5:
        print("⚠️ WARNING: Very low airway ratio (<5%)")
        print("   → May need threshold adjustment or different preprocessing")
    else:
        print("⚠️ WARNING: Very high airway ratio (>40%)")
        print("   → May be detecting too much as airway")
        print("   → Consider ROI focusing or threshold tuning")

    print(f"\n{'='*70}")
    print("🎯 Next Steps:")
    print("   1. Review saved image: results/segmentation_test.png")
    print("   2. If segmentation looks good → Generate pseudo-labels")
    print("   3. If adjustment needed → Tune thresholds/ROI")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    test_initial_segmentation()
