#!/usr/bin/env python3
"""Test HDDB loader"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.hddb_data_loader import HDDBLoader

def main():
    print("Testing HDDB Loader...")
    print("=" * 70)

    # Initialize loader
    data_root = '/mnt/HDDB/dataset/my_dataset/dataset'
    loader = HDDBLoader(data_root)

    # Print stats
    stats = loader.get_statistics()
    print(f"\nDataset Statistics:")
    print(f"  Subjects: {stats['num_subjects']}")
    print(f"  Utterances: {stats['num_utterances']}")
    print(f"  Avg utterances/subject: {stats['avg_utterances_per_subject']:.1f}")

    # Test loading one utterance
    print(f"\n{'='*70}")
    print("Testing single utterance load...")
    utterance_list = loader.get_utterance_list()
    if utterance_list:
        test_utt = utterance_list[0]
        print(f"Loading: {test_utt}")

        data = loader.load_utterance(test_utt)

        print(f"\nResults:")
        print(f"  Subject: {data['subject']}")
        print(f"  MRI shape: {data['mri_shape']}")
        print(f"  Num frames: {data['num_frames']}")
        print(f"  Audio shape: {data['audio'].shape}")
        print(f"  Audio SR: {data['audio_sr']}")
        print(f"  Duration: {data['duration']:.2f}s")
        print(f"  FPS: {data['fps']:.2f}")

        print(f"\n{'='*70}")
        print("✅ Loader test PASSED!")
    else:
        print("❌ No utterances found")

if __name__ == '__main__':
    main()
