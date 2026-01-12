#!/usr/bin/env python3
"""
Analyze HDDB USC-TIMIT Dataset Structure
Quick analysis script to understand the dataset.
"""

import json
from pathlib import Path
import h5py
import soundfile as sf
import numpy as np
from tqdm import tqdm

def analyze_dataset(data_root):
    """Analyze HDDB dataset structure and statistics."""
    data_root = Path(data_root)

    print("=" * 70)
    print("HDDB USC-TIMIT Dataset Analysis")
    print("=" * 70)

    # Find all subjects
    subjects = sorted([d.name for d in data_root.iterdir() if d.is_dir() and d.name.startswith('sub')])
    print(f"\nTotal subjects: {len(subjects)}")
    print(f"Subjects: {', '.join(subjects[:10])}..." if len(subjects) > 10 else f"Subjects: {', '.join(subjects)}")

    stats = {
        'total_subjects': len(subjects),
        'subjects': subjects,
        'utterances': {},
        'total_utterances': 0,
        'total_frames': 0,
        'sample_info': {}
    }

    # Analyze first 5 subjects in detail
    print(f"\n{'='*70}")
    print("Analyzing first 5 subjects in detail...")
    print(f"{'='*70}")

    for subject in tqdm(subjects[:5], desc="Subjects"):
        subject_path = data_root / subject / '2drt'

        if not subject_path.exists():
            print(f"Warning: {subject_path} not found")
            continue

        # Find H5 files (MRI recon)
        recon_dir = subject_path / 'recon'
        audio_dir = subject_path / 'audio'

        h5_files = sorted(recon_dir.glob('*.h5'))
        wav_files = sorted(audio_dir.glob('*.wav'))

        stats['utterances'][subject] = {
            'h5_count': len(h5_files),
            'wav_count': len(wav_files),
            'utterance_stats': []
        }

        # Sample first utterance
        if h5_files and wav_files:
            h5_file = h5_files[0]
            wav_file = wav_files[0]

            try:
                # Load MRI
                with h5py.File(h5_file, 'r') as f:
                    # Find the main dataset key
                    keys = list(f.keys())
                    main_key = None
                    for key in keys:
                        if isinstance(f[key], h5py.Dataset):
                            shape = f[key].shape
                            if len(shape) >= 3:  # Should be (frames, height, width) or similar
                                main_key = key
                                break

                    if main_key:
                        mri_shape = f[main_key].shape
                        mri_dtype = f[main_key].dtype
                        num_frames = mri_shape[0]
                    else:
                        mri_shape = "Unknown"
                        mri_dtype = "Unknown"
                        num_frames = 0

                # Load audio
                audio, sr = sf.read(wav_file)
                audio_duration = len(audio) / sr

                stats['utterances'][subject]['utterance_stats'].append({
                    'name': h5_file.stem,
                    'mri_shape': str(mri_shape),
                    'mri_dtype': str(mri_dtype),
                    'num_frames': int(num_frames),
                    'audio_duration': float(audio_duration),
                    'audio_sr': int(sr),
                    'fps': float(num_frames / audio_duration) if audio_duration > 0 else 0
                })

                stats['total_frames'] += num_frames

            except Exception as e:
                print(f"Error processing {h5_file.name}: {e}")

        stats['total_utterances'] += len(h5_files)

    # Quick stats for remaining subjects (just count)
    print(f"\nCounting remaining subjects...")
    for subject in tqdm(subjects[5:], desc="Counting", leave=False):
        subject_path = data_root / subject / '2drt' / 'recon'
        if subject_path.exists():
            h5_count = len(list(subject_path.glob('*.h5')))
            stats['total_utterances'] += h5_count
            stats['utterances'][subject] = {'h5_count': h5_count}

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total subjects: {stats['total_subjects']}")
    print(f"Total utterances (estimated): {stats['total_utterances']}")
    print(f"Total frames (5 subjects sample): {stats['total_frames']:,}")
    print(f"Estimated total frames (all 27 subjects): {stats['total_frames'] * 27 // 5:,}")

    # Sample info
    if stats['utterances']:
        first_subj = list(stats['utterances'].keys())[0]
        if stats['utterances'][first_subj].get('utterance_stats'):
            sample = stats['utterances'][first_subj]['utterance_stats'][0]
            print(f"\nSample utterance info:")
            print(f"  Name: {sample['name']}")
            print(f"  MRI shape: {sample['mri_shape']}")
            print(f"  MRI dtype: {sample['mri_dtype']}")
            print(f"  Num frames: {sample['num_frames']}")
            print(f"  Audio duration: {sample['audio_duration']:.2f}s")
            print(f"  Audio sample rate: {sample['audio_sr']} Hz")
            print(f"  MRI FPS: {sample['fps']:.2f}")

    # Save to JSON
    output_file = Path('/home/Project_Sullivan/data/hddb_dataset_stats.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nStatistics saved to: {output_file}")
    print(f"{'='*70}\n")

    return stats

if __name__ == '__main__':
    data_root = '/mnt/HDDB/dataset/my_dataset/dataset'
    analyze_dataset(data_root)
