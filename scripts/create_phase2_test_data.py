"""
Create synthetic test data for Phase 2 pipeline validation

Generates fake but realistic audio features and articulatory parameters
to test the Phase 2 pipeline without requiring full dataset processing.

Usage:
    python scripts/create_phase2_test_data.py --output_dir data/processed/phase2_test
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.parameter_extraction import ArticulatoryParameters
from src.modeling.audio_features import AudioFeatureConfig


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Create Phase 2 test data")

    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for test data')
    parser.add_argument('--num_train', type=int, default=50,
                       help='Number of training samples')
    parser.add_argument('--num_val', type=int, default=10,
                       help='Number of validation samples')
    parser.add_argument('--num_test', type=int, default=10,
                       help='Number of test samples')
    parser.add_argument('--min_duration', type=float, default=1.0,
                       help='Minimum utterance duration (seconds)')
    parser.add_argument('--max_duration', type=float, default=3.0,
                       help='Maximum utterance duration (seconds)')
    parser.add_argument('--audio_feature_type', type=str, default='mel',
                       choices=['mel', 'mfcc'],
                       help='Type of audio features to generate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    return parser.parse_args()


def generate_synthetic_parameters(num_frames: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate synthetic articulatory parameters

    Creates realistic-looking parameter trajectories with temporal smoothness

    Args:
        num_frames: Number of frames to generate
        seed: Random seed

    Returns:
        (num_frames, 10) array of parameters
    """
    if seed is not None:
        np.random.seed(seed)

    # Initialize parameters
    params = np.zeros((num_frames, 10))

    # Generate each parameter with smooth temporal evolution
    for i in range(10):
        # Generate smooth trajectory using sum of sinusoids
        t = np.linspace(0, 1, num_frames)

        # Base frequency components
        freq1 = np.random.uniform(1, 3)
        freq2 = np.random.uniform(4, 8)
        freq3 = np.random.uniform(10, 15)

        # Amplitudes
        amp1 = np.random.uniform(0.2, 0.4)
        amp2 = np.random.uniform(0.1, 0.2)
        amp3 = np.random.uniform(0.05, 0.1)

        # Phase offsets
        phase1 = np.random.uniform(0, 2 * np.pi)
        phase2 = np.random.uniform(0, 2 * np.pi)
        phase3 = np.random.uniform(0, 2 * np.pi)

        # Generate smooth signal
        signal = (amp1 * np.sin(2 * np.pi * freq1 * t + phase1) +
                 amp2 * np.sin(2 * np.pi * freq2 * t + phase2) +
                 amp3 * np.sin(2 * np.pi * freq3 * t + phase3))

        # Normalize to [0, 1] range
        signal = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)

        # Add small noise
        noise = np.random.normal(0, 0.02, num_frames)
        signal = signal + noise

        # Clip to [0, 1]
        signal = np.clip(signal, 0, 1)

        params[:, i] = signal

    return params.astype(np.float32)


def generate_synthetic_audio_features(num_frames: int,
                                     feature_type: str = 'mel',
                                     seed: Optional[int] = None) -> np.ndarray:
    """
    Generate synthetic audio features

    Args:
        num_frames: Number of frames
        feature_type: 'mel' or 'mfcc'
        seed: Random seed

    Returns:
        (num_frames, n_features) array of audio features
    """
    if seed is not None:
        np.random.seed(seed)

    config = AudioFeatureConfig()

    if feature_type == 'mel':
        n_features = config.n_mels  # 80
    elif feature_type == 'mfcc':
        # 13 MFCCs + 13 deltas + 13 delta-deltas
        n_features = config.n_mfcc * 3  # 39
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

    # Generate features with some temporal structure
    features = np.zeros((num_frames, n_features))

    for i in range(n_features):
        # Generate smooth trajectory
        t = np.linspace(0, 1, num_frames)
        freq = np.random.uniform(2, 10)
        phase = np.random.uniform(0, 2 * np.pi)
        amp = np.random.uniform(10, 30)

        signal = amp * np.sin(2 * np.pi * freq * t + phase)

        # Add noise
        noise = np.random.normal(0, 5, num_frames)
        signal = signal + noise

        features[:, i] = signal

    # Scale to typical mel-spectrogram range (in dB)
    if feature_type == 'mel':
        features = features - 50  # Shift to [-80, -20] dB range

    return features.astype(np.float32)


def create_sample(sample_idx: int,
                 subject_id: str,
                 duration: float,
                 feature_type: str,
                 mri_fps: float = 50.0,
                 seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a single synthetic sample

    Args:
        sample_idx: Sample index
        subject_id: Subject identifier
        duration: Duration in seconds
        feature_type: Audio feature type
        mri_fps: MRI frame rate
        seed: Random seed

    Returns:
        (audio_features, parameters) tuple
    """
    # Calculate number of frames
    num_frames = int(duration * mri_fps)

    # Generate parameters
    params = generate_synthetic_parameters(num_frames, seed=seed)

    # Generate audio features
    audio = generate_synthetic_audio_features(num_frames, feature_type, seed=seed)

    return audio, params


def main():
    """Main function"""
    args = parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ['train', 'val', 'test']:
        (output_dir / split).mkdir(exist_ok=True)

    print("=" * 80)
    print("Creating Phase 2 Test Data")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Train samples: {args.num_train}")
    print(f"Val samples: {args.num_val}")
    print(f"Test samples: {args.num_test}")
    print(f"Audio feature type: {args.audio_feature_type}")
    print("=" * 80)

    # Generate samples for each split
    splits_config = {
        'train': args.num_train,
        'val': args.num_val,
        'test': args.num_test
    }

    all_sample_info = {}

    for split, num_samples in splits_config.items():
        print(f"\nGenerating {split} split ({num_samples} samples)...")

        split_dir = output_dir / split
        sample_list = []

        for i in tqdm(range(num_samples)):
            # Generate sample metadata
            subject_id = f"sub{(i % 15 + 1):03d}"  # Cycle through 15 subjects
            utterance_id = f"utt{i:04d}"
            utterance_name = f"{subject_id}_{utterance_id}"

            # Random duration
            duration = np.random.uniform(args.min_duration, args.max_duration)

            # Generate sample data
            audio, params = create_sample(
                sample_idx=i,
                subject_id=subject_id,
                duration=duration,
                feature_type=args.audio_feature_type,
                seed=args.seed + i
            )

            # Save audio features
            audio_path = split_dir / f"{utterance_name}_audio_{args.audio_feature_type}.npy"
            np.save(audio_path, audio)

            # Save parameters
            param_path = split_dir / f"{utterance_name}_parameters.npy"
            np.save(param_path, params)

            # Save sample info
            sample_info = {
                'audio_path': str(audio_path),
                'param_path': str(param_path),
                'utterance_name': utterance_name,
                'subject_id': subject_id,
                'duration': float(duration),
                'num_frames': len(params)
            }
            sample_list.append(sample_info)

        # Save sample list
        sample_list_path = output_dir / f"{split}_samples.json"
        with open(sample_list_path, 'w') as f:
            json.dump(sample_list, f, indent=2)

        all_sample_info[split] = sample_list
        print(f"  Saved {len(sample_list)} samples to {split_dir}")

    # Compute and save normalization statistics
    print("\nComputing normalization statistics from train set...")

    all_audio = []
    all_params = []

    for sample in all_sample_info['train']:
        audio = np.load(sample['audio_path'])
        params = np.load(sample['param_path'])
        all_audio.append(audio)
        all_params.append(params)

    all_audio = np.concatenate(all_audio, axis=0)
    all_params = np.concatenate(all_params, axis=0)

    stats = {
        'audio_mean': np.mean(all_audio, axis=0).astype(np.float32),
        'audio_std': np.std(all_audio, axis=0).astype(np.float32),
        'param_mean': np.mean(all_params, axis=0).astype(np.float32),
        'param_std': np.std(all_params, axis=0).astype(np.float32)
    }

    # Prevent division by zero
    stats['audio_std'] = np.where(stats['audio_std'] < 1e-6, 1.0, stats['audio_std'])
    stats['param_std'] = np.where(stats['param_std'] < 1e-6, 1.0, stats['param_std'])

    # Save statistics
    stats_path = output_dir / 'normalization_stats.npz'
    np.savez(stats_path, **stats)

    print(f"  Saved normalization statistics to {stats_path}")
    print(f"\n  Audio: mean={np.mean(stats['audio_mean']):.4f}, std={np.mean(stats['audio_std']):.4f}")
    print(f"  Params: mean={np.mean(stats['param_mean']):.4f}, std={np.mean(stats['param_std']):.4f}")

    # Save dataset info
    dataset_info = {
        'num_train': args.num_train,
        'num_val': args.num_val,
        'num_test': args.num_test,
        'total_samples': args.num_train + args.num_val + args.num_test,
        'audio_feature_type': args.audio_feature_type,
        'duration_range': [args.min_duration, args.max_duration],
        'parameter_dim': 10,
        'audio_dim': 80 if args.audio_feature_type == 'mel' else 39,
        'created_at': str(Path(__file__).stat().st_mtime),
        'seed': args.seed
    }

    info_path = output_dir / 'dataset_info.json'
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)

    print(f"\n✓ Test data generation complete!")
    print(f"  Dataset info saved to {info_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
