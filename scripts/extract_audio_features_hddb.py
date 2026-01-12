#!/usr/bin/env python3
"""
HDDB Audio Feature Extraction
===============================

Extract audio features (Mel-spectrogram + MFCC) from HDDB dataset.

Features:
- 80-dimensional Mel-spectrogram (primary features)
- 13-dimensional MFCC (alternative features)

Author: Claude & Research Assistant
Date: 2026-01-11
"""

import os
import sys
import json
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime
import librosa

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger


class AudioFeatureExtractor:
    """Extract audio features from HDDB dataset"""

    def __init__(
        self,
        output_dir: str,
        sr: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: int = 80,
        n_mfcc: int = 13
    ):
        """
        Initialize extractor

        Args:
            output_dir: Output directory for audio features
            sr: Target sample rate (16kHz)
            n_fft: FFT window size
            hop_length: Hop length (160 samples = 10ms at 16kHz)
            n_mels: Number of mel bands
            n_mfcc: Number of MFCC coefficients
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger("AudioFeatureExtractor")

        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc

        self.logger.info(f"Audio feature extractor initialized:")
        self.logger.info(f"  Sample rate: {sr} Hz")
        self.logger.info(f"  FFT size: {n_fft}")
        self.logger.info(f"  Hop length: {hop_length} ({hop_length / sr * 1000:.1f} ms)")
        self.logger.info(f"  Mel bands: {n_mels}")
        self.logger.info(f"  MFCC coeffs: {n_mfcc}")

    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract log mel-spectrogram

        Args:
            audio: (num_samples,) audio waveform

        Returns:
            mel_spec: (num_frames, n_mels) log mel-spectrogram
        """
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=0,
            fmax=self.sr // 2
        )

        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Transpose to (time, freq)
        mel_spec_db = mel_spec_db.T

        return mel_spec_db.astype(np.float32)

    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features

        Args:
            audio: (num_samples,) audio waveform

        Returns:
            mfcc: (num_frames, n_mfcc) MFCC features
        """
        # Compute MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        # Transpose to (time, freq)
        mfcc = mfcc.T

        return mfcc.astype(np.float32)

    def process_utterance(
        self,
        wav_path: Path,
        extract_mel: bool = True,
        extract_mfcc: bool = True
    ) -> dict:
        """
        Process single utterance audio

        Args:
            wav_path: Path to WAV audio file
            extract_mel: Extract mel-spectrogram
            extract_mfcc: Extract MFCC

        Returns:
            stats: Processing statistics
        """
        utterance_name = wav_path.stem.replace('_audio', '_recon')  # Replace _audio with _recon to match H5 naming

        # Load audio from WAV file
        audio, original_sr = librosa.load(wav_path, sr=None)

        # Resample to target sample rate if needed
        if original_sr != self.sr:
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=self.sr)

        # Extract features
        features = {}

        if extract_mel:
            mel_spec = self.extract_mel_spectrogram(audio)
            features['mel_spectrogram'] = mel_spec

        if extract_mfcc:
            mfcc = self.extract_mfcc(audio)
            features['mfcc'] = mfcc

        # Save features
        output_path = self.output_dir / f"{utterance_name}_audio.npz"
        np.savez_compressed(
            output_path,
            **features,
            utterance_name=utterance_name,
            sample_rate=self.sr,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            num_frames_mel=mel_spec.shape[0] if extract_mel else 0,
            num_frames_mfcc=mfcc.shape[0] if extract_mfcc else 0
        )

        stats = {
            'utterance_name': utterance_name,
            'audio_duration': len(audio) / self.sr,
            'num_frames_mel': mel_spec.shape[0] if extract_mel else 0,
            'num_frames_mfcc': mfcc.shape[0] if extract_mfcc else 0,
            'output_path': str(output_path)
        }

        return stats

    def process_dataset(
        self,
        data_root: Path,
        subjects: list,
        extract_mel: bool = True,
        extract_mfcc: bool = True
    ):
        """
        Process entire HDDB dataset

        Args:
            data_root: Root path to HDDB dataset
            subjects: List of subject IDs
            extract_mel: Extract mel-spectrogram
            extract_mfcc: Extract MFCC
        """
        self.logger.info(f"Processing audio from {len(subjects)} subjects")

        all_stats = []
        total_duration = 0.0

        for subject_id in tqdm(subjects, desc="Subjects"):
            # Use audio directory (has WAV files)
            audio_dir = data_root / subject_id / "2drt" / "audio"

            if not audio_dir.exists():
                self.logger.warning(f"Audio directory not found: {audio_dir}")
                continue

            self.logger.info(f"Processing {subject_id}")

            # Find all WAV files
            wav_files = sorted(audio_dir.glob("*_audio.wav"))

            for wav_path in tqdm(wav_files, desc=f"{subject_id} utterances", leave=False):
                try:
                    stats = self.process_utterance(
                        wav_path,
                        extract_mel=extract_mel,
                        extract_mfcc=extract_mfcc
                    )
                    all_stats.append(stats)
                    total_duration += stats['audio_duration']
                except Exception as e:
                    self.logger.error(f"Failed to process {wav_path.name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_subjects': len(subjects),
            'total_utterances': len(all_stats),
            'total_duration': total_duration,
            'sample_rate': self.sr,
            'n_mels': self.n_mels if extract_mel else 0,
            'n_mfcc': self.n_mfcc if extract_mfcc else 0,
            'output_dir': str(self.output_dir),
            'subjects': subjects,
            'utterance_stats': all_stats
        }

        summary_path = self.output_dir / 'audio_feature_extraction_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Audio feature extraction complete!")
        self.logger.info(f"  Subjects processed: {len(subjects)}")
        self.logger.info(f"  Utterances processed: {len(all_stats)}")
        self.logger.info(f"  Total duration: {total_duration / 60:.1f} minutes")
        self.logger.info(f"  Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract audio features from HDDB dataset")
    parser.add_argument(
        '--data-root',
        type=str,
        default='/mnt/HDDB/dataset/my_dataset/dataset',
        help='Root path to HDDB dataset'
    )
    parser.add_argument(
        '--subjects',
        type=str,
        default='all',
        help='Comma-separated subject IDs or "all"'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed/audio_features',
        help='Output directory for audio features'
    )
    parser.add_argument(
        '--no-mel',
        action='store_true',
        help='Skip mel-spectrogram extraction'
    )
    parser.add_argument(
        '--no-mfcc',
        action='store_true',
        help='Skip MFCC extraction'
    )

    args = parser.parse_args()

    # Get list of subjects
    if args.subjects == 'all':
        data_root = Path(args.data_root)
        subjects = sorted([d.name for d in data_root.iterdir() if d.is_dir() and d.name.startswith('sub')])
    else:
        subjects = args.subjects.split(',')

    print(f"Processing {len(subjects)} subjects: {', '.join(subjects[:5])}{'...' if len(subjects) > 5 else ''}")

    # Create extractor
    extractor = AudioFeatureExtractor(output_dir=args.output_dir)

    # Process dataset
    extractor.process_dataset(
        data_root=Path(args.data_root),
        subjects=subjects,
        extract_mel=not args.no_mel,
        extract_mfcc=not args.no_mfcc
    )


if __name__ == '__main__':
    main()
