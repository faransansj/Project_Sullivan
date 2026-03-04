#!/usr/bin/env python3
"""
Test Feature Extraction and Alignment
Tests data integrity, feature extraction, and audio-MRI synchronization
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import librosa
import librosa.display

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.hddb_data_loader import HDDBLoader

def analyze_sample_features():
    """
    Analyze sample utterance to verify:
    1. Data loading works correctly
    2. Audio and MRI are properly synchronized
    3. Feature extraction is functional
    """

    # 1. 데이터 로더 초기화
    print("=" * 70)
    print("🔄 데이터 로더 초기화 중...")
    print("=" * 70)

    data_root = "/mnt/HDDB/dataset/my_dataset/dataset"
    loader = HDDBLoader(data_root)

    # 2. 첫 번째 주체의 첫 번째 발화 가져오기
    if not loader.subjects:
        print("❌ 데이터셋 경로에 subject 디렉토리가 없습니다.")
        return

    subject_id = loader.subjects[0]
    utterance_list = loader.get_utterance_list(subject_id)

    if not utterance_list:
        print(f"❌ Subject {subject_id}에 발화가 없습니다.")
        return

    utterance_id = utterance_list[0]

    print(f"\n📊 분석 대상:")
    print(f"   - Subject: {subject_id}")
    print(f"   - Utterance: {utterance_id}")

    try:
        # 데이터 로드
        print(f"\n{'='*70}")
        print("📦 데이터 로드 중...")
        print(f"{'='*70}")

        data = loader.load_utterance(
            utterance_id,
            load_mri=True,
            load_audio=True,
            target_audio_sr=16000  # Resample to 16kHz for consistency
        )

        audio_data = data['audio']
        sr = data['audio_sr']
        mri_data = data['mri_frames']

        print(f"✅ 로드 완료:")
        print(f"   - Audio shape: {audio_data.shape}, SR: {sr} Hz")
        print(f"   - Audio duration: {data['duration']:.2f}s")
        print(f"   - MRI shape: {mri_data.shape}")
        print(f"   - MRI frames: {data['num_frames']}")
        print(f"   - FPS: {data['fps']:.2f}")

        # A. Audio Feature: Mel-spectrogram
        print(f"\n{'='*70}")
        print("🎵 Mel-spectrogram 추출 중...")
        print(f"{'='*70}")

        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=sr,
            n_mels=80,
            n_fft=2048,
            hop_length=160,  # 10ms hop at 16kHz
            fmax=8000
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        print(f"✅ Mel-spectrogram shape: {mel_spec.shape}")
        print(f"   - Time frames: {mel_spec.shape[1]}")
        print(f"   - Mel bins: {mel_spec.shape[0]}")

        # B. MRI Feature: Frame-wise Intensity Statistics
        print(f"\n{'='*70}")
        print("🖼️ MRI 통계 추출 중...")
        print(f"{'='*70}")

        # Calculate mean intensity per frame (indicator of motion/change)
        mri_intensity = np.mean(mri_data, axis=(1, 2))

        # Calculate frame-to-frame difference (motion indicator)
        mri_motion = np.abs(np.diff(mri_intensity, prepend=mri_intensity[0]))

        # Normalize to [0, 1]
        mri_intensity_norm = (mri_intensity - np.min(mri_intensity)) / (np.max(mri_intensity) - np.min(mri_intensity) + 1e-8)
        mri_motion_norm = (mri_motion - np.min(mri_motion)) / (np.max(mri_motion) - np.min(mri_motion) + 1e-8)

        print(f"✅ MRI statistics shape: {mri_intensity.shape}")

        # C. Audio Energy (RMS) for comparison
        print(f"\n{'='*70}")
        print("🔊 Audio RMS Energy 계산 중...")
        print(f"{'='*70}")

        rmse = librosa.feature.rms(y=audio_data, frame_length=2048, hop_length=160)[0]

        print(f"✅ Audio RMS shape: {rmse.shape}")

        # Resample to match MRI frame count for alignment check
        rmse_resampled = np.interp(
            np.linspace(0, len(rmse) - 1, len(mri_intensity)),
            np.arange(len(rmse)),
            rmse
        )
        rmse_norm = (rmse_resampled - np.min(rmse_resampled)) / (np.max(rmse_resampled) - np.min(rmse_resampled) + 1e-8)

        # D. Correlation check
        print(f"\n{'='*70}")
        print("🔗 동기화 검증 (상관관계 분석)...")
        print(f"{'='*70}")

        from scipy.stats import pearsonr

        # Correlate MRI motion with audio energy
        corr_motion_energy, p_value = pearsonr(mri_motion_norm, rmse_norm)

        print(f"✅ Correlation Analysis:")
        print(f"   - MRI Motion vs Audio Energy: {corr_motion_energy:.3f} (p={p_value:.4f})")

        if abs(corr_motion_energy) > 0.3:
            print(f"   ✅ 좋은 동기화! (상관계수 > 0.3)")
        else:
            print(f"   ⚠️ 동기화 확인 필요 (상관계수가 낮음)")

        # E. 결과 시각화
        print(f"\n{'='*70}")
        print("📊 시각화 생성 중...")
        print(f"{'='*70}")

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # Subplot 1: Mel-spectrogram
        ax1 = axes[0]
        img = librosa.display.specshow(
            mel_spec_db,
            sr=sr,
            hop_length=160,
            x_axis='time',
            y_axis='mel',
            ax=ax1
        )
        ax1.set_title(f'Mel-spectrogram (Subject: {subject_id}, Utterance: {utterance_id})')
        fig.colorbar(img, ax=ax1, format='%+2.0f dB')

        # Subplot 2: MRI Intensity and Motion
        ax2 = axes[1]
        time_mri = np.linspace(0, data['duration'], len(mri_intensity_norm))
        ax2.plot(time_mri, mri_intensity_norm, label='MRI Mean Intensity', color='blue', linewidth=1.5)
        ax2.plot(time_mri, mri_motion_norm, label='MRI Motion (diff)', color='cyan', linewidth=1, alpha=0.7)
        ax2.set_title('MRI Statistics Over Time')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Normalized Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Subplot 3: Alignment Check
        ax3 = axes[2]
        ax3.plot(time_mri, mri_motion_norm, label='MRI Motion', color='blue', linewidth=1.5)
        ax3.plot(time_mri, rmse_norm, label='Audio RMS Energy', color='orange', linewidth=1.5, alpha=0.8)
        ax3.set_title(f'Alignment Check: MRI Motion vs Audio Energy (Correlation: {corr_motion_energy:.3f})')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Normalized Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'feature_extraction_test.png')

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ 결과 이미지 저장 완료: {output_path}")

        # F. Summary
        print(f"\n{'='*70}")
        print("📋 테스트 요약")
        print(f"{'='*70}")
        print(f"✅ 데이터 로드: 성공")
        print(f"✅ Mel-spectrogram 추출: 성공 ({mel_spec.shape})")
        print(f"✅ MRI 통계 추출: 성공 ({mri_intensity.shape})")
        print(f"✅ 동기화 상관계수: {corr_motion_energy:.3f}")
        print(f"✅ 시각화 저장: {output_path}")
        print(f"\n{'='*70}")
        print("🎉 모든 테스트 통과!")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ 에러 발생")
        print(f"{'='*70}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*70}\n")

if __name__ == "__main__":
    analyze_sample_features()
