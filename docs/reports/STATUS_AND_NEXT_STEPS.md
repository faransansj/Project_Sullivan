# Project Sullivan - 현재 상황 및 다음 단계

**날짜:** 2026-01-11
**작성자:** Claude Code

---

## ✅ 완료된 작업

### 1. 환경 설정 성공
- ✅ UV 패키지 관리자 설치
- ✅ PyTorch **CPU 버전** 설치 (AMD 서버, NVIDIA CUDA 제외)
- ✅ 필수 라이브러리 설치 완료:
  - torch 2.9.1+cpu
  - numpy 2.2.6
  - librosa 0.11.0
  - opencv-python-headless 4.12.0
  - h5py 3.15.1
  - scipy, scikit-learn, lightning 등

### 2. 데이터셋 통합 성공
- ✅ 600GB USC-TIMIT 데이터 접근 설정
  - 원본: `/mnt/HDDB/dataset/my_dataset/dataset/`
  - 링크: `/home/Project_Sullivan/data/raw/usc_timit_full/`
  - 링크: `/home/Project_Sullivan/data/raw/usc_timit_data/` (스크립트 호환성)
- ✅ 27명 피험자, ~840개 utterances 확인

### 3. 데이터 구조 파악
- ✅ **오디오와 MRI가 이미 분리되어 있음** (ffmpeg 불필요!)
  ```
  sub011/2drt/
  ├── audio/
  │   └── sub011_2drt_01_vcv1_r1_audio.wav (20kHz, ~29초)
  ├── recon/
  │   └── sub011_2drt_01_vcv1_r1_recon.h5 (2373 frames, 84×84)
  └── video/
      └── sub011_2drt_01_vcv1_r1_video.mp4
  ```

---

## ⚠️ 발견된 문제

### 전처리 데이터 부재
- ❌ **Aligned HDF5 파일**: 0개
- ❌ **Segmentation masks**: 없음
- ❌ **Articulatory parameters**: 없음
- ❌ **Audio features**: 없음
- ❌ **Train/Val/Test splits**: 없음

**원인:**
- `data/processed/aligned/`에 metadata JSON만 존재
- 실제 HDF5 파일이 한 번도 생성되지 않음
- Git에 전처리 데이터가 포함되지 않았음

### U-Net 모델 부재
- ❌ **Segmentation model weights**: 없음
- ✅ **Model code**: 존재 (`src/segmentation/unet.py`)
- ⚠️ **Pretrained weights 폴더**: 있으나 파일 없음

---

## 🔄 현재 상황 요약

```
[현재 위치]
데이터 다운로드 ✓ → 환경 설정 ✓ → 전처리 ❌ → 학습 ⬜

[필요한 작업]
1. 전처리 (Phase 1) - 5-10시간
2. Segmentation - 1-2시간
3. Parameter Extraction - 1시간
4. Audio Feature Extraction - 1시간
5. 학습 (Phase 2) - 10-20시간
```

**예상 총 소요 시간:** 18-34시간 (CPU 환경)

---

## 🎯 옵션 및 권장 사항

### 옵션 A: 전체 파이프라인 실행 (최선, 시간 많이 소요)

**장점:** 완전한 데이터 활용 (~840 utterances)
**단점:** 18-34시간 소요

**단계:**

#### 1. MRI 데이터 직접 로드 스크립트 작성
```python
# h5 + wav를 직접 로드하는 새 스크립트
# batch_preprocess_h5.py
```

#### 2. U-Net 학습 또는 대안
- **옵션 2-A:** U-Net 처음부터 학습 (2-3시간, GPU 권장)
- **옵션 2-B:** Simple threshold 기반 segmentation (빠름, 정확도↓)
- **옵션 2-C:** Pretrained model 다운로드 (있다면)

#### 3. 전체 파이프라인 실행
```bash
# 1. h5 → segments
# 2. segments → parameters
# 3. wav → audio features
# 4. 학습
```

---

### 옵션 B: 소규모 테스트 (권장, 빠른 검증)

**장점:** 2-4시간 내 결과
**단점:** 제한된 데이터 (1-2명, ~32 utterances)

**단계:**

1. **1-2명 피험자만 수동 전처리**
   ```bash
   # sub011, sub012만 처리
   # 총 64 utterances
   ```

2. **간단한 segmentation**
   - Threshold 기반 또는 Edge detection
   - U-Net 없이 진행

3. **빠른 학습**
   - Baseline LSTM으로 테스트
   - 성능 확인 후 확장 결정

---

### 옵션 C: 기존 전처리 데이터 다운로드 (최선, 가능하다면)

**확인 필요:**
```bash
# Google Drive, Figshare 등에 전처리 데이터가 공유되어 있는지 확인
# researcher_manual.md 또는 README에 링크가 있을 수 있음
```

**장점:** 즉시 학습 시작
**단점:** 데이터 소스 찾기 어려움

---

## 💡 즉시 실행 가능한 작업

### 1. 간단한 데이터 로더 작성 (30분)

```python
# quick_data_loader.py
import h5py
import librosa
import numpy as np

def load_utterance(subject_id, utterance_name):
    """h5와 wav 직접 로드"""
    base = f"data/raw/usc_timit_data/{subject_id}/2drt"

    # MRI
    h5_path = f"{base}/recon/{utterance_name}_recon.h5"
    with h5py.File(h5_path, 'r') as f:
        mri = f['recon'][:]  # (T, H, W)

    # Audio
    wav_path = f"{base}/audio/{utterance_name}_audio.wav"
    audio, sr = librosa.load(wav_path, sr=20000)

    return mri, audio, sr
```

### 2. Simple segmentation (1시간)

```python
# simple_segmentation.py
def threshold_segment(mri_frame, threshold=0.5):
    """간단한 threshold 기반 분할"""
    normalized = (mri_frame - mri_frame.min()) / (mri_frame.max() - mri_frame.min())
    mask = normalized > threshold
    return mask.astype(np.uint8)
```

### 3. Geometric feature 추출 (30분)

```python
# extract_simple_features.py
from skimage import measure

def extract_geometric_features(mask):
    """마스크에서 기하학적 특징 추출"""
    props = measure.regionprops(mask)[0]
    features = [
        props.area,
        props.centroid[0],
        props.centroid[1],
        props.major_axis_length,
        props.minor_axis_length,
        # ... 등
    ]
    return np.array(features)
```

---

## 🚀 추천 진행 방안

### 단계 1: 소규모 검증 (오늘, 2-4시간)
1. sub011 1개 utterance로 전체 파이프라인 테스트
2. 간단한 segmentation으로 parameter 추출
3. Audio feature 추출
4. 초소형 모델로 학습 가능성 검증

### 단계 2: 중규모 확장 (내일, 6-8시간)
1. 5명 피험자 × 10 utterances = 50 samples
2. 검증된 파이프라인으로 처리
3. Baseline 모델 학습

### 단계 3: 전체 확장 (주말, 20-30시간)
1. 전체 데이터셋 처리
2. 고성능 모델 학습
3. 목표 RMSE < 0.10 달성

---

## 📋 다음 질문

**선택해주세요:**

1. **옵션 A**: 전체 파이프라인 구현 (시간 많이 소요, 완전한 결과)
2. **옵션 B**: 소규모 테스트 먼저 (빠른 검증, 점진적 확장) ⭐ **권장**
3. **옵션 C**: 기존 전처리 데이터 찾기

**또는:**
- "sub011 1개 utterance로 빠른 테스트 해줘"
- "간단한 전처리 스크립트부터 작성해줘"
- "U-Net 없이 threshold segmentation으로 시작해줘"

---

**작성 완료:** 2026-01-11 08:40
**환경:** AMD 서버, CPU only, 8GB RAM
**데이터:** 600GB USC-TIMIT 접근 가능

다음 명령을 기다리고 있습니다! 🎯
