# HDDB 데이터셋 파이프라인 진행 보고서

**날짜**: 2026-01-11
**상태**: Phase 0-1A 완료, Phase 1B 진행 중

---

## ✅ 완료된 작업

### Phase 0: 환경 및 데이터 분석
**소요 시간**: 약 30분

#### 환경 설정
- ✅ 새로운 Python 가상환경 생성 (`.venv_new`)
- ✅ 모든 필수 패키지 설치:
  - PyTorch 2.9.1 (CPU)
  - PyTorch Lightning 2.6.0
  - h5py, scipy, scikit-learn
  - librosa, soundfile, opencv-python
  - matplotlib, seaborn, tqdm

#### 데이터셋 분석
- ✅ HDDB 데이터셋 구조 파악
  - 위치: `/mnt/HDDB/dataset/my_dataset/dataset/`
  - 27명 피험자 (sub010 ~ sub039)
  - 800개 utterances 발견

- ✅ 데이터 형식 확인
  - MRI: HDF5 (.h5) 형식, shape (T, 84, 84), T~2490 frames
  - 오디오: WAV 형식, 20kHz 샘플레이트
  - FPS: ~81.5 fps

- ✅ 데이터셋 통계 생성
  - 파일: `data/hddb_dataset_stats.json`
  - 평균 29.6 utterances/피험자

**출력 파일**:
- `data/hddb_dataset_stats.json` - 전체 데이터셋 통계
- `scripts/analyze_hddb_dataset.py` - 데이터 분석 스크립트

---

### Phase 1-A: HDDB 전용 데이터 로더 개발
**소요 시간**: 약 20분

#### 구현 내용
- ✅ `src/preprocessing/hddb_data_loader.py` 작성
  - `HDDBLoader` 클래스: HDDB H5+WAV 형식 지원
  - MRI H5 파일 로드: `load_mri_from_h5()`
  - 오디오 WAV 파일 로드: `load_audio()`
  - Utterance 로드: `load_utterance()`
  - Subject 로드: `load_subject_utterances()`

#### 테스트 결과
- ✅ 로더 테스트 통과
  - 테스트 스크립트: `scripts/test_hddb_loader.py`
  - 샘플 utterance 로드 성공
  - MRI: (2490, 84, 84), FPS: 81.49
  - 오디오: 611,124 samples @ 20kHz, 30.56s

**출력 파일**:
- `src/preprocessing/hddb_data_loader.py` - HDDB 데이터 로더
- `scripts/test_hddb_loader.py` - 로더 테스트 스크립트

---

## 🚧 현재 상태 및 차단 요소

### Phase 1-B: 세그멘테이션 파이프라인 준비
**상태**: 🔴 차단됨

#### 문제
**U-Net 세그멘테이션 모델이 존재하지 않음**

프로젝트 문서 (README.md, CLAUDE.md)에서 81.8% Dice score의 사전 학습된 U-Net 모델을 언급했지만, 실제 모델 파일이 없습니다:
- `models/segmentation/unet_best.pth` - **존재하지 않음**
- `models/unet_scratch/` - **디렉토리 없음**

#### 영향
Articulatory parameter 추출을 위해서는 vocal tract segmentation이 필수입니다. 세그멘테이션 없이는 다음 단계 진행 불가:
- ❌ Geometric features 추출 불가
- ❌ PCA features 추출 불가
- ❌ 모델 훈련 불가

---

## 🎯 다음 단계 옵션

### 옵션 A: U-Net 모델 학습 (완전한 파이프라인)
**예상 시간**: 1-2일

#### 필요 작업
1. **Pseudo-label 생성** (5-10시간)
   - Traditional CV 기법으로 고품질 마스크 생성
   - 150-200 프레임 수동 선택 및 검증
   - 스크립트: `scripts/generate_pseudo_labels.py`

2. **U-Net 훈련** (2-3시간)
   - 5-class segmentation (background, tongue, jaw, lips, velum)
   - 목표: Dice score > 70% (가능하면 80%+)
   - 스크립트: `scripts/train_unet.py`

3. **모델 평가 및 검증** (30분)
   - Test set 평가
   - 시각화 및 품질 검증
   - 스크립트: `scripts/evaluate_unet.py`

**장점**:
- ✅ 완전한 파이프라인 구축
- ✅ 고품질 articulatory parameters
- ✅ 재현 가능한 워크플로우

**단점**:
- ⏱️ 시간 소요 큼 (1-2일)
- 💻 GPU 필요 (또는 CPU로 매우 느림)

---

### 옵션 B: 세그멘테이션 없이 직접 학습 (실험적)
**예상 시간**: 몇 시간

#### 접근 방법
MRI 프레임을 직접 CNN으로 처리하여 articulatory features 추출
- Input: 오디오 (mel-spectrogram)
- Target: MRI 프레임 자체 또는 간단한 통계적 features
- 모델: CNN-LSTM 또는 Transformer with vision encoder

**장점**:
- ⏱️ 빠른 시작
- 🔬 실험적 접근

**단점**:
- ❓ 성능 미지수
- ❌ Interpretability 부족
- ❌ Geometric features 없음

---

### 옵션 C: 단계별 접근 (권장)
**예상 시간**: 유동적

#### 접근 방법
1. **현재 상태 문서화** ✅ (완료)
2. **간단한 baseline 테스트**
   - 800개 utterance 중 1개로 end-to-end 테스트
   - 데이터 로드 → 간단한 feature 추출 → 저장
   - 파이프라인 검증
3. **U-Net 학습 결정**
   - Baseline 결과 확인 후 결정
   - 필요하면 옵션 A 진행
4. **점진적 확장**
   - 1개 → 5개 → 27개 utterances

**장점**:
- ✅ 단계별 검증
- ✅ 문제 조기 발견
- ✅ 유연한 접근

**단점**:
- ⏱️ 전체 시간은 비슷

---

## 📦 생성된 파일 및 스크립트

### 데이터 분석
- `scripts/analyze_hddb_dataset.py` - 데이터셋 통계 수집
- `data/hddb_dataset_stats.json` - 데이터셋 통계 결과

### 데이터 로더
- `src/preprocessing/hddb_data_loader.py` - HDDB 데이터 로더
- `scripts/test_hddb_loader.py` - 로더 테스트

### 문서
- `WORKFLOW_HDDB_DATASET.md` - 전체 워크플로우 상세 가이드
- `HDDB_QUICK_START.md` - 빠른 시작 가이드
- `scripts/hddb_pipeline.sh` - 자동화 파이프라인 스크립트 (테스트 필요)
- `HDDB_PROGRESS_REPORT.md` - 이 파일

---

## 💡 권장 사항

### 즉시 실행 가능한 작업

#### 1. 간단한 파이프라인 테스트 (30분)
현재 로더와 인프라로 1개 utterance를 처리해보기:

```bash
source .venv_new/bin/activate
python << EOF
from src.preprocessing.hddb_data_loader import HDDBLoader

# Load data
loader = HDDBLoader('/mnt/HDDB/dataset/my_dataset/dataset')
data = loader.load_utterance('sub010_2drt_01_vcv1_r1')

print(f"MRI shape: {data['mri_shape']}")
print(f"Audio shape: {data['audio'].shape}")
print(f"Duration: {data['duration']:.2f}s")
print(f"FPS: {data['fps']:.2f}")

# 간단한 처리 예시
import numpy as np

# MRI에서 간단한 feature 추출 (예: 평균 intensity)
mri_features = np.mean(data['mri_frames'], axis=(1,2))  # (T,)
print(f"MRI features shape: {mri_features.shape}")

# 오디오에서 mel-spectrogram 추출
import librosa
mel = librosa.feature.melspectrogram(
    y=data['audio'],
    sr=data['audio_sr'],
    n_mels=80,
    hop_length=160
)
print(f"Mel spectrogram shape: {mel.shape}")

print("\n✅ Basic pipeline test passed!")
EOF
```

#### 2. 전체 피험자 데이터 로드 가능 여부 확인 (10분)
```bash
source .venv_new/bin/activate
python << EOF
from src.preprocessing.hddb_data_loader import HDDBLoader

loader = HDDBLoader('/mnt/HDDB/dataset/my_dataset/dataset')

# 각 피험자당 utterance 수 확인
for subject in loader.get_subject_list()[:5]:
    utts = loader.get_utterance_list(subject)
    print(f"{subject}: {len(utts)} utterances")
EOF
```

---

## 📈 예상 타임라인

### 시나리오 1: 옵션 A (완전한 파이프라인)
```
Day 1-2: U-Net 학습
  - Pseudo-label 생성: 5-10h
  - U-Net 훈련: 2-3h
  - 검증: 0.5h

Day 3-5: 데이터 전처리 (5명)
  - 세그멘테이션: 2-3h (GPU)
  - Parameter 추출: 1.5h
  - Audio features: 0.5h
  - 분할: 0.1h

Day 6-7: 모델 훈련
  - Baseline LSTM: 3-5h
  - Transformer: 6-8h
  - 평가: 1-2h

Total: 7일
```

### 시나리오 2: 옵션 C (단계별)
```
Day 1: 테스트 및 검증
  - Basic pipeline test: 0.5h
  - 1 utterance 처리: 1h
  - 결과 분석: 0.5h

Day 2+: 필요에 따라 확장
```

---

## 🔧 기술적 세부 사항

### 환경
- **Python**: 3.13.11
- **가상환경**: `/home/Project_Sullivan/.venv_new`
- **PyTorch**: 2.9.1+cpu
- **디스크 여유 공간**: 81GB available (14% used)

### 데이터셋
- **경로**: `/mnt/HDDB/dataset/my_dataset/dataset/`
- **피험자**: 27명 (sub010-sub039, sub036 누락)
- **Utterances**: 800개 (일부 오디오 누락)
- **MRI 형식**: HDF5, (T, 84, 84), T~2490, dtype float64
- **오디오 형식**: WAV, 20kHz, mono

### 하드웨어
- **CPU**: 사용 가능
- **GPU**: 상태 미확인 (torch.cuda.is_available() 필요)
- **RAM**: 충분 (대용량 MRI 데이터 처리 가능)

---

## 📞 다음 행동

**결정 필요**: 어떤 옵션으로 진행할지 선택

1. **옵션 A**: 완전한 파이프라인 (U-Net 학습부터)
2. **옵션 B**: 실험적 접근 (세그멘테이션 없이)
3. **옵션 C**: 단계별 접근 (권장)

**현재 준비 상태**:
- ✅ 환경 완비
- ✅ 데이터 로더 완성
- ✅ 데이터셋 분석 완료
- ⏸️ 다음 단계 대기 중

---

**보고서 작성**: 2026-01-11
**작성자**: Claude Code Assistant
**버전**: 1.0
