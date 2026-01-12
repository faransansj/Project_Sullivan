# Project Sullivan - HDDB 데이터셋 활용 워크플로우

**작성일**: 2026-01-11
**목적**: `/mnt/HDDB/dataset/my_dataset/` 데이터셋을 활용한 음성-발음기관 파라미터 예측 모델 개발
**데이터셋**: USC-TIMIT (27명 피험자, 약 879GB)

---

## 📊 데이터셋 현황

### 위치 및 크기
```
/mnt/HDDB/dataset/my_dataset/dataset/
├── 피험자: 27명 (sub010~sub019, sub030~sub039)
├── 총 용량: 879GB
└── 구조: 각 피험자당 2drt (2D real-time MRI) + 3d 데이터
```

### 데이터 구조
```
sub031/
├── 2drt/                   # 2D Real-time MRI (Phase 2 학습용)
│   ├── audio/             # 동기화된 오디오 (*.wav)
│   ├── raw/               # Raw k-space MRI 데이터
│   ├── recon/             # 재구성된 MRI 프레임 (*.mat)
│   └── video/             # MRI 비디오
└── 3d/                     # 3D volumetric MRI (Phase 3용)
    └── ...
```

### 예상 데이터량
- **총 발화 수**: 약 800-900개 utterances (27명 × 30-35 utterances/person)
- **총 프레임 수**: 약 4-5M frames (기존 75 utterances = 186K frames 기준 비례 계산)
- **학습 가능 규모**: 기존 대비 **20-25배 증가**

---

## 🎯 전체 워크플로우 개요

```
Phase 0: 환경 및 데이터 준비 (1-2일)
    ↓
Phase 1: 데이터 전처리 및 세그멘테이션 (2-3주)
    ↓
Phase 2: 모델 훈련 및 평가 (1-2주)
    ↓
Phase 3: 모델 최적화 및 배포 (1주)
```

**총 예상 소요 기간**: 5-7주

---

## Phase 0: 환경 및 데이터 준비 (1-2일)

### Task 0.1: 환경 검증 및 설정

**목적**: Project Sullivan 개발 환경 검증 및 대용량 데이터 처리 준비

#### 0.1.1 가상환경 활성화 및 의존성 확인
```bash
cd /home/Project_Sullivan
source venv_sullivan/bin/activate
pip list  # 주요 패키지 확인: torch, lightning, librosa, opencv-python
```

#### 0.1.2 디스크 공간 확인
```bash
# 처리된 데이터 저장 공간 확인 (최소 500GB 필요)
df -h /home/Project_Sullivan/data/

# 필요시 심볼릭 링크로 대용량 스토리지 연결
# ln -s /mnt/HDDB/processed_data /home/Project_Sullivan/data/processed_hddb
```

#### 0.1.3 데이터셋 접근 확인
```bash
# 데이터셋 접근 권한 확인
ls -lh /mnt/HDDB/dataset/my_dataset/dataset/

# 피험자 수 확인
ls /mnt/HDDB/dataset/my_dataset/dataset/ | wc -l  # 27명 확인

# 샘플 데이터 구조 확인
ls -lh /mnt/HDDB/dataset/my_dataset/dataset/sub031/2drt/
```

**체크포인트**:
- [ ] 가상환경 활성화 완료
- [ ] 디스크 공간 500GB+ 확보
- [ ] 데이터셋 읽기 권한 확인
- [ ] 27명 피험자 데이터 접근 가능

---

### Task 0.2: 데이터 구조 분석 및 메타데이터 생성

**목적**: 전체 데이터셋의 구조와 특성 파악

#### 0.2.1 데이터셋 통계 수집
```bash
# 스크립트 생성 또는 수정
python scripts/collect_dataset_stats.py \
  --data-root /mnt/HDDB/dataset/my_dataset/dataset \
  --output-file data/hddb_dataset_stats.json
```

**수집할 정보**:
- 각 피험자별 utterance 수
- MRI 프레임 수 (frame count per utterance)
- MRI 프레임 크기 및 FPS
- 오디오 샘플레이트 및 duration
- 총 처리 가능한 데이터량

#### 0.2.2 샘플 데이터 검증
```bash
# 샘플 utterance 로드 테스트
python -c "
from pathlib import Path
import scipy.io as sio
import soundfile as sf

# 샘플 MRI 파일 로드
mat_file = '/mnt/HDDB/dataset/my_dataset/dataset/sub031/2drt/recon/sub031_2drt_01_vcv1_r1_recon.mat'
audio_file = '/mnt/HDDB/dataset/my_dataset/dataset/sub031/2drt/audio/sub031_2drt_01_vcv1_r1_audio.wav'

# MRI 데이터 로드
mri_data = sio.loadmat(mat_file)
print(f'MRI keys: {mri_data.keys()}')
print(f'MRI shape: {mri_data[\"img\"].shape if \"img\" in mri_data else \"check keys\"}')

# 오디오 로드
audio, sr = sf.read(audio_file)
print(f'Audio shape: {audio.shape}, SR: {sr}')
"
```

**체크포인트**:
- [ ] 데이터셋 통계 JSON 파일 생성
- [ ] MRI 데이터 로드 가능 확인
- [ ] 오디오 데이터 로드 가능 확인
- [ ] 데이터 형식 (shape, dtype) 검증

---

### Task 0.3: 처리 전략 수립

**목적**: 27명 전체 또는 부분 처리 전략 결정

#### 전략 A: 단계적 처리 (권장)
```
1단계: 5명 피험자로 파이프라인 검증 (1-2일)
2단계: 15명으로 확장 (1주)
3단계: 전체 27명 처리 (2주)
```

#### 전략 B: 전체 일괄 처리
```
- 장점: 한 번에 모든 데이터 처리
- 단점: 오류 발생 시 재처리 비용 높음
- 예상 시간: 3-4주
```

**권장**: 전략 A (단계적 처리)
- 파이프라인 오류 조기 발견
- 중간 결과로 모델 훈련 시작 가능
- 디스크 공간 관리 용이

**체크포인트**:
- [ ] 처리 전략 선택
- [ ] 1단계 피험자 선정 (예: sub010~sub014)
- [ ] 처리 일정 수립

---

## Phase 1: 데이터 전처리 및 세그멘테이션 (2-3주)

### Task 1.1: MRI/오디오 정렬 및 전처리

**목적**: MRI 프레임과 오디오를 시간 동기화하고 노이즈 제거

#### 1.1.1 전처리 스크립트 수정
```bash
# src/preprocessing/data_loader.py 수정
# - /mnt/HDDB 경로 지원 추가
# - 2drt 데이터 구조 처리 로직 추가
```

**data_loader.py 수정 사항**:
```python
class USCTIMITLoader:
    def __init__(self, data_root='/mnt/HDDB/dataset/my_dataset/dataset'):
        self.data_root = Path(data_root)

    def load_utterance(self, subject, utterance_name):
        """
        Load 2drt MRI recon and audio
        subject: 'sub031'
        utterance_name: 'sub031_2drt_01_vcv1_r1'
        """
        # MRI recon 경로
        mri_path = self.data_root / subject / '2drt' / 'recon' / f'{utterance_name}_recon.mat'
        # Audio 경로
        audio_path = self.data_root / subject / '2drt' / 'audio' / f'{utterance_name}_audio.wav'

        # Load and return
        ...
```

#### 1.1.2 배치 전처리 실행 (1단계: 5명)
```bash
# 설정 파일 수정: configs/preprocess.yaml
# raw_data_path: /mnt/HDDB/dataset/my_dataset/dataset
# subjects: [sub010, sub011, sub012, sub013, sub014]
# output_path: data/processed_hddb/aligned

python scripts/batch_preprocess.py --config configs/preprocess_hddb.yaml
```

**처리 내용**:
- MRI/Audio 시간 정렬 (cross-correlation)
- MRI 프레임 denoising (Gaussian + Median filter)
- 정규화 및 크기 조정 (256×256)
- HDF5 형식으로 저장

**예상 시간**: 5명 × 30 utterances × 2분/utterance = **약 5시간**

#### 1.1.3 전처리 결과 검증
```bash
# 시각화 스크립트로 정렬 검증
python scripts/visualize_alignment.py \
  --input data/processed_hddb/aligned/sub010/sub010_2drt_01_vcv1_r1.h5 \
  --output results/alignment_check/
```

**체크포인트**:
- [ ] 5명 피험자 전처리 완료 (~150 utterances)
- [ ] HDF5 파일 생성 확인
- [ ] MRI/오디오 정렬 품질 검증
- [ ] 다음 10명 처리 시작 결정

---

### Task 1.2: U-Net 세그멘테이션

**목적**: 전처리된 MRI 프레임에서 성도(vocal tract) 세그멘테이션

#### 1.2.1 사전 훈련된 U-Net 모델 확인
```bash
# 기존 학습된 U-Net 모델 사용
ls -lh models/segmentation/unet_best.pth

# 모델 성능: Dice Score 81.8% (tongue 96.5%)
```

#### 1.2.2 세그멘테이션 실행 (1단계: 5명)
```bash
python scripts/segment_subset.py \
  --data-root data/processed_hddb/aligned \
  --subjects sub010,sub011,sub012,sub013,sub014 \
  --output-dir data/processed_hddb/segmentations \
  --checkpoint models/segmentation/unet_best.pth \
  --batch-size 32 \
  --device cuda  # GPU 사용 권장
```

**처리 속도**:
- CPU: ~22.8 fps
- GPU (T4): ~150-200 fps (예상)

**예상 시간** (GPU 기준):
- 5명 × 30 utterances × 200 frames / 150 fps = **약 2-3시간**

#### 1.2.3 세그멘테이션 품질 검증
```bash
# 랜덤 샘플 시각화
python scripts/visualize_segmentation.py \
  --input data/processed_hddb/segmentations/sub010/sub010_2drt_01_vcv1_r1_mask.npz \
  --output results/segmentation_check/
```

**체크포인트**:
- [ ] 5명 피험자 세그멘테이션 완료
- [ ] NPZ 마스크 파일 생성 확인
- [ ] 세그멘테이션 품질 육안 검증
- [ ] 다음 단계 진행 가능 여부 판단

---

### Task 1.3: 발음 기관 파라미터 추출

**목적**: 세그멘테이션 마스크로부터 저차원 articulatory parameters 추출

#### 1.3.1 파라미터 추출 실행
```bash
python scripts/extract_articulatory_params.py \
  --segmentation-dir data/processed_hddb/segmentations \
  --output-dir data/processed_hddb/parameters \
  --subjects sub010,sub011,sub012,sub013,sub014 \
  --param-type both  # geometric + pca
```

**추출되는 파라미터**:
1. **Geometric Features (14차원)**:
   - Tongue position (x, y, angle)
   - Jaw opening
   - Lip aperture
   - Velum position
   - etc.

2. **PCA Features (10차원)**:
   - 전체 마스크의 주성분 분석
   - 저차원 압축 표현

**출력 형식**: NPZ 파일 (shape: [T, 14] or [T, 10])

**예상 시간**: 5명 × 30 utterances × 30초/utterance = **약 1.5시간**

#### 1.3.2 파라미터 통계 분석
```bash
python scripts/analyze_parameters.py \
  --input-dir data/processed_hddb/parameters \
  --output results/parameter_stats.json
```

**분석 내용**:
- 각 파라미터의 평균, 표준편차, 범위
- 파라미터 간 상관관계
- 이상치(outlier) 검출

**체크포인트**:
- [ ] 파라미터 NPZ 파일 생성 확인
- [ ] 파라미터 차원 검증 (14 or 10)
- [ ] 통계 분석 완료
- [ ] 이상치 처리 방안 수립

---

### Task 1.4: 오디오 특징 추출

**목적**: 오디오 신호에서 mel-spectrogram 및 MFCC 추출

#### 1.4.1 오디오 특징 추출 실행
```bash
python scripts/extract_audio_features.py \
  --audio-dir data/processed_hddb/aligned \
  --output-dir data/processed_hddb/audio_features \
  --subjects sub010,sub011,sub012,sub013,sub014 \
  --feature-type mel  # or mfcc
  --n-mels 80 \
  --hop-length 160  # 10ms at 16kHz
```

**추출 특징**:
1. **Mel-spectrogram (80차원)** [주 특징]
   - Frequency bins: 80
   - Window: 25ms (400 samples @ 16kHz)
   - Hop: 10ms (160 samples)

2. **MFCC (13차원)** [대안 특징]
   - Cepstral coefficients: 13

**출력 형식**: NPZ 파일 (shape: [T, 80] or [T, 13])

**예상 시간**: 5명 × 30 utterances × 10초/utterance = **약 30분**

#### 1.4.2 특징 정렬 검증
```bash
# 오디오 특징과 articulatory 파라미터의 시간 정렬 확인
python scripts/verify_feature_alignment.py \
  --audio-features data/processed_hddb/audio_features/sub010/sub010_2drt_01_vcv1_r1_mel.npz \
  --parameters data/processed_hddb/parameters/sub010/sub010_2drt_01_vcv1_r1_geometric.npz
```

**체크포인트**:
- [ ] 오디오 특징 NPZ 파일 생성 확인
- [ ] 특징 차원 검증 (80 or 13)
- [ ] 시간 정렬 검증 (audio features ≈ parameters in time)
- [ ] 다음 단계 데이터 준비 완료

---

### Task 1.5: 데이터셋 분할 (Train/Val/Test)

**목적**: 학습/검증/테스트 세트 생성 (subject-level split)

#### 1.5.1 데이터셋 분할 실행
```bash
python scripts/create_dataset_splits.py \
  --data-root data/processed_hddb \
  --output-dir data/processed_hddb/splits \
  --split-ratios 0.7 0.15 0.15 \
  --split-by subject  # 피험자 단위 분할 (중요!)
  --seed 42
```

**분할 전략**:
- **Train**: 70% (약 19명) → ~570 utterances
- **Validation**: 15% (약 4명) → ~120 utterances
- **Test**: 15% (약 4명) → ~120 utterances

**중요**: Subject-level split으로 speaker-independent 모델 보장

**출력**: JSON 파일
```json
// data/processed_hddb/splits/train.json
[
  "sub010/sub010_2drt_01_vcv1_r1",
  "sub010/sub010_2drt_02_vcv2_r1",
  ...
]
```

#### 1.5.2 분할 검증
```bash
# 분할 통계 확인
python scripts/check_splits.py --splits-dir data/processed_hddb/splits

# Expected output:
# Train: 19 subjects, 570 utterances, ~1.4M frames
# Val: 4 subjects, 120 utterances, ~300K frames
# Test: 4 subjects, 120 utterances, ~300K frames
```

**체크포인트**:
- [ ] Train/Val/Test JSON 파일 생성
- [ ] 피험자 중복 없음 확인
- [ ] 각 세트의 데이터량 확인
- [ ] Phase 1 완료 ✅

---

### Task 1.6: 전체 데이터셋 확장 (선택 사항)

**목적**: 5명 → 27명 전체 처리 (성능 개선 필요시)

```bash
# 전체 27명 처리 (Task 1.1~1.5 반복)
# 예상 시간: 2-3주 (병렬 처리 시)

# 단계별 확장 권장:
# 1단계: 5명 완료 → 모델 학습 시작
# 2단계: 15명 추가 (총 20명) → 모델 개선
# 3단계: 나머지 7명 추가 (총 27명) → 최종 모델
```

**체크포인트**:
- [ ] 확장 필요성 판단 (5명 학습 결과 기반)
- [ ] 추가 처리 일정 수립

---

## Phase 2: 모델 훈련 및 평가 (1-2주)

### Task 2.1: Baseline LSTM 모델 훈련

**목적**: 간단한 Bi-LSTM 모델로 파이프라인 검증 및 baseline 성능 확보

#### 2.1.1 설정 파일 수정
```bash
# configs/baseline_config_hddb.yaml 생성
cp configs/baseline_config.yaml configs/baseline_config_hddb.yaml

# 수정 내용:
# data:
#   splits_dir: data/processed_hddb/splits
#   audio_feature_dir: data/processed_hddb/audio_features
#   parameter_dir: data/processed_hddb/parameters
```

#### 2.1.2 Quick Test 실행
```bash
# 작은 데이터로 파이프라인 검증
python scripts/train_baseline.py \
  --config configs/baseline_quick_test.yaml \
  --fast-dev-run

# 문제 없으면 전체 훈련
```

#### 2.1.3 전체 훈련 실행 (CPU 가능)
```bash
# CPU 훈련 (느림, 1-2일 예상)
python scripts/train_baseline.py --config configs/baseline_config_hddb.yaml

# GPU 훈련 권장 (수 시간)
python scripts/train_baseline.py \
  --config configs/baseline_config_hddb.yaml \
  --gpus 1
```

**모델 구조**:
```
Input: [Batch, Time, 80] (mel-spectrogram)
  ↓
Bi-LSTM (2 layers, 128 hidden)
  ↓
FC Layer
  ↓
Output: [Batch, Time, 14] (articulatory params)

Loss: MSE with mask
Parameters: ~613K
```

#### 2.1.4 TensorBoard 모니터링
```bash
# 별도 터미널에서 실행
bash scripts/start_tensorboard.sh

# 브라우저에서 http://localhost:6006 접속
# - Training/Validation loss 확인
# - RMSE, MAE, Pearson correlation 확인
```

**예상 성능** (5명 데이터 기준):
- **Target**: RMSE < 0.15, Pearson > 0.50 (M2)
- **Baseline 예상**: RMSE ~0.4-0.6, Pearson ~0.3-0.4
- **기존 baseline (75 utterances)**: RMSE 1.011, Pearson 0.105

**예상 시간**:
- GPU: 3-5시간 (50 epochs)
- CPU: 1-2일

**체크포인트**:
- [ ] 훈련 완료 (early stopping)
- [ ] 최종 checkpoint 저장
- [ ] Validation 성능 기록
- [ ] Test set 평가 완료

---

### Task 2.2: Transformer 모델 훈련

**목적**: 더 강력한 Transformer 아키텍처로 M2 목표 달성

#### 2.2.1 설정 파일 수정
```bash
# configs/transformer_config_hddb.yaml 생성
cp configs/transformer_config.yaml configs/transformer_config_hddb.yaml

# 수정 내용:
# data:
#   splits_dir: data/processed_hddb/splits
#   audio_feature_dir: data/processed_hddb/audio_features
#   parameter_dir: data/processed_hddb/parameters
```

#### 2.2.2 Transformer 훈련 (GPU 필수)
```bash
# GPU 필요 (5M parameters)
python scripts/train_transformer.py \
  --config configs/transformer_config_hddb.yaml \
  --gpus 1
```

**모델 구조**:
```
Input: [Batch, Time, 80]
  ↓
Input Projection: 80 → 256 (d_model)
  ↓
Positional Encoding (learnable)
  ↓
Transformer Encoder (4 layers, 8 heads)
  ↓
Output Projection: 256 → 14
  ↓
Output: [Batch, Time, 14]

Parameters: ~5M
```

**하이퍼파라미터**:
- d_model: 256
- num_layers: 4
- num_heads: 8
- d_ff: 1024
- dropout: 0.1
- learning_rate: 5e-4 (AdamW)
- batch_size: 16 (GPU), 8 (CPU)

**예상 성능** (5명 데이터):
- **Target**: RMSE < 0.15, Pearson > 0.50
- **예상 달성**: RMSE ~0.12-0.18, Pearson ~0.45-0.60

**예상 시간**:
- GPU (T4): 6-8시간 (50 epochs)
- GPU (V100): 3-4시간

#### 2.2.3 Google Colab 훈련 (대안)

무료 GPU 사용을 원하면:

```bash
# 1. 데이터 아카이브 생성
bash scripts/prepare_data_for_colab.sh

# 2. Google Drive에 업로드
# colab_data_archives/processed_data_all.tar.gz

# 3. Colab 노트북 사용
# notebooks/Project_Sullivan_Transformer_Training.ipynb
```

**체크포인트**:
- [ ] Transformer 훈련 완료
- [ ] M2 목표 달성 여부 확인
- [ ] Best model checkpoint 저장
- [ ] Test set 평가 및 성능 분석

---

### Task 2.3: 모델 평가 및 분석

**목적**: Test set에서 최종 성능 측정 및 상세 분석

#### 2.3.1 Test Set 평가
```bash
# Baseline 평가
python scripts/evaluate_model.py \
  --checkpoint models/baseline_lstm_hddb/best.ckpt \
  --config configs/baseline_config_hddb.yaml \
  --split test \
  --output results/baseline_hddb_evaluation.json

# Transformer 평가
python scripts/evaluate_model.py \
  --checkpoint models/transformer_hddb/best.ckpt \
  --config configs/transformer_config_hddb.yaml \
  --split test \
  --output results/transformer_hddb_evaluation.json
```

**측정 지표**:
1. **RMSE** (Root Mean Square Error) - 전체 및 파라미터별
2. **MAE** (Mean Absolute Error)
3. **Pearson Correlation** - 전체 및 파라미터별
4. **R² Score**

#### 2.3.2 파라미터별 상세 분석
```bash
python scripts/analyze_predictions.py \
  --predictions results/transformer_hddb_predictions.npz \
  --ground-truth data/processed_hddb/parameters/test/ \
  --output results/parameter_analysis/
```

**분석 내용**:
- Tongue position prediction accuracy
- Jaw opening prediction accuracy
- Lip aperture prediction accuracy
- 파라미터 간 오차 상관관계

#### 2.3.3 시각화
```bash
python scripts/visualize_predictions.py \
  --predictions results/transformer_hddb_predictions.npz \
  --ground-truth data/processed_hddb/parameters/test/ \
  --audio-features data/processed_hddb/audio_features/test/ \
  --output results/visualizations/ \
  --num-samples 10
```

**생성 결과**:
- 예측 vs 실제 시계열 그래프
- 오차 히트맵
- 파라미터별 산점도
- Attention weights 시각화 (Transformer)

**체크포인트**:
- [ ] Test RMSE, Pearson correlation 계산
- [ ] M2 목표 달성 여부 판단 (RMSE < 0.15, PCC > 0.50)
- [ ] 파라미터별 성능 분석 완료
- [ ] 시각화 결과 생성

---

### Task 2.4: 성능 개선 (필요시)

**목적**: M2 목표 미달성 시 추가 개선

#### 2.4.1 하이퍼파라미터 튜닝
```bash
# 학습률, 모델 크기, dropout 등 조정
# configs/transformer_config_hddb_v2.yaml 생성
```

**튜닝 대상**:
- Learning rate: 1e-4 ~ 1e-3
- Model size: d_model=128/256/512
- Layers: 2/4/6
- Batch size: 8/16/32

#### 2.4.2 데이터 증강
```python
# 구현 필요: src/modeling/augmentation.py
- Time stretching (0.9x ~ 1.1x)
- Pitch shifting (±2 semitones)
- Noise injection (SNR 20-40dB)
```

#### 2.4.3 전체 데이터셋 활용
```bash
# 5명 → 27명 확장 (Task 1.6 실행)
# 20-25배 데이터 증가로 성능 대폭 향상 예상
```

**예상 개선**:
- 5명 → 15명: RMSE 10-15% 감소
- 5명 → 27명: RMSE 20-30% 감소

**체크포인트**:
- [ ] 개선 전략 선택
- [ ] 추가 실험 완료
- [ ] M2 목표 달성 ✅

---

## Phase 3: 모델 최적화 및 배포 (1주)

### Task 3.1: 모델 최적화

**목적**: 실시간 추론을 위한 모델 경량화 및 속도 개선

#### 3.1.1 모델 양자화 (Optional)
```bash
# PyTorch quantization
python scripts/quantize_model.py \
  --checkpoint models/transformer_hddb/best.ckpt \
  --output models/transformer_hddb/quantized.ckpt \
  --dtype int8
```

**효과**:
- 모델 크기 50-75% 감소
- 추론 속도 1.5-2배 향상
- 성능 저하 < 2%

#### 3.1.2 ONNX 변환
```bash
# ONNX 형식으로 변환 (다양한 플랫폼 지원)
python scripts/export_to_onnx.py \
  --checkpoint models/transformer_hddb/best.ckpt \
  --output models/transformer_hddb/model.onnx
```

**체크포인트**:
- [ ] 양자화 모델 생성 (선택)
- [ ] ONNX 모델 생성 (선택)
- [ ] 추론 속도 측정

---

### Task 3.2: 추론 파이프라인 구축

**목적**: 실시간 오디오 → 발음 기관 파라미터 변환 시스템

#### 3.2.1 추론 스크립트 작성
```python
# scripts/infer_realtime.py
import torch
from src.modeling.transformer import TransformerModel
from src.audio_features.mel_spectrogram import extract_mel

def infer(audio_path, model_path):
    # Load model
    model = TransformerModel.load_from_checkpoint(model_path)
    model.eval()

    # Load audio
    audio, sr = load_audio(audio_path)

    # Extract features
    mel = extract_mel(audio, sr)

    # Predict
    with torch.no_grad():
        params = model(mel)

    return params
```

#### 3.2.2 실시간 추론 테스트
```bash
python scripts/infer_realtime.py \
  --audio test_audio.wav \
  --checkpoint models/transformer_hddb/best.ckpt \
  --output predictions.npz
```

**성능 목표**:
- 추론 속도: > 10x realtime (2초 오디오를 0.2초 이내 처리)
- Latency: < 100ms

**체크포인트**:
- [ ] 추론 스크립트 작성 완료
- [ ] 실시간 성능 검증
- [ ] 배포 준비 완료

---

### Task 3.3: 문서화 및 최종 보고서

**목적**: 프로젝트 결과 문서화 및 아카이빙

#### 3.3.1 최종 보고서 작성
```markdown
# docs/HDDB_PROJECT_REPORT.md

1. Executive Summary
   - 목표: 음성 → 발음기관 파라미터 예측
   - 데이터: 27명 USC-TIMIT (5명으로 시작)
   - 결과: Test RMSE = X.XXX, PCC = X.XXX

2. Dataset
   - 데이터셋 구조 및 통계
   - 전처리 과정

3. Methodology
   - 세그멘테이션 (U-Net)
   - 특징 추출 (Mel-spectrogram, Geometric parameters)
   - 모델 아키텍처 (Transformer)

4. Results
   - 성능 지표
   - 파라미터별 분석
   - 시각화

5. Conclusion & Future Work
```

#### 3.3.2 코드 정리 및 주석
```bash
# 코드 포맷팅
black src/ scripts/

# Docstring 추가
# 각 함수에 명확한 설명 추가
```

#### 3.3.3 모델 및 데이터 아카이빙
```bash
# 최종 모델 저장
mkdir -p models/final_release/
cp models/transformer_hddb/best.ckpt models/final_release/transformer_hddb_v1.ckpt

# 메타데이터 저장
cat > models/final_release/model_info.json << EOF
{
  "model": "Transformer",
  "version": "1.0",
  "train_subjects": 19,
  "test_rmse": 0.XXX,
  "test_pearson": 0.XXX,
  "date": "2026-01-XX"
}
EOF
```

**체크포인트**:
- [ ] 최종 보고서 작성
- [ ] 코드 정리 및 문서화
- [ ] 모델 아카이빙
- [ ] 프로젝트 완료 ✅

---

## 📊 리소스 요구사항

### 컴퓨팅 리소스

**Phase 1 (전처리)**:
- CPU: 8+ cores 권장
- RAM: 32GB+ (대용량 MRI 데이터 처리)
- Storage: 500GB+ free space
- 예상 시간: 2-3주 (5명 기준 5-7일)

**Phase 2 (훈련)**:
- GPU: 16GB+ VRAM (NVIDIA T4/V100/RTX 3090)
- RAM: 16GB+
- 예상 시간:
  - Baseline LSTM: 3-5시간 (GPU)
  - Transformer: 6-8시간 (GPU)

**대안 (GPU 없는 경우)**:
- Google Colab (무료 T4 GPU, 12시간 세션)
- AWS/GCP GPU 인스턴스 (시간당 $1-3)

### 스토리지 예상 사용량

```
원본 데이터: 879GB (read-only)
전처리 데이터:
  - Aligned HDF5: ~50GB (5명) / ~270GB (27명)
  - Segmentations: ~30GB (5명) / ~160GB (27명)
  - Parameters: ~5GB (5명) / ~27GB (27명)
  - Audio features: ~10GB (5명) / ~54GB (27명)

Total: ~95GB (5명) / ~511GB (27명)

모델 체크포인트: ~5GB
로그 및 결과: ~10GB

총 필요 공간: ~600GB+ (여유 공간 포함)
```

---

## ⚠️ 주요 주의사항

### 1. 데이터 무결성
- 전처리 과정에서 원본 데이터 수정 금지
- 각 단계마다 체크포인트 파일 생성
- 정기적인 백업 수행

### 2. 피험자 단위 Split
- **반드시 subject-level split 사용**
- Utterance-level split은 data leakage 발생 (동일 화자가 train/test에 중복)
- Speaker-independent 모델 보장

### 3. 메모리 관리
- 대용량 데이터 처리 시 OOM 발생 가능
- Batch processing 사용
- Streaming dataset 모드 활용

### 4. GPU 메모리
- Transformer 훈련 시 16GB VRAM 필요
- OOM 발생 시:
  - Batch size 감소 (16 → 8)
  - Gradient accumulation 증가
  - Model size 축소 (d_model=256 → 128)

### 5. 실험 추적
- 모든 실험에 고유 ID 부여
- TensorBoard 로그 보존
- 재현성을 위한 random seed 고정 (42)

---

## 🎯 성공 기준 (Milestone M2)

### 정량적 지표
- [x] **Test RMSE < 0.15** (목표)
- [x] **Test Pearson Correlation > 0.50** (목표)
- [x] 파라미터별 correlation > 0.40 (모든 파라미터)

### 정성적 지표
- [x] 시각화 결과 육안 검증 통과
- [x] Attention weights가 의미 있는 패턴 보임
- [x] 재현 가능한 파이프라인 구축

### 문서화
- [x] 전체 워크플로우 문서화
- [x] 코드 주석 및 docstring 완비
- [x] 최종 보고서 작성

---

## 📝 체크리스트 요약

### Phase 0: 환경 준비 (1-2일)
- [ ] 가상환경 활성화 및 의존성 확인
- [ ] 디스크 공간 500GB+ 확보
- [ ] 데이터셋 접근 확인 (27명 피험자)
- [ ] 데이터 구조 분석 및 메타데이터 생성
- [ ] 처리 전략 수립 (단계적 vs 일괄)

### Phase 1: 데이터 전처리 (2-3주 for 전체, 5-7일 for 5명)
- [ ] MRI/오디오 정렬 및 전처리 (5명)
- [ ] U-Net 세그멘테이션 (5명)
- [ ] 발음 기관 파라미터 추출 (5명)
- [ ] 오디오 특징 추출 (5명)
- [ ] 데이터셋 분할 (Train/Val/Test)
- [ ] (선택) 전체 27명으로 확장

### Phase 2: 모델 훈련 (1-2주)
- [ ] Baseline LSTM 훈련 및 평가
- [ ] Transformer 훈련 및 평가
- [ ] M2 목표 달성 확인 (RMSE < 0.15, PCC > 0.50)
- [ ] (필요시) 성능 개선 및 재훈련

### Phase 3: 최적화 및 배포 (1주)
- [ ] 모델 최적화 (양자화, ONNX)
- [ ] 추론 파이프라인 구축
- [ ] 최종 보고서 작성
- [ ] 프로젝트 완료 🎉

---

## 📚 참고 자료

### 내부 문서
- `CLAUDE.md` - 개발 가이드
- `README.md` - 프로젝트 개요
- `docs/BASELINE_PERFORMANCE_REPORT.md` - Baseline 성능 분석
- `docs/COLAB_TRAINING_GUIDE.md` - Colab 훈련 가이드

### 외부 자료
- USC-TIMIT Dataset: https://doi.org/10.6084/m9.figshare.13725546.v1
- Paper: https://arxiv.org/abs/2102.07896
- U-Net Paper: https://arxiv.org/abs/1505.04597
- Transformer Paper: https://arxiv.org/abs/1706.03762

---

**작성자**: Claude Code Assistant
**최종 수정**: 2026-01-11
**버전**: 1.0
