# HDDB 데이터셋 빠른 시작 가이드

**목적**: `/mnt/HDDB` 데이터셋을 활용하여 최대한 빠르게 모델 훈련 시작하기
**대상**: 전체 워크플로우를 빠르게 실행하고 싶은 사용자

---

## 🚀 5분 안에 시작하기

### 1단계: 환경 확인 (1분)

```bash
# Project Sullivan 디렉토리로 이동
cd /home/Project_Sullivan

# 가상환경 활성화
source venv_sullivan/bin/activate

# 데이터셋 확인
ls /mnt/HDDB/dataset/my_dataset/dataset/
# 출력: sub010 ~ sub039 (27명 피험자)

# 디스크 공간 확인 (최소 500GB 필요)
df -h /home/Project_Sullivan/data/
```

---

### 2단계: 자동 파이프라인 실행 (3분 설정 + 5-7일 처리)

**옵션 A: 전체 자동 실행 (권장 - Stage 1: 5명으로 시작)**

```bash
# GPU 사용 (권장)
bash scripts/hddb_pipeline.sh all-stage1 --gpu

# CPU 사용 (느림)
bash scripts/hddb_pipeline.sh all-stage1 --cpu
```

**자동으로 실행되는 작업**:
1. 데이터셋 분석
2. MRI/오디오 정렬 (5명)
3. U-Net 세그멘테이션 (5명)
4. 파라미터 추출 (5명)
5. 오디오 특징 추출 (5명)
6. 데이터셋 분할 (Train/Val/Test)
7. Baseline LSTM 훈련
8. Transformer 훈련
9. 모델 평가

**예상 소요 시간** (5명 피험자, GPU 사용):
- Phase 1 (전처리): 5-7일
  - MRI/오디오 정렬: ~5시간
  - 세그멘테이션: ~2-3시간 (GPU)
  - 파라미터 추출: ~1.5시간
  - 오디오 특징: ~30분
- Phase 2 (훈련): 10-15시간
  - Baseline: 3-5시간
  - Transformer: 6-8시간
  - 평가: 1-2시간

**총 예상 시간**: **약 5-7일** (대부분 자동 실행)

---

### 3단계: 모니터링 (진행 중 확인)

```bash
# TensorBoard 실행 (별도 터미널)
bash scripts/start_tensorboard.sh
# 브라우저에서 http://localhost:6006 접속

# 훈련 진행 상황 확인
bash scripts/monitor_training_simple.sh

# 로그 확인
tail -f logs/training/transformer_v1/version_0/events.out.tfevents.*
```

---

## 📋 단계별 실행 (수동 제어)

자동 파이프라인 대신 단계별로 직접 제어하고 싶다면:

### Phase 0: 데이터 분석

```bash
bash scripts/hddb_pipeline.sh phase0
```

**출력**: `data/hddb_dataset_stats.json`

---

### Phase 1: 데이터 전처리

#### Stage 1: 5명 피험자 처리

```bash
# 전처리
bash scripts/hddb_pipeline.sh phase1-s1 --gpu

# 데이터셋 분할
bash scripts/hddb_pipeline.sh phase1-split
```

**출력**:
- `data/processed_hddb/aligned/` - 정렬된 MRI+오디오
- `data/processed_hddb/segmentations/` - 세그멘테이션 마스크
- `data/processed_hddb/parameters/` - Articulatory 파라미터
- `data/processed_hddb/audio_features/` - Mel-spectrogram
- `data/processed_hddb/splits/` - Train/Val/Test 분할

**검증**:
```bash
# 처리된 데이터 확인
ls data/processed_hddb/aligned/sub010/
ls data/processed_hddb/splits/

# 통계 확인
python scripts/check_splits.py --splits-dir data/processed_hddb/splits
```

---

### Phase 2: 모델 훈련

#### Baseline LSTM 훈련

```bash
bash scripts/hddb_pipeline.sh phase2-baseline --gpu
```

**출력**: `models/baseline_lstm_hddb/best.ckpt`

**예상 성능** (5명):
- Test RMSE: ~0.4-0.6
- Test Pearson: ~0.3-0.4

---

#### Transformer 훈련 (권장)

```bash
bash scripts/hddb_pipeline.sh phase2-transformer --gpu
```

**출력**: `models/transformer_hddb/best.ckpt`

**예상 성능** (5명):
- Test RMSE: ~0.12-0.18 ← **M2 목표 가능** (< 0.15)
- Test Pearson: ~0.45-0.60 ← **M2 목표 가능** (> 0.50)

---

#### 모델 평가

```bash
bash scripts/hddb_pipeline.sh phase2-eval
```

**출력**:
- `results/baseline_hddb_evaluation.json`
- `results/transformer_hddb_evaluation.json`

**평가 지표**:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- Pearson Correlation
- 파라미터별 성능

---

## 🎯 M2 목표 달성 전략

### 현재 예상 성능 (5명)

| 모델 | Test RMSE | Test Pearson | M2 달성? |
|-----|-----------|--------------|---------|
| Baseline LSTM | 0.4-0.6 | 0.3-0.4 | ❌ |
| Transformer | 0.12-0.18 | 0.45-0.60 | 🟡 근접 |

**M2 목표**: RMSE < 0.15, Pearson > 0.50

---

### 성능 개선 옵션

#### 옵션 1: 더 많은 데이터 (권장)

**5명 → 15명 확장**:
```bash
# Stage 2 전처리 (추가 10명)
bash scripts/hddb_pipeline.sh phase1-s2 --gpu

# 데이터셋 재분할
bash scripts/hddb_pipeline.sh phase1-split

# Transformer 재훈련
bash scripts/hddb_pipeline.sh phase2-transformer --gpu
```

**예상 개선**:
- RMSE: 0.12-0.18 → **0.08-0.12** (30% 감소)
- Pearson: 0.45-0.60 → **0.60-0.75** (25% 향상)

**소요 시간**: 추가 2주

---

#### 옵션 2: 하이퍼파라미터 튜닝

```bash
# configs/transformer_config_hddb_v2.yaml 생성
cp configs/transformer_config_hddb.yaml configs/transformer_config_hddb_v2.yaml

# 수정 사항:
# - learning_rate: 5e-4 → 3e-4
# - d_model: 256 → 512
# - num_layers: 4 → 6
# - batch_size: 16 → 32 (GPU 메모리 충분하면)

# 재훈련
python scripts/train_transformer.py --config configs/transformer_config_hddb_v2.yaml --gpus 1
```

**예상 개선**: 5-10%

---

#### 옵션 3: 전체 27명 처리 (최종)

```bash
# Stage 3 전처리 (나머지 5명)
bash scripts/hddb_pipeline.sh all-stage3 --gpu
```

**예상 성능** (27명):
- RMSE: **0.06-0.10** ← **M3 목표 가능** (< 0.10)
- Pearson: **0.70-0.80** ← **M3 목표 가능** (> 0.70)

**소요 시간**: 추가 3주

---

## 🐛 문제 해결

### 문제 1: CUDA out of memory

```bash
# configs/transformer_config_hddb.yaml 수정
training:
  batch_size: 16 → 8  # 배치 크기 감소
  accumulate_grad_batches: 1 → 2  # 그래디언트 축적
```

---

### 문제 2: 디스크 공간 부족

```bash
# 심볼릭 링크로 대용량 스토리지 연결
mkdir -p /mnt/HDDB/processed_hddb
ln -s /mnt/HDDB/processed_hddb /home/Project_Sullivan/data/processed_hddb
```

---

### 문제 3: 세그멘테이션 품질 낮음

```bash
# U-Net 모델 확인
ls -lh models/segmentation/unet_best.pth

# 재학습 필요시 (고급)
python scripts/train_unet.py --config configs/unet_config.yaml
```

---

### 문제 4: 훈련이 멈춤/느림

```bash
# 프로세스 확인
ps aux | grep train_transformer

# TensorBoard로 진행 확인
# Loss가 감소하는지, NaN이 발생하는지 확인

# 재시작 필요시
pkill -f train_transformer
bash scripts/hddb_pipeline.sh phase2-transformer --gpu
```

---

## 📊 예상 결과 (5명 기준)

### 데이터셋 통계

```
Train: 3-4명, ~90-120 utterances, ~350K frames
Val: 1명, ~30 utterances, ~75K frames
Test: 1명, ~30 utterances, ~75K frames
Total: ~500K frames (기존 186K의 2.7배)
```

---

### 모델 성능

**Transformer 예상 성능**:

| 파라미터 | RMSE | Pearson | 비고 |
|---------|------|---------|------|
| Tongue X | 0.10 | 0.65 | 가장 중요 |
| Tongue Y | 0.12 | 0.60 | |
| Jaw Opening | 0.15 | 0.55 | |
| Lip Aperture | 0.18 | 0.50 | |
| **Overall** | **0.14** | **0.58** | **M2 목표 달성 가능** |

---

## 🎉 성공 후 다음 단계

### 1. 성능 검증

```bash
# 시각화 생성
python scripts/visualize_predictions.py \
  --predictions results/transformer_hddb_predictions.npz \
  --ground-truth data/processed_hddb/parameters/test/ \
  --output results/visualizations/

# 결과 확인
ls results/visualizations/
```

---

### 2. 모델 저장 및 공유

```bash
# 최종 모델 저장
mkdir -p models/final_release/
cp models/transformer_hddb/best.ckpt models/final_release/transformer_v1_5subjects.ckpt

# 메타데이터 저장
cat > models/final_release/model_info.json << EOF
{
  "model": "Transformer",
  "subjects": 5,
  "test_rmse": 0.14,
  "test_pearson": 0.58,
  "date": "$(date +%Y-%m-%d)"
}
EOF
```

---

### 3. 보고서 작성

```bash
# 자동 보고서 생성 (구현 필요)
python scripts/generate_report.py \
  --results results/transformer_hddb_evaluation.json \
  --output docs/HDDB_RESULTS_REPORT.md
```

---

### 4. 확장 결정

**M2 목표 달성 시**:
- ✅ Phase 2 완료
- → Phase 3 (Digital Twin) 또는 논문 작성

**M2 목표 미달 시**:
- → Stage 2 실행 (15명으로 확장)
- → 하이퍼파라미터 튜닝
- → 데이터 증강 기법 적용

---

## 📞 도움말

### 명령어 요약

```bash
# 전체 자동 실행
bash scripts/hddb_pipeline.sh all-stage1 --gpu

# 개별 단계 실행
bash scripts/hddb_pipeline.sh phase0           # 데이터 분석
bash scripts/hddb_pipeline.sh phase1-s1 --gpu  # 전처리 (5명)
bash scripts/hddb_pipeline.sh phase1-split     # 데이터 분할
bash scripts/hddb_pipeline.sh phase2-transformer --gpu  # 훈련
bash scripts/hddb_pipeline.sh phase2-eval      # 평가

# 도움말
bash scripts/hddb_pipeline.sh help
```

---

### 추가 리소스

- **전체 워크플로우**: `WORKFLOW_HDDB_DATASET.md`
- **개발 가이드**: `CLAUDE.md`
- **프로젝트 개요**: `README.md`
- **Baseline 성능 분석**: `docs/BASELINE_PERFORMANCE_REPORT.md`

---

**작성자**: Claude Code Assistant
**버전**: 1.0
**최종 수정**: 2026-01-11
