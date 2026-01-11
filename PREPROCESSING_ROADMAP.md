# 데이터 전처리 로드맵 - Preprocessing Roadmap

**날짜:** 2026-01-11
**목적:** 600GB 데이터셋 활용을 위한 단계별 실행 계획

---

## 📊 현재 상태 요약

### ✅ 완료된 작업
1. **데이터셋 통합**: `/mnt/HDDB/dataset/` → `data/raw/usc_timit_full/` (심볼릭 링크)
2. **데이터 확인**: 27 subjects, ~840 utterances 접근 가능
3. **문서 작성**: 통합 보고서, 사용 가이드, 환경 설정 가이드

### ⚠️ 현재 장애물
1. **Python 환경 미설정**: numpy, torch 등 필수 라이브러리 미설치
2. **전처리 데이터 부재**: HDF5 파일 없음 (metadata만 존재)
3. **패키지 관리자 없음**: pip, uv 미설치

### 🎯 목표
- **1단계**: 5명 피험자 전처리 (~200 utterances)
- **2단계**: 모델 재학습 및 성능 비교
- **3단계**: 전체 데이터셋 확장 (~840 utterances)

---

## 🔧 1단계: 환경 설정 (선행 필수)

### 방법 1: UV 패키지 관리자 (권장)

```bash
# UV 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 의존성 설치
cd /home/Project_Sullivan
uv sync

# 가상환경 활성화
source .venv/bin/activate

# 확인
python -c "import torch, numpy, librosa; print('✓ 환경 준비 완료')"
```

### 방법 2: Docker 환경 (안정적)

```bash
# PyTorch Docker 이미지 사용
docker run --gpus all -it \
  -v /home/Project_Sullivan:/workspace \
  -v /mnt/HDDB:/mnt/HDDB \
  pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# 컨테이너 내부에서
cd /workspace
pip install -r requirements.txt
```

### 방법 3: System pip (간단)

```bash
# pip 설치
python3 -m ensurepip --upgrade

# 의존성 설치
cd /home/Project_Sullivan
pip install -r requirements.txt
```

---

## 📋 2단계: 소규모(5명) 샘플 전처리 및 검증

**환경 설정 완료 후 실행**

### 2-1. 데이터 정렬 및 전처리

```bash
cd /home/Project_Sullivan

# Logging 디렉토리 생성
mkdir -p logs/preprocessing

# 5명 피험자 전처리 (sub011-sub015)
python scripts/batch_preprocess.py \
  --data-root data/raw/usc_timit_full \
  --subjects sub011 sub012 sub013 sub014 sub015 \
  --output-dir data/processed/aligned \
  --max-utterances 32 \
  > logs/preprocessing/stage1_alignment_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 진행 상황 모니터링
tail -f logs/preprocessing/stage1_alignment_*.log
```

**예상 시간:** 2-5시간 (5명 × 32 utterances)

**체크포인트:**
```bash
# 전처리 완료 확인
ls data/processed/aligned/sub011/*.h5 | wc -l  # 32개 예상
ls data/processed/aligned/sub012/*.h5 | wc -l  # 32개 예상

# Batch summary 업데이트 확인
cat data/processed/aligned/batch_summary.json | grep "total_utterances"
```

### 2-2. 오류 검증

```bash
# 전처리 결과 검증 스크립트 실행
python scripts/test_preprocessing_pipeline.py \
  --batch-summary data/processed/aligned/batch_summary.json \
  --check-alignment \
  --check-missing

# 결과 확인
cat logs/preprocessing_validation.log
```

**확인 항목:**
- ✓ Missing files: 0
- ✓ Alignment correlation > 0.3
- ✓ Audio-video sync errors: 0

### 2-3. Segmentation (U-Net)

```bash
# MRI 세그먼트화 (성도 분할)
python scripts/segment_subset.py \
  --batch-summary data/processed/aligned/batch_summary.json \
  --model models/unet_scratch/unet_final.pth \
  --output-dir data/processed/segmentations \
  --max-per-subject 10 \
  --device cuda \
  > logs/preprocessing/stage1_segmentation_$(date +%Y%m%d_%H%M%S).log 2>&1
```

**예상 시간:** 1-2시간 (5명 × 10 utterances × ~200 frames @ 22.8 fps)

**진행 상황:**
```bash
# 리소스 모니터링
watch -n 5 'nvidia-smi; echo "---"; du -sh data/processed/segmentations/'

# 완료 확인
find data/processed/segmentations -name "*.png" | wc -l
```

### 2-4. Articulatory Parameter 추출

```bash
# Geometric parameters 추출
python scripts/extract_articulatory_params.py \
  --segmentation-dir data/processed/segmentations \
  --output-dir data/processed/parameters \
  --method geometric \
  > logs/preprocessing/stage1_parameters_$(date +%Y%m%d_%H%M%S).log 2>&1
```

**출력:**
- `data/processed/parameters/*.npy` (geometric features: 14차원)

### 2-5. Audio Feature 추출

```bash
# Mel-spectrogram 추출
python scripts/extract_audio_features.py \
  --data-dir data/processed/aligned \
  --output-dir data/processed/audio_features \
  --feature-type mel \
  --subjects sub011 sub012 sub013 sub014 sub015 \
  > logs/preprocessing/stage1_audio_features_$(date +%Y%m%d_%H%M%S).log 2>&1
```

**출력:**
- `data/processed/audio_features/*.npy` (mel-spectrogram: 80 bins)

---

## 📈 3단계: 증량 데이터로 모델 학습

**전처리 완료 후 실행**

### 3-1. 데이터셋 분할 업데이트

```bash
# Train/Val/Test split 재생성
python scripts/create_splits.py \
  --parameter-dir data/processed/parameters \
  --audio-dir data/processed/audio_features \
  --output-dir data/processed/splits \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --subject-level
```

### 3-2. 설정 파일 확인

```yaml
# configs/transformer_config.yaml
data:
  splits_dir: data/processed/splits
  audio_feature_dir: data/processed/audio_features
  parameter_dir: data/processed/parameters

training:
  batch_size: 16  # GPU VRAM에 따라 조정 (8-32)
  num_epochs: 50
  precision: 16  # Mixed precision (GPU 필수)
```

**메모리 최적화:**
- GPU 8GB → batch_size: 8, accumulate_grad_batches: 2
- GPU 16GB → batch_size: 16, accumulate_grad_batches: 1
- GPU 24GB+ → batch_size: 32

### 3-3. 모델 학습 시작

```bash
# TensorBoard 시작 (별도 터미널)
bash scripts/start_tensorboard.sh

# Transformer 학습
python scripts/train_transformer.py \
  --config configs/transformer_config.yaml \
  > logs/training/stage2_transformer_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 모니터링
bash scripts/monitor_training_simple.sh
```

**TensorBoard 확인:**
- URL: http://localhost:6006
- Metrics: train_loss, val_loss, val_rmse, val_pearson

### 3-4. 성능 비교

```bash
# 학습 완료 후 평가
python scripts/evaluate_model.py \
  --checkpoint models/transformer/best.ckpt \
  --test-data data/processed/splits/test \
  --output-dir results/evaluation

# 결과 확인
cat results/evaluation/metrics.json
```

**예상 성능:**

| 데이터셋 크기 | RMSE (목표) | PCC (목표) | 현재 상태 |
|--------------|-------------|-----------|----------|
| 75 utterances | < 1.0 | > 0.50 | Baseline (현재) |
| ~200 utterances (5명) | < 0.5 | > 0.65 | **1단계 목표** |
| ~400 utterances (10명) | < 0.3 | > 0.75 | 2단계 목표 |
| ~840 utterances (25명) | < 0.10 | > 0.80 | **최종 목표 (M3)** |

---

## 🚀 4단계: 전체 데이터셋 확장

**중간 검증 완료 후 실행**

### 4-1. Batch Processing 스크립트 생성

```bash
# 전체 피험자 리스트 생성
cat > /home/Project_Sullivan/scripts/process_all_subjects.sh << 'EOF'
#!/bin/bash
set -e

SUBJECTS=(sub010 sub011 sub012 sub013 sub014 sub015 sub016 sub017 sub018 sub019 \
          sub030 sub031 sub032 sub033 sub034 sub035 sub037 sub038 sub039 \
          sub043 sub050 sub052 sub053 sub054 sub056 sub058)

LOG_DIR="logs/preprocessing/full_dataset"
mkdir -p $LOG_DIR

for subject in "${SUBJECTS[@]}"; do
  echo "[$(date)] Processing $subject..."

  python scripts/batch_preprocess.py \
    --data-root data/raw/usc_timit_full \
    --subjects $subject \
    --output-dir data/processed/aligned \
    --max-utterances 32 \
    > $LOG_DIR/${subject}_$(date +%Y%m%d_%H%M%S).log 2>&1

  if [ $? -eq 0 ]; then
    echo "[$(date)] ✓ $subject completed"
  else
    echo "[$(date)] ✗ $subject failed" >> $LOG_DIR/failed_subjects.log
  fi
done

echo "[$(date)] All subjects processed"
EOF

chmod +x /home/Project_Sullivan/scripts/process_all_subjects.sh
```

### 4-2. 병렬 처리 (선택)

```bash
# GNU Parallel 사용 (속도 향상)
parallel -j 4 --bar \
  'python scripts/batch_preprocess.py --data-root data/raw/usc_timit_full --subjects {} --output-dir data/processed/aligned --max-utterances 32 > logs/preprocessing/full/{}_$(date +%Y%m%d_%H%M%S).log 2>&1' \
  ::: sub010 sub011 sub012 sub013 sub014 sub015 # ... (모든 피험자)
```

### 4-3. 전처리 완료 후 정리

```bash
# 전체 통계
find data/processed/aligned -name "*.h5" | wc -l

# 디스크 사용량
du -sh data/processed/

# 아카이빙 (선택)
tar -czf data/processed/aligned_full_dataset_$(date +%Y%m%d).tar.gz data/processed/aligned/
mv data/processed/aligned_full_dataset_*.tar.gz /mnt/HDDA/backups/
```

---

## 📊 리소스 모니터링

### CPU/GPU 모니터링

```bash
# Terminal 1: GPU 모니터링
watch -n 1 nvidia-smi

# Terminal 2: CPU/RAM 모니터링
htop

# Terminal 3: 디스크 모니터링
watch -n 10 'df -h /home/Project_Sullivan; echo "---"; du -sh data/processed/*'
```

### 로그 분석

```bash
# 전처리 진행 상황
grep -r "Processing utterance" logs/preprocessing/*.log | tail -20

# 오류 확인
grep -i "error\|failed\|exception" logs/preprocessing/*.log

# 성공률 계산
total=$(grep -c "Processing utterance" logs/preprocessing/stage1_alignment_*.log)
success=$(grep -c "saved successfully" logs/preprocessing/stage1_alignment_*.log)
echo "Success rate: $((success * 100 / total))%"
```

---

## ✅ 체크리스트

### 환경 설정
- [ ] Python 환경 설정 완료
- [ ] 필수 라이브러리 설치 확인 (numpy, torch, librosa)
- [ ] GPU 사용 가능 확인 (선택)

### 1단계 (5명 피험자)
- [ ] Alignment 및 전처리 완료 (sub011-sub015)
- [ ] Segmentation 완료
- [ ] Parameter & Audio feature 추출
- [ ] 오류 검증 (missing files, sync errors)

### 2단계 (모델 학습)
- [ ] Train/Val/Test split 생성
- [ ] Transformer 학습 시작
- [ ] TensorBoard 모니터링 설정
- [ ] 성능 평가 (RMSE, PCC)

### 3단계 (전체 확장)
- [ ] Batch processing 스크립트 준비
- [ ] 전체 피험자 전처리 (~25명)
- [ ] 최종 모델 학습
- [ ] 목표 성능 달성 (RMSE < 0.10, PCC > 0.70)

---

## 🔗 참고 문서

- `ENVIRONMENT_SETUP_REQUIRED.md` - 환경 설정 가이드
- `DATASET_USAGE_GUIDE.md` - 데이터셋 활용 가이드
- `DATASET_INTEGRATION_REPORT.md` - 데이터 통합 보고서
- `researcher_manual.md` - 연구 매뉴얼

---

## 📞 문제 해결

### Q: "ModuleNotFoundError: No module named 'numpy'"
**A:** 환경 설정 필요. `ENVIRONMENT_SETUP_REQUIRED.md` 참조

### Q: "CUDA out of memory"
**A:** `batch_size` 감소, `accumulate_grad_batches` 증가

### Q: "Alignment correlation too low"
**A:** 정상. correlation > 0.3이면 사용 가능, 낮은 것은 자동 필터링됨

### Q: "Processing too slow"
**A:** GPU 사용, 병렬 처리, 또는 피험자 수 감소

---

**다음 단계:** 환경 설정 → 1단계 실행 → 성능 검증 → 2-3단계 진행

**Good luck! 🚀**
