# USC-TIMIT Full Dataset 활용 가이드

**목적:** 600GB USC-TIMIT 데이터셋을 Project Sullivan 학습에 활용하는 방법

---

## 1. 현재 상태

### ✅ 완료된 작업
- 데이터셋 접근 경로 설정: `data/raw/usc_timit_full/` (심볼릭 링크)
- 27명 피험자, ~840개 utterances 사용 가능
- 기존 데이터 로더와 호환 확인 완료

### 📊 현재 프로젝트 상태
- **Phase:** 2-B (Transformer 모델 훈련)
- **현재 학습 데이터:** 75 utterances (186K frames)
- **사용 가능 데이터:** ~840 utterances (예상 2M+ frames)

---

## 2. 데이터 활용 옵션

### 옵션 A: 추가 데이터 전처리 후 학습 (권장)

전체 데이터셋으로 모델 성능을 크게 향상시킬 수 있습니다.

**단계:**

#### Step 1: 피험자 선택

현재 프로젝트는 일부 피험자만 전처리했습니다. 추가 피험자를 전처리하세요:

```bash
# 현재 전처리된 피험자 확인
ls data/processed/aligned/

# 새로운 피험자 추가 (예: sub011, sub012, sub013)
# researcher_manual.md의 Phase 1 절차 참조
```

#### Step 2: Segmentation (U-Net으로 성도 분할)

```bash
# 추가 피험자 분할 (예시)
python scripts/segment_subset.py \
  --data-root data/raw/usc_timit_full \
  --subjects sub011,sub012,sub013,sub014,sub015 \
  --output-dir data/processed/segmentations \
  --checkpoint models/segmentation/unet_best.pth

# 진행 상황 모니터링
# (프로젝트에서 22.8 fps 속도로 처리됨)
```

**예상 시간:** 피험자당 ~2-3시간 (32 utterances × ~200 frames)

#### Step 3: Articulatory Parameter 추출

```bash
python scripts/extract_articulatory_params.py \
  --segmentation-dir data/processed/segmentations \
  --output-dir data/processed/parameters \
  --method geometric  # or 'pca'
```

#### Step 4: Audio Feature 추출

```bash
# Audio feature 추출 스크립트 사용
# (프로젝트 scripts/ 디렉토리 확인)
python scripts/extract_audio_features.py \
  --data-root data/raw/usc_timit_full \
  --output-dir data/processed/audio_features \
  --feature-type mel  # mel-spectrogram
```

#### Step 5: Train/Val/Test Split 재구성

```bash
# 더 많은 데이터로 split 재생성
# (프로젝트 스크립트 확인)
python scripts/create_splits.py \
  --parameter-dir data/processed/parameters \
  --audio-dir data/processed/audio_features \
  --output-dir data/processed/splits \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --subject-level  # 피험자 단위 분할
```

#### Step 6: 모델 재학습

```bash
# Transformer 모델 학습
python scripts/train_transformer.py \
  --config configs/transformer_config.yaml

# TensorBoard 모니터링
tensorboard --logdir logs/training/
```

**예상 성능 향상:**
- 현재: 75 utterances → 예상 RMSE ~1.0
- 확장: ~840 utterances → 예상 RMSE ~0.5 이하 (목표: < 0.10)

---

### 옵션 B: 점진적 확장 (추천 - 단계적 접근)

대용량 데이터셋을 한 번에 처리하는 것은 시간이 오래 걸립니다. 점진적으로 확장하세요:

**Phase 1: 소규모 확장 (빠른 검증)**
```bash
# 5명 피험자 추가 (현재 75 → ~200 utterances)
# sub011, sub012, sub013, sub014, sub015
```

**Phase 2: 중규모 확장**
```bash
# 10명 피험자 사용 (현재 → ~400 utterances)
```

**Phase 3: 전체 데이터셋**
```bash
# 25명 피험자 사용 (~800 utterances)
```

각 단계에서 모델 성능을 평가하고 개선 효과를 확인하세요.

---

### 옵션 C: 기존 데이터로 계속 학습

현재 75 utterances로 먼저 Transformer 모델을 완성하고, 나중에 데이터 확장:

```bash
# 현재 설정으로 계속 진행
python scripts/train_transformer.py \
  --config configs/transformer_config.yaml

# Baseline 성능 확인 후 데이터 확장 결정
```

---

## 3. 빠른 시작 (권장 워크플로우)

### 3.1. 데이터 탐색

먼저 새 데이터셋을 탐색하세요:

```python
from src.preprocessing.data_loader import USCTIMITLoader

# 데이터셋 로드
loader = USCTIMITLoader("data/raw/usc_timit_full")

# 통계 확인
stats = loader.get_statistics()
print(f"Total subjects: {stats['num_subjects']}")
print(f"Subject IDs: {stats['subject_ids']}")

# 특정 피험자 로드
subject_data = loader.load_subject("sub011", load_mri=True, load_audio=True)
print(f"Utterances: {subject_data['num_utterances']}")
print(f"Utterance files: {subject_data['utterance_files'][:3]}")
```

### 3.2. 샘플 데이터 전처리 테스트

1명 피험자로 먼저 테스트:

```bash
# 1. Segmentation 테스트
python scripts/segment_subset.py \
  --data-root data/raw/usc_timit_full \
  --subjects sub011 \
  --output-dir data/processed/segmentations_test

# 2. Parameter 추출 테스트
python scripts/extract_articulatory_params.py \
  --segmentation-dir data/processed/segmentations_test/sub011 \
  --output-dir data/processed/parameters_test

# 3. 결과 확인
ls data/processed/parameters_test/
```

### 3.3. 전체 파이프라인 실행

테스트가 성공하면 전체 데이터로 확장:

```bash
# Phase 1 전처리 스크립트 일괄 실행
# (scripts/ 디렉토리에서 batch 스크립트 확인)
```

---

## 4. 데이터셋 관리 팁

### 4.1. 디스크 공간 관리

전처리된 데이터는 원본만큼 크거나 더 클 수 있습니다:

```bash
# 디스크 사용량 확인
du -sh data/processed/*
df -h /home/Project_Sullivan

# 불필요한 중간 파일 정리
# (segmentation raw outputs, temporary files 등)
```

### 4.2. 데이터 백업

중요한 전처리 결과는 백업하세요:

```bash
# 전처리 완료 데이터 아카이브
tar -czf processed_data_backup_$(date +%Y%m%d).tar.gz data/processed/

# 외부 저장소로 복사
cp processed_data_backup_*.tar.gz /mnt/HDDA/backups/
```

### 4.3. 피험자 선택 전략

모든 피험자를 사용할 필요는 없습니다. 품질 좋은 피험자를 선택하세요:

```bash
# 프로젝트에서 권장하는 피험자 목록 확인
cat data/raw/recommended_subjects.json
```

---

## 5. 성능 모니터링

### 5.1. 데이터셋 크기별 성능 추적

| 데이터셋 크기 | RMSE 목표 | PCC 목표 | 상태 |
|--------------|-----------|---------|------|
| 75 utterances | < 0.15 | > 0.50 | Baseline (M2) |
| 200 utterances | < 0.12 | > 0.60 | Small expansion |
| 400 utterances | < 0.10 | > 0.70 | Medium expansion (M3 목표) |
| 800 utterances | < 0.08 | > 0.80 | Full dataset |

### 5.2. 학습 모니터링

```bash
# TensorBoard 시작
bash scripts/start_tensorboard.sh

# 학습 진행 상황 모니터링
bash scripts/monitor_training_simple.sh
```

---

## 6. 트러블슈팅

### Q1: 메모리 부족 (OOM) 에러

**해결책:**
```yaml
# configs/transformer_config.yaml 수정
training:
  batch_size: 8  # 16에서 8로 감소
  accumulate_grad_batches: 2  # Gradient accumulation 사용
```

### Q2: 전처리 속도가 너무 느림

**해결책:**
```bash
# 피험자를 나눠서 병렬 처리
# Terminal 1
python scripts/segment_subset.py --subjects sub011,sub012,sub013

# Terminal 2
python scripts/segment_subset.py --subjects sub014,sub015,sub016
```

### Q3: 디스크 공간 부족

**해결책:**
1. 중간 파일 삭제 (raw segmentation masks 등)
2. 전처리된 데이터를 /mnt/HDDA로 이동
3. 심볼릭 링크로 연결

---

## 7. 참고 문서

- **연구 매뉴얼:** `researcher_manual.md` - Phase 1 전처리 상세 가이드
- **프로젝트 README:** `README.md` - 프로젝트 전체 구조
- **데이터 통합 보고서:** `DATASET_INTEGRATION_REPORT.md` - 데이터셋 상세 정보

---

## 8. 다음 단계 체크리스트

- [ ] **옵션 선택:** A (전체 확장) / B (점진적) / C (현재 유지)
- [ ] **피험자 선택:** 추가할 피험자 ID 결정
- [ ] **Segmentation 실행:** U-Net으로 MRI 분할
- [ ] **Parameter 추출:** Articulatory parameters 생성
- [ ] **Audio Feature 추출:** Mel-spectrogram 생성
- [ ] **Split 재구성:** Train/Val/Test split 업데이트
- [ ] **모델 재학습:** Transformer 학습 시작
- [ ] **성능 평가:** RMSE, PCC 측정 및 목표 달성 확인

---

**작성일:** 2026-01-11
**업데이트:** 데이터셋 통합 완료, 사용 가이드 제공
**문의:** 프로젝트 README 참조

---

## 빠른 명령어 요약

```bash
# 1. 데이터 확인
ls data/raw/usc_timit_full/

# 2. 피험자 추가 전처리 (예시)
python scripts/segment_subset.py --subjects sub011,sub012,sub013

# 3. Parameter 추출
python scripts/extract_articulatory_params.py

# 4. 모델 학습
python scripts/train_transformer.py --config configs/transformer_config.yaml

# 5. 모니터링
tensorboard --logdir logs/training/
```

**Good luck with your research! 🚀**
