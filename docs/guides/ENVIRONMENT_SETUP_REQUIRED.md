# 환경 설정 필요 - Environment Setup Required

**날짜:** 2026-01-11
**상태:** ⚠️ Python 환경 설정 필요

---

## 🔍 현재 상황

### 데이터셋
- ✅ **대용량 USC-TIMIT 데이터 접근 가능**: `data/raw/usc_timit_full/` (879 GB, 27 subjects, ~840 utterances)
- ✅ **데이터 구조 확인 완료**: 비디오, 오디오, MRI 파일 모두 접근 가능
- ⚠️ **전처리 스크립트 존재**: `scripts/batch_preprocess.py`, `segment_subset.py` 등

### Python 환경
- ✅ **Python 3.13 설치됨**: `/bin/python`
- ❌ **필수 라이브러리 미설치**: numpy, scipy, torch, opencv, librosa 등
- ❌ **패키지 관리자 없음**: pip, uv 미설치
- ❌ **가상 환경 없음**: venv 미생성

### 전처리 상태
- ⚠️ **Metadata만 존재**: `data/processed/aligned/sub*/` 폴더에 JSON만 있음
- ❌ **실제 전처리 데이터 없음**: HDF5 파일(*.h5) 0개
- ❌ **Parameter 데이터 없음**: articulatory parameters 미추출

---

## 🛠️ 해결 방법

### 옵션 A: Docker/Conda 환경 사용 (권장)

프로젝트에 필요한 모든 라이브러리를 포함한 환경을 사용하세요.

```bash
# Docker 사용 (권장)
docker run -v /home/Project_Sullivan:/workspace -v /mnt/HDDB:/mnt/HDDB \
  pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime bash

# 또는 Conda 환경
conda create -n sullivan python=3.10
conda activate sullivan
pip install -r /home/Project_Sullivan/requirements.txt
```

### 옵션 B: UV 패키지 관리자 설치

프로젝트에 `uv.lock` 파일이 있으므로, UV를 사용할 수 있습니다.

```bash
# UV 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 환경 생성 및 패키지 설치
cd /home/Project_Sullivan
uv sync

# 활성화
source .venv/bin/activate
```

### 옵션 C: pip 설치 및 requirements 설치

```bash
# pip 설치
python3 -m ensurepip --upgrade

# 의존성 설치
cd /home/Project_Sullivan
pip install -r requirements.txt
```

---

## 📦 필요한 주요 라이브러리

`requirements.txt`에 명시된 라이브러리:

```
# Deep Learning
torch>=2.0.0
torchvision>=0.15.0
lightning>=2.0.0

# Data Processing
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
pandas>=2.0.0

# Audio Processing
librosa>=0.10.0
soundfile>=0.12.0

# Image/Video Processing
opencv-python>=4.8.0
scikit-image>=0.21.0

# Medical Image
nibabel>=5.0.0
h5py>=3.8.0

# Utilities
tqdm>=4.65.0
matplotlib>=3.7.0
pyyaml>=6.0
tensorboard>=2.13.0
```

---

## 🚀 환경 설정 후 작업 순서

### 1. 환경 확인

```bash
# Python 환경 확인
python --version
python -c "import torch, numpy, librosa; print('환경 OK')"
```

### 2. 샘플 전처리 테스트

```bash
# 1명 피험자로 테스트
cd /home/Project_Sullivan
python scripts/batch_preprocess.py \
  --data-root data/raw/usc_timit_full \
  --subjects sub011 \
  --output-dir data/processed/aligned_new \
  --max-utterances 5
```

### 3. 5명 피험자 전처리 (1단계)

```bash
# sub011-sub015 전처리
python scripts/batch_preprocess.py \
  --data-root data/raw/usc_timit_full \
  --subjects sub011 sub012 sub013 sub014 sub015 \
  --output-dir data/processed/aligned \
  --max-utterances 32  # 전체 utterances
```

**예상 시간:** ~5-10시간 (5명 × 32 utterances × ~200 frames)

### 4. Segmentation (U-Net)

```bash
# MRI 세그먼트화
python scripts/segment_subset.py \
  --batch-summary data/processed/aligned/batch_summary.json \
  --model models/unet_scratch/unet_final.pth \
  --output-dir data/processed/segmentations \
  --max-per-subject 10
```

### 5. Parameter & Audio Feature 추출

```bash
# Articulatory parameters
python scripts/extract_articulatory_params.py \
  --segmentation-dir data/processed/segmentations \
  --output-dir data/processed/parameters \
  --method geometric

# Audio features
python scripts/extract_audio_features.py \
  --data-dir data/processed/aligned \
  --output-dir data/processed/audio_features \
  --feature-type mel
```

### 6. 모델 학습

```bash
# Transformer 모델 학습
python scripts/train_transformer.py \
  --config configs/transformer_config.yaml
```

---

## 💡 현재 할 수 있는 작업 (Python 없이)

환경 설정 전에 할 수 있는 작업들:

### 1. 데이터셋 탐색

```bash
# 피험자별 파일 수 확인
for subject in /home/Project_Sullivan/data/raw/usc_timit_full/sub*/; do
  echo "$(basename $subject): $(ls $subject/2drt/video/*.mp4 2>/dev/null | wc -l) videos"
done

# 전체 데이터 크기
du -sh /home/Project_Sullivan/data/raw/usc_timit_full/
```

### 2. 로그 및 문서 검토

```bash
# 프로젝트 문서 읽기
cat /home/Project_Sullivan/README.md
cat /home/Project_Sullivan/researcher_manual.md

# 기존 실험 로그 확인
ls /home/Project_Sullivan/logs/
```

### 3. 설정 파일 수정

```bash
# 학습 설정 확인/수정
vim /home/Project_Sullivan/configs/transformer_config.yaml
```

---

## 📊 예상 리소스 요구사항

### 전처리 (Phase 1)
- **CPU**: 4+ cores
- **RAM**: 16GB+ (32GB 권장)
- **Disk**: 100-200GB (전처리 데이터)
- **시간**: ~20-30시간 (전체 25명 피험자)

### 학습 (Phase 2)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (권장: RTX 3070 이상)
- **RAM**: 16GB+
- **Disk**: 50GB (모델 체크포인트, 로그)
- **시간**: ~10-20시간 (전체 데이터셋)

---

## 🔗 참고 문서

- **프로젝트 README**: `README.md`
- **연구 매뉴얼**: `researcher_manual.md`
- **데이터셋 가이드**: `DATASET_USAGE_GUIDE.md`
- **통합 보고서**: `DATASET_INTEGRATION_REPORT.md`
- **Requirements**: `requirements.txt`

---

## ❓ 다음 단계

1. **환경 선택**: Docker / UV / pip 중 선택
2. **환경 설정**: 위의 옵션 A/B/C 중 하나 실행
3. **테스트**: 샘플 전처리 실행
4. **본격 작업**: 5명 → 10명 → 전체 순차 확장

---

**작성:** Claude Code
**목적:** Python 환경 설정 필요성 및 해결 방법 안내
**다음 작업:** 환경 설정 후 1단계 전처리 시작
