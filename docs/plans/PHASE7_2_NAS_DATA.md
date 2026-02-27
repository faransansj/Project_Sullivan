# Phase 7-2: 대용량 데이터 학습 전략 (NAS 600GB+ 연계)

**Phase**: 7-2
**Status**: ⬜ Planning
**Last Update**: 2026-02-27
**Estimated Duration**: 2–3주
**Dependency**: Phase 7-1 (GPU 서버 환경) 완료 후 진행

---

## 1. 목표

NAS 서버에 저장된 600GB+ USC-TIMIT 데이터셋을 외부 GPU 서버에서 효율적으로 학습할 수 있는 데이터 파이프라인을 구축한다.

### 핵심 제약사항
- **NAS 서버**: 780M GPU — 학습 불가, **스토리지 전용**
- **GPU 서버**: A100/A6000 — 충분한 컴퓨팅, **로컬 스토리지 제한**
- **데이터 크기**: 600GB+ (전체 복사 비현실적일 수 있음)

---

## 2. 데이터 접근 전략 비교

| 방식 | 장점 | 단점 | 적합 상황 |
|------|------|------|----------|
| **rsync 전체 복사** | 최고 I/O 성능 | 600GB 전송 시간, 추가 스토리지 필요 | GPU 서버 SSD 여유 시 |
| **rsync 부분 복사** | 빠른 시작 | 데이터 제한 | 소규모 실험, 검증 |
| **NFS Mount** | 투명한 접근 | 네트워크 I/O 병목 | 같은 네트워크 내 |
| **Streaming DataLoader** | 최소 스토리지 | 구현 복잡도 | 대규모 데이터 |
| **HDF5 Sharding** | 효율적 랜덤 접근 | 전처리 필요 | 안정적 학습 |

### 권장 전략: **하이브리드 접근**
1. **1차**: rsync로 핵심 subset 전송 (100–200GB, 핵심 피험자)
2. **2차**: Streaming DataLoader로 나머지 데이터 점진적 학습
3. **3차**: 전처리된 HDF5 shard 파일로 최적화

---

## 3. 구현 계획

### Step 1: 데이터 전송 스크립트

**파일**: `scripts/infra/sync_data.sh`

```bash
#!/bin/bash
# ============================================
# NAS → GPU Server Data Sync
# ============================================
# Usage: ./scripts/infra/sync_data.sh <mode> <gpu-server>
# Modes: subset, full, incremental

MODE=${1:-subset}
GPU_SERVER=${2:-sullivan-gpu}
NAS_DATA="/mnt/HDDB/dataset/my_dataset/dataset"
REMOTE_DATA="~/Project_Sullivan/data/raw/usc_timit_full"

case $MODE in
    subset)
        echo "=== Syncing priority subjects (estimated: ~150GB) ==="
        # 품질 검증된 상위 10명 피험자
        SUBJECTS="sub011 sub012 sub013 sub014 sub015 sub016 sub017 sub018 sub019 sub020"
        for SUB in $SUBJECTS; do
            echo "Syncing ${SUB}..."
            rsync -avz --progress \
                ${NAS_DATA}/${SUB}/ \
                ${GPU_SERVER}:${REMOTE_DATA}/${SUB}/
        done
        ;;
    full)
        echo "=== Full dataset sync (estimated: 600GB+) ==="
        rsync -avz --progress \
            ${NAS_DATA}/ \
            ${GPU_SERVER}:${REMOTE_DATA}/
        ;;
    incremental)
        echo "=== Incremental sync (new/changed files only) ==="
        rsync -avz --progress --update \
            ${NAS_DATA}/ \
            ${GPU_SERVER}:${REMOTE_DATA}/
        ;;
esac

echo "=== Sync complete ==="
ssh ${GPU_SERVER} "du -sh ${REMOTE_DATA}"
```

**작업 항목:**
- [ ] `scripts/infra/sync_data.sh` 작성
- [ ] subset / full / incremental 모드 지원
- [ ] 전송 진행률 및 검증 로직

---

### Step 2: Streaming DataLoader 구현

**파일**: `src/modeling/streaming_dataset.py`

현재 `src/modeling/dataset.py`의 `ArticulatoryDataset`은 모든 데이터를 메모리에 로드. 
600GB+ 데이터에 대응하는 streaming 버전 구현.

```python
"""
Streaming Dataset for Large-Scale Training.

메모리에 전체 데이터를 로드하지 않고, 필요한 시점에
개별 utterance를 디스크에서 읽어오는 방식.
"""

import h5py
import librosa
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, IterableDataset
from typing import Optional, Tuple


class StreamingArticulatoryDataset(Dataset):
    """
    Lazy-loading dataset for 600GB+ USC-TIMIT data.
    
    기존 ArticulatoryDataset과 동일한 인터페이스를 유지하면서
    개별 샘플을 요청 시점에 디스크에서 로드.
    
    Args:
        manifest_path: JSON manifest (utterance ID 목록)
        data_root: raw 데이터 루트 경로
        features_root: 전처리된 feature 캐시 경로
        audio_feature_type: 'mel' or 'mfcc'
        max_seq_len: 최대 시퀀스 길이
        cache_size: LRU 캐시 크기 (utterance 단위)
    """
    
    def __init__(
        self,
        manifest_path: str,
        data_root: str,
        features_root: str,
        audio_feature_type: str = "mel",
        max_seq_len: int = 500,
        cache_size: int = 100,
    ):
        self.data_root = Path(data_root)
        self.features_root = Path(features_root)
        self.audio_feature_type = audio_feature_type
        self.max_seq_len = max_seq_len
        
        # Load manifest
        import json
        with open(manifest_path, 'r') as f:
            self.utterance_ids = json.load(f)
        
        # LRU cache for recently accessed samples
        from functools import lru_cache
        self._load_sample = lru_cache(maxsize=cache_size)(self._load_sample_impl)
    
    def __len__(self) -> int:
        return len(self.utterance_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        utterance_id = self.utterance_ids[idx]
        audio_features, articulatory_params = self._load_sample(utterance_id)
        
        # Truncate or pad to max_seq_len
        seq_len = min(len(audio_features), self.max_seq_len)
        
        padded_features = np.zeros((self.max_seq_len, audio_features.shape[1]))
        padded_params = np.zeros((self.max_seq_len, articulatory_params.shape[1]))
        mask = np.zeros(self.max_seq_len)
        
        padded_features[:seq_len] = audio_features[:seq_len]
        padded_params[:seq_len] = articulatory_params[:seq_len]
        mask[:seq_len] = 1.0
        
        return (
            torch.FloatTensor(padded_features),
            torch.FloatTensor(padded_params),
            torch.FloatTensor(mask),
        )
    
    def _load_sample_impl(self, utterance_id: str):
        """디스크에서 개별 샘플 로드 (캐시됨)"""
        # Audio features
        feature_path = self.features_root / self.audio_feature_type / f"{utterance_id}.npy"
        audio_features = np.load(feature_path)
        
        # Articulatory parameters
        param_path = self.features_root / "parameters" / f"{utterance_id}.npy"
        params = np.load(param_path)
        
        return audio_features, params
```

**작업 항목:**
- [ ] `src/modeling/streaming_dataset.py` 구현
- [ ] LRU 캐시 기반 lazy loading
- [ ] 기존 `ArticulatoryDataset`과 호환 인터페이스
- [ ] `num_workers > 0` 멀티프로세스 데이터 로딩 테스트
- [ ] 단위 테스트 작성

---

### Step 3: 전처리 파이프라인 (NAS → 전처리 Feature)

NAS의 raw 데이터를 GPU 서버에서 사용 가능한 전처리된 feature로 변환.
전처리 자체는 CPU 작업이므로 NAS 서버 또는 별도 CPU 서버에서 실행 가능.

**파일**: `scripts/infra/batch_preprocess_remote.sh`

```bash
#!/bin/bash
# ============================================
# Remote Batch Preprocessing
# ============================================
# NAS 서버에서 CPU로 전처리 후, 결과를 GPU 서버로 전송
#
# 전처리 결과 (feature 파일)는 raw 데이터의 ~5-10% 크기
# 600GB raw → ~30-60GB features

NAS_SERVER=${1:-nas-server}
GPU_SERVER=${2:-sullivan-gpu}

echo "=== [1/3] NAS에서 전처리 실행 ==="
ssh ${NAS_SERVER} "cd Project_Sullivan && \
    uv run python scripts/batch_preprocess.py \
        --config configs/preprocess.yaml \
        --output-dir data/processed/"

echo "=== [2/3] Feature 추출 ==="
ssh ${NAS_SERVER} "cd Project_Sullivan && \
    uv run python scripts/extract_audio_features.py && \
    uv run python scripts/extract_articulatory_params.py"

echo "=== [3/3] Feature를 GPU 서버로 전송 ==="
# Feature 파일만 전송 (raw 데이터 제외)
rsync -avz --progress \
    ${NAS_SERVER}:~/Project_Sullivan/data/processed/ \
    ${GPU_SERVER}:~/Project_Sullivan/data/processed/

echo "=== Complete ==="
```

**작업 항목:**
- [ ] 전처리 → GPU 서버 전송 파이프라인
- [ ] 전처리 결과 크기 측정 및 최적화
- [ ] Checksum 기반 무결성 검증

---

### Step 4: 학습 재개 (Resume) 인프라

네트워크 불안정, 서버 재부팅 등에 대비하여 학습 재개 기능 강화.

**파일 수정**: `scripts/train_transformer.py`

```python
# 추가할 CLI 인자
parser.add_argument('--resume-from', type=str, default=None,
                    help='Path to checkpoint to resume training from')
parser.add_argument('--auto-resume', action='store_true',
                    help='Automatically resume from latest checkpoint')
```

**작업 항목:**
- [ ] `--resume-from` 및 `--auto-resume` 옵션 추가
- [ ] 최신 checkpoint 자동 탐색 로직
- [ ] Epoch, optimizer state, scheduler state 완전 복원
- [ ] 학습 로그 연속성 보장

---

### Step 5: 데이터 Subset 샘플링 전략

전체 600GB를 사용하기 전, 효율적인 subset 선정 전략.

**파일**: `configs/data_subsets.yaml`

```yaml
# 데이터 subset 정의
subsets:
  # Tier 1: 고품질 핵심 데이터 (~100GB)
  core:
    subjects: [sub011, sub012, sub013, sub014, sub015]
    description: "품질 검증 완료, 세그멘테이션 우수"
    estimated_size: "100GB"
    
  # Tier 2: 확장 데이터 (~250GB)  
  extended:
    subjects: [sub011-sub020]
    description: "Tier 1 + 추가 피험자"
    estimated_size: "250GB"
    
  # Tier 3: 전체 데이터 (~600GB)
  full:
    subjects: "all"
    description: "USC-TIMIT 전체 (27명)"
    estimated_size: "600GB+"

# 학습 전략
training_strategy:
  phase1: "core subset으로 기본 성능 확인"
  phase2: "extended로 일반화 성능 향상"
  phase3: "full dataset으로 최종 모델 학습"
```

**작업 항목:**
- [ ] `configs/data_subsets.yaml` 작성
- [ ] Tier별 subset 정의 및 우선순위 
- [ ] 학습 스크립트에서 subset 선택 기능

---

## 4. 디렉터리 구조 (신규)

```
scripts/infra/
├── sync_data.sh               # NAS → GPU 서버 데이터 전송
└── batch_preprocess_remote.sh # NAS 전처리 → GPU 전송

src/modeling/
└── streaming_dataset.py       # Streaming DataLoader (신규)

configs/
└── data_subsets.yaml          # 데이터 subset 정의
```

---

## 5. 성능 목표

| 지표 | 현재 | 목표 |
|------|------|------|
| 학습 가능 데이터양 | ~75 utterances | 840+ utterances (27명) |
| 데이터 로딩 속도 | N/A (메모리) | ≥ 100 samples/sec (streaming) |
| GPU 활용률 | N/A | ≥ 80% |
| 학습 재개 시간 | N/A | < 30초 (checkpoint 복원) |

---

## 6. 리스크 및 대응

| 리스크 | 영향 | 대응 |
|--------|------|------|
| NAS ↔ GPU 네트워크 대역폭 부족 | 데이터 전송 지연 | 전처리된 feature만 전송 (10x 압축) |
| GPU 서버 스토리지 부족 | 전체 데이터 복사 불가 | Streaming DataLoader + 부분 rsync |
| 전처리 불일치 | 학습 성능 저하 | Checksum 검증, 버전 관리 |
| 네트워크 끊김 | 학습 중단 | Auto-resume, checkpoint 주기 단축 |
