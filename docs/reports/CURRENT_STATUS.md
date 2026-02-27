# Current Status

**Last Update**: 2026-02-27 (Phase 구조 재편성)
**Current Phase**: Phase 4 (정확도 개선) Active, Phase 5 (인프라) Planning

---

## ✅ Completed Milestones

### Phase 1: Data Preprocessing & Segmentation
- U-Net 81.8% Dice Score, 468 utterances processed.

### Phase 2: Baseline Model
- Bi-LSTM: RMSE 1.011, PCC 0.105 — task feasibility 입증.

### Phase 3: Core Goal & High-Resolution Shape Recovery
- Full-Scale Training + PCA Reconstruction.
- **Global PCC 0.1982** (7.6x improvement over Phase 2).
- 21.5M parameter Transformer Encoder, 24-dim output (14 Geometric + 10 PCA).
- Jaw Opening PCC 0.50, Tongue Fronting PCC 0.46.

---

## 🔄 Phase 4: 정확도 개선 파이프라인 (Active)

Core Goal 달성 이후, 모델 정확도를 더 높이기 위한 파이프라인 구축.

### 4-1: Inference Engine Wrapper
- **Status**: ⏳ Pending
- **Goal**: `src/inference/engine.py`로 모델 로딩 및 예측 로직 정리
- **Tasks**:
  - [ ] 모델 체크포인트 로딩 추상화
  - [ ] 배치/단일 추론 인터페이스
  - [ ] 오디오 → 파라미터 end-to-end 파이프라인

### 4-2: 고성능 오디오 피처 (HuBERT)
- **Status**: ⏳ Pending
- **Goal**: HuBERT 사전학습 모델로 기존 Mel-spectrogram 대체
- **Tasks**:
  - [ ] `src/audio_features/hubert_extractor.py` 구현
  - [ ] HuBERT feature 추출 + 캐싱
  - [ ] 기존 파이프라인과 통합

### 4-3: Conformer Architecture Upgrade
- **Status**: ⏳ Pending
- **Goal**: Transformer → Conformer (Conv + Attention)로 아키텍처 개선
- **Tasks**:
  - [ ] `src/modeling/conformer.py` 구현
  - [ ] Conformer config 작성
  - [ ] Transformer 대비 성능 비교 실험

### 4-4: A100 High-Performance Training 🚀
- **Status**: 🟢 Config Created
- **Goal**: A100 GPU에서 대규모 학습으로 PCC > 0.4 달성
- **Tasks**:
  - [ ] Mixed precision (FP16/BF16) 학습
  - [ ] 대규모 배치 사이즈 실험
  - [ ] Learning rate schedule 최적화
  - [ ] 목표: **PCC > 0.4** (Phase 3 대비 2x 개선)

### Phase 4 성능 목표

| 지표 | Phase 3 (현재) | Phase 4 목표 |
|------|---------------|-------------|
| Global PCC | 0.1982 | > 0.40 |
| Jaw Opening PCC | 0.50 | > 0.70 |
| Tongue Fronting PCC | 0.46 | > 0.65 |

---

## ⬜ Phase 5: 인프라 구축 & 프로덕션 (Planning)

학습 환경 확장, 대용량 데이터 연계, 웹 기반 모니터링 구축.

### 5-1: 외부 GPU 서버 환경 구성 (A100 / A6000)
- **Detailed Plan**: [PHASE5_1_GPU_SERVER.md](../plans/PHASE5_1_GPU_SERVER.md)
- **Status**: ⬜ Planning
- **Goal**: 외부 GPU 서버에서 UV 기반 재현 가능한 학습 환경 실행
- **Key Requirements**:
  - UV 기반 환경 관리 (`uv sync`, `uv run`)
  - SSH 원격 학습 워크플로우 자동화
  - `pyproject.toml` CUDA extras 설정
- **Tasks**:
  - [ ] 서버 접근 환경 설정 (SSH key, 방화벽)
  - [ ] UV 환경 초기화 스크립트 (`scripts/infra/setup_remote_env.sh`)
  - [ ] GPU 호환 `pyproject.toml` 업데이트
  - [ ] 원격 학습 실행 스크립트 (`scripts/infra/remote_train.sh`)
  - [ ] TensorBoard 원격 접근 설정

### 5-2: 대용량 데이터 학습 전략 (NAS 600GB+ 연계)
- **Detailed Plan**: [PHASE5_2_NAS_DATA.md](../plans/PHASE5_2_NAS_DATA.md)
- **Status**: ⬜ Planning
- **Goal**: NAS 600GB+ 데이터셋을 GPU 서버에서 효율적으로 학습
- **Current Constraint**: NAS 컴퓨팅(780M)은 학습 불가, 스토리지 전용
- **Key Requirements**:
  - NAS → GPU 서버 데이터 전송 전략 (rsync / NFS / streaming)
  - Streaming DataLoader 구현
  - Checkpoint 기반 학습 재개
- **Tasks**:
  - [ ] NAS 데이터 접근 방식 결정
  - [ ] Streaming Dataset 클래스 구현
  - [ ] 데이터 전처리 파이프라인 최적화
  - [ ] 학습 재개(resume) 인프라 구축
  - [ ] 데이터 subset 샘플링 전략

### 5-3: 웹 기반 데모 & 모니터링 대시보드
- **Detailed Plan**: [PHASE5_3_WEB_DEMO.md](../plans/PHASE5_3_WEB_DEMO.md)
- **Status**: ⬜ Planning
- **Goal**: 데이터셋 품질 검증 + 학습 모니터링 웹 인터페이스
- **Key Requirements**:
  - 데이터셋 시각화: MRI 프레임, segmentation, 파라미터 오버레이
  - 학습 모니터링: Loss/PCC 그래프, epoch 진행률
  - 추론 데모: 오디오 → articulatory 파라미터 예측
- **Tasks**:
  - [ ] Gradio 기반 통합 대시보드 구현
  - [ ] 데이터셋 탐색 뷰어
  - [ ] 학습 로그 실시간 모니터링
  - [ ] 추론 데모 페이지
  - [ ] 배포 환경 구성

---

## 📅 Overall Roadmap

```
Phase 1: Data Pipeline              ✅ Complete
Phase 2: Baseline Model             ✅ Complete
Phase 3: Core Goal & Shape Recovery  ✅ Complete (PCC 0.1982)
Phase 4: 정확도 개선 파이프라인       🔄 Active
  ├── 4-1: Inference Engine
  ├── 4-2: HuBERT Features
  ├── 4-3: Conformer Upgrade
  └── 4-4: A100 Training (PCC > 0.4)
Phase 5: 인프라 구축 & 프로덕션       ⬜ Planning
  ├── 5-1: 외부 GPU 서버 (A100/A6000)
  ├── 5-2: NAS 600GB 데이터 연계
  └── 5-3: 웹 데모 & 모니터링
```
