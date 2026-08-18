# Current Status

**Last Update**: 2026-03-20 (Phase 4 종결)
**Current Phase**: Phase 4 Complete, Phase 5 data expansion recommended

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

## ✅ Phase 4: 정확도 개선 파이프라인 (Complete)

HuBERT, Conformer, SpecAugment, Curriculum Loss를 포함한 9개 변형을 A100에서 비교했습니다.

| 모델 | 파라미터 | Test RMSE | Test PCC |
|------|---------:|----------:|---------:|
| Phase 3 Transformer | 21.5M | — | **0.1982** |
| Phase 4 HuBERT Small Conformer | 6.3M | **0.1200** | **0.1212** |

- RMSE M2 목표(< 0.15)는 달성했습니다.
- PCC M2 목표(> 0.50)는 달성하지 못했습니다.
- 대형 모델, SpecAugment, Curriculum Loss는 일반화를 개선하지 못했습니다.
- 현재 병목은 모델 구조보다 약 330개인 훈련 발화 규모입니다.
- 상세 결과: `papers/phase4_research_journal.md`

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
Phase 4: 정확도 개선 파이프라인       ✅ Complete (RMSE 0.1200, PCC 0.1212)
Phase 5: 데이터 확장 & 프로덕션        ⬜ Planning
  ├── 5-1: 외부 GPU 서버 (A100/A6000)
  ├── 5-2: NAS 600GB 데이터 연계
  └── 5-3: 웹 데모 & 모니터링
```
