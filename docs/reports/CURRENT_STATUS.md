# Current Status

**Last Update**: 2026-02-27 (Phase 7 Roadmap Added)
**Current Phase**: Phase 5/6 (Web Demo & A100 Training) → Phase 7 Planning

---

## ✅ Completed Milestones

### Phase 1: Data Preprocessing & Segmentation
- **Outcome**: 81.8% Dice Score U-Net, 468 utterances processed.

### Phase 2: Audio-to-Parameter Model
- **Baseline (LSTM)**: RMSE 1.011, PCC 0.105.
- **Transformer**: Implemented and trained.

### Phase 3: Full-Scale Training
- **Outcome**: Scaled to full dataset, RMSE optimization.

### Phase 4: High-Resolution Shape Recovery
- **Outcome**: **Global PCC 0.1982**, 21.5M parameter Transformer Encoder, 24-dim output.
- **Deliverables**: [Final Report](../reports/PHASE4_FINAL_REPORT.md)

---

## 🔄 Active Tasks (Phase 5 & 6)

### M1: Inference Engine Wrapper (Phase 5)
- **Status**: ⏳ Pending
- **Goal**: Build `src/inference/engine.py` for model loading and prediction.

### M2: A100 Hyper-Performance Raid (Phase 6) 🚀
- **Status**: 🟢 In Progress (Config Created)
- **Goal**: Achieve PCC > 0.4 using A100 GPU and HuBERT features.
- **Task**: Implement `src/audio_features/hubert_extractor.py` and upgrade to Conformer.

### M3: Gradio UI
- **Status**: ⏳ Pending
- **Goal**: Create `scripts/app.py` for the web interface.

---

## 🚀 Phase 7 Roadmap (Next Steps)

### 7-1: 외부 GPU 서버 환경 구성 (A100 / A6000)
- **Detailed Plan**: [PHASE7_1_GPU_SERVER.md](../plans/PHASE7_1_GPU_SERVER.md)
- **Status**: ⬜ Planning
- **Goal**: 외부 GPU 서버(A100, A6000)에서 학습 파이프라인 실행
- **Key Requirements**:
  - UV 기반 환경 관리 (`uv sync`, `uv run`)로 재현 가능한 파이프라인 구성
  - SSH 기반 원격 학습 워크플로우 구축
  - `pyproject.toml` + `uv.lock`으로 의존성 완전 고정
  - CUDA 12.x 호환 PyTorch 설치 자동화
- **Tasks**:
  - [ ] 서버 접근 환경 설정 (SSH key, 방화벽)
  - [ ] UV 환경 초기화 스크립트 작성 (`scripts/setup_remote_env.sh`)
  - [ ] GPU 호환 `pyproject.toml` 업데이트 (CUDA extras)
  - [ ] 원격 학습 실행 스크립트 (`scripts/remote_train.sh`)
  - [ ] TensorBoard 원격 접근 설정

### 7-2: 대용량 데이터 학습 전략 (NAS 600GB+ 연계)
- **Detailed Plan**: [PHASE7_2_NAS_DATA.md](../plans/PHASE7_2_NAS_DATA.md)
- **Status**: ⬜ Planning
- **Goal**: NAS에 저장된 600GB+ 데이터셋을 GPU 서버에서 효율적으로 학습
- **Current Constraint**: NAS 서버 컴퓨팅(780M)으로는 학습 불가능, 스토리지 전용
- **Key Requirements**:
  - NAS → GPU 서버 데이터 전송 전략 (rsync / NFS mount / streaming)
  - Streaming DataLoader 구현 (전체 데이터 로컬 복사 불필요)
  - 데이터 샤딩 및 분산 전처리
  - Checkpoint 기반 중간 저장/재개 지원
- **Tasks**:
  - [ ] NAS 데이터 접근 방식 결정 (NFS vs rsync vs WebDAV)
  - [ ] Streaming 기반 Dataset 클래스 구현
  - [ ] 데이터 전처리 파이프라인 최적화 (on-the-fly vs pre-processed)
  - [ ] 학습 재개(resume) 인프라 구축
  - [ ] 데이터 subset 샘플링 전략 구현

### 7-3: 웹 기반 데모 & 모니터링 대시보드
- **Detailed Plan**: [PHASE7_3_WEB_DEMO.md](../plans/PHASE7_3_WEB_DEMO.md)
- **Status**: ⬜ Planning
- **Goal**: 데이터셋 품질 검증 + 학습 모니터링을 위한 웹 인터페이스
- **Key Requirements**:
  - 데이터셋 시각화: MRI 프레임, segmentation 결과, 파라미터 오버레이
  - 학습 진행 모니터링: Loss/PCC 그래프, epoch 진행률
  - 추론 데모: 오디오 입력 → 실시간 articulatory 파라미터 예측
  - 반응형 웹 UI (데스크톱 + 모바일)
- **Tasks**:
  - [ ] 프레임워크 선정 (Gradio / Streamlit / Next.js)
  - [ ] 데이터셋 탐색 뷰어 구현
  - [ ] 학습 로그 실시간 모니터링 연동
  - [ ] 추론 데모 페이지 구현
  - [ ] 배포 환경 구성 (Docker + Cloudflare/Vercel)

---

## 📅 Overall Roadmap

```
Phase 1-4: ✅ Complete (Data Pipeline → Shape Recovery)
Phase 5:   🔄 Inference Engine & Gradio UI
Phase 6:   🔄 A100 + HuBERT (High Performance)
Phase 7:   ⬜ Infrastructure & Monitoring
  ├── 7-1: 외부 GPU 서버 (A100/A6000) + UV 환경
  ├── 7-2: NAS 600GB 데이터 ↔ GPU 서버 연계
  └── 7-3: 웹 기반 데모 & 모니터링 대시보드
```
