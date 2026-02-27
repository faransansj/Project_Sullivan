# Project Sullivan: Project Overview & Status Report

**Project**: Acoustic-to-Articulatory Parameter Inference
**Current Status**: Phase 1–3 Complete, Phase 4 Active, Phase 5 Planning
**Last Update**: 2026-02-27

---

## 1. Project Goal
Project Sullivan aims to synthesize low-dimensional articulatory parameters (tongue, jaw, lips) from speech audio by leveraging real-time MRI (rtMRI) data from the USC-TIMIT and HDDB datasets.

## 2. Phase 1: Data Preprocessing & Segmentation ✅
- **MRI/Audio Alignment**: Synchronized audio with MRI frames for 468 utterances.
- **U-Net Segmentation**: **81.8% Mean Dice Score**, **96.5% tongue region**.

## 3. Phase 2: Baseline Model ✅
- **Bi-LSTM Baseline**: RMSE 1.011, PCC 0.105 — proved task feasibility.
- **Transformer**: Encoder architecture implemented.

## 4. Phase 3: Core Goal & High-Resolution Shape Recovery ✅
- Full-Scale Training + PCA Reconstruction.
- **Global PCC 0.1982** (7.6x improvement over Phase 2).
- 21.5M parameter Transformer Encoder, 24-dim output (14 Geometric + 10 PCA).
- High-fidelity tracking: Jaw Opening PCC 0.50, Tongue Fronting PCC 0.46.

---

## 5. Phase 4: 정확도 개선 파이프라인 🔄

Core Goal 달성 후, 모델 정확도를 더 높이기 위한 파이프라인.

### 4-1: Inference Engine
- [ ] 모델 로딩/예측 추상화 (`src/inference/engine.py`)

### 4-2: HuBERT Features
- [ ] 사전학습 오디오 피처로 Mel-spectrogram 대체
- [ ] `src/audio_features/hubert_extractor.py` 구현

### 4-3: Conformer Architecture
- [ ] Transformer → Conformer (Conv + Attention) 업그레이드
- [ ] 성능 비교 실험

### 4-4: A100 High-Performance Training
- [ ] Mixed precision + 대규모 배치 학습
- [ ] 목표: **PCC > 0.4**

---

## 6. Phase 5: 인프라 구축 & 프로덕션 ⬜

### 5-1: 외부 GPU 서버 환경 (A100/A6000)
- [ ] UV 기반 재현 가능한 학습 환경
- [ ] SSH 원격 학습 워크플로우
- [ ] 상세 계획: [PHASE5_1_GPU_SERVER.md](plans/PHASE5_1_GPU_SERVER.md)

### 5-2: 대용량 데이터 학습 (NAS 600GB+)
- [ ] NAS → GPU 서버 데이터 전송 전략
- [ ] Streaming DataLoader 구현
- [ ] 상세 계획: [PHASE5_2_NAS_DATA.md](plans/PHASE5_2_NAS_DATA.md)

### 5-3: 웹 기반 데모 & 모니터링
- [ ] 데이터셋 품질 검증 뷰어
- [ ] 학습 모니터링 대시보드
- [ ] 추론 데모 페이지
- [ ] 상세 계획: [PHASE5_3_WEB_DEMO.md](plans/PHASE5_3_WEB_DEMO.md)
