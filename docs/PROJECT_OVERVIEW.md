# Project Sullivan: Project Overview & Status Report

**Project**: Acoustic-to-Articulatory Parameter Inference
**Current Status**: Phase 1–4 Complete, Phase 5/6 Active, Phase 7 Planning
**Last Update**: 2026-02-27

---

## 1. Project Goal
Project Sullivan aims to synthesize low-dimensional articulatory parameters (tongue, jaw, lips) from speech audio by leveraging real-time MRI (rtMRI) data from the USC-TIMIT and HDDB datasets.

## 2. Phase 1: Data Preprocessing & Segmentation ✅
- **MRI/Audio Alignment**: Synchronized audio with MRI frames for 468 utterances.
- **U-Net Segmentation**: **81.8% Mean Dice Score**, **96.5% tongue region**.

## 3. Phase 2: Baseline Model ✅
- **Bi-LSTM Baseline**: RMSE 1.011, PCC 0.105 — proved task feasibility.
- **Transformer**: Encoder architecture implemented and training infrastructure set up.

## 4. Phase 3: Full-Scale Training ✅
- Scaled to full dataset with RMSE optimization and data augmentation strategies.

## 5. Phase 4: High-Resolution Shape Recovery ✅
- **Global PCC 0.1982** (7.6x improvement over Phase 2 baseline).
- 21.5M parameter Transformer Encoder, 24-dim output (14 Geometric + 10 PCA).
- High-fidelity tracking: Jaw Opening PCC 0.50, Tongue Fronting PCC 0.46.

## 6. Phase 5/6: Inference & High Performance 🔄
- **Phase 5**: Inference Engine (`src/inference/engine.py`) + Gradio web demo.
- **Phase 6**: A100 GPU training with HuBERT features and Conformer upgrade.

---

## 7. Phase 7 Roadmap (Next Steps)

### 7-1: 외부 GPU 서버 환경 (A100/A6000)
- [ ] UV 기반 재현 가능한 학습 환경 구축 (`uv sync` + `uv.lock`)
- [ ] SSH 원격 학습 워크플로우 및 자동화 스크립트
- [ ] CUDA 호환 `pyproject.toml` 업데이트

### 7-2: 대용량 데이터 학습 (NAS 600GB+)
- [ ] NAS → GPU 서버 데이터 전송 전략 (rsync / NFS / streaming)
- [ ] Streaming DataLoader 구현 (전체 복사 불필요)
- [ ] NAS 컴퓨팅(780M) 한계로 인한 전처리 분리 전략

### 7-3: 웹 기반 데모 & 모니터링
- [ ] 데이터셋 품질 검증 뷰어 (MRI + segmentation 시각화)
- [ ] 학습 진행 모니터링 대시보드 (Loss/PCC 그래프)
- [ ] 추론 데모 페이지 (오디오 → articulatory 파라미터)
