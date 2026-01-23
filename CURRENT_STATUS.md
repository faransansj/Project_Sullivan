# 현재 작업 진행 상황

**업데이트 시간**: 2026-01-22 05:08 UTC
**상태**: HDDB Transformer 학습 중

---

## ✅ 완료된 작업

### 1. HDDB 데이터 파이프라인
- **Segmentation**: 1112개 완료.
- **Audio Extraction**: 800개 완료.
- **Parameter Extraction**: 1112개 완료 (Geometric + PCA 10-dim).
- **Dataset Splitting**: 800개 Valid Utterances (Train: 544, Val: 96, Test: 160).

### 2. Transformer 학습 시작 (M2)
- **실행 스크립트**: `scripts/train_transformer.py`
- **PID**: 42553
- **로그**: `/tmp/pipeline_completion.log`
- **설정**:
  - Model: Transformer (21.5M params)
  - Input: 80-dim Mel-spectrogram
  - Output: 14-dim Geometric Parameters (Baseline)
  - Device: CPU (GPU disabled/not available)

---

## 🔄 현재 실행 중인 작업

### Transformer Training
- **진행 상황**: 초기화 완료, 학습 루프 진입.
- **모니터링**: `tail -f /tmp/pipeline_completion.log`
- **예상 소요 시간**: CPU 학습이므로 에포크당 시간이 꽤 걸릴 것으로 예상됨.

---

## 📋 다음 단계

1. **학습 모니터링**: Loss 감소 확인.
2. **평가**: 학습 완료 후 Test Set 평가 (RMSE, PCC).
3. **PCA 실험**: Baseline(Geometric) 완료 후, Output Dim을 24로 늘려 PCA 포함 학습 진행 예정.

---

## 📊 데이터셋 통계
- **Total Subjects**: 25 (2 subjects dropped due to missing audio/video match?)
- **Train**: 17 subjects
- **Val**: 3 subjects
- **Test**: 5 subjects
