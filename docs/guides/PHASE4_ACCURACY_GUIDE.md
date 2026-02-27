# Phase 4: 정확도 개선 파이프라인 가이드

**Phase**: 4
**Last Update**: 2026-02-27
**Target**: PCC 0.1982 → PCC > 0.40

---

## 1. 개요

Phase 3에서 달성한 Global PCC 0.1982를 더 높이기 위한 정확도 개선 파이프라인입니다.

### 파이프라인 구조

```
Phase 4 정확도 개선 파이프라인

  4-1  Inference Engine        모델 로딩/예측 추상화
  4-2  HuBERT Features         고성능 오디오 피처
  4-3  Conformer Architecture  Conv + Attention 아키텍처
  4-4  A100 Training           대규모 GPU 학습
```

### 핵심 파일

| 구성 요소 | 파일 | 설명 |
|----------|------|------|
| Conformer 모델 | `src/modeling/conformer_model.py` | Conformer 아키텍처 (512d, 12 layers) |
| HuBERT 추출기 | `src/audio_features/hubert_extractor.py` | 사전학습 오디오 피처 |
| Inference Engine | `src/inference/engine.py` | 통합 추론 인터페이스 |
| 학습 스크립트 | `scripts/train_conformer.py` | Conformer 학습 실행 |
| A100 설정 | `configs/conformer_a100_config.yaml` | GPU 최적화 config |

---

## 2. Quick Start

### 2-1. 환경 준비

```bash
# UV 환경 동기화
uv sync

# GPU 확인
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 2-2. Conformer 학습 실행

```bash
# CPU (테스트용)
uv run python scripts/train_conformer.py \
    --config configs/conformer_a100_config.yaml \
    --gpus 0 \
    --fast-dev-run

# GPU (A100/A6000)
uv run python scripts/train_conformer.py \
    --config configs/conformer_a100_config.yaml \
    --gpus 1

# 학습 재개
uv run python scripts/train_conformer.py \
    --config configs/conformer_a100_config.yaml \
    --gpus 1 \
    --auto-resume
```

### 2-3. TensorBoard 모니터링

```bash
tensorboard --logdir logs/training
# http://localhost:6006 에서 확인
```

### 2-4. 추론 실행

```python
from src.inference.engine import InferenceEngine

# Transformer 모델 (기존)
engine = InferenceEngine(
    model_path="models/transformer/final_model.ckpt",
    config_path="configs/transformer_config.yaml",
    model_type="transformer",
    feature_type="mel"
)

# Conformer 모델 (Phase 4)
engine = InferenceEngine(
    model_path="models/conformer/final_model.ckpt",
    config_path="configs/conformer_a100_config.yaml",
    model_type="conformer",
    feature_type="mel"
)

# 예측
params = engine.predict("path/to/audio.wav")
print(f"Output shape: {params.shape}")  # (Time, 24)

# 상세 예측 (파라미터 이름 포함)
result = engine.predict_with_details("path/to/audio.wav")
for name, values in zip(result['param_names'], result['predictions'].T):
    print(f"  {name}: mean={values.mean():.4f}")
```

---

## 3. 아키텍처 상세

### 3-1. Conformer vs Transformer 비교

| 항목 | Transformer (Phase 3) | Conformer (Phase 4) |
|------|----------------------|---------------------|
| Self-Attention | ✅ | ✅ |
| Convolution | ❌ | ✅ (Depthwise) |
| 로컬 패턴 | 약함 | 강함 (Conv kernel=31) |
| 파라미터 수 | 21.5M | ~50M (12 layers) |
| Loss | Hybrid (MSE+PCC+Temporal) | Hybrid (동일) |
| Optimizer | AdamW + CosineAnnealing | AdamW + OneCycleLR |

### 3-2. Hybrid Loss 구성

```
Total Loss = mse_weight × MSE_loss
           + pcc_weight × PCC_loss
           + velocity_weight × Velocity_loss
           + acceleration_weight × Acceleration_loss
```

**기본 가중치 (A100 config):**
- `mse_weight`: 0.8
- `pcc_weight`: 2.0 (PCC 개선에 집중)
- `velocity_weight`: 1.2
- `acceleration_weight`: 0.5

### 3-3. HuBERT Features (선택적)

기존 Mel-spectrogram (80-dim) 대신 HuBERT-Large (1024-dim)을 사용하면
자기지도학습으로 학습된 풍부한 음성 표현을 활용할 수 있습니다.

```python
from src.audio_features.hubert_extractor import HuBERTExtractor

extractor = HuBERTExtractor(
    model_name="hubert_large_ll60k",
    layer_index=12,
    device="cuda"
)

features = extractor.extract(
    audio=audio_array,       # 16kHz numpy array
    num_mri_frames=500,      # MRI 프레임 수
    mri_fps=83.3             # MRI fps
)
# features.shape: (500, 1024)
```

**사용 시 config 변경:**
```yaml
data:
  audio_feature_type: hubert    # 'mel' → 'hubert'
model:
  input_dim: 1024               # 80 → 1024
```

---

## 4. 실험 전략

### 4-1. 단계별 실험 계획

| 실험 | 피처 | 모델 | 예상 PCC |
|------|------|------|---------|
| Baseline | Mel (80d) | Transformer 6L | 0.1982 (현재) |
| Exp-1 | Mel (80d) | Conformer 12L | > 0.25 |
| Exp-2 | HuBERT (1024d) | Transformer 6L | > 0.30 |
| Exp-3 | HuBERT (1024d) | Conformer 12L | > 0.40 |

### 4-2. A100 학습 세팅

```yaml
# A100 최적화 설정
training:
  batch_size: 32          # A100 80GB: 32~64 가능
  precision: "16-mixed"   # BF16 mixed precision
  num_workers: 8          # CPU 코어 활용
```

### 4-3. 학습 모니터링 체크리스트

- [ ] `train_loss` 안정적 감소 확인
- [ ] `val_loss` < `train_loss`가 아닌지 확인 (과적합 방지)
- [ ] `val_pearson` 지속적 상승 확인
- [ ] `val_mse_geo` vs `val_mse_pca` 균형 확인
- [ ] Learning rate schedule 정상 작동 확인

---

## 5. CLI 레퍼런스

### train_conformer.py

```
Options:
  --config PATH      Config 파일 경로 (default: configs/conformer_a100_config.yaml)
  --gpus N           GPU 수 (default: 1, 0=CPU)
  --fast-dev-run     1 batch 디버그 모드
  --resume-from PATH 특정 checkpoint에서 재개
  --auto-resume      최신 checkpoint 자동 재개
```

### 추론

```python
from src.inference.engine import InferenceEngine

engine = InferenceEngine(
    model_path="path/to/model.ckpt",
    config_path="path/to/config.yaml",
    model_type="conformer",   # 'transformer' | 'conformer'
    feature_type="mel",       # 'mel' | 'hubert'
    device="cpu"              # 'cpu' | 'cuda'
)
```
