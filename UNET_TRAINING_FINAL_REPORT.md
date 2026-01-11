# U-Net 훈련 완료 보고서

**훈련 완료 일시**: 2026-01-11
**훈련 모델**: U-Net (Vocal Tract Segmentation)
**데이터셋**: Pseudo Labels (200 샘플)

---

## 📊 최종 성능 결과

### Validation 성능 (최고)
- **Epoch**: 95
- **Validation Dice Score**: **0.9219** (92.19%)
- **Validation Loss**: 0.128

### Test 성능 (최종 평가)
- **Test Dice Score**: **0.9142** (91.42%)
- **Test Loss**: 0.1350

### Training 성능 (마지막 epoch)
- **Train Dice Score**: 0.898 (89.8%)
- **Train Loss**: 0.158

---

## 📈 훈련 진행 상황

**총 훈련 기간**: Epoch 0-49 (이전) + Epoch 50-99 (재개)
- 시작: Epoch 49, val_dice=0.9196
- 종료: Epoch 99, val_dice=0.9218
- 최고: Epoch 95, val_dice=0.9219

**성능 개선**:
- Resume 시작 (Epoch 49): val_dice = 0.9196
- 최종 최고 성능 (Epoch 95): val_dice = 0.9219
- **개선율**: +0.0023 (0.25% 향상)

---

## 💾 저장된 파일

### 체크포인트 (Top 3)
1. `models/unet_scratch/checkpoints/unet-epoch=95-val_dice=0.9219.ckpt` ⭐ (최고 성능)
2. `models/unet_scratch/checkpoints/unet-epoch=98-val_dice=0.9218.ckpt`
3. `models/unet_scratch/checkpoints/unet-epoch=48-val_dice=0.9186.ckpt`
4. `models/unet_scratch/checkpoints/last.ckpt` (마지막 epoch)

### 최종 모델
- `models/unet_scratch/unet_best.pth` (테스트 완료 모델)

### 로그
- TensorBoard 로그: `models/unet_scratch/logs/unet_training/version_6/`

---

## 🎯 모델 사용법

### 1. 체크포인트에서 로드
```python
from src.segmentation.unet_lightning import UNetLightning

# 최고 성능 모델 로드
model = UNetLightning.load_from_checkpoint(
    'models/unet_scratch/checkpoints/unet-epoch=95-val_dice=0.9219.ckpt'
)
model.eval()
```

### 2. State Dict에서 로드
```python
import torch

# 최종 모델 로드
model = UNetLightning(n_channels=1, n_classes=1)
model.load_state_dict(torch.load('models/unet_scratch/unet_best.pth'))
model.eval()
```

### 3. 추론 (Inference)
```python
import torch
from PIL import Image
import numpy as np

# 이미지 로드 및 전처리
image = Image.open('path/to/image.png').convert('L')
image = np.array(image).astype(np.float32) / 255.0
image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

# 추론
with torch.no_grad():
    output = model(image)
    mask = (output > 0.5).float()  # Binary mask
```

---

## 🔍 분석 및 인사이트

### 강점
1. ✅ **높은 Dice Score**: 테스트 데이터에서 91.42% 달성
2. ✅ **안정적인 훈련**: Validation과 Test 성능 차이가 작음 (0.77%p)
3. ✅ **일관된 개선**: Resume 후에도 계속 성능 향상

### 개선 가능 영역
1. **Train-Val Gap**: Train dice(89.8%) vs Val dice(92.2%)
   - Validation이 더 높음 → 데이터 증강 효과 또는 작은 validation set
2. **Val-Test Gap**: Val dice(92.2%) vs Test dice(91.4%)
   - 약간의 overfitting 가능성

### 권장사항
1. 더 많은 데이터로 훈련 시 성능 향상 예상
2. Test augmentation (TTA) 적용 시 성능 개선 가능
3. Ensemble 방법 적용 고려

---

## 📝 훈련 설정

**하이퍼파라미터**:
- Learning Rate: 1e-4
- Batch Size: 8
- Max Epochs: 100
- Optimizer: Adam (기본 설정)
- Loss: Combined Loss (BCE 50% + Dice 50%)

**데이터 분할**:
- Train: 140 samples (70%)
- Validation: 30 samples (15%)
- Test: 30 samples (15%)

**하드웨어**:
- Device: CPU
- Training Speed: ~3-4초/epoch

---

## ✅ 다음 단계

1. ✅ **훈련 완료** - U-Net 모델 준비됨
2. 🔄 **다음 작업**:
   - USC-TIMIT 전체 데이터셋에 세그멘테이션 적용
   - 추가 데이터로 모델 재훈련 (선택사항)
   - 다운스트림 task에 모델 활용

---

**생성 일시**: 2026-01-11
**모델 상태**: Production Ready ✅
