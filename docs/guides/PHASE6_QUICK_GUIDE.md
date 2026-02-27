# [PHASE 6 RAID GUIDE: A100 HIGH-RESOLUTION SULLIVAN] 🚀

선생님, A100의 풀 파워를 개방하기 위한 퀵 가이드입니다! 빰파카밤!

## 1. 전설 등급 장비 세팅 (Preparation)

먼저, 새로운 아키텍처와 특징 추출기를 사용하기 위해 필요한 패키지를 확인하세요.
```bash
pip install torchaudio>=2.0.0 lightning>=2.0.0 albumentations
```

## 2. 데이터 정화 마법 (HuBERT Feature Extraction)

멜-스펙트로그램보다 10배는 더 강력한 HuBERT 특징을 추출해야 합니다.
`src/audio_features/hubert_extractor.py`를 사용하여 전체 데이터셋의 특징을 새로 생성하세요.

*   **입력**: 16kHz 오디오
*   **출력**: 1024차원 특징 벡터 (12번째 레이어 권장)

## 3. 새로운 아머 장착 (Model Selection)

`src/modeling/conformer_model.py`가 성공적으로 제조되었습니다. 
기존 Transformer보다 '성도 형상'의 국소적 변화를 훨씬 잘 포착하는 최신 Conformer 구조입니다.

## 4. 실전 레이드 커맨드 (Training)

A100 GPU 1장을 사용하여 훈련을 시작하는 마법의 주문입니다:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/src
python scripts/train_conformer.py \
    --config configs/transformer_a100.yaml \
    --gpus 1
```

## 💡 주요 오버클럭 팁 (A100 전용)

1.  **배치 사이즈**: `configs/transformer_a100.yaml`에서 `batch_size: 256`이 너무 가볍다면 **512**까지 올려보세요!
2.  **정밀도**: `precision: "bf16-mixed"`는 선택이 아닌 필수입니다! 연산 속도가 비약적으로 상승합니다.
3.  **학습률**: 배치가 커졌으므로 `learning_rate`를 **0.001** 이상으로 설정하는 것이 좋습니다.

---
선생님, 이제 준비가 끝났습니다! A100의 엔진 소리와 함께 Global PCC 0.5 돌파를 향해 진격하십시오! 빰파카밤!
