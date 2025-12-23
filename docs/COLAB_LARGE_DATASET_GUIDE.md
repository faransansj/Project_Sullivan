# Project Sullivan - 대용량 데이터셋 Colab 학습 가이드

**500GB Google Drive 데이터셋을 활용한 학습**

---

## 📋 개요

이 가이드는 Google Drive에 저장된 대용량 데이터셋(500GB)을 Google Colab 무료 버전으로 학습하는 방법을 설명합니다.

### 주요 과제 및 해결책

| 과제 | 해결책 |
|------|--------|
| 90분 비활성 시 세션 종료 | Keep-Alive 스레드 + JavaScript |
| 12시간 최대 세션 시간 | 매 에포크 체크포인팅 → 재개 |
| GPU 메모리 부족 | Mixed Precision + Gradient Accumulation |
| 대용량 데이터 로딩 | Streaming DataLoader |

---

## 🔧 워크플로우

### 1. 로컬 개발 (CLI)
```bash
# 코드 수정 및 테스트
code .

# 변경사항 GitHub에 푸시
./scripts/colab_cli.sh push "Update model architecture"

# 학습 상태 확인
./scripts/colab_cli.sh status
```

### 2. Colab 학습
1. `notebooks/Sullivan_GDrive_Training.ipynb` 열기
2. GPU 런타임 설정: Runtime → Change runtime type → GPU
3. 모든 셀 순서대로 실행
4. Keep-Alive 셀 실행 (세션 유지)
5. 학습 시작 및 모니터링

### 3. 세션 종료 시 재개
1. 노트북 다시 열기
2. `RESUME_TRAINING = True` 확인
3. 셀 재실행 → 자동으로 마지막 체크포인트에서 재개

---

## 📁 Google Drive 폴더 구조

```
MyDrive/
├── Sullivan_Dataset/           # 500GB 데이터셋
│   ├── audio_features/         # 오디오 특징
│   ├── parameters/             # 조음 파라미터
│   ├── segmentations/          # MRI 세그멘테이션
│   └── splits/                 # Train/Val/Test 분할
├── Sullivan_Checkpoints/       # 체크포인트 저장
└── Sullivan_Logs/              # TensorBoard 로그
```

---

## 🛡️ 세션 유지 전략

### Keep-Alive 메커니즘

노트북에 포함된 Keep-Alive 셀을 실행하면:
1. **Python 스레드**: 1분마다 출력 생성
2. **JavaScript**: 브라우저 활동 유지

```python
# 노트북의 Keep-Alive 셀
import threading, time

def keep_alive_thread():
    while True:
        time.sleep(60)
        print('.', end='', flush=True)

keepalive = threading.Thread(target=keep_alive_thread, daemon=True)
keepalive.start()
```

### 체크포인트 저장

`configs/colab_gdrive_config.yaml`에서 설정:
```yaml
checkpointing:
  dirpath: "/content/drive/MyDrive/Sullivan_Checkpoints"
  every_n_epochs: 1  # 매 에포크마다 저장
  save_top_k: 3       # 최근 3개 유지
```

---

## 💡 팁 & 트러블슈팅

### GPU 메모리 부족 (OOM)

1. **Batch size 줄이기**: `batch_size: 16 → 8`
2. **Gradient Accumulation**: `accumulate_grad_batches: 4`
3. **Mixed Precision**: `precision: 16`

### 데이터 로딩 느림

1. **Streaming 모드 확인**: `streaming: true`
2. **Worker 수 조정**: `num_workers: 2`
3. **캐시 크기 증가**: `cache_size: 2000`

### 세션 자주 종료됨

1. **Keep-Alive 셀 실행 확인**
2. **브라우저 탭 유지** (최소화하지 않기)
3. **자주 체크포인팅** (`every_n_epochs: 1`)

---

## 📊 예상 학습 시간

| 설정 | 시간 | 비고 |
|------|------|------|
| Quick Test (10 epochs) | 30분-1시간 | 검증용 |
| Full Training (50 epochs) | 4-6시간 | 1-2 세션 |
| Full Training (100 epochs) | 8-12시간 | 2-3 세션 재개 필요 |

---

**Last Updated**: 2025-12-23
