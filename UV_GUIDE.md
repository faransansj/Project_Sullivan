# ⚡ Project Sullivan: uv 패키지 매니저 가이드라인

이 프로젝트는 초고속 파이썬 패키지 매니저인 `uv`를 사용하여 환경을 관리합니다. `uv`를 사용하면 기존 `pip`나 `conda`보다 훨씬 빠른 속도로 의존성을 설치하고 실행할 수 있으며, `ValueError: numpy.dtype size changed`와 같은 바이너리 호환성 문제를 방지할 수 있습니다.

## 1. uv 설치
`uv`가 설치되어 있지 않다면 아래 명령어로 설치하십시오.
```bash
pip install uv
```

## 2. 환경 구축 및 동기화
`pyproject.toml`과 `uv.lock` 파일을 기반으로 최적화된 가상환경을 생성하고 패키지를 동기화합니다.
```bash
uv sync
```
이 명령어는 `.venv` 폴더를 생성하고 모든 의존성을 자동으로 설치합니다.

## 3. 스크립트 실행
가상환경을 수동으로 활성화하지 않아도 `uv run` 명령어를 통해 환경이 적용된 상태로 스크립트를 실행할 수 있습니다.

### Pseudo-label 생성
```bash
uv run scripts/generate_pseudo_labels.py \
    --data-root /mnt/HDDB/dataset/my_dataset/dataset \
    --output-dir data/pseudo_labels \
    --num-subjects 10 \
    --frames-per-subject 50
```

### U-Net 모델 학습
```bash
uv run scripts/train_unet.py \
    --data-dir data/pseudo_labels \
    --output-dir models/unet_scratch \
    --batch-size 32 \
    --max-epochs 100
```

### HDDB 파이프라인 실행
```bash
uv run ./scripts/hddb_pipeline.sh phase1-s1 --gpu
```

## 4. 새로운 패키지 추가
새로운 마법 재료(패키지)가 필요할 때는 아래 명령어를 사용하십시오.
```bash
uv add <package_name>
```

---
빰파카밤! 이제 `uv`와 함께라면 어떤 환경 오류도 두렵지 않습니다! 용사여, 퀘스트를 계속하십시오!
