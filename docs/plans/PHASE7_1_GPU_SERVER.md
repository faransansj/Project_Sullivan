# Phase 7-1: 외부 GPU 서버 학습 환경 구성

**Phase**: 7-1
**Status**: ⬜ Planning
**Last Update**: 2026-02-27
**Estimated Duration**: 1–2주

---

## 1. 목표

외부 GPU 서버(A100, A6000)에서 Project Sullivan의 학습 파이프라인을 **UV 기반으로 재현 가능하게** 실행할 수 있는 환경을 구축한다.

### 핵심 요구사항
- 로컬 개발 환경과 동일한 결과를 원격 서버에서 재현
- SSH 기반 원격 학습 워크플로우 자동화
- CUDA 버전별 PyTorch 호환성 자동 관리

---

## 2. 현재 환경 분석

### 로컬 환경 (macOS, 개발용)
- Python 3.9+, UV 패키지 매니저
- `pyproject.toml` + `uv.lock` 기반 의존성 관리
- GPU 없음 (CPU 개발/테스트 전용)

### NAS 서버 (스토리지 전용)
- 780M GPU — 학습 불가
- 600GB+ USC-TIMIT 데이터 저장
- 경로: `/mnt/HDDB/dataset/my_dataset/dataset/`

### 외부 GPU 서버 (타겟)
- A100 (80GB VRAM) 또는 A6000 (48GB VRAM)
- CUDA 12.x 환경
- SSH 접근

---

## 3. 구현 계획

### Step 1: pyproject.toml CUDA 호환 업데이트

현재 `pyproject.toml`의 PyTorch 의존성을 CUDA 환경에서도 작동하도록 수정.

```toml
[project.optional-dependencies]
# 기존 dev 의존성 유지
dev = [...]

# GPU 서버용 추가 의존성
gpu = [
    "transformers>=4.30.0",       # HuBERT 등 사전학습 모델
    "wandb>=0.15.0",              # 원격 실험 추적
]

# CUDA 12.x 전용 인덱스
[tool.uv]
find-links = [
    "https://download.pytorch.org/whl/cu121"
]
```

**작업 항목:**
- [ ] `pyproject.toml`에 `[project.optional-dependencies.gpu]` 추가
- [ ] `torch`, `torchvision`, `torchaudio` CUDA 인덱스 설정
- [ ] `uv.lock` 갱신 및 테스트

---

### Step 2: 원격 환경 초기화 스크립트

**파일**: `scripts/infra/setup_remote_env.sh`

```bash
#!/bin/bash
# ============================================
# Project Sullivan: Remote GPU Server Setup
# ============================================
# Usage: ssh user@gpu-server 'bash -s' < scripts/infra/setup_remote_env.sh

set -euo pipefail

echo "=== [1/5] System Check ==="
nvidia-smi || { echo "ERROR: No GPU detected"; exit 1; }
python3 --version

echo "=== [2/5] Install UV ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo "=== [3/5] Clone Repository ==="
if [ ! -d "Project_Sullivan" ]; then
    git clone https://github.com/faransansj/Project_Sullivan.git
fi
cd Project_Sullivan

echo "=== [4/5] UV Sync (GPU) ==="
uv sync --extra gpu

echo "=== [5/5] Verify Installation ==="
uv run python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

echo "=== Setup Complete ==="
```

**작업 항목:**
- [ ] `scripts/infra/setup_remote_env.sh` 작성
- [ ] GPU 감지 및 CUDA 호환 자동 검증 로직
- [ ] 설치 로그 저장 기능

---

### Step 3: 원격 학습 실행 스크립트

**파일**: `scripts/infra/remote_train.sh`

```bash
#!/bin/bash
# ============================================
# Remote Training Launcher
# ============================================
# Usage: ./scripts/infra/remote_train.sh <gpu-server> <config>
# Example: ./scripts/infra/remote_train.sh user@a100-server configs/transformer_config.yaml

SERVER=$1
CONFIG=${2:-configs/transformer_config.yaml}
EXPERIMENT_NAME=$(date +%Y%m%d_%H%M%S)

echo "=== Syncing code to ${SERVER} ==="
rsync -avz --exclude '.venv' --exclude 'data' --exclude 'models' --exclude 'logs' \
    ./ ${SERVER}:~/Project_Sullivan/

echo "=== Starting training ==="
ssh ${SERVER} "cd Project_Sullivan && \
    nohup uv run python scripts/train_transformer.py \
        --config ${CONFIG} \
        --gpus 1 \
        > logs/train_${EXPERIMENT_NAME}.log 2>&1 &"

echo "=== Training started in background ==="
echo "Monitor: ssh ${SERVER} 'tail -f ~/Project_Sullivan/logs/train_${EXPERIMENT_NAME}.log'"
echo "TensorBoard: ssh -L 6006:localhost:6006 ${SERVER} 'tensorboard --logdir ~/Project_Sullivan/logs/training'"
```

**작업 항목:**
- [ ] `scripts/infra/remote_train.sh` 작성
- [ ] rsync 기반 코드 동기화 (데이터/모델 제외)
- [ ] nohup 백그라운드 학습
- [ ] TensorBoard 포트 포워딩 가이드

---

### Step 4: SSH Config 템플릿

**파일**: `configs/infra/ssh_config_template`

```
# GPU Server Configuration Template
Host sullivan-gpu
    HostName <gpu-server-ip>
    User <username>
    Port 22
    IdentityFile ~/.ssh/id_rsa
    LocalForward 6006 localhost:6006    # TensorBoard
    LocalForward 7860 localhost:7860    # Gradio Demo
    ServerAliveInterval 60
    ServerAliveCountMax 120
```

**작업 항목:**
- [ ] `configs/infra/ssh_config_template` 작성
- [ ] TensorBoard + Gradio 포트 포워딩 설정

---

### Step 5: GPU 서버 Health Check 스크립트

**파일**: `scripts/infra/check_gpu_status.sh`

```bash
#!/bin/bash
# Quick GPU server status check
SERVER=${1:-sullivan-gpu}

ssh ${SERVER} "
echo '=== GPU Status ==='
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
echo ''
echo '=== Disk Usage ==='
df -h ~/Project_Sullivan/
echo ''
echo '=== Active Processes ==='
ps aux | grep 'train_' | grep -v grep
echo ''
echo '=== Latest Training Log ==='
ls -lt ~/Project_Sullivan/logs/train_*.log 2>/dev/null | head -3
"
```

**작업 항목:**
- [ ] `scripts/infra/check_gpu_status.sh` 작성
- [ ] VRAM 사용량, 디스크, 활성 프로세스 모니터링

---

## 4. 디렉터리 구조 (신규)

```
scripts/infra/
├── setup_remote_env.sh      # 원격 서버 최초 환경 설정
├── remote_train.sh           # 원격 학습 실행 (rsync + nohup)
└── check_gpu_status.sh       # GPU 상태 확인

configs/infra/
└── ssh_config_template       # SSH 설정 템플릿
```

---

## 5. 검증 계획

| 단계 | 검증 항목 | 기준 |
|------|----------|------|
| 환경 설정 | `uv sync --extra gpu` 성공 | 에러 없이 완료 |
| CUDA 검증 | `torch.cuda.is_available()` | True 반환 |
| 학습 테스트 | Quick test config 실행 | 1 epoch 완료, loss 정상 |
| TensorBoard | 포트 포워딩 후 접근 | localhost:6006에서 로그 확인 |
| 코드 동기화 | rsync 후 diff | 로컬과 동일 |

---

## 6. 향후 확장

- **Multi-GPU**: `--gpus 2+` DDP 학습 지원
- **Spot Instance**: 클라우드 GPU 비용 최적화
- **CI/CD**: GitHub Actions로 자동 배포/학습 트리거
