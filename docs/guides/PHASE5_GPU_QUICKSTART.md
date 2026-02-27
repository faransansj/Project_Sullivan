# Phase 5-1: GPU 서버 Quick Start 가이드

**Phase**: 5-1
**Last Update**: 2026-02-27

---

## 0. 전체 워크플로우

```
[로컬 Mac]                  [GPU 서버 (A100/A6000)]
     │                              │
     │  1. SSH 설정                  │
     ├─────────────────────────────→│
     │                              │
     │  2. 환경 초기화 스크립트       │
     ├─────────────────────────────→│ setup_remote_env.sh
     │                              │  └→ UV 설치 + 의존성 동기화
     │                              │
     │  3. 데이터 전송 (rsync)       │
     ├─────────────────────────────→│ NAS → GPU 서버
     │                              │
     │  4. 학습 실행                 │
     ├─────────────────────────────→│ remote_train.sh
     │                              │  └→ nohup 백그라운드 학습
     │                              │
     │  5. 모니터링                  │
     │←─────────────────────────────┤ TensorBoard (포트 6006)
     │                              │ check_gpu_status.sh
     │                              │
     │  6. 결과 회수                 │
     │←─────────────────────────────┤ rsync models/ logs/
```

---

## 1. SSH 설정

`~/.ssh/config`에 아래 내용 추가:

```
Host sullivan-gpu
    HostName <GPU_SERVER_IP>
    User <USERNAME>
    IdentityFile ~/.ssh/id_rsa
    LocalForward 6006 localhost:6006    # TensorBoard
    LocalForward 7860 localhost:7860    # Gradio
    ServerAliveInterval 60
    ServerAliveCountMax 120
```

> 📄 템플릿: `configs/infra/ssh_config_template`

---

## 2. 서버 환경 초기화

**최초 1회만 실행**하면 됩니다.

```bash
# 방법 1: 로컬에서 직접 전송 & 실행
ssh sullivan-gpu 'bash -s' < scripts/infra/setup_remote_env.sh

# 방법 2: 파일 복사 후 실행
scp scripts/infra/setup_remote_env.sh sullivan-gpu:~/
ssh sullivan-gpu 'bash ~/setup_remote_env.sh'
```

이 스크립트가 하는 일:
1. GPU / CUDA 감지
2. UV 패키지 매니저 설치
3. 레포지트리 클론 (또는 pull)
4. `uv sync --extra gpu` 의존성 설치
5. PyTorch + CUDA 작동 검증
6. 작업 디렉터리 생성

---

## 3. 데이터 전송

```bash
# NAS → GPU 서버 (core subset만 전송, 약 50GB)
ssh sullivan-gpu 'mkdir -p ~/Project_Sullivan/data/processed'

rsync -avz --progress \
    /path/to/nas/data/processed/ \
    sullivan-gpu:~/Project_Sullivan/data/processed/
```

---

## 4. 학습 실행

### 자동 방식 (추천)

```bash
# Conformer 학습 (기본)
./scripts/infra/remote_train.sh sullivan-gpu

# 특정 config
./scripts/infra/remote_train.sh sullivan-gpu configs/conformer_a100_config.yaml train_conformer.py

# 학습 재개
./scripts/infra/remote_train.sh sullivan-gpu configs/conformer_a100_config.yaml train_conformer.py --auto-resume
```

### 수동 방식

```bash
# 1. 코드 동기화
rsync -avz --exclude '.venv' --exclude 'data' --exclude 'models' \
    ./ sullivan-gpu:~/Project_Sullivan/

# 2. SSH 접속 후 직접 실행
ssh sullivan-gpu
cd ~/Project_Sullivan
uv run python scripts/train_conformer.py \
    --config configs/conformer_a100_config.yaml \
    --gpus 1
```

---

## 5. 모니터링

### GPU 상태

```bash
./scripts/infra/check_gpu_status.sh sullivan-gpu
```

### TensorBoard

```bash
# 로컬에서 실행 (SSH 포워딩 필요)
ssh sullivan-gpu    # LocalForward 6006 설정 필요
# GPU 서버에서:
cd ~/Project_Sullivan && tensorboard --logdir logs/training

# → 로컬 브라우저에서 http://localhost:6006
```

### 학습 로그

```bash
ssh sullivan-gpu 'tail -f ~/Project_Sullivan/logs/train_*.log'
```

---

## 6. 결과 회수

```bash
# 모델 다운로드
rsync -avz sullivan-gpu:~/Project_Sullivan/models/ ./models/

# 로그 다운로드
rsync -avz sullivan-gpu:~/Project_Sullivan/logs/ ./logs/
```

---

## 핵심 파일 목록

| 파일 | 용도 |
|------|------|
| `scripts/infra/setup_remote_env.sh` | 서버 최초 환경 설정 |
| `scripts/infra/remote_train.sh` | 원격 학습 실행 |
| `scripts/infra/check_gpu_status.sh` | GPU 상태 확인 |
| `configs/infra/ssh_config_template` | SSH 설정 템플릿 |
| `configs/conformer_a100_config.yaml` | A100 학습 설정 |
| `pyproject.toml` → `[gpu]` extras | GPU 의존성 |

---

## 트러블슈팅

### UV 명령어를 찾을 수 없는 경우
```bash
export PATH="$HOME/.local/bin:$PATH"
# ~/.bashrc 에 추가하면 영구 적용
```

### CUDA 버전 불일치
```bash
# 서버의 CUDA 버전 확인
nvidia-smi | grep 'CUDA Version'

# PyTorch CUDA 버전 확인
uv run python -c "import torch; print(torch.version.cuda)"
```

### 학습 중 SSH 끊김
- `nohup`으로 실행하므로 학습은 계속됨
- `--auto-resume` 옵션으로 자동 재개 가능
- `tmux` 또는 `screen` 사용 권장
