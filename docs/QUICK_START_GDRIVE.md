# Project Sullivan - Quick Start Guide (Google Drive Dataset)

**500GB Dataset이 이미 Google Drive에 있는 경우**

---

## ✅ 현재 상태
- Google Drive 경로: `MyDrive/Project_Sullivan/Dataset` (644MB zip)
- rclone 연결: 완료
- SSH 도구: 설치 완료

---

## 🚀 Quick Start

### 1. Colab 노트북 실행
기존 `Sullivan_GDrive_Training.ipynb` 대신 간단한 Python 스크립트로 시작:

```python
# Colab에서 실행
!git clone https://github.com/faransansj/Project_Sullivan.git
%cd Project_Sullivan  
!python scripts/extract_gdrive_dataset.py
```

### 2. 자동 진행 과정
- Google Drive 마운트
- `Project_Sullivan/Dataset` 압축 해제
- `/content/sullivan_data/`에 데이터 추출
- 학습 준비 완료

### 3. 학습 시작
데이터 추출 후:
```python
!python scripts/train_transformer.py --config configs/colab_gdrive_config.yaml
```

---

## 📁 예상 파일 구조

압축 해제 후:
```
/content/sullivan_data/
├── audio_features/
├── parameters/
├── segmentations/
└── splits/
```

---

## 🔌 SSH 원격 제어 (선택사항)

로컬 터미널에서 Colab 제어를 원하면:

**Colab에서:**
```python
!pip install colab-ssh
from colab_ssh import launch_ssh_cloudflared
launch_ssh_cloudflared(password="sullivan2025")
```

**로컬에서:**
```bash
./scripts/colab_connect.sh save    # 연결 정보 저장
./scripts/colab_connect.sh connect # SSH 접속
```

---

**Last Updated**: 2025-12-23
