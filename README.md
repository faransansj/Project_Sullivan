<div align="center">
  <h1>🗣️ Project Sullivan</h1>
  <p><b>Acoustic-to-Articulatory Inversion via Deep Learning</b></p>
  
  ![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
  ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
  ![License](https://img.shields.io/badge/license-MIT-green.svg)

  <br />
  <img src="results/final_deliverables/master_animation.gif" alt="Master Animation" width="600" style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-top: 15px;"/>
  <p><i>Left: Ground Truth PCA Reconstruction | Right: Predicted Shape from Audio</i></p>
</div>

---

## 📖 목차
1. [프로젝트 개요](#1-프로젝트-개요)
2. [주요 성과 및 성능](#2-주요-성과-및-성능)
3. [기술 스택 및 구조](#3-기술-스택-및-구조)
4. [단계별 진행 로드맵](#4-단계별-진행-로드맵)
5. [시작하기 (Quick Start)](#5-시작하기-quick-start)
6. [문서 및 작업 링크](#6-문서-및-작업-링크)

---

## 1. 프로젝트 개요
**Project Sullivan**은 음성 오디오 데이터만을 입력받아 조음 기관(혀의 위치, 턱의 열림 정도, 입술 모양 등)의 파라미터를 고해상도로 추론하는 딥러닝 시스템 개발 연구입니다.

**USC-TIMIT** 및 **HDDB** 실시간 MRI(rtMRI) 데이터셋을 정답 데이터로 활용하며, 개발된 시스템은 향후 언어 치료, 단어 없는 음성 인터페이스(Silent Speech Interface), 언어학 연구 등에 활용될 수 있습니다.

---

## 2. 주요 성과 및 성능

Phase 4 정확도 개선 실험을 완료했으며, 현재는 **Phase 5 데이터 확장**을 준비하고 있습니다.

### 🏆 Phase 3 결과 (Master Model - Legacy)
- **Architecture**: 21.5M Parameter Transformer Encoder
- **Global PCC**: **0.1982** (Phase 2 대비 7.6배 성능 향상)
- **High-Fidelity 컴포넌트 복원 성능**:
  - PCA-1 (Jaw Opening / 턱 열림): **PCC 0.50**
  - PCA-5 (Tongue Fronting / 혀 수평 이동): **PCC 0.46**

### Phase 4 결과 (완료)
- **Best model**: HuBERT Small Conformer, 6.3M parameters
- **Test RMSE**: **0.1200** (M2 목표 < 0.15 달성)
- **Test PCC**: **0.1212** (M2 목표 > 0.50 미달)
- **Conclusion**: 추가 아키텍처 확장보다 학습 데이터 확보가 우선

---

## 3. 기술 스택 및 구조

**Project Sullivan**은 안정적인 학습 파이프라인과 생산성을 위해 최신 도구들을 사용합니다.

- **Package Manager**: UV (`uv run`, `uv sync`)
- **Deep Learning**: PyTorch 2.0+, PyTorch Lightning
- **Data Engineering**: OpenCV, Librosa, HDF5, Segmentation Models (U-Net)
- **Experiment Tracking**: TensorBoard, Weights & Biases

자세한 소스 코드 및 폴더 구조는 [📂 Repository Structure](docs/PROJECT_STRUCTURE.md)를 참고하세요.

---

## 4. 단계별 진행 로드맵

전체 개발 단계(Phase 1~5)의 진행 상황입니다. 

| Phase | 단계 명칭 | 상태 | 핵심 요약 및 산출물 |
|:---:|:---|:---:|:---|
| **1-3** | **기반 인프라 및 기준선 확보** | ✅ 완료 | MRI 세그멘테이션 파이프라인(Dice 81.8%), Bi-LSTM 기준선, 21.5M Transformer를 통한 초기 고해상도 형상 복원(PCC 0.1982). |
| **4** | **정확도 개선 (Conformer)** | ✅ 완료 | 9개 변형 비교 결과 HuBERT Small이 최고(RMSE 0.1200, PCC 0.1212). 데이터 규모가 핵심 병목으로 확인됨. ([Research Journal](papers/phase4_research_journal.md)) |
| **5-1** | **원격 GPU 서버 구축** | ✅ 준비 완료 | A100/A6000 서버용 SSH 스크립트, UV 환경 초기화 로직 구현 완료. ([GPU Quick Start](docs/guides/PHASE5_GPU_QUICKSTART.md)) |
| **5-2** | **NAS 데이터 스트리밍 연계** | ⏳ 대기 중 | 600GB+ 데이터를 NAS에서 파싱 후 GPU 서버로 전송하는 하이브리드 자동 전처리 파이프라인 스크립트 완료. NAS 실제 연동 테스트 대기. |
| **5-3** | **웹 데모 대시보드 시각화** | ⬜ 계획 중 | 추론 결과 시각화 및 학습 진행 과정 모니터링을 위한 Gradio 앱 대시보드 기획안 작성 완료. ([Web Demo Plan](docs/plans/PHASE5_3_WEB_DEMO.md)) |

---

## 5. 시작하기 (Quick Start)

### 💻 1. 환경 설정 (UV 기반)
```bash
git clone https://github.com/faransansj/Project_Sullivan.git
cd Project_Sullivan
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra gpu   # GPU 환경 설치 (개발시에는 uv sync)
```

### 🎬 2. 데이터 전처리 (MP4 → 학습 데이터)

USC-TIMIT `dataset_2drt_video_only` (2,371개 MP4, 75명)를 사용하는 파이프라인입니다.
`dl_data/dataset_2drt_video_only/` 폴더가 있으면 바로 실행할 수 있습니다.

**전체 파이프라인 한 번에 실행:**
```bash
bash scripts/run_pipeline_mp4.sh
```

**일부 피험자만 테스트:**
```bash
bash scripts/run_pipeline_mp4.sh sub001,sub002,sub003
```

**GPU 서버에서 (병렬 추출 + CUDA 세그멘테이션):**
```bash
# 인자 순서: subjects(빈칸=all), workers, device
bash scripts/run_pipeline_mp4.sh "" 8 cuda
```

파이프라인은 5단계로 구성됩니다:

| 단계 | 스크립트 | 출력 |
|:---:|:---|:---|
| 1 | `extract_frames_from_mp4.py` | `data/processed/aligned/*.h5` |
| 2 | `segment_mp4_dataset.py` | `data/processed/segmentations/*.npz` |
| 3 | `extract_articulatory_params.py` | `data/processed/parameters/*.npy` |
| 4 | `extract_audio_features.py` | `data/processed/audio_features/*.npy` |
| 5 | `create_dataset_splits.py` | `data/processed/splits/` |

> **주의**: Step 2는 U-Net 체크포인트가 필요합니다. 기본 경로: `models/unet_scratch/unet_best.pth`

**단계별 개별 실행도 가능합니다:**
```bash
# Step 1: MP4 → HDF5 (병렬 처리)
uv run python scripts/extract_frames_from_mp4.py \
    --data-root dl_data/dataset_2drt_video_only \
    --workers 4 --skip-existing

# Step 2: HDF5 → 세그멘테이션 NPZ
uv run python scripts/segment_mp4_dataset.py \
    --model models/unet_scratch/unet_best.pth \
    --device cuda --skip-existing

# Step 3: 세그멘테이션 → 조음 파라미터
uv run python scripts/extract_articulatory_params.py --method geometric

# Step 4: MP4 오디오 → Mel-spectrogram
uv run python scripts/extract_audio_features.py --features mel

# Step 5: 학습/검증/테스트 분할 생성
uv run python scripts/create_dataset_splits.py
```

### 🧠 3. Conformer 모델 학습 시작 (Phase 4)
```bash
uv run python scripts/train_conformer.py --config configs/conformer_a100_config.yaml --gpus 1
```

### 📡 4. 외부 GPU 서버로 원격 전송 및 학습 (Phase 5-1)
```bash
# 코드 동기화 후 백그라운드에서 학습 시작
./scripts/infra/remote_train.sh user@sullivan-gpu configs/conformer_a100_config.yaml train_conformer.py
```

---

## 6. 문서 및 작업 링크

과거의 작업 이력이나 구체적인 실험 계획, 세팅 방법론은 아래의 문서 아카이브에서 확인할 수 있습니다.

### 📚 가이드 및 계획 (Guides & Plans)
- 🚀 **[Phase 5 GPU 학습 환경 퀵스타트](docs/guides/PHASE5_GPU_QUICKSTART.md)**
- 🎯 **[Phase 4 모델 학습 가이드](docs/guides/PHASE4_ACCURACY_GUIDE.md)**
- 📂 **[전처리 NAS-GPU 자동화 워크플로우](docs/PROJECT_STRUCTURE.md)** 

### 📈 리포트 및 이력 (Reports)
- 🏆 **[Phase 4 진행 현황 리포트](docs/reports/PHASE4_FINAL_REPORT.md)**
- 🧠 **[MRI 세그멘테이션 데이터 방법론](docs/reports/METHODOLOGY_SEGMENTATION_PIPELINE.md)**

> ⚠️ *구형 코드 및 Phase 1~3의 테스트 스크립트는 `legacy/` 폴더에서 확인하실 수 있습니다.*

---
<div align="center">
  <b>Project Lead</b>: Midori (AI Agent) • <b>Last Update</b>: March 2026
</div>