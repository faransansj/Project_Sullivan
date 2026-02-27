# Project Sullivan - Researcher Manual
## 음성 기반 발음 기관 파라미터 추론 (Acoustic-to-Articulatory Inversion)

**Version:** 1.1
**Last Updated:** 2025-11-25
**Project Type:** Long-term Multimodal Research

---

## 📋 목차 (Table of Contents)

1. [프로젝트 개요](#1-프로젝트-개요)
2. [연구 수행 매뉴얼](#2-연구-수행-매뉴얼)
3. [상세 연구 계획](#3-상세-연구-계획)
4. [선행 연구 분석](#4-선행-연구-분석)
5. [평가 지표](#5-평가-지표)
6. [초기 설정 및 당면 과제](#6-초기-설정-및-당면-과제)
7. [부록](#7-부록)

---

## 1. 프로젝트 개요

### 1.1. 연구 목표

#### 목표 1 (Primary Goal) - 핵심 연구 목표 ⭐

**음성 신호로부터 발음 기관 파라미터 추론 (Acoustic-to-Articulatory Inversion)**

음성(오디오) 신호만을 입력받아 **발음 기관의 위치, 형태, 움직임을 나타내는 저차원 파라미터(Articulatory Parameters)**를 정확하게 추론하는 AI 모델을 개발하는 것이 본 연구의 **핵심 목표**입니다.

**Input**: 음성 파형 (Audio Waveform)
**Output**: 발음 기관 파라미터 (혀 위치, 턱 개방도, 입술 모양 등)

이는 음성학, 언어병리학, 음성 합성, 인간-컴퓨터 상호작용 등 다양한 분야에서 활용될 수 있는 기초 기술입니다.

#### 목표 2 (Secondary Goal) - 향후 확장 연구 목표

**디지털 트윈 구축 및 음향 재합성**

목표 1에서 추론된 발음 기관 파라미터를 기반으로 3D 발음 기관 모델(Digital Twin)을 생성하고, 이를 통해 물리 기반 음향 시뮬레이션으로 소리를 재합성하는 연구입니다. 이는 **목표 1이 성공적으로 달성된 후** 진행될 확장 연구입니다.

### 1.2. 핵심 데이터셋

- **USC-TIMIT / USC Speech MRI Dataset**
  - Real-time MRI (rtMRI) 영상
  - 동기화된 오디오 데이터
  - 발음 중 성도(Vocal Tract)의 동적 움직임 포착

### 1.3. 연구의 핵심 가치

- **재현성 (Reproducibility)**: 모든 실험은 추적 가능하고 재현 가능해야 함
- **협업성 (Collaboration)**: 팀원 누구나 이전 작업을 이해하고 이어갈 수 있어야 함
- **과학적 엄밀성 (Scientific Rigor)**: 정량적 지표로 모든 진척을 측정

---

## 2. 연구 수행 매뉴얼

### 2.1. 연구 로그(Research Log) 작성 원칙

모든 연구원은 작업 종료 시 또는 주요 마일스톤 달성 시 **즉시** 연구 로그를 작성해야 합니다.

#### 2.1.1. 작성 시점
- 일일 작업 종료 시
- 주요 마일스톤 달성 시
- 예상치 못한 결과 발생 시
- 다른 팀원에게 작업을 인계할 때

#### 2.1.2. 필수 포함 항목

```markdown
## Research Log Entry

**Date/Time**: YYYY-MM-DD HH:MM (KST)
**Researcher**: [이름]
**Commit Hash**: [Git 커밋 해시] (코드 변경 시)
**Experiment ID**: EXP-YYYYMMDD-NN

### Parameters
- [변경된 하이퍼파라미터 또는 설정값]
- [예: learning_rate = 0.001, batch_size = 32]

### Objective
- [이번 실험/작업의 목표]

### Method
- [사용한 방법론 간단 요약]
- [참고한 논문 또는 코드 링크]

### Results
- **Status**: [Success/Failed/Partial]
- **Quantitative**: [정량적 지표 - Loss, Accuracy, etc.]
- **Qualitative**: [정성적 관찰 사항]
- **Output Files**: [생성된 파일 경로]

### Analysis
- [결과에 대한 해석]
- [예상과의 차이점]

### Next Steps
- [ ] [다음에 수행할 작업 1]
- [ ] [다음에 수행할 작업 2]

### Notes
- [특이사항, 에러, 주의사항]
```

#### 2.1.3. 로그 저장 위치

```
/logs/
├── YYYY-MM/
│   ├── YYYYMMDD_researcher_name.md
│   └── experiments/
│       ├── EXP-YYYYMMDD-01.json
│       └── EXP-YYYYMMDD-02.json
```

---

### 2.2. 작업 할당 및 업데이트 프로세스

#### 2.2.1. 작업 정의 (To-Do)

**이슈 트래커(GitHub Issue/Jira)에 다음 정보를 명시하여 등록:**

```markdown
### Task Title
[명확한 작업 제목]

### Objective
[작업의 목표와 의의]

### Input
- [필요한 입력 데이터/파일]
- [의존성 있는 이전 작업]

### Expected Output
- [예상되는 결과물]
- [생성될 파일 형식 및 위치]

### Acceptance Criteria
- [ ] [완료 조건 1]
- [ ] [완료 조건 2]

### Deadline
YYYY-MM-DD

### References
- [관련 논문 링크]
- [참고 코드 링크]

### Assignee
[@username]
```

#### 2.2.2. 진행 (In-Progress)

작업 시작 시:
1. 이슈 상태를 **"In Progress"**로 변경
2. 작업 시작 시간을 댓글로 기록
3. 참고 중인 레퍼런스를 링크로 추가

#### 2.2.3. 완료 및 업데이트 (Done)

작업 종료 시 **결과 보고서**를 이슈 댓글로 작성:

```markdown
## Completion Report

### What was done
- [수행한 작업 내용]

### Method
- [사용한 코드/스크립트 경로]
- [핵심 알고리즘 설명]

### Output
- **Files Generated**:
  - `/path/to/output1.npy` - [설명]
  - `/path/to/output2.png` - [설명]
- **Metrics**:
  - Accuracy: XX%
  - Loss: XX

### Code Changes
- Commit: [commit hash]
- Files modified:
  - `src/module.py` - [변경 내용]

### Challenges & Solutions
- [발생한 문제점과 해결 방법]

### Next Steps for Other Researchers
- [다음 작업자가 알아야 할 사항]
```

---

### 2.3. 코드 및 데이터 관리 규칙

#### 2.3.1. 디렉토리 구조

```
Project_Sullivan/
├── data/                      # 데이터 저장소 (Git에서 제외)
│   ├── raw/                   # 원본 데이터 (수정 금지)
│   │   └── usc_speech_mri/
│   ├── processed/             # 전처리된 데이터
│   │   ├── segmented/
│   │   └── parameters/
│   └── experiments/           # 실험별 데이터
│       └── EXP-YYYYMMDD-NN/
├── src/                       # 소스 코드
│   ├── preprocessing/         # Phase 1: 전처리
│   ├── modeling/              # Phase 2: 모델링
│   ├── simulation/            # Phase 3: 시뮬레이션
│   └── utils/                 # 공통 유틸리티
├── notebooks/                 # Jupyter 노트북 (EDA, 시각화)
├── configs/                   # 설정 파일 (YAML, JSON)
├── logs/                      # 실험 로그
├── docs/                      # 문서
│   ├── researcher_manual.md   # 본 문서
│   ├── literature_review.md   # 논문 리뷰 정리
│   └── meeting_notes/         # 회의록
├── models/                    # 학습된 모델 체크포인트
├── results/                   # 결과 이미지, 그래프
└── tests/                     # 단위 테스트
```

#### 2.3.2. Git 커밋 규칙

```bash
# 커밋 메시지 형식
[TYPE] Brief description

Detailed explanation (optional)

- Related Issue: #123
- Experiment ID: EXP-20251125-01

# TYPE 종류:
# [DATA] - 데이터 처리 관련
# [MODEL] - 모델 구현/수정
# [EXP] - 실험 수행
# [FIX] - 버그 수정
# [DOCS] - 문서 작성/수정
# [REFACTOR] - 코드 리팩토링
```

**예시:**
```bash
git commit -m "[MODEL] Implement Bi-LSTM articulatory parameter predictor

- Added src/modeling/articulation_predictor.py
- Input: Mel-spectrogram (80 bins)
- Output: 10-dim articulatory parameters
- Related Issue: #15
- Experiment ID: EXP-20251125-03"
```

#### 2.3.3. 브랜치 전략

```
main (또는 master)
├── develop                    # 개발 통합 브랜치
│   ├── feature/data-preprocessing
│   ├── feature/audio-to-param-model
│   ├── feature/3d-reconstruction
│   └── experiment/exp-20251125-01
```

---

### 2.4. 코드 작성 규칙

#### 2.4.1. Python 코딩 스타일

- **PEP 8** 준수
- 함수/클래스에 **Docstring** 필수 작성
- Type Hints 사용 권장

```python
def extract_mfcc(
    audio_path: str,
    n_mfcc: int = 13,
    sr: int = 16000
) -> np.ndarray:
    """
    Extract MFCC features from audio file.

    Args:
        audio_path: Path to the audio file
        n_mfcc: Number of MFCC coefficients to extract
        sr: Sample rate for audio loading

    Returns:
        MFCC feature matrix of shape (n_mfcc, time_steps)

    Raises:
        FileNotFoundError: If audio file doesn't exist

    Example:
        >>> mfcc = extract_mfcc("data/audio.wav", n_mfcc=13)
        >>> print(mfcc.shape)
        (13, 100)
    """
    # Implementation
    pass
```

#### 2.4.2. 설정 파일 사용

하드코딩 금지! 모든 하이퍼파라미터는 설정 파일로 관리:

```yaml
# configs/phase2_training.yaml
model:
  name: "BiLSTM_Articulation_Predictor"
  architecture:
    input_dim: 80  # Mel-spectrogram bins
    hidden_dim: 256
    num_layers: 3
    output_dim: 10  # Articulatory parameters
    dropout: 0.3

training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 100
  optimizer: "Adam"

data:
  train_path: "data/processed/train"
  val_path: "data/processed/val"
  test_path: "data/processed/test"
```

---

## 3. 상세 연구 계획

### 연구 우선순위 및 마일스톤

본 연구는 **2개의 핵심 Phase(Phase 1-2)**와 **1개의 확장 Phase(Phase 3)**로 구성됩니다.

```
[핵심 연구: 목표 1 달성]

┌─────────────────┐         ┌──────────────────┐
│    Phase 1      │────────▶│     Phase 2      │
│  Data Prep &    │         │ Audio-to-Param   │
│ Parameterization│         │  Model Training  │
└─────────────────┘         └──────────────────┘
        ▲                            │
        │                            │
   rtMRI Data                        ▼
   + Audio                    Articulatory
                              Parameters ✓


[확장 연구: 목표 2 (향후)]

┌──────────────────┐
│     Phase 3      │
│  3D Digital Twin │
│  & Synthesis     │
└──────────────────┘
```

**현재 집중 작업: Phase 1 → Phase 2**
- Phase 1-2 완료가 최우선 과제
- Phase 3는 Phase 2가 성공적으로 완료된 후 착수

---

### 마일스톤 (Milestones)

| Milestone | Target | 완료 조건 | 예상 기간 |
|-----------|--------|----------|-----------|
| **M1: Data Pipeline** | Phase 1 완료 | MRI-Audio 쌍 데이터셋 생성 완료 (Train/Val/Test split) | 4-6주 |
| **M2: Baseline Model** | Phase 2 초기 | 베이스라인 모델 학습 완료 (RMSE < 0.15) | 2-3주 |
| **M3: Core Goal Achievement** | Phase 2 완료 | 목표 성능 달성 (RMSE < 0.10, PCC > 0.70) | 8-12주 |
| **M4: Digital Twin (Optional)** | Phase 3 완료 | 3D 모델 생성 및 음향 재합성 | TBD |

---

### Phase 1: 데이터 전처리 및 파라미터 추출

#### 목표
Raw MRI 영상에서 발음 기관의 '움직임'을 대표하는 저차원 파라미터(Latent Vector) 추출

#### 3.1.1. Step 1: 데이터 로딩 및 탐색

**작업 내용:**
- USC Speech MRI 데이터셋 압축 해제 및 구조 파악
- MRI 영상 메타데이터 확인 (해상도, fps, 포맷)
- 오디오 메타데이터 확인 (샘플링 레이트, 길이, 포맷)
- 오디오-영상 동기화 상태 검증

**산출물:**
```
data/raw/usc_speech_mri/
├── README.md                  # 데이터셋 설명
├── subjects/
│   ├── subject_01/
│   │   ├── mri_frames/        # MRI 영상 프레임
│   │   ├── audio.wav          # 동기화된 오디오
│   │   └── metadata.json      # 메타정보
│   └── ...

notebooks/01_EDA.ipynb         # 탐색적 데이터 분석 노트북
docs/data_statistics.md        # 데이터 통계 보고서
```

**체크리스트:**
- [ ] MRI 프레임 수, 해상도, fps 확인
- [ ] 오디오 샘플링 레이트, 길이 확인
- [ ] 오디오-MRI 동기화 offset 계산
- [ ] 데이터 품질 이슈 확인 (노이즈, 결측치)

---

#### 3.1.2. Step 2: 전처리 (Denoising & Alignment)

**작업 내용:**
- MRI 영상 노이즈 제거 (Gaussian/Median filtering)
- 오디오 노이즈 제거 (Spectral subtraction)
- 오디오-영상 정밀 정렬 (Cross-correlation 기반)
- 프레임 레이트 통일 (Interpolation)

**구현 예시:**
```python
# src/preprocessing/denoising.py

import cv2
import numpy as np
from scipy import signal

def denoise_mri_frame(frame: np.ndarray, method: str = "gaussian") -> np.ndarray:
    """MRI 프레임 노이즈 제거"""
    if method == "gaussian":
        return cv2.GaussianBlur(frame, (5, 5), 0)
    elif method == "median":
        return cv2.medianBlur(frame, 5)
    else:
        raise ValueError(f"Unknown method: {method}")

def align_audio_mri(
    audio: np.ndarray,
    mri_timestamps: np.ndarray,
    audio_sr: int
) -> tuple[np.ndarray, np.ndarray]:
    """오디오와 MRI 타임스탬프 정렬"""
    # Cross-correlation으로 최적 offset 찾기
    # ...
    return aligned_audio, aligned_timestamps
```

**산출물:**
```
data/processed/aligned/
├── subject_01/
│   ├── mri_denoised/          # 노이즈 제거된 MRI
│   ├── audio_clean.wav        # 노이즈 제거된 오디오
│   └── alignment_info.json    # 정렬 정보
```

---

#### 3.1.3. Step 3: ROI 분할 (Segmentation)

**작업 내용:**
- 성도(Vocal Tract), 혀, 턱, 입술 영역을 프레임 단위로 분할
- Deep Learning 기반 Segmentation 모델 학습 또는 사용

**모델 선택지:**
- **U-Net**: 의료 영상 분할의 표준
- **SegFormer**: Transformer 기반 고성능 분할 모델
- **Mask R-CNN**: Instance segmentation 필요 시

**Ground Truth 생성:**
- 소수의 프레임을 수동으로 라벨링 (Labelme, CVAT 사용)
- Semi-supervised learning으로 라벨 확장

**구현 예시:**
```python
# src/preprocessing/segmentation.py

import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

class VocalTractSegmenter:
    def __init__(self, num_classes: int = 5):
        """
        num_classes: 배경 + 혀 + 턱 + 입술 + 연구개 = 5
        """
        self.model = deeplabv3_resnet50(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def segment(self, mri_frame: np.ndarray) -> np.ndarray:
        """MRI 프레임을 입력받아 분할 마스크 출력"""
        # Preprocessing
        x = self.preprocess(mri_frame)

        # Inference
        with torch.no_grad():
            output = self.model(x)['out']
            mask = output.argmax(dim=1).cpu().numpy()

        return mask
```

**산출물:**
```
data/processed/segmented/
├── subject_01/
│   ├── frame_0001_mask.png
│   ├── frame_0002_mask.png
│   └── ...
models/segmentation/
└── vocal_tract_segmenter_v1.pth
```

**평가 지표:**
- Dice Coefficient: > 0.85
- IoU (Intersection over Union): > 0.80

---

#### 3.1.4. Step 4: 형상 파라미터화 (Parameter Extraction)

**작업 내용:**
분할된 고차원 이미지를 소수의 제어 파라미터로 변환

**방법론 1: PCA (Principal Component Analysis)**
```python
# src/preprocessing/parameterization.py

from sklearn.decomposition import PCA
import numpy as np

def extract_pca_parameters(
    segmentation_masks: np.ndarray,  # Shape: (num_frames, H, W)
    n_components: int = 10
) -> np.ndarray:
    """
    PCA를 사용하여 저차원 파라미터 추출

    Returns:
        parameters: Shape (num_frames, n_components)
    """
    # Flatten masks
    num_frames, H, W = segmentation_masks.shape
    flattened = segmentation_masks.reshape(num_frames, -1)

    # PCA
    pca = PCA(n_components=n_components)
    parameters = pca.fit_transform(flattened)

    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

    return parameters, pca
```

**방법론 2: Autoencoder**
```python
class ArticulatoryAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 10):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),   # (H, W) -> (H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # -> (H/4, W/4)
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # -> (H/8, W/8)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * (H//8) * (W//8), latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * (H//8) * (W//8)),
            nn.ReLU(),
            nn.Unflatten(1, (128, H//8, W//8)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)  # Latent parameters
        x_recon = self.decoder(z)
        return x_recon, z
```

**Ground Truth 데이터셋 생성:**
```
data/processed/parameters/
├── train/
│   ├── subject_01_audio_mfcc.npy      # (num_frames, 13)
│   ├── subject_01_parameters.npy      # (num_frames, 10)
│   └── ...
├── val/
└── test/
```

**각 파일 형식:**
- `audio_mfcc.npy`: Mel-Frequency Cepstral Coefficients (MFCC)
- `parameters.npy`: 발음 기관 파라미터 (PCA 또는 Autoencoder의 latent vector)

---

### Phase 2: 오디오-파라미터 매핑 모델 (핵심 목표)

#### 목표
**이것이 본 연구의 핵심입니다!**
음성 신호만 입력했을 때, Phase 1에서 정의한 발음 기관 파라미터를 정확하게 추론하는 AI 모델 개발

```
Input: Audio Waveform
   ↓
Mel-Spectrogram Extraction
   ↓
Deep Learning Model (Bi-LSTM / Transformer)
   ↓
Output: Articulatory Parameters (10-dim vector per frame)
```

---

#### 3.2.1. 모델 아키텍처 선택

**옵션 1: Bi-LSTM (Bidirectional LSTM)**
- 장점: 시계열 데이터 처리에 강함, 상대적으로 가벼움
- 단점: 장기 의존성 처리에 한계

```python
# src/modeling/articulation_predictor.py

class BiLSTMArticulationPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int = 80,    # Mel-spectrogram bins
        hidden_dim: int = 256,
        num_layers: int = 3,
        output_dim: int = 10,   # Articulatory parameters
        dropout: float = 0.3
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional

    def forward(self, x):
        # x: (batch, time, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch, time, hidden_dim*2)
        output = self.fc(lstm_out)   # (batch, time, output_dim)
        return output
```

**옵션 2: Conformer (Convolution-augmented Transformer)**
- 장점: SOTA 성능, local + global context 모두 포착
- 단점: 계산 비용 높음

---

#### 3.2.2. 손실 함수 (Loss Function)

**1. MSE Loss (기본)**
```python
mse_loss = nn.MSELoss()
loss = mse_loss(predicted_params, target_params)
```

**2. Smoothness Loss (시간적 연속성)**
```python
def smoothness_loss(predictions, alpha=0.1):
    """
    연속된 프레임 간의 급격한 변화를 패널티
    """
    diff = predictions[:, 1:, :] - predictions[:, :-1, :]
    smooth_loss = torch.mean(diff ** 2)
    return alpha * smooth_loss
```

**3. Total Loss**
```python
total_loss = mse_loss(pred, target) + smoothness_loss(pred)
```

---

#### 3.2.3. 학습 파이프라인

**데이터 로더:**
```python
# src/modeling/dataset.py

from torch.utils.data import Dataset, DataLoader
import numpy as np

class ArticulatoryDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train"):
        self.audio_files = sorted(glob(f"{data_dir}/{split}/*_audio_mfcc.npy"))
        self.param_files = sorted(glob(f"{data_dir}/{split}/*_parameters.npy"))

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio = np.load(self.audio_files[idx])    # (time, 13)
        params = np.load(self.param_files[idx])   # (time, 10)

        # Convert to torch tensors
        audio = torch.FloatTensor(audio)
        params = torch.FloatTensor(params)

        return audio, params

# Usage
train_dataset = ArticulatoryDataset("data/processed/parameters", split="train")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

**학습 스크립트:**
```python
# src/modeling/train.py

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for audio, params in dataloader:
        audio, params = audio.to(device), params.to(device)

        optimizer.zero_grad()

        # Forward pass
        predictions = model(audio)

        # Loss calculation
        loss = criterion(predictions, params) + smoothness_loss(predictions)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# Main training loop
def main():
    # Load config
    config = yaml.safe_load(open("configs/phase2_training.yaml"))

    # Initialize model
    model = BiLSTMArticulationPredictor(**config['model']['architecture'])
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # Training
    for epoch in range(config['training']['num_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        # Save checkpoint
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), f"models/phase2_best.pth")
```

---

#### 3.2.4. 평가

**정량적 지표:**
```python
# src/modeling/evaluate.py

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

def evaluate_model(model, test_loader, device):
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for audio, params in test_loader:
            audio = audio.to(device)
            predictions = model(audio).cpu().numpy()

            all_predictions.append(predictions)
            all_targets.append(params.numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Metrics
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)

    # Per-parameter Pearson correlation
    correlations = []
    for i in range(targets.shape[1]):
        corr, _ = pearsonr(targets[:, i].flatten(), predictions[:, i].flatten())
        correlations.append(corr)

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Mean Pearson Correlation: {np.mean(correlations):.4f}")

    return {
        'rmse': rmse,
        'mae': mae,
        'correlations': correlations
    }
```

**목표 성능 (Milestone M3):**
- **RMSE**: < 0.10 (normalized parameters)
- **Pearson Correlation**: > 0.70 per parameter
- **MAE**: < 0.08

**이 목표를 달성하면 본 연구의 핵심 과제가 완료됩니다!**

---

## 4. 향후 확장 계획 (Phase 3: Digital Twin)

**주의: 이 섹션은 Phase 1-2가 성공적으로 완료된 후에 진행하는 확장 연구입니다.**
**현재는 Phase 1-2에 집중해주세요!**

### Phase 3: 디지털 트윈 구축 및 음향 시뮬레이션 (목표 2)

#### 목표
추론된 파라미터로 3D Mesh를 변형하고, 이를 통해 소리를 합성

```
Articulatory Parameters (10-dim)
   ↓
3D Vocal Tract Reconstruction
   ↓
Physics-based Acoustic Simulation
   ↓
Synthesized Audio Waveform
```

---

#### 3.3.1. 3D Mesh 생성

**방법론 1: MRI 슬라이스 적층 (Stacking)**
```python
# src/simulation/mesh_generator.py

import numpy as np
from scipy.interpolate import interp1d
import trimesh

def stack_mri_slices_to_3d(
    segmentation_masks: np.ndarray,  # (num_slices, H, W)
    slice_thickness: float = 2.0     # mm
) -> trimesh.Trimesh:
    """
    2D MRI 분할 마스크를 적층하여 3D Mesh 생성
    """
    # Extract contours from each slice
    contours = []
    for i, mask in enumerate(segmentation_masks):
        contour = extract_contour(mask)  # Returns (N, 2) points
        z_coord = i * slice_thickness
        # Add z-coordinate
        contour_3d = np.column_stack([contour, np.full(len(contour), z_coord)])
        contours.append(contour_3d)

    # Connect contours to form mesh
    vertices, faces = connect_contours(contours)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    return mesh
```

**방법론 2: 파라메트릭 모델 변형**
```python
def deform_vocal_tract_template(
    template_mesh: trimesh.Trimesh,
    articulatory_params: np.ndarray  # (10,)
) -> trimesh.Trimesh:
    """
    Template mesh를 articulatory parameters로 변형
    """
    # Parameters control specific deformation modes
    # e.g., params[0] -> tongue height
    #       params[1] -> tongue frontness
    #       params[2] -> jaw opening

    deformed_vertices = template_mesh.vertices.copy()

    # Apply deformations (simplified example)
    for i, param in enumerate(articulatory_params):
        deformation_vector = deformation_basis[i]  # Pre-computed basis
        deformed_vertices += param * deformation_vector

    deformed_mesh = trimesh.Trimesh(
        vertices=deformed_vertices,
        faces=template_mesh.faces
    )

    return deformed_mesh
```

---

#### 3.3.2. 음향 시뮬레이션

**방법론 1: VocalTractLab 연동**

[VocalTractLab](http://www.vocaltractlab.de/)은 발음 기관의 물리 기반 시뮬레이터입니다.

```python
# src/simulation/acoustic_synthesizer.py

import subprocess
import numpy as np

class VocalTractLabSynthesizer:
    def __init__(self, vtl_path: str = "/usr/local/bin/VocalTractLab"):
        self.vtl_path = vtl_path

    def synthesize(
        self,
        vocal_tract_params: np.ndarray,  # (time, num_params)
        output_path: str
    ) -> np.ndarray:
        """
        VocalTractLab을 사용하여 음향 합성
        """
        # Write parameters to VTL format
        param_file = "temp_params.txt"
        self.write_vtl_params(vocal_tract_params, param_file)

        # Call VocalTractLab
        subprocess.run([
            self.vtl_path,
            "--synthesize",
            "--input", param_file,
            "--output", output_path
        ])

        # Load synthesized audio
        audio, sr = librosa.load(output_path, sr=16000)
        return audio
```

**방법론 2: FEM (Finite Element Method) 기반 시뮬레이션**

```python
# src/simulation/fem_acoustic.py

def simulate_acoustic_fem(
    mesh: trimesh.Trimesh,
    glottal_source: np.ndarray,  # (time,) - 성대 파형
    sampling_rate: int = 16000
) -> np.ndarray:
    """
    유한 요소법으로 성도 내부 음파 전파 시뮬레이션
    """
    # 1. Mesh를 FEM solver 형식으로 변환
    # 2. 경계 조건 설정 (입술 = 방사 경계)
    # 3. 파동 방정식 수치 해석
    # 4. 입술 위치에서 음압 기록 -> 출력 오디오

    # Simplified pseudo-code
    solver = AcousticFEMSolver(mesh)
    output_audio = solver.solve(glottal_source, sampling_rate)

    return output_audio
```

**방법론 3: Neural Vocoder (End-to-End 학습)**

물리 시뮬레이션 대신 뉴럴 네트워크로 직접 학습:

```python
class ArticulatoryNeuralVocoder(nn.Module):
    """
    Articulatory parameters -> Audio waveform
    """
    def __init__(self, param_dim: int = 10):
        super().__init__()

        # Upsample parameters to audio rate
        self.upsampler = nn.ConvTranspose1d(param_dim, 256, kernel_size=400, stride=200)

        # WaveNet-style generator
        self.wavenet = WaveNetDecoder(channels=256)

    def forward(self, params):
        # params: (batch, time, param_dim)
        params = params.transpose(1, 2)  # (batch, param_dim, time)

        upsampled = self.upsampler(params)  # (batch, 256, audio_time)
        audio = self.wavenet(upsampled)      # (batch, 1, audio_time)

        return audio
```

---

#### 3.3.3. End-to-End 파이프라인

**전체 시스템 통합:**
```python
# src/pipeline/end_to_end.py

class DigitalTwinPipeline:
    def __init__(self):
        # Phase 2: Audio -> Parameters
        self.param_predictor = BiLSTMArticulationPredictor()
        self.param_predictor.load_state_dict(torch.load("models/phase2_best.pth"))

        # Phase 3: Parameters -> Audio
        self.synthesizer = VocalTractLabSynthesizer()

    def synthesize_from_audio(self, input_audio_path: str, output_audio_path: str):
        """
        입력 오디오를 받아 디지털 트윈으로 재합성
        """
        # 1. Extract features
        mfcc = extract_mfcc(input_audio_path)

        # 2. Predict articulatory parameters
        with torch.no_grad():
            params = self.param_predictor(torch.FloatTensor(mfcc).unsqueeze(0))
            params = params.squeeze(0).numpy()

        # 3. Synthesize audio
        synthesized_audio = self.synthesizer.synthesize(params, output_audio_path)

        print(f"Synthesis complete: {output_audio_path}")
        return synthesized_audio

# Usage
pipeline = DigitalTwinPipeline()
pipeline.synthesize_from_audio("input.wav", "output_synthesized.wav")
```

**Phase 3 체크리스트 (착수 전 확인):**
- [ ] Phase 2 모델이 목표 성능 달성 (RMSE < 0.10, PCC > 0.70)
- [ ] Phase 2 모델이 다양한 화자에게 일반화됨을 검증
- [ ] 논문 초안 작성 완료 (Phase 1-2 결과)
- [ ] 프로젝트 리더의 Phase 3 착수 승인

---

## 5. 선행 연구 분석

### 4.1. 논문 리뷰 프로세스

모든 팀원은 할당된 논문을 읽고 다음 템플릿으로 정리:

```markdown
## Paper Review: [논문 제목]

**Reviewer**: [이름]
**Date**: YYYY-MM-DD
**Paper Link**: [URL or DOI]

### 1. Summary (3-5 문장)
[논문의 핵심 내용 요약]

### 2. Key Contributions
- [기여 1]
- [기여 2]

### 3. Methodology
[사용한 방법론 상세 설명]

### 4. Results
- **Dataset**: [사용한 데이터셋]
- **Metrics**: [평가 지표 및 결과]

### 5. Relevance to Our Project
[우리 연구에 어떻게 적용할 수 있는지]

### 6. Code/Resources
- Code: [GitHub link if available]
- Dataset: [Download link]

### 7. Limitations
[논문의 한계점]

### 8. Future Work / Our Improvements
[우리가 개선할 수 있는 부분]
```

저장 위치: `docs/literature_review/YYYYMMDD_paper_title.md`

---

### 4.2. 필수 리뷰 대상 논문

#### 5.2.1. 데이터셋 & 전처리 (Phase 1 - 우선 리뷰 ⭐)

1. **"A Real-Time MRI Database for Speech Production"** (Narayanan et al.)
   - USC-TIMIT 데이터셋의 원본 논문
   - 담당자: [이름]
   - 마감: [날짜]
   - **우선순위: 높음**

2. **"Automatic Segmentation of the Vocal Tract from Real-Time MRI"**
   - MRI 분할 방법론
   - 담당자: [이름]
   - **우선순위: 높음**

#### 5.2.2. Articulatory Inversion (Phase 2 - 필수 리뷰 ⭐⭐)

3. **"Deep Learning for Acoustic-to-Articulatory Inversion"** (Ribeiro et al., 2019)
   - LSTM 기반 음향-조음 변환
   - 담당자: [이름]
   - **우선순위: 최고 (본 연구의 핵심)**

4. **"Transformer-based Acoustic-to-Articulatory Speech Inversion"**
   - 최신 Transformer 응용
   - 담당자: [이름]
   - **우선순위: 최고**

5. **"Learning Acoustic-Articulatory Mapping with LSTM Networks"**
   - 시계열 매핑 학습 방법론
   - 담당자: [이름]
   - **우선순위: 높음**

#### 5.2.3. 음향 합성 (Phase 3 - 향후 참고용)

6. **"VocalTractLab: An Articulatory Speech Synthesizer"** (Birkholz, 2013)
   - 물리 기반 합성의 표준
   - 담당자: [이름]
   - **우선순위: 낮음 (Phase 3 착수 시 리뷰)**

7. **"Neural Vocoding with Articulatory Features"**
   - 뉴럴 보코더 접근
   - 담당자: [이름]
   - **우선순위: 낮음 (Phase 3 착수 시 리뷰)**

---

### 4.3. 키워드별 검색 전략

정기적으로 다음 키워드로 최신 논문 검색:

- **arXiv / Google Scholar 검색어 (우선순위별):**
  - **Phase 1-2 (필수):**
    - "acoustic to articulatory inversion"
    - "speech to vocal tract parameters"
    - "articulatory feature extraction from audio"
    - "vocal tract MRI segmentation"
    - "real-time MRI speech"
  - **Phase 3 (향후 참고):**
    - "articulatory speech synthesis"
    - "digital twin vocal tract"
    - "physical speech synthesis"

- **주요 학회:**
  - INTERSPEECH
  - ICASSP
  - IEEE Transactions on Audio, Speech, and Language Processing

---

## 6. 평가 지표

### 6.1. Phase 1 평가 (데이터 전처리)

| 지표 | 목표 | 측정 방법 |
|------|------|----------|
| Segmentation Dice Coefficient | > 0.85 | Ground truth와 비교 |
| Segmentation IoU | > 0.80 | Ground truth와 비교 |
| Parameter Reconstruction Error | < 5% | Autoencoder reconstruction loss |

---

### 6.2. Phase 2 평가 (Audio-to-Parameter) - 핵심 평가 지표 ⭐

**이것이 본 연구의 성공 여부를 판단하는 핵심 지표입니다!**

#### 6.2.1. 기하학적 정확도 (Geometric Accuracy)

```python
# src/evaluation/metrics.py

def compute_rmse(predictions, targets):
    """Root Mean Squared Error"""
    return np.sqrt(np.mean((predictions - targets) ** 2))

def compute_mae(predictions, targets):
    """Mean Absolute Error"""
    return np.mean(np.abs(predictions - targets))

def compute_pearson_correlation(predictions, targets):
    """Pearson Correlation Coefficient per parameter"""
    from scipy.stats import pearsonr

    correlations = []
    for i in range(predictions.shape[1]):  # For each parameter
        corr, p_value = pearsonr(
            predictions[:, i].flatten(),
            targets[:, i].flatten()
        )
        correlations.append(corr)

    return np.array(correlations)
```

**목표 성능 (Milestone M3):**
- **RMSE**: < 0.10 (normalized) - **필수 달성**
- **MAE**: < 0.08 - **필수 달성**
- **Pearson Correlation**: > 0.70 (per parameter) - **필수 달성**

**베이스라인 성능 (Milestone M2):**
- RMSE < 0.15
- Pearson Correlation > 0.50

**성능 평가 주기:**
- 매 epoch마다 validation set 평가
- 매주 test set으로 전체 평가 및 결과 기록

---

### 6.3. Phase 3 평가 (음향 합성) - 향후 참고용

**주의: Phase 1-2 완료 후에만 적용되는 지표입니다.**

#### 6.3.1. 음향학적 정확도 (Acoustic Accuracy)

```python
def compute_lsd(original_audio, synthesized_audio, sr=16000):
    """
    Log-Spectral Distance (LSD)
    낮을수록 좋음 (< 1.0 dB가 목표)
    """
    from scipy.signal import spectrogram

    # Compute spectrograms
    _, _, S_orig = spectrogram(original_audio, fs=sr)
    _, _, S_synth = spectrogram(synthesized_audio, fs=sr)

    # Log-spectral distance
    lsd = np.mean(np.sqrt(np.mean((10 * np.log10(S_orig + 1e-10) -
                                     10 * np.log10(S_synth + 1e-10)) ** 2, axis=0)))
    return lsd

def compute_pesq(original_audio, synthesized_audio, sr=16000):
    """
    PESQ (Perceptual Evaluation of Speech Quality)
    범위: -0.5 ~ 4.5 (높을수록 좋음)
    """
    from pesq import pesq
    score = pesq(sr, original_audio, synthesized_audio, 'wb')  # wideband
    return score

def compute_stoi(original_audio, synthesized_audio, sr=16000):
    """
    STOI (Short-Time Objective Intelligibility)
    범위: 0 ~ 1 (높을수록 좋음)
    """
    from pystoi import stoi
    score = stoi(original_audio, synthesized_audio, sr, extended=False)
    return score
```

**목표 성능:**
- **LSD**: < 1.5 dB
- **PESQ**: > 3.0
- **STOI**: > 0.75

---

### 6.4. 실험 결과 기록 형식

모든 실험 결과는 다음 형식으로 JSON에 저장:

```json
{
  "experiment_id": "EXP-20251125-01",
  "date": "2025-11-25",
  "phase": 2,
  "milestone": "M2",  // M1, M2, M3, M4
  "model": "BiLSTM_Articulation_Predictor",
  "config": {
    "hidden_dim": 256,
    "num_layers": 3,
    "learning_rate": 0.001,
    "batch_size": 32
  },
  "dataset": {
    "train_size": 8000,
    "val_size": 1000,
    "test_size": 1000
  },
  "metrics": {
    "train_loss": 0.0523,
    "val_loss": 0.0687,
    "test_rmse": 0.0891,
    "test_mae": 0.0712,
    "test_pearson_correlation": [0.78, 0.82, 0.75, 0.79, 0.81, 0.77, 0.80, 0.76, 0.83, 0.79]
  },
  "notes": "Added smoothness loss with alpha=0.1"
}
```

---

## 7. 초기 설정 및 당면 과제

**현재 우선순위: Phase 1 완료 → Milestone M1 달성**

### 7.1. 환경 설정 (Environment Setup)

#### 6.1.1. Python 가상환경 생성

```bash
# Create virtual environment
python3 -m venv venv_sullivan

# Activate
source venv_sullivan/bin/activate  # Linux/Mac
# venv_sullivan\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

#### 6.1.2. 필수 라이브러리 (requirements.txt)

```txt
# Deep Learning
torch>=2.0.0
torchvision>=0.15.0
lightning>=2.0.0

# Data Processing
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
pandas>=2.0.0

# Audio Processing
librosa>=0.10.0
soundfile>=0.12.0
pydub>=0.25.0

# Image/Video Processing
opencv-python>=4.8.0
scikit-image>=0.21.0
pillow>=10.0.0

# Medical Image Processing
nibabel>=5.0.0  # NIfTI format
pydicom>=2.4.0  # DICOM format

# 3D Processing
trimesh>=3.23.0
open3d>=0.17.0

# Evaluation
pesq>=0.0.4
pystoi>=0.3.3

# Utilities
tqdm>=4.65.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyyaml>=6.0
tensorboard>=2.13.0

# Notebook
jupyter>=1.0.0
ipywidgets>=8.0.0
```

---

### 7.2. 데이터셋 초기 설정

**목표: Milestone M1 달성을 위한 데이터 파이프라인 구축**

#### Task 1: 데이터 압축 해제 및 구조 파악 (Phase 1 - Step 1)

```bash
# 압축 해제
cd /home/midori/Develop/Project_Sullivan
unzip usc_speech_mri-master.zip -d data/raw/

# 구조 확인
tree data/raw/usc_speech_mri-master -L 2
```

**체크리스트:**
- [ ] 압축 해제 완료
- [ ] 디렉토리 구조 문서화
- [ ] 샘플 데이터 로딩 테스트
- [ ] 메타데이터 파일 확인

---

#### Task 2: 탐색적 데이터 분석 (EDA) (Phase 1 - Step 1)

**목표: 데이터셋 특성 파악 및 전처리 전략 수립**

**Jupyter Notebook 생성:** `notebooks/01_EDA.ipynb`

```python
# EDA 필수 확인 사항

import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# 1. 데이터 파일 수 확인
mri_files = glob("data/raw/usc_speech_mri-master/**/*.png", recursive=True)
audio_files = glob("data/raw/usc_speech_mri-master/**/*.wav", recursive=True)

print(f"Total MRI frames: {len(mri_files)}")
print(f"Total audio files: {len(audio_files)}")

# 2. 샘플 MRI 프레임 로드
sample_mri = plt.imread(mri_files[0])
print(f"MRI frame shape: {sample_mri.shape}")
print(f"MRI dtype: {sample_mri.dtype}")
print(f"MRI value range: [{sample_mri.min()}, {sample_mri.max()}]")

# 3. 샘플 오디오 로드
import librosa
audio, sr = librosa.load(audio_files[0], sr=None)
print(f"Audio length: {len(audio)} samples")
print(f"Sample rate: {sr} Hz")
print(f"Duration: {len(audio) / sr:.2f} seconds")

# 4. 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].imshow(sample_mri, cmap='gray')
axes[0].set_title("Sample MRI Frame")
axes[1].plot(audio[:sr])  # First second
axes[1].set_title("Audio Waveform (1 sec)")
plt.tight_layout()
plt.savefig("results/eda_sample.png")
```

**출력 문서:** `docs/data_statistics.md`

---

#### Task 3: Baseline 모델 구축 (Phase 2 - Milestone M2)

**목표:** 간단한 DNN으로 Audio → Articulatory Parameters 예측 베이스라인 확립
**성공 기준:** RMSE < 0.15, PCC > 0.50

```python
# src/baseline/simple_predictor.py

import torch
import torch.nn as nn

class SimpleBaselinePredictor(nn.Module):
    """
    최소 기능 베이스라인
    Input: MFCC (13,)
    Output: Flattened MRI pixels or PCA components
    """
    def __init__(self, input_dim=13, output_dim=100):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Training
model = SimpleBaselinePredictor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ... training loop ...
```

**성공 기준:**
- Loss가 수렴하는지 확인
- Validation RMSE < 0.15
- 이 베이스라인이 향후 개선의 기준점이 됨

---

### 7.3. 초기 작업 체크리스트 (Milestone M1 달성 경로)

**Phase 1 작업 순서:**
- [ ] Task 1: 데이터 압축 해제 및 구조 파악 (1주)
- [ ] Task 2: EDA 수행 및 데이터 특성 파악 (1주)
- [ ] Step 2: MRI/Audio 전처리 및 정렬 (1-2주)
- [ ] Step 3: MRI ROI 분할 모델 학습 또는 적용 (2-3주)
- [ ] Step 4: 발음 기관 파라미터 추출 (PCA/Autoencoder) (1-2주)
- [ ] 데이터셋 분할 (Train/Val/Test) 및 저장

**Phase 2 작업 순서 (M1 완료 후):**
- [ ] Task 3: 베이스라인 모델 구축 (Milestone M2) (2-3주)
- [ ] 모델 개선 및 하이퍼파라미터 튜닝 (4-6주)
- [ ] 목표 성능 달성 (Milestone M3)

---

### 7.4. 팀 역할 분담 (권장)

**Phase 1-2 집중 역할:**

| 역할 | 담당자 | 책임 범위 | 현재 우선순위 |
|------|--------|----------|--------------|
| **Project Lead** | [이름] | 전체 진행 관리, 마일스톤 추적 | ⭐⭐⭐ |
| **Data Engineer** | [이름] | Phase 1 전처리 파이프라인 (M1 달성) | ⭐⭐⭐ |
| **ML Engineer 1** | [이름] | Phase 2 모델 개발 (M2, M3 달성) | ⭐⭐⭐ |
| **ML Engineer 2** | [이름] | 모델 최적화, 하이퍼파라미터 튜닝 | ⭐⭐ |
| **Research Analyst** | [이름] | 논문 리뷰, 평가 지표 분석 | ⭐⭐ |

**Phase 3 (향후):**
| **Simulation Engineer** | [이름] | Phase 3 3D 및 음향 시뮬레이션 | (Phase 1-2 완료 후 참여) |

---

### 7.5. 주간 회의 프로토콜

**매주 [요일] [시간]에 진행**

#### 회의 전 준비사항
- 각자 작업 로그 업데이트 완료
- **현재 마일스톤 진척도** 체크
- 주요 결과 슬라이드 1-2장 준비

#### 회의 안건 (30-60분)
1. **마일스톤 진척도 확인** (5분)
   - 현재 마일스톤: M?
   - 목표 달성률: ?%

2. 지난 주 작업 리뷰 (각 10분)
   - 완료한 작업
   - 주요 결과 및 지표 (특히 RMSE, PCC)
   - 발생한 문제

3. 이슈 토론 (15분)
   - 블로킹 이슈 해결
   - 기술적 난제 브레인스토밍

4. 다음 주 계획 (10분)
   - 작업 할당
   - 마감 기한 설정
   - **다음 마일스톤 달성 전략**

#### 회의록 작성
- 담당: 순번제
- 저장 위치: `docs/meeting_notes/YYYYMMDD_meeting.md`

---

## 8. 부록

### 8.1. 용어 사전 (Glossary)

| 용어 | 설명 |
|------|------|
| **Articulatory Parameters** | 발음 기관(혀, 턱, 입술 등)의 위치와 형태를 나타내는 파라미터 |
| **Vocal Tract** | 성도, 성대에서 입술까지의 공기 통로 |
| **rtMRI** | Real-time MRI, 실시간 자기공명영상 |
| **MFCC** | Mel-Frequency Cepstral Coefficients, 오디오 특징 추출 방법 |
| **Digital Twin** | 실제 물리 시스템을 디지털로 재현한 모델 |
| **LSD** | Log-Spectral Distance, 스펙트럼 거리 지표 |
| **PESQ** | Perceptual Evaluation of Speech Quality, 음질 평가 지표 |
| **IoU** | Intersection over Union, 분할 정확도 지표 |

---

| **Acoustic-to-Articulatory Inversion** | 음성 신호로부터 발음 기관 파라미터를 역추론하는 기술 (본 연구의 핵심) |

### 8.2. 트러블슈팅 (Troubleshooting)

#### Issue 1: MRI 데이터 로딩 실패
**증상:** `FileNotFoundError` 또는 이미지 로딩 에러

**해결책:**
```python
# 파일 경로 확인
import os
assert os.path.exists(mri_path), f"File not found: {mri_path}"

# 이미지 포맷 확인 (PNG, DICOM, NIfTI 등)
from PIL import Image
img = Image.open(mri_path)
```

---

#### Issue 2: CUDA Out of Memory
**증상:** `RuntimeError: CUDA out of memory`

**해결책:**
```python
# 1. Batch size 줄이기
batch_size = 16  # 32에서 16으로

# 2. Gradient accumulation 사용
accumulation_steps = 4
for i, (audio, params) in enumerate(dataloader):
    loss = model(audio, params)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 3. Mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    loss = model(audio, params)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

#### Issue 3: 모델이 수렴하지 않음
**체크리스트:**
- [ ] Learning rate가 적절한가? (0.001 ~ 0.0001 시도)
- [ ] 데이터 정규화가 되어있는가? (Mean 0, Std 1)
- [ ] Loss가 NaN이 되지 않는가? (Gradient clipping 적용)
- [ ] 베이스라인보다 복잡한 모델인가? (단순 모델부터 시작)

```python
# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

### 8.3. 유용한 리소스

#### 7.3.1. 공개 데이터셋
- **USC-TIMIT**: https://sail.usc.edu/span/usc-timit/
- **MNGU0 Articulatory Corpus**: http://www.mngu0.org/
- **EMA Database**: http://www.cs.toronto.edu/~hinton/ema/

#### 7.3.2. 오픈소스 도구
- **VocalTractLab**: http://www.vocaltractlab.de/
- **Praat** (음성 분석): https://www.fon.hum.uva.nl/praat/
- **Audacity** (오디오 편집): https://www.audacityteam.org/

#### 7.3.3. 학습 자료
- **Speech Signal Processing (Course)**: https://www.coursera.org/learn/audio-signal-processing
- **Articulatory Phonetics**: http://www.phonetics.ucla.edu/

---

### 8.4. 라이선스 및 윤리

#### 데이터 사용 규정
- USC-TIMIT 데이터셋은 **연구 목적으로만** 사용 가능
- 상업적 사용 시 별도 라이선스 필요
- 논문 발표 시 데이터셋 출처 명시 필수

#### 인용 형식
```
Narayanan, S., Byrd, D., & Kaun, A. (1999).
"Speech production data for research and education."
Journal of the Acoustical Society of America.
```

---

### 8.5. 버전 관리

본 매뉴얼은 연구 진행에 따라 지속적으로 업데이트됩니다.

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-11-25 | 초기 버전 작성 | [이름] |
| 1.1 | 2025-11-25 | 연구 우선순위 명확화 (목표 1 중심), 마일스톤 추가 | [이름] |
| | | | |

---

## 📞 연락처 및 지원

**프로젝트 리더:**
- Name: [이름]
- Email: [이메일]
- Slack: @username

**긴급 이슈 보고:**
GitHub Issues: https://github.com/[org]/Project_Sullivan/issues

**정기 회의:**
매주 [요일] [시간] @ [장소/Zoom]

---

## 🎯 연구 목표 재확인

### 목표 1 (현재 집중) ⭐⭐⭐

> **"음성 신호만 입력하면, 발음 기관의 위치와 움직임을 나타내는 파라미터를 정확하게 추론한다."**

**Input**: 음성 파형 (Audio Waveform)
**Output**: 발음 기관 파라미터 (Articulatory Parameters)
- 혀의 위치 (높이, 전후 위치)
- 턱의 개방도
- 입술의 모양 (원순성, 개방도)
- 연구개, 인두 등의 상태

**성공 기준**: RMSE < 0.10, Pearson Correlation > 0.70

---

### 목표 2 (향후 확장)

> **"추론된 파라미터로 3D 발음 기관을 재현하고, 물리 기반 시뮬레이션으로 소리를 합성한다."**

*이 목표는 목표 1이 성공적으로 달성된 후 진행합니다.*

---

**모든 연구원은 먼저 목표 1 달성에 집중합니다!**

**현재 마일스톤**: M1 (Data Pipeline 구축)
**다음 마일스톤**: M2 (Baseline Model)
**최종 목표**: M3 (Core Goal Achievement)

---

*End of Researcher Manual v1.1*
