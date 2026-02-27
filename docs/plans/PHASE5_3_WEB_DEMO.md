# Phase 5-3: 웹 기반 데모 & 모니터링 대시보드

**Phase**: 5-3
**Status**: ⬜ Planning
**Last Update**: 2026-02-27
**Estimated Duration**: 2–3주
**Dependency**: Phase 5 (Inference Engine) 부분 완료 시 병행 가능

---

## 1. 목표

데이터셋 품질 검증, 학습 진행 모니터링, 추론 결과 시각화를 위한 **웹 기반 통합 대시보드**를 구축한다.

### 3가지 핵심 기능
1. **데이터셋 탐색 뷰어** — MRI 프레임, segmentation 결과, 파라미터 시각화
2. **학습 모니터링 대시보드** — Loss/PCC 실시간 그래프, epoch 진행률
3. **추론 데모 페이지** — 오디오 입력 → articulatory 파라미터 예측 시각화

---

## 2. 기술 스택 선정

| 옵션 | 장점 | 단점 | 판정 |
|------|------|------|------|
| **Gradio** | Python 네이티브, ML 친화적, 빠른 프로토타이핑 | 커스터마이징 제한 | ⭐ **추천** |
| Streamlit | 데이터 시각화 우수 | 컴포넌트 제한 | 대안 |
| Next.js | 완전한 커스터마이징 | 프론트엔드 개발 필요 | 오버엔지니어링 |

### 선정: **Gradio**
- 이미 Phase 5에서 `scripts/app.py`로 Gradio 사용 중
- Python 코드만으로 인터랙티브 UI 구현 가능
- Hugging Face Spaces 무료 배포 가능
- 기존 PyTorch 모델과 직접 연동

---

## 3. 구현 계획

### Page 1: 데이터셋 탐색 뷰어

**목적**: 학습 데이터의 품질을 시각적으로 검증

**파일**: `scripts/web/dataset_viewer.py`

```python
"""
Dataset Quality Inspector
- MRI 프레임 브라우징
- Segmentation 마스크 오버레이
- Articulatory 파라미터 시계열 그래프
- 오디오 재생 + 스펙트로그램 동기화
"""

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_sample(subject_id: str, utterance_idx: int):
    """피험자/발화 선택 시 데이터 로드"""
    # MRI frames, segmentation, audio, parameters 로드
    ...

def render_mri_with_overlay(frame_idx: int, show_segmentation: bool):
    """MRI 프레임 + 세그멘테이션 오버레이 렌더링"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # 원본 MRI / 세그멘테이션 오버레이
    ...
    return fig

def render_parameter_timeline(utterance_data, current_frame: int):
    """Articulatory 파라미터 시계열 + 현재 프레임 표시"""
    fig, ax = plt.subplots(figsize=(12, 4))
    # 14 geometric + 10 PCA 파라미터 그래프
    # 현재 프레임 위치 표시 (수직선)
    ...
    return fig

# Gradio Interface
with gr.Blocks(title="Dataset Inspector") as dataset_viewer:
    gr.Markdown("# 🔍 Dataset Quality Inspector")
    
    with gr.Row():
        subject_dropdown = gr.Dropdown(
            choices=get_subjects(), label="Subject"
        )
        utterance_slider = gr.Slider(
            minimum=0, maximum=100, step=1, label="Utterance"
        )
    
    with gr.Row():
        frame_slider = gr.Slider(
            minimum=0, maximum=500, step=1, label="Frame"
        )
        show_seg = gr.Checkbox(label="Show Segmentation", value=True)
    
    with gr.Row():
        mri_plot = gr.Plot(label="MRI + Segmentation")
        param_plot = gr.Plot(label="Articulatory Parameters")
    
    audio_player = gr.Audio(label="Audio")
    spectrogram_plot = gr.Plot(label="Spectrogram")
    
    # Statistics summary
    with gr.Accordion("📊 Sample Statistics", open=False):
        stats_table = gr.Dataframe(label="Parameter Statistics")
```

**작업 항목:**
- [ ] 피험자/발화 선택 인터페이스
- [ ] MRI 프레임 브라우저 (슬라이더 기반)
- [ ] Segmentation 마스크 오버레이 토글
- [ ] Articulatory 파라미터 시계열 그래프
- [ ] 오디오 재생 + 스펙트로그램 동기화
- [ ] 샘플 통계 요약 테이블

---

### Page 2: 학습 모니터링 대시보드

**목적**: 실시간으로 학습 진행 상황을 모니터링

**파일**: `scripts/web/training_monitor.py`

```python
"""
Training Monitor Dashboard
- Loss / PCC 실시간 그래프
- Epoch 진행률
- GPU 상태 모니터링
- Best checkpoint 요약
- 학습 로그 뷰어
"""

import gradio as gr
import json
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_training_logs(log_dir: str):
    """TensorBoard 로그에서 메트릭 추출"""
    ea = EventAccumulator(log_dir)
    ea.Reload()
    
    metrics = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        metrics[tag] = {
            'steps': [e.step for e in events],
            'values': [e.value for e in events],
        }
    return metrics

def render_loss_chart(metrics: dict):
    """Loss 곡선 (train + validation)"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(metrics['train_loss']['steps'], 
            metrics['train_loss']['values'], label='Train')
    ax.plot(metrics['val_loss']['steps'], 
            metrics['val_loss']['values'], label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.set_title('Training & Validation Loss')
    return fig

def render_pcc_chart(metrics: dict):
    """PCC (Pearson Correlation) 추이"""
    fig, ax = plt.subplots(figsize=(10, 5))
    if 'val_pearson' in metrics:
        ax.plot(metrics['val_pearson']['steps'],
                metrics['val_pearson']['values'], 
                label='Global PCC', color='green')
    ax.axhline(y=0.1982, color='red', linestyle='--', 
               label='Phase 4 Best (0.1982)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PCC')
    ax.legend()
    ax.set_title('Pearson Correlation Coefficient')
    return fig


with gr.Blocks(title="Training Monitor") as training_monitor:
    gr.Markdown("# 📈 Training Monitor")
    
    log_dir_input = gr.Textbox(
        value="logs/training", label="Log Directory"
    )
    refresh_btn = gr.Button("🔄 Refresh")
    
    with gr.Row():
        loss_chart = gr.Plot(label="Loss Curve")
        pcc_chart = gr.Plot(label="PCC Trend")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🏆 Best Checkpoint")
            best_epoch = gr.Number(label="Best Epoch")
            best_loss = gr.Number(label="Best Val Loss")
            best_pcc = gr.Number(label="Best PCC")
        
        with gr.Column():
            gr.Markdown("### ⚡ Current Status")
            current_epoch = gr.Number(label="Current Epoch")
            elapsed_time = gr.Textbox(label="Elapsed Time")
            gpu_usage = gr.Textbox(label="GPU Memory")
    
    # Training log viewer
    with gr.Accordion("📜 Training Log", open=False):
        log_viewer = gr.Textbox(
            lines=20, label="Recent Log Lines",
            interactive=False
        )
```

**작업 항목:**
- [ ] TensorBoard 로그 파싱 유틸리티
- [ ] Loss / PCC 실시간 차트
- [ ] Best checkpoint 요약 카드
- [ ] GPU 상태 표시 (nvidia-smi 연동)
- [ ] 자동 새로고침 기능 (30초 간격)

---

### Page 3: 추론 데모

**목적**: 학습된 모델로 실시간 추론 결과를 시각화

**파일**: `scripts/web/inference_demo.py` (기존 `scripts/app.py` 확장)

```python
"""
Inference Demo Page
- 오디오 파일 업로드 또는 마이크 입력
- 실시간 articulatory 파라미터 예측
- PCA → 2D 형상 복원 시각화
- Ground Truth 비교 (테스트 데이터 사용 시)
"""

import gradio as gr
import torch
import numpy as np


def predict_from_audio(audio, model_checkpoint: str):
    """오디오 → articulatory 파라미터 예측"""
    # 1. Audio feature 추출 (mel-spectrogram)
    # 2. 모델 추론
    # 3. 14 geometric + 10 PCA 파라미터 반환
    ...

def render_vocal_tract(pca_params: np.ndarray):
    """PCA 파라미터에서 2D 보컬 트랙트 형상 복원"""
    # PCA 역변환으로 마스크 복원
    # 윤곽선 추출 및 시각화
    ...

def compare_with_ground_truth(predicted, ground_truth):
    """예측 vs 실제 파라미터 비교"""
    ...


with gr.Blocks(title="Inference Demo") as inference_demo:
    gr.Markdown("# 🎤 Real-Time Inference Demo")
    
    with gr.Row():
        audio_input = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Audio Input"
        )
        model_selector = gr.Dropdown(
            choices=get_available_models(),
            label="Model Checkpoint"
        )
    
    predict_btn = gr.Button("🚀 Predict", variant="primary")
    
    with gr.Row():
        param_chart = gr.Plot(label="Predicted Parameters")
        vocal_tract_plot = gr.Plot(label="Vocal Tract Shape")
    
    # Per-parameter details
    with gr.Accordion("📊 Parameter Details", open=False):
        param_table = gr.Dataframe(
            headers=["Parameter", "Value", "Confidence"],
            label="Individual Parameters"
        )
```

**작업 항목:**
- [ ] 기존 `scripts/app.py` 리팩터링
- [ ] 오디오 업로드 + 마이크 입력 지원
- [ ] 모델 체크포인트 선택 기능
- [ ] PCA → 2D vocal tract 시각화
- [ ] Ground truth 비교 오버레이 (선택적)

---

### 통합 앱 엔트리포인트

**파일**: `scripts/web/app.py`

```python
"""
Project Sullivan Web Dashboard
3개 페이지를 탭으로 통합하는 메인 앱.
"""

import gradio as gr
from dataset_viewer import dataset_viewer
from training_monitor import training_monitor
from inference_demo import inference_demo


app = gr.TabbedInterface(
    interface_list=[dataset_viewer, training_monitor, inference_demo],
    tab_names=["🔍 Dataset Viewer", "📈 Training Monitor", "🎤 Inference Demo"],
    title="Project Sullivan Dashboard"
)

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,     # 공개 URL 생성
        auth=None,       # 필요 시 인증 추가
    )
```

**작업 항목:**
- [ ] `scripts/web/app.py` 통합 엔트리포인트
- [ ] 3개 탭 (Dataset / Training / Inference) 통합
- [ ] 공개 URL 생성 (`share=True`)

---

## 4. 디렉터리 구조 (신규)

```
scripts/web/
├── app.py                  # 통합 대시보드 엔트리포인트
├── dataset_viewer.py       # 데이터셋 품질 뷰어
├── training_monitor.py     # 학습 모니터링 대시보드
└── inference_demo.py       # 추론 데모 페이지
```

---

## 5. 배포 계획

### 옵션 A: 로컬 실행 (개발/검증)
```bash
uv run python scripts/web/app.py
# → http://localhost:7860
```

### 옵션 B: SSH 포트 포워딩 (GPU 서버)
```bash
ssh -L 7860:localhost:7860 sullivan-gpu
# → http://localhost:7860 (로컬 브라우저)
```

### 옵션 C: Hugging Face Spaces (공개 데모)
- 무료 배포, GPU 서버 불필요 (추론용)
- `README.md`에 Space 링크 추가
- 경량 모델 + 샘플 데이터 패키징

### 옵션 D: Docker + Cloudflare Tunnel (자체 호스팅)
```dockerfile
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install uv && uv sync
EXPOSE 7860
CMD ["uv", "run", "python", "scripts/web/app.py"]
```

---

## 6. UI/UX 설계 원칙

| 원칙 | 구현 |
|------|------|
| **직관적 탐색** | 드롭다운/슬라이더로 데이터 탐색 |
| **실시간 피드백** | 슬라이더 조작 즉시 시각화 갱신 |
| **비교 기능** | 예측 vs Ground Truth 오버레이 |
| **반응형** | 데스크톱 + 태블릿 지원 |
| **성능** | 캐싱으로 반복 요청 최적화 |

---

## 7. 검증 계획

| 항목 | 검증 방법 | 기준 |
|------|----------|------|
| 데이터셋 뷰어 | 10개 샘플 탐색 | 모든 시각화 정상 렌더링 |
| 학습 모니터 | 기존 로그 로딩 | TensorBoard 로그 파싱 성공 |
| 추론 데모 | 5초 오디오 예측 | < 2초 이내 결과 반환 |
| 탭 전환 | 3개 탭 순환 | 에러 없이 전환 |
| 배포 | `share=True`로 공개 URL | 외부 접근 성공 |
