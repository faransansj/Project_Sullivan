# Phase 4 실험 전체 기록 (2026-03-17 ~ 2026-03-19)

## 개요

**목표**: Acoustic-to-Articulatory Inversion에서 PCC > 0.50 달성 (M2 목표)
**데이터**: USC-TIMIT Speech MRI Dataset, ~330 훈련 발화
**출력**: 14차원 기하학적 조음 파라미터 (혀 위치, 턱 개방, 입술 모양 등)

---

## 실험 요약 테이블

| 실험 | 모델 | 피처 | 파라미터 수 | precision | dropout | weight_decay | lr | test_RMSE | test_PCC |
|------|------|------|------------|-----------|---------|-------------|-----|-----------|---------|
| Phase 3 | Transformer | Mel 80 | 21.5M | fp32 | — | — | — | — | **0.198** |
| v7 | Conformer | Mel 80 | 72.9M | bf16-mixed | 0.1 | 0.01 | 5e-4 | 0.1213 | 0.1175 |
| v8 | Conformer | Mel 80 | 72.9M | 32-true | 0.1 | 0.01 | 1e-4 | 0.1244 | 0.0950 |
| HuBERT v2 | Conformer | HuBERT 1024 | 73.4M | 32-true | 0.1 | 0.01 | 1e-4 | 0.1225 | 0.1048 |
| HuBERT Small | Conformer | HuBERT 1024 | 6.3M | 32-true | 0.3 | 0.1 | 3e-4 | 0.1200 | **0.1212** |
| HuBERT Medium | Conformer | HuBERT 1024 | 21.5M | 32-true | 0.2 | 0.05 | 2e-4 | 0.1219 | 0.1097 |
| HuBERT Small v2 | Conformer | HuBERT 1024 | 6.3M | 32-true | 0.4 | 0.15 | 2e-4 | — | — |
| HuBERT Small aug | Conformer+Aug | HuBERT 1024 | 6.3M | 32-true | 0.3 | 0.1 | 3e-4 | 0.2099 | 0.1151 |

---

## 실험별 상세 기록

### 1. v7 — Conformer Mel, 구 loss weights (실패)

**설정**
- 모델: d_model=512, 12L, 8H, d_ff=2048
- 피처: Mel 80-dim
- Loss weights: mse=0.8, pcc=2.0, vel=1.2, acc=0.5
- precision: bf16-mixed
- epoch=04 체크포인트에서 resume

**결과**
- test_loss: 0.290, test_RMSE: 0.1213, test_PCC: 0.1175
- Early stop: epoch 25 (val_loss best 2.130)

**문제 진단**
- val_loss 2.1의 81%가 PCC loss 기여 → loss가 PCC loss에 지배됨
- `PCC loss = (1-0.13) × 2.0 = 1.74` vs `MSE = 0.016 × 0.8 = 0.011`
- Resume 시 OneCycleLR 스케줄러 불일치로 LR 왜곡

---

### 2. v8 — Conformer Mel, loss weights 수정

**설정 변경 (v7 대비)**
- mse_weight: 0.8 → 1.0
- pcc_weight: **2.0 → 0.3** (핵심 변경)
- velocity_weight: 1.2 → 0.2
- acceleration_weight: 0.5 → 0.1
- precision: bf16-mixed → **32-true**
- lr: 5e-4 → **1e-4**
- Resume 제거, 처음부터 재학습

**결과**
- test_loss: 0.236, test_RMSE: 0.1244, test_PCC: 0.0950
- Early stop: epoch 40 (val_loss best 0.510)
- val_loss가 2.1 → **0.51**로 75% 감소 (loss 척도 정상화 확인)

**문제**: PCC가 epoch 9(0.133) 이후 하락 → train_loss는 계속 감소 → **오버피팅**
- 73M 파라미터 + ~330 발화 = 구조적 오버피팅

---

### 3. HuBERT v2 — Conformer Large + HuBERT, v8 개선 적용

**설정**
- 피처: HuBERT-Large 1024-dim (`facebook/hubert-large-ls960-ft`)
- 모델: d_model=512, 12L (v8과 동일 크기, 73.4M params)
- v8 loss weights 동일 적용

**결과**
- test_PCC: 0.1048 (v8 mel 0.0950 대비 +0.01 개선)
- val_PCC 최고 0.161 → test_PCC 0.105 (gap = 0.056)
- Early stop: epoch 25

**관찰**: HuBERT가 mel보다 소폭 우수하나 오버피팅 문제 지속

---

### 4. HuBERT Small — 모델 축소 실험

**설정 변경 (HuBERT v2 대비)**
- d_model: 512 → **256**
- num_layers: 12 → **6**
- d_ff: 2048 → **512**
- 파라미터: 73.4M → **6.3M** (1/12 축소)
- dropout: 0.1 → **0.3**
- weight_decay: 0.01 → **0.1**
- lr: 1e-4 → **3e-4**

**결과**
- test_loss: 0.223, test_RMSE: **0.1200**, test_PCC: **0.1212**
- Early stop: epoch 31 (val_loss best 0.503)
- val_PCC 최고 **0.160**
- **Phase 4 최고 성능** (test_PCC 기준)

**핵심 발견**: 모델 크기보다 정규화 강도가 더 중요. 1/12 크기에서 최고 성능.

---

### 5. HuBERT Medium — 중간 크기 탐색

**설정**
- d_model: 384, 8L, 4H, d_ff=1024
- 파라미터: **21.5M** (Phase 3 Transformer와 동일)
- dropout: 0.2, weight_decay: 0.05

**결과**
- test_PCC: 0.1097 (Small 0.1212보다 낮음)
- 결론: 정규화 약화 시 성능 하락 확인 → Small이 최적점에 더 가까움

---

### 6. HuBERT Small v2 — 정규화 강화 튜닝

**설정 변경 (Small v1 대비)**
- dropout: 0.3 → **0.4**
- weight_decay: 0.1 → **0.15**
- lr: 3e-4 → **2e-4**
- patience: 30 → **40**
- num_epochs: 200 → **300**

**결과**
- test_PCC: 결과 미기록 (실험 로그 참조)
- 목표였던 val→test gap 축소 여부 확인 필요

---

### 7. HuBERT Small aug — SpecAugment 데이터 증강 (완료)

**설정 (Small v1 기반 + 증강)**
- augmentation: enabled=true
- time_mask_max_len=30, time_mask_num=2 (최대 30프레임=0.6초 마스킹 2개)
- freq_mask_max_len=128, freq_mask_num=2 (HuBERT 1024-dim의 12.5% 마스킹 2개)
- noise_std=0.01 (Gaussian noise)
- patience: 40, num_epochs: 300

**결과**
- test_loss: 0.658, test_RMSE: **0.2099**, test_PCC: **0.1151**
- val_loss best: 0.714 (Small v1: 0.503)
- Early stop: epoch 40 (val_loss 40 epoch 무개선)

**분석: 증강이 오히려 성능 하락**

| 지표 | Small v1 | Small aug | 변화 |
|------|---------|-----------|------|
| test_PCC | **0.1212** | 0.1151 | -0.006 ❌ |
| test_RMSE | **0.1200** | 0.2099 | +0.090 ❌ |
| val_loss best | 0.503 | 0.714 | 악화 ❌ |

**문제 원인 분석**
1. **과도한 증강 강도**: freq_mask 128-dim은 HuBERT 특징 공간의 12.5%를 마스킹 — HuBERT의 맥락적 표현이 특징 차원 간 고도로 상관되어 있어 일부 마스킹이 표현력을 심각하게 훼손
2. **소규모 데이터에서 증강 효과 반감**: ~330 발화에서는 증강이 오히려 학습 신호를 약화시켜 수렴 불안정 유발
3. **val_loss가 0.714로 상승**: 증강 없는 val에서도 train 분포 왜곡이 일반화에 부정적 영향

**결론**: SpecAugment는 Mel-spectrogram에 효과적이나 HuBERT 특징에는 부적합. HuBERT의 밀도 높은 표현에는 증강보다 데이터 확보가 우선.

---

## 핵심 발견 요약

### 1. Loss 불균형 문제
```
기존 val_loss(≈2.1) 분해:
  PCC loss = (1-0.13) × 2.0 ≈ 1.74  (81%)
  MSE loss = 0.014 × 0.8  ≈ 0.01   (0.5%)
  vel+acc                 ≈ 0.35   (16%)
```
→ pcc_weight를 2.0→0.3으로 조정하여 val_loss 0.50 수준으로 정상화

### 2. 오버피팅과 모델 크기의 관계
```
데이터: ~330 훈련 발화 (매우 소규모)

73.4M params (Large):  test_PCC 0.105  ← 오버피팅 심함
21.5M params (Medium): test_PCC 0.110
 6.3M params (Small):  test_PCC 0.121  ← 현재 최고
```
→ 파라미터 수보다 **정규화 강도**가 핵심

### 3. HuBERT vs Mel
| 피처 | 모델 크기 | test_PCC |
|------|---------|---------|
| Mel 80-dim | 72.9M | 0.095 |
| HuBERT 1024-dim | 73.4M | 0.105 |
| HuBERT 1024-dim | 6.3M | **0.121** |

→ HuBERT가 mel 대비 소폭 유리, 작은 모델과 결합 시 효과 극대화

---

## Epoch 속도 비교

| 모델 | 파라미터 | epoch/시간 |
|------|---------|-----------|
| Conformer Large (73M) | 73.4M | ~2분 16초 |
| Conformer Medium (21.5M) | 21.5M | ~56초 |
| Conformer Small (6.3M) | 6.3M | ~30초 |

---

## M2 목표 대비 현재 성능

| 지표 | Phase 2-A | Phase 3 | Phase 4 최고 | M2 목표 |
|------|----------|---------|------------|---------|
| RMSE | 1.011 | — | **0.120** | < 0.15 ✅ |
| PCC | 0.105 | 0.198 | **0.121** | > 0.50 ❌ |

RMSE는 M2 목표 달성. PCC는 아직 0.50에 한참 미치지 못함.

---

## 향후 실험 방향

1. ~~**Small v2 결과 확인**~~ — 완료 (결과 기록 필요)
2. ~~**SpecAugment 데이터 증강**~~ — 완료, HuBERT에 부적합 확인
3. **Curriculum Loss** — MSE만으로 초기 학습 후 PCC 항 점진적 추가
4. **Speaker normalization** — inter-speaker variance 감소로 PCC 향상 가능성
5. **더 많은 데이터** — USC-TIMIT 추가 피험자 데이터 확보 (Phase 5)
6. **Mel+HuBERT 병렬 입력** — 두 피처를 연결(concat)하여 상호보완 정보 활용
7. **CTC/Connectionist 기반 alignment loss** — 음소 레벨 supervision 추가

---

*실험 환경: NVIDIA A100-SXM4-80GB, PyTorch Lightning, torchaudio Conformer*
*로그 위치: `logs/conformer_*.log`*
*체크포인트: `models/conformer*/checkpoints/`*
*마지막 업데이트: 2026-03-19 (Small aug 완료)*
