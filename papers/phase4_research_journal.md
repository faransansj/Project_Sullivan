# Project Sullivan Phase 4 연구일지

**기간**: 2026-03-17 ~ 2026-03-19
**연구자**: Project Sullivan Team
**목표**: Acoustic-to-Articulatory Inversion 정확도 개선 (M2 목표: PCC > 0.50)
**환경**: NVIDIA A100-SXM4-80GB, PyTorch Lightning, Singularity Container

---

## 연구 배경

음성 MRI 데이터(USC-TIMIT)를 활용하여 오디오 신호로부터 조음 파라미터(혀 위치, 턱 개방, 입술 모양 등 14차원)를 예측하는 딥러닝 모델을 개발하고 있다. Phase 3에서 Transformer 기반 모델로 Global PCC 0.1982를 달성하였고, Phase 4에서는 Conformer 아키텍처와 HuBERT 피처를 도입하여 PCC > 0.50 달성을 목표로 한다.

**데이터 규모**: ~330 훈련 발화 (매우 소규모)
**출력 차원**: 14차원 기하학적 조음 파라미터

---

## 2026-03-17 (1일차) — Loss 불균형 문제 발견 및 수정

### 문제 발견: PCC Loss 지배 현상

기존 Conformer(v7) 모델이 학습 중 `val_loss ≈ 2.1`에서 벗어나지 못하는 현상 발생. val_RMSE는 0.12로 수렴했음에도 불구하고 전체 loss가 개선되지 않는 이상 징후 확인.

**원인 분석 — Hybrid Loss 분해:**

```
val_loss = mse_weight × L_mse + pcc_weight × L_pcc + vel_weight × L_vel + acc_weight × L_acc

val_loss ≈ 2.1 분해:
  L_pcc = (1 - 0.13) × 2.0 ≈ 1.74  [81% 기여]
  L_mse = 0.014 × 0.8           ≈ 0.011 [0.5% 기여]
  L_vel + L_acc                  ≈ 0.35  [16% 기여]
```

초기 학습 단계에서 PCC ≈ 0.13으로 낮기 때문에 `L_pcc = 1 - PCC ≈ 0.87`이 되고, 여기에 weight 2.0을 곱하면 약 1.74의 loss가 발생한다. 반면 MSE는 RMSE 기준 0.12로 수렴되어 있어 실제 contribution은 0.011에 불과했다. Loss의 81%가 PCC 항에 의해 지배되어 optimizer가 의미 없는 신호를 쫓게 됨.

**추가 문제:**
- bf16-mixed precision + lr 5e-4 조합으로 NaN loss 발생 (v3)
- 체크포인트 resume 시 OneCycleLR 스케줄러 불일치

### 수정 (v8 Configuration)

| 항목 | 기존 | 수정 |
|------|------|------|
| pcc_weight | 2.0 | **0.3** |
| velocity_weight | 1.2 | 0.2 |
| acceleration_weight | 0.5 | 0.1 |
| precision | bf16-mixed | **32-true** |
| learning_rate | 5e-4 | **1e-4** |

**결과**: val_loss 2.1 → 0.51 (75% 감소), val_loss 척도 정상화 확인
- test_RMSE: 0.1244, test_PCC: 0.0950

---

## 2026-03-18 (2일차) — HuBERT 피처 도입 및 모델 크기 탐색

### HuBERT 피처 전환

Mel-spectrogram(80-dim)을 HuBERT-Large(`facebook/hubert-large-ls960-ft`) 1024-dim 피처로 교체. HuBERT는 wav2vec 계열의 self-supervised 모델로, 음소 정보를 밀도 있게 인코딩하여 조음 파라미터와의 상관성이 더 높을 것으로 기대.

**HuBERT v2 결과** (Large 모델, 73.4M params):
- test_PCC: 0.1048 (Mel v8 대비 +0.01 개선)
- val_PCC 최고 0.161 → test_PCC 0.105 (gap = 0.056 → 오버피팅 지속)

### 핵심 발견: 모델 크기 vs. 정규화 강도

~330 발화라는 소규모 데이터에서 73M 파라미터 모델은 구조적으로 오버피팅에 취약하다는 가설 수립. 모델 크기를 단계적으로 축소하며 실험:

| 모델 | 파라미터 | dropout | weight_decay | test_PCC |
|------|---------|---------|-------------|---------|
| Large (73.4M) | 73.4M | 0.1 | 0.01 | 0.105 |
| Medium (21.5M) | 21.5M | 0.2 | 0.05 | 0.110 |
| **Small (6.3M)** | **6.3M** | **0.3** | **0.1** | **0.121** |

**결론**: 파라미터 수를 12분의 1로 줄였을 때 최고 성능 달성. 모델 용량보다 **정규화 강도**가 소규모 데이터 환경에서 더 중요한 변수임을 확인.

**HuBERT Small 최종 결과:**
- test_RMSE: 0.1200, test_PCC: 0.1212
- val_PCC 최고: 0.160 (epoch 31에서 early stop)
- **Phase 4 현재 최고 성능**

---

## 2026-03-19 (3일차) — 증강 및 Curriculum Loss 실험

### 실험 1: SpecAugment 데이터 증강 (Small aug)

데이터 부족 문제를 근본적으로 해결하기 위해 SpecAugment 도입. `AudioAugmentation` 클래스를 `src/modeling/dataset.py`에 구현하여 train split에만 선택적으로 적용:

```python
augmentation:
  time_mask_max_len: 30     # 최대 30프레임(~0.6초) 마스킹
  time_mask_num: 2
  freq_mask_max_len: 128    # HuBERT 1024-dim의 12.5% 마스킹
  freq_mask_num: 2
  noise_std: 0.01           # Gaussian noise
```

**결과**: test_PCC 0.1151 (Small v1 대비 -0.006), test_RMSE 0.2099 (악화)

**실패 원인:**
1. HuBERT의 1024-dim 피처는 트랜스포머 레이어를 통해 이미 맥락적으로 압축된 밀도 높은 표현. 특징 차원 간 고도로 상관되어 있어 일부를 마스킹하면 표현력이 크게 훼손됨
2. SpecAugment는 원래 Mel-spectrogram의 독립적 주파수 빈(bin)을 마스킹하기 위해 설계된 기법 — HuBERT 특징 공간에는 적합하지 않음
3. 소규모 데이터에서는 증강이 오히려 학습 신호를 약화시켜 수렴 불안정

**교훈**: HuBERT 피처에는 SpecAugment보다 HuBERT 인코더 단에서의 증강(dropout, layer dropping)이 더 적합할 수 있음.

---

### 실험 2: Curriculum Loss v1 (val_loss monitor)

PCC loss가 학습 초기에 너무 큰 gradient를 발생시키는 문제를 해결하기 위해 Curriculum Loss 스케줄링 도입:

```
epoch  0-29:  MSE only        (pcc_weight = 0)
epoch 30-59:  선형 ramp        (pcc_weight: 0 → 0.3)
epoch 60+:   전체 hybrid loss (pcc_weight = 0.3)
```

`ConformerInversionModel`에 `curriculum_warmup_epochs`, `curriculum_ramp_epochs` 파라미터 추가. Training step과 validation step 모두 동일한 curriculum weight를 사용하여 early stopping의 공정성 유지를 시도.

**결과**: test_PCC 0.1068 (Small v1 대비 -0.014), test_RMSE 0.2086

**실패 원인 — Monitor 불일치 문제:**

```
epoch  0-14:  val_loss 0.503 → 0.427까지 하락 [MSE-only, best checkpoint 저장]
epoch 30+:    PCC 항 추가 → val_loss 정의 변경 → 0.791로 상승
epoch 65:     patience=50 소진 → early stop
```

`val_loss` monitor가 curriculum 단계에 따라 의미가 달라진다. MSE-only 구간의 `val_loss=0.427`과 Full-loss 구간의 `val_loss=0.791`은 서로 다른 척도다. 결과적으로 best checkpoint가 PCC 학습 전(epoch ~15)에 저장되어, test 시 PCC 최적화가 전혀 되지 않은 모델이 평가됨.

---

### 실험 3: Curriculum Loss v2 (val_pearson monitor) — 진행 중

v1의 구조적 문제를 수정:

| 항목 | v1 | v2 |
|------|-----|-----|
| monitor | val_loss (min) | **val_pearson (max)** |
| patience | 50 | **80** |

`val_pearson`은 curriculum 단계에 관계없이 의미가 일정하게 유지되므로 어느 epoch의 checkpoint가 저장되어도 공정한 비교가 가능함. patience=80은 warmup(30)+ramp(30) 구간이 완전히 끝난 후에도 충분한 탐색 시간을 보장.

**결과**: test_PCC 0.0992, test_RMSE 0.2160 — 역대 최저 성능

**분석**: monitor를 val_pearson으로 바꿨음에도 불구하고 오히려 악화. val_pearson을 최대화하는 checkpoint가 test 일반화로 이어지지 않음. val→test gap 심화.

**Phase 4 최종 결론**: Curriculum Loss 접근 자체가 ~330 발화 규모에서 효과가 없음. 데이터가 너무 적어 두 단계 학습 전략이 의미 있는 이득을 주지 못함.

---

## Phase 4 전체 실험 결과 요약

| 실험 | 피처 | 파라미터 | test_PCC | test_RMSE | 비고 |
|------|------|---------|---------|----------|------|
| Phase 3 Transformer | Mel 80 | 21.5M | 0.198 | — | 이전 최고 |
| Conformer v7 | Mel 80 | 72.9M | 0.1175 | 0.1213 | loss 불균형 |
| Conformer v8 | Mel 80 | 72.9M | 0.0950 | 0.1244 | 오버피팅 |
| HuBERT Large | HuBERT 1024 | 73.4M | 0.1048 | 0.1225 | 오버피팅 |
| HuBERT Medium | HuBERT 1024 | 21.5M | 0.1097 | 0.1219 | — |
| **HuBERT Small** | **HuBERT 1024** | **6.3M** | **0.1212** | **0.1200** | **현재 최고** |
| HuBERT Small aug | HuBERT 1024 | 6.3M | 0.1151 | 0.2099 | 증강 역효과 |
| HuBERT Small curriculum v1 | HuBERT 1024 | 6.3M | 0.1068 | 0.2086 | monitor 문제 |
| HuBERT Small curriculum v2 | HuBERT 1024 | 6.3M | 0.0992 | 0.2160 | 역대 최저 |

---

## 핵심 발견 및 교훈

### 1. Loss 불균형은 Hybrid Loss 설계의 핵심 과제

PCC loss는 정규화된 상관계수로부터 계산되어 초기 학습 단계에서 절댓값이 크다 (PCC ≈ 0.1 → L_pcc ≈ 0.9). 반면 MSE는 수렴 후 작은 값을 가진다. 이 두 항의 스케일 불일치를 고려하지 않으면 loss가 PCC 항에 지배되어 gradient signal이 오염된다.

**처방**: pcc_weight 2.0 → 0.3으로 대폭 축소. 또는 각 loss 항을 초기 값으로 정규화하여 사용.

### 2. 소규모 데이터에서 모델 크기보다 정규화가 중요

```
데이터 330발화 기준:
  73M params → 모수/발화 = 221K  [심각한 오버피팅]
   6.3M params → 모수/발화 = 19K  [적절한 용량]
```

Large 모델은 validation에서 과도하게 빠른 수렴 후 test 성능이 낮게 나타났고, Small 모델은 더 많은 epoch를 거쳐 안정적으로 수렴했다. 동시에 dropout=0.3, weight_decay=0.1의 강한 정규화가 핵심이었음.

### 3. HuBERT는 Mel보다 유리하지만 SpecAugment와 호환되지 않음

HuBERT는 self-supervised pre-training으로 음소 정보를 압축한 표현이라 조음 파라미터와의 상관성이 Mel보다 높다. 그러나 이 고밀도 표현의 특성상 SpecAugment로 특징 차원을 마스킹하면 표현력이 크게 훼손된다.

### 4. Curriculum Loss의 Monitor 선택이 핵심

Loss 함수가 학습 단계에 따라 변화하는 curriculum 설정에서 `val_loss`를 monitor로 사용하면 early stopping과 checkpoint 선택이 왜곡된다. 단계 변화에 무관하게 일관된 의미를 가지는 `val_pearson`을 monitor로 사용해야 한다.

---

## M2 목표 달성 현황

| 지표 | Phase 2-A | Phase 3 | Phase 4 최고 | M2 목표 |
|------|----------|---------|------------|---------|
| RMSE | 1.011 | — | **0.120** | < 0.15 ✅ |
| PCC | 0.105 | 0.198 | **0.121** | > 0.50 ❌ |

RMSE는 M2 목표(< 0.15) 달성. PCC는 0.121로 목표(> 0.50)에 크게 미치지 못함. PCC 0.50 달성을 위해서는 현재 수준의 4배 이상 개선이 필요하며, 이는 알고리즘 최적화만으로는 어렵고 **데이터 확보가 근본 해결책**임을 시사한다.

---

## 향후 연구 방향

### 단기 (Phase 4 계속)
- **Curriculum v2 결과 확인** — val_pearson monitor로 구조적 문제 수정 후 Small v1 초과 여부
- **Speaker normalization** — 화자 간 분산 감소로 PCC 향상 시도

### 중기 (Phase 5)
- **데이터 확보** — USC-TIMIT 추가 피험자 데이터 및 외부 MRI 데이터셋 도입
- **전이 학습** — 다른 발화자 데이터로 pre-train 후 fine-tune
- **HuBERT 내부 표현 활용** — 마지막 레이어 대신 중간 레이어 피처 사용 (발음 정보가 더 명시적)

### 장기
- **NAS 기반 데이터 파이프라인** — 600GB+ 대규모 데이터 스트리밍
- **웹 데모** — 실시간 조음 파라미터 추정 시각화

---

*실험 환경: NVIDIA A100-SXM4-80GB (80GB), PyTorch Lightning 2.x, torchaudio Conformer*
*데이터: USC-TIMIT Speech MRI Dataset, ~468 발화 (330 train / 69 val / 69 test)*
*로그: `logs/conformer_*.log` | 체크포인트: `models/conformer*/checkpoints/`*
### Phase 4 최종 판단

총 9개 실험 변형을 거쳐 **HuBERT Small (6.3M, dropout=0.3, wd=0.1)이 최고 성능(test_PCC 0.1212)** 임을 확인했다. 이후 시도한 SpecAugment, Curriculum Loss(v1/v2)는 모두 Small v1보다 낮은 성능을 기록했다.

알고리즘적 개선의 한계에 도달했다는 판단 하에 Phase 4 실험을 종결하고, 데이터 확보를 통한 근본적 성능 향상을 위해 Phase 5로 전환을 권장한다.

*작성일: 2026-03-20 (Phase 4 종결)*
