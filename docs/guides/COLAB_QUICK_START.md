# 🚀 Google Colab 빠른 시작 가이드

Project Sullivan Transformer 모델을 Google Colab에서 학습하기 위한 빠른 가이드입니다.

---

## ✅ 준비물

1. **Google 계정** (Gmail)
2. **Google Drive** (15GB 무료 - 78MB 데이터 저장 가능)
3. **GitHub 계정** (선택사항, 권장)

---

## 📝 단계별 진행 (총 소요시간: ~15분 설정 + 2-3시간 학습)

### 1단계: 데이터 압축 (로컬 환경)

```bash
# Project Sullivan 디렉토리에서 실행
bash scripts/prepare_data_for_colab.sh
```

**결과**: `colab_data_archives/processed_data_all.tar.gz` (78MB) 생성됨

---

### 2단계: Google Drive에 업로드

1. **Google Drive** 열기: https://drive.google.com
2. **폴더 생성** (선택사항): `Project_Sullivan_Data`
3. **파일 업로드**: `processed_data_all.tar.gz` 드래그 앤 드롭
4. **공유 링크 생성**:
   - 파일 우클릭 → 공유
   - "액세스 권한 변경" → "링크가 있는 모든 사용자"
   - 링크 복사

5. **File ID 추출**:
   ```
   https://drive.google.com/file/d/1a2B3c4D5e6F7g8H9i0J_EXAMPLE_ID/view
                                    ↑ 이 부분이 FILE_ID입니다
   ```

---

### 3단계: GitHub에 코드 푸시

```bash
# GitHub 저장소가 없다면 생성
gh repo create Project_Sullivan --public --source=. --remote=origin

# 코드 푸시
git push -u origin main
```

**또는** GitHub 웹에서 수동으로 저장소 생성 후 푸시

---

### 4단계: Colab 노트북 열기

**방법 A: Google Drive에서**
1. `notebooks/Project_Sullivan_Transformer_Training.ipynb` 파일을 Drive에 업로드
2. 우클릭 → "연결 앱" → "Google Colaboratory"

**방법 B: 직접 링크** (GitHub에 푸시한 경우)
```
https://colab.research.google.com/github/YOUR_USERNAME/Project_Sullivan/blob/main/notebooks/Project_Sullivan_Transformer_Training.ipynb
```

---

### 5단계: 노트북 설정

**Configuration 셀에서 업데이트**:

```python
# 2단계에서 복사한 File ID 입력
GDRIVE_FILE_ID_ALL = '1a2B3c4D5e6F7g8H9i0J_EXAMPLE_ID'  # ← 여기 수정!

# GitHub 사용자명 입력
GITHUB_REPO = 'YOUR_USERNAME/Project_Sullivan'  # ← 여기 수정!

# 학습 모드 선택
QUICK_TEST = False  # True=10에폭 테스트, False=50에폭 전체 학습
```

---

### 6단계: GPU 런타임 설정

**⚠️ 중요!** 이 단계를 건너뛰면 학습이 매우 느립니다!

1. **메뉴**: 런타임 → 런타임 유형 변경
2. **하드웨어 가속기**: GPU 선택
3. **GPU 유형**: T4 (무료)
4. **저장** 클릭

**확인**:
```python
import torch
print(torch.cuda.is_available())  # True가 나와야 함
```

---

### 7단계: 셀 실행

**위에서 아래로 순서대로 실행** (Shift+Enter):

1. ✅ GPU 확인
2. ✅ 저장소 클론
3. ✅ 의존성 설치 (~2분)
4. ✅ 데이터 다운로드 & 압축 해제 (~3-5분)
5. ✅ **학습 시작** (~2-3시간)

---

### 8단계: 학습 모니터링

**옵션 A: 출력 로그**
- 학습 셀의 출력에서 진행상황 확인
- Epoch, Loss, RMSE, PCC 표시됨

**옵션 B: TensorBoard**
```python
%load_ext tensorboard
%tensorboard --logdir logs/training/
```

**예상 성능** (학습 종료 시):
- RMSE: 0.20-0.30 (목표: 베이스라인 대비 3-5배 향상)
- PCC: 0.30-0.45 (목표: 베이스라인 대비 3-4배 향상)

---

### 9단계: 결과 다운로드

**방법 A: 직접 다운로드**
```python
# "Download Results" 셀 실행
# ZIP 파일이 컴퓨터로 다운로드됨
```

**방법 B: Google Drive 저장**
```python
# "Save to Google Drive" 셀 실행
# 결과가 Drive/Project_Sullivan_Results/ 폴더에 저장됨
```

---

## ⏱️ 예상 소요시간

| 단계 | 시간 | 비고 |
|------|------|------|
| 데이터 압축 (로컬) | 1분 | 한 번만 |
| Drive 업로드 | 2-3분 | 인터넷 속도 의존 |
| GitHub 푸시 | 1분 | 선택사항 |
| Colab 설정 | 5분 | 처음만 |
| 의존성 설치 | 2분 | 자동 |
| 데이터 다운로드 | 3-5분 | 자동 |
| **빠른 테스트 (10 에폭)** | **20-30분** | QUICK_TEST=True |
| **전체 학습 (50 에폭)** | **2-3시간** | QUICK_TEST=False |

---

## 💡 팁

### 처음 사용하시나요?
1. **먼저 빠른 테스트 실행**: `QUICK_TEST = True` 설정
2. 모든 것이 정상 작동하는지 확인 (~30분)
3. 전체 학습 시작: `QUICK_TEST = False` 설정

### Colab 세션 유지
- Colab은 90분 동안 활동이 없으면 연결 해제될 수 있음
- 브라우저 탭을 열어두기
- 가끔 마우스 움직이기 또는 자동 클리커 사용

### 세션이 끊겼다면?
- 걱정 마세요! 체크포인트가 에폭마다 저장됨
- 노트북 재실행하면 마지막 체크포인트부터 재개

---

## ❓ 문제 해결

### GPU를 사용할 수 없습니다
- 런타임 → 런타임 유형 변경 → GPU 확인
- 1분 정도 기다리기
- Colab 무료 사용량 초과 시 나중에 재시도

### 다운로드 실패
- File ID가 정확한지 확인
- 공유 설정이 "링크가 있는 모든 사용자"인지 확인
- 개별 파일로 시도: `USE_COMBINED_ARCHIVE = False`

### Out of Memory
- 배치 사이즈 줄이기 (config 파일에서 16 → 8)
- 그라디언트 누적 사용

---

## 📚 자세한 가이드

더 자세한 정보는 다음 문서를 참고하세요:
- **전체 가이드**: `docs/COLAB_TRAINING_GUIDE.md`
- **문제 해결**: 위 문서의 Troubleshooting 섹션

---

## ✅ 체크리스트

학습 시작 전 확인:

- [ ] 데이터 압축 완료 (`processed_data_all.tar.gz`)
- [ ] Google Drive에 업로드 완료
- [ ] File ID 복사 완료
- [ ] GitHub에 코드 푸시 완료 (선택사항)
- [ ] Colab 노트북 열림
- [ ] Configuration 셀 수정 (FILE_ID, GITHUB_REPO)
- [ ] GPU 런타임 설정 완료
- [ ] `torch.cuda.is_available()` = True 확인

**모두 체크했다면 학습 시작!** 🚀

---

**마지막 업데이트**: 2025-12-01
**예상 성공률**: 95%+ (가이드 준수 시)
