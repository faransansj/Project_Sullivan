# USC-TIMIT Full Dataset Integration Report

**Date:** 2026-01-11
**Status:** ✅ Successfully Integrated

---

## Summary

600GB USC-TIMIT 데이터셋이 성공적으로 프로젝트에 통합되었습니다.

## Dataset Location

- **원본 위치:** `/mnt/HDDB/dataset/my_dataset/dataset/`
- **프로젝트 접근 경로:** `/home/Project_Sullivan/data/raw/usc_timit_full/` (symbolic link)
- **총 크기:** ~879 GB (압축 해제 후)

## Dataset Statistics

### Subjects
- **총 피험자 수:** 27명
- **피험자 ID:** sub010-sub019, sub030-sub039, sub043, sub050, sub052-sub054, sub056, sub058

### Data Files
- **비디오 파일 (MRI):** ~840개 (대부분 피험자당 32개)
- **오디오 파일:** ~840개 (각 비디오에 대응하는 오디오)
- **데이터 타입:** 2D real-time MRI (2drt), 3D volumetric MRI (3d), T2-weighted MRI (t2w)

### File Distribution by Subject
```
sub010: 32 videos
sub011: 32 videos
sub012: 32 videos
sub013: 32 videos
sub014: 32 videos
sub015: 32 videos
sub016: 32 videos
sub017: 32 videos
sub018: 32 videos
sub019: 32 videos
sub030: 32 videos
sub031: 32 videos
sub032: 32 videos
sub033: 32 videos
sub034: 32 videos
sub035: 32 videos
sub036: 0 videos (⚠ no data)
sub037: 28 videos
sub038: 32 videos
sub039: 32 videos
sub043: 32 videos
sub050: 32 videos
sub052: 32 videos
sub053: 32 videos
sub054: 32 videos
sub056: 32 videos
sub058: 32 videos
```

## Data Structure

각 피험자 디렉토리는 다음과 같은 구조를 가집니다:

```
sub010/
├── 2drt/                    # 2D real-time MRI
│   ├── audio/               # 동기화된 오디오 파일
│   │   ├── sub010_2drt_01_vcv1_r1_audio.wav
│   │   └── ...
│   ├── raw/                 # 원시 MRI 데이터
│   ├── recon/               # 재구성된 MRI 이미지
│   └── video/               # MRI 비디오
│       ├── sub010_2drt_01_vcv1_r1_video.mp4
│       └── ...
├── 3d/                      # 3D volumetric MRI
└── t2w/                     # T2-weighted MRI
```

## Integration Method

**심볼릭 링크 생성 방식 사용:**
- ✅ 장점: 디스크 공간 절약 (복사 불필요)
- ✅ 장점: 원본 데이터 보존
- ✅ 장점: 기존 코드 수정 불필요
- ⚠️ 주의: 원본 데이터 이동 시 링크 깨짐

## Compatibility with Project

프로젝트의 `USCTIMITLoader` (src/preprocessing/data_loader.py)는 이 데이터 구조를 완벽하게 지원합니다:

- ✅ 2drt/video/ 디렉토리 자동 인식
- ✅ 다중 utterance 파일 지원
- ✅ 오디오/비디오 동기화 처리
- ✅ 메타데이터 자동 로딩

## Usage Examples

### 데이터 로더 사용 예제

```python
from src.preprocessing.data_loader import USCTIMITLoader

# 데이터셋 로드
loader = USCTIMITLoader("data/raw/usc_timit_full")

# 피험자 목록 확인
subject_ids = loader.get_subject_ids()
print(f"Found {len(subject_ids)} subjects")

# 통계 확인
stats = loader.get_statistics()
print(stats)

# 특정 피험자 데이터 로드
subject_data = loader.load_subject("sub010", load_mri=True, load_audio=True)
print(f"Loaded {subject_data['num_utterances']} utterances")
```

### 데이터셋 통계 스크립트

프로젝트에서 제공하는 스크립트 사용:
```bash
python scripts/collect_dataset_stats.py --data-root data/raw/usc_timit_full
```

## Next Steps

### 1. 데이터 전처리 (Phase 1)
현재 프로젝트는 Phase 2-B (Transformer 훈련)에 있지만, 새로운 대용량 데이터셋으로 다음을 수행할 수 있습니다:

```bash
# 1. 추가 피험자 데이터 전처리
python scripts/segment_subset.py --subjects sub010,sub011,sub012

# 2. Articulatory parameter 추출
python scripts/extract_articulatory_params.py

# 3. Audio feature 추출
# (프로젝트 스크립트 참조)
```

### 2. 학습 데이터셋 확장
현재 75개 utterance로 학습 중이지만, 이제 ~840개 utterance를 사용할 수 있습니다:

- **현재:** 75 utterances (186,124 frames)
- **가능:** ~840 utterances (예상 2M+ frames)

### 3. 모델 재학습
더 많은 데이터로 모델 성능 향상 가능:

```bash
# Transformer 모델 재학습 (더 많은 데이터)
python scripts/train_transformer.py --config configs/transformer_config.yaml
```

## Configuration Updates

현재는 설정 변경이 필요하지 않습니다. 데이터 경로만 지정하면 됩니다:

```yaml
# configs/transformer_config.yaml 예시
data:
  data_root: "data/raw/usc_timit_full"  # 새 데이터셋 경로
  processed_dir: "data/processed"
  splits_dir: "data/processed/splits"
```

## Notes

- ⚠️ **sub036** 피험자는 비디오 데이터가 없음 (사용 불가)
- ⚠️ **sub037** 피험자는 28개 비디오만 있음 (다른 피험자보다 적음)
- ✅ 나머지 25명 피험자는 각 32개 비디오 보유

## References

- **Dataset:** USC-TIMIT Speech MRI Dataset
- **Paper:** Lim et al. (2021), Scientific Data
- **DOI:** https://doi.org/10.6084/m9.figshare.13725546.v1
- **프로젝트 문서:** researcher_manual.md

---

**작성자:** Claude Code
**검증 완료:** ✅ 심볼릭 링크 생성, 데이터 구조 확인, 파일 카운트 검증
