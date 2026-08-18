# Project Sullivan AAI repository audit

**Audit branch:** `research/aai-phase0-phase1`  
**Starting commit:** `02d20660f3303bf97dc3e3f6d844b2ffc2b9a657`  
**Research definition:** speaker-independent acoustic-to-articulatory inversion
(AAI): reconstruct physically interpretable motion of tongue, lips, jaw, velum,
and related vocal-tract structures for unseen speakers.

## Repository ground truth

The checked-out repository does not contain a runnable research dataset. Local
inspection found metadata directories for 15 aligned subjects under
`data/processed/aligned/`, but no aligned HDF5, segmentation NPZ, parameter
NPY/NPZ, HuBERT arrays, split JSON manifests, or contour annotations. Models and
reported results may exist outside Git, but this checkout alone cannot reproduce
them. Annot-16 has not been downloaded.

### Implemented pipeline

| Stage | Code ground truth |
|---|---|
| Raw MRI/audio loading | `src/preprocessing/data_loader.py:USCTIMITLoader`, `src/preprocessing/hddb_data_loader.py` |
| Audio–MRI alignment | `src/preprocessing/alignment.py:AudioMRIAligner`; motion-energy/audio-envelope cross-correlation |
| Segmentation | `src/segmentation/unet.py:UNet`, `unet_lightning.py`, `traditional_cv.py` |
| Geometric targets | `src/parameter_extraction/geometric_features.py:GeometricFeatureExtractor` |
| PCA targets | `src/parameter_extraction/pca_features.py:PCAFeatureExtractor`; orchestration in `scripts/extract_articulatory_params.py` |
| Audio features | `src/audio_features/{mel_spectrogram,mfcc,hubert_extractor}.py` |
| Paired training data | `src/modeling/dataset.py:ArticulatoryDataset`, `create_dataloaders`, `collate_fn` |
| Models | `src/modeling/baseline_lstm.py`, `transformer.py:TransformerModel`, `conformer_model.py:ConformerInversionModel` |
| Losses | model-local hybrid MSE/PCC/velocity/acceleration and `src/modeling/losses.py` |
| Training | `scripts/train_transformer.py`, `scripts/train_conformer.py`, YAML in `configs/` |
| Evaluation | model `validation_step`/`test_step`, legacy `src/modeling/evaluate.py`, evaluation scripts in `scripts/` |
| Visualization | `scripts/visualize_predictions.py`, `compare_reconstruction.py`, `visualize_alignment.py` |
| Tests | `tests/unit`, `tests/integration` |

The default splitter `scripts/create_dataset_splits.py` writes nested text lists,
while `src/modeling/dataset.py:create_dataloaders` expects root-level
`train.json`, `val.json`, and `test.json`. The Phase-4B splitter writes JSON but
uses a separate contract. A single canonical research manifest is therefore now
specified in `src/research/split_manifest.py`.

## Current target representations

### Geometric target

`GeometricFeatureExtractor` derives normalized image quantities from categorical
segmentation labels: tongue area/centroid/tip/dorsum/width, jaw area/centroid/opening,
lip area/centroid/aperture, and row-width constriction degree/location. Output is
currently 14 float values per frame, mostly clipped to `[0,1]`. These are
interpretable image-derived proxies, not calibrated millimetre tract variables.
There is no exact inverse transformation to a contour or mask.

### Legacy mask-PCA target

`PCAFeatureExtractor.fit` flattens an `[N,H,W]` categorical label image to
`[N,H×W]` float values and applies sklearn PCA. It predicts 10 coefficients in
common configs and can inverse-transform to a continuous image, then clips to
`[0,3]` and truncates to `uint8`.

This is a valid legacy compression baseline but has methodological limitations:

- class IDs become ordered scalar intensities, creating arbitrary distances
  between labels;
- PCA mixes articulators and does not preserve per-articulator topology;
- a small boundary shift can create many pixel changes while an anatomically
  large thin-structure shift can be underrepresented;
- clipping/truncation creates reconstruction artifacts;
- PCA axes need not correspond to physical articulation modes.

Historically, `scripts/extract_articulatory_params.py` fitted PCA on every
utterance before splitting, leaking validation/test shape distributions into the
basis. Phase 0 changes require a canonical manifest and fit only train-assigned
utterances. Existing PCA code is retained for fair comparison.

### Segmentation and contour status

Segmentation labels are intermediate supervision, not a current AAI output.
There is no landmark or direct-contour target/loader in the historical pipeline.
Phase 1 adds only a canonical contour contract, synthetic loader/resampling, and
masked metric/loss interfaces; no real contour result is claimed.

## Split and leakage audit

- Subject grouping is intended in split scripts using the prefix before `_`.
- No checked-in manifest proves which speakers or utterances produced historical
  scores.
- Canonical loader normalization now shares train target statistics with val/test,
  but streaming mode estimates them from only the first 100 generated samples.
- Audio feature normalization provenance is not consistently recorded.
- Historical PCA fitting included all discovered data.
- Missing paired files are silently skipped; fixed-length chunking drops remainder
  frames and all utterances shorter than the configured sequence length.
- No historical check prevents speaker, utterance, or sample overlap.

Phase 0 utilities reject duplicate samples, utterance overlap, speaker overlap,
and missing splits. Train-only normalization records count, dtype, epsilon,
dataset version, manifest hash, and config hash.

## Alignment audit

MRI defaults vary between 50 fps and about 83.28 fps. Audio is commonly 16 kHz;
Mel hop configuration is usually 160 samples, while HuBERT is treated as 50 Hz.
`ArticulatoryDataset` silently interpolates features by endpoint index whenever
feature and target lengths differ. This ignores timestamps, acquisition offset,
dropped frames, trimming, and hop-center convention. Inference hardcodes about
83.3 MRI fps for HuBERT output length.

`AudioMRIAligner` performs non-causal, whole-sequence cross-correlation between
MRI motion energy and a smoothed audio envelope. Its validation helper hardcodes
83.28 fps for duration. These assumptions mean a good training metric cannot by
itself establish correct timing.

`src/research/alignment_diagnostic.py` and
`scripts/diagnose_alignment_lag.py` provide a validation-only `-300..+300 ms`
lag sweep that trims overlap rather than padding. It diagnoses but does not
silently correct lag. Real execution remains blocked by absent predictions,
timestamps, and manifests.

## Metric audit

Historical code uses several incompatible definitions:

- model PCC loss: per sequence and dimension, then sequence average;
- model metric: pooled valid frames per batch, dimension-wise PCC, then Lightning
  averages batches;
- `src/modeling/evaluate.py`: flattened valid frames with constant dimensions set
  to zero, but expects dictionary batches incompatible with active tuple loaders;
- reported RMSE is normally in normalized target units, not pixels or mm.

Authoritative Phase 0 definitions in `src/research/metrics.py` are:

- **RMSE/MAE:** mean over all valid observations and dimensions;
- **global PCC:** flatten all valid frame-dimension values once;
- **dimension-mean PCC:** PCC per target dimension, macro-average finite values;
- **utterance-mean PCC:** PCC per utterance, macro-average utterances;
- **speaker-mean PCC:** dimension-mean PCC per speaker, macro-average speakers;
- **contour RMSE:** square root of mean squared Euclidean point distance;
- **Chamfer/Hausdorff:** symmetric set distances over valid points;
- **articulator error:** contour RMSE reported separately per named structure.

Constant-target PCC is undefined (`NaN`) and excluded from macro averages. Pixel
spacing is required before any contour value is labelled mm.

## Severity-ranked findings

### Critical

1. No local arrays/manifests/contours make historical training or direct-contour
   execution irreproducible from this checkout.
2. Historical PCA fitting used held-out data (`scripts/extract_articulatory_params.py`).
3. Split producer and active loader contracts disagree.

### High

1. `parameter_type: combined` configs request 24 outputs, but
   `ArticulatoryDataset` routes every non-geometric NPY target to PCA; no actual
   geometric+PCA concatenation exists.
2. Length mismatch is hidden by timestamp-free interpolation.
3. Streaming NPZ paths hardcode Mel/geometric keys regardless of configured type.
4. Missing samples and remainder frames are silently dropped.
5. Historical RMSE/PCC aggregation and units are not stable across batch size and
   evaluation paths.

### Medium

1. Target normalization is train-shared but streaming statistics are a biased
   prefix and audio normalization is not covered.
2. Transformer YAML loss weights are not all wired into model construction.
3. Test evaluation is automatically invoked after training, making repeated test
   inspection easy rather than explicitly authorized.
4. The integration loader suite has known contract/fixture failures.

## Evidence from foundational work

- Azzouz, Vuissoz & Laprie predict eight structures × 50 ordered points = 800
  normalized coordinates/frame and evaluate denormalized mm contour error plus
  tract variables. Their corpus is single-speaker, so it motivates contour
  representation but does not solve Sullivan's unseen-speaker question.
  [arXiv:2603.28723](https://arxiv.org/html/2603.28723v1)
- Annot-16 supplies phonetic alignments, semi-automatic contour tracks for 16 USC
  speakers, and only 160 manually curated QA frames. Its paper reports strong
  seen/unseen-speaker degradation. Archive format must be verified after download.
  [Interspeech paper](https://www.isca-archive.org/interspeech_2025/shi25g_interspeech.pdf),
  [official page](https://sail.usc.edu/span/75speakers_annot/)
- Bandekar & Ghosh pretrain phone, place/manner/height/backness and critical-
  articulator heads, then fine-tune 12 standardized EMA channels. This supports a
  later auxiliary-task phase, not immediate contour implementation.
  [Interspeech paper](https://www.isca-archive.org/interspeech_2025/bandekar25_interspeech.pdf)
- Speech2rtMRI generates 10-frame 64×64 clips and evaluates FVD/SSIM; its authors
  note these do not measure sound–articulator correctness. It is a contrast, not
  Sullivan's primary target. [arXiv:2409.15525](https://arxiv.org/html/2409.15525v1)

## Interpretation of existing results

Phase 3/4 scores remain useful historical engineering evidence but are not a
speaker-independent direct-contour benchmark. They lack a checked-in manifest,
train-only PCA provenance, stable metric specification, and physical-unit contour
evaluation. They must not be compared numerically with contour papers' mm RMSE or
EMA papers' standardized-unit RMSE.
