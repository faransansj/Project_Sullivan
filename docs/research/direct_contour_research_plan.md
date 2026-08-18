# Speaker-independent direct-contour AAI research plan

## Decision

**Gate 1 update:** Annot-16 acquisition and handmade-contour feasibility are **GO**. Provenance, IDs, canonical conversion, coordinate semantics, timing, spacing, and ten MRI overlays were validated in [`annot16_feasibility_report.md`](annot16_feasibility_report.md). Formal model comparison remains blocked until the dense MAT label adapter, corresponding source video/audio, and audio–MRI alignment are validated.

## Research question and hypotheses

**Question:** With the same acoustic representation, temporal encoder, speaker
split, and training budget, does direct ordered-contour regression outperform
segmentation-mask PCA for unseen-speaker vocal-tract reconstruction?

Primary hypothesis: direct contours reduce physical contour and derived
tract-variable error relative to reconstructed PCA geometry.

Secondary hypotheses:

1. low PCA coefficient error does not guarantee low reconstructed contour error;
2. direct targets expose articulator-specific failure modes;
3. frozen HuBERT improves unseen-speaker generalization over Mel/MFCC, but target
   representation may have the larger effect;
4. BiLSTM or small Conformer is more stable than a large Conformer at low data scale.

## Phase 0: evaluation integrity

Implemented interfaces:

- canonical speaker-disjoint manifest: `src/research/split_manifest.py`;
- train-only normalization artifact: `src/research/normalization.py`;
- explicit scalar/contour metrics: `src/research/metrics.py`;
- validation-only lag sweep: `src/research/alignment_diagnostic.py` and CLI;
- reproducibility record: `src/research/reproducibility.py`;
- PCA extraction requires a manifest and fits train utterances only.

Before a formal experiment, the actual dataset conversion must produce a valid
manifest containing train, validation, and test assignments. The test split is
sealed until one authorized final evaluation. Alignment and threshold choices use
validation only.

### Gate 0 GO criteria

All must hold:

- zero sample, utterance, and speaker overlap;
- train-only normalization and PCA artifacts match manifest/config hashes;
- padding and point masks pass tests;
- metric aggregation and units are fixed;
- lag sweep recovers synthetic offset and runs on real validation data;
- smoke run records seed, resolved config, commit SHA/dirty state, dataset version,
  hashes, parameter count, status, and checkpoint rule;
- the same config/seed reproduces manifest/artifact hashes.

Current state: **partial GO for utilities; NO-GO for formal comparison**.

## Phase 1: representation comparison

### Targets

1. existing 14-D geometric target (secondary interpretable baseline);
2. existing 10-D legacy mask-PCA target, fit on train only;
3. direct ordered articulator contours.

If geometric values are derived from contours they are evaluation variables, not
an independent annotation source. PCA is evaluated both in coefficient space and,
after inverse transformation/contour extraction, geometry space.

### Models and features

Do not add a new large architecture. Start with pre-extracted/frozen HuBERT and
the existing BiLSTM. Its flat output dimension becomes `A × P × 2` and is reshaped
outside the encoder. Use only masked point error plus first-order velocity error:

```text
L = L_point + 0.1 L_velocity
```

After the target comparison:

1. best target: BiLSTM vs existing small Conformer;
2. best target/model: Mel or MFCC vs HuBERT.

`configs/research/phase1_hubert_bilstm_target_comparison.yaml` is a scaffold, not
an executable claim. Required dataset/version/hash/point-count values deliberately
remain unresolved.

### Fair comparison contract

Hold fixed: speaker manifest, usable annotated utterances, context, optimizer,
maximum steps, early stopping, seed, coordinate normalization, masks, and
checkpoint selection. Use at least seeds 17/42/101 once data exists. Report mean,
standard deviation, per-speaker and per-articulator distributions. Speaker macro
is primary so prolific speakers cannot dominate.

### Metrics

Primary:

- speaker-macro contour RMSE;
- per-articulator contour RMSE;
- mm point error only with verified spacing.

Secondary: symmetric Chamfer, Hausdorff, velocity error, global/dimension/
utterance/speaker PCC, and explicitly defined tract variables. Initial tract-
variable candidates are lip aperture/protrusion, tongue tip/body/dorsum position,
velum opening, constriction location and degree. Each needs a geometry definition,
coordinate convention, and unit before implementation.

### Gate 1: contour feasibility

GO only when:

- licensed Annot-16 or equivalent files are acquired;
- source schema, IDs, contour names/order/orientation, crop transforms, timestamps,
  missing values, and spacing are verified rather than inferred;
- canonical conversion and masks validate;
- contours overlay correctly on source MRI;
- a tiny real subset can be overfit;
- validation beats a train-mean/static-contour predictor.

Current state: **GO for handmade contour feasibility; dense training adapter not yet validated**.

### Gate 2: representation utility

Continue if direct contours achieve at least one of:

- lower primary contour-space error than PCA reconstruction;
- similar average error with clearer articulator diagnosis;
- improved tract-variable accuracy;
- improved unseen-speaker robustness;
- removal of material PCA reconstruction artifacts.

Failure triggers diagnosis of annotation noise, correspondence/order, anatomy
normalization, timing, loss scale, data quantity, output dimension, and sampling;
it does not justify silently changing test data or metrics.

## Data scaling after Phase 1

Keep validation/test speakers fixed. Generate nested train-speaker subsets of
1/5/10/25/50/75 where available. Separately compare duration-matched conditions:
few speakers/more utterances versus more speakers/fewer utterances. Persist sampler
seed, selected speakers/utterances, frame counts, and total duration. Do not start
this study with the current 15 metadata-only local subjects.

## Phase 2 interface only: multi-target learning

Reuse one SSL/temporal encoder with optional training heads for phone,
place/manner/voicing/vowel features, tract variables, and later a speaker-
invariance objective. Priority:

1. contour + phone;
2. contour + categorical articulatory features;
3. contour + derived tract variables;
4. speaker-adversarial objective.

Phone targets require forced alignment and manual/automatic alignment confidence.
Place/manner/voicing can derive from training phone labels but inherit alignment
noise. Tract variables may derive from contours without extra annotation but are
not independent ground truth. Critical-articulator labels require a Sullivan-
specific mapping; Bandekar & Ghosh's EMA-channel mapping cannot be copied.
Speaker IDs may be used for adversarial training only from training speakers.
No auxiliary labels or transforms may be fit using validation/test observations.

## Expected contribution, conditional on results

Only supported results may justify claims about representation, physical
articulator-aware evaluation, low-resource auxiliary supervision, or annotation-
budget scaling. The scaffold itself is infrastructure, not a research finding.

## Failure risks

- Annot-16 archive may differ from paper-level descriptions or lack dense manual
  labels for all structures.
- Semi-automatic contours may impose tracker-specific errors.
- Point correspondence or reversed direction can dominate coordinate loss.
- Anatomy-relative normalization may erase clinically useful speaker anatomy.
- Static structures can make mean-shape predictors look strong.
- Validation lag tuning can overfit if repeatedly revised.
- Historical normalized RMSE is not comparable to contour mm RMSE.

## Next minimum experiment

Acquire and inventory Zenodo Annot-16 record 18931763 without training. Implement
one source adapter for a single speaker/utterance, verify license and metadata,
convert a handful of frames to the canonical contract, and render contour overlays
on original MRI. If and only if that passes Gate 1, overfit the existing BiLSTM on
a tiny real HuBERT/direct-contour subset and compare against a static mean-contour
validation baseline.
