# Phase 2A actual audio–MRI alignment report

## Source and provenance

The mounted file follows the USC 75-Speaker corpus layout and has SHA256
`c53042a781f3bc33d338bdc3040d5b20bd08bfbaf0107f737f501e66047413a4`. Primary corpus/acquisition publications document synchronized
75-Speaker audio/rtMRI and USC shared-clock acquisition. No primary manifest was
found that authenticates this local MP4 or its AAC/H.264 transcoding chain, so
those publications are acquisition context—not proof that this derivative
preserved relative timing.

## Container and decoder timeline

Video has 2,745 frames at exact rate `65040/781`
with max affine PTS residual 1.421e-14 s. AAC begins with
2048 skip samples and ends with 414 discard-padding
samples. ffmpeg emits 727,650 samples in 711 contiguous
decoded frames from sample PTS 0; decoded frame `nb_samples` sum exactly matches
the emitted count. Tool versions and commands are recorded in the inventory.

## Offset and drift diagnostics

The corrected sign convention correlates `label(t)` with
`log_RMS(t + feature_time_offset)`, using overlap-only z-normalization. The
TextGrid/audio global peak is
-0.005 s
(r=0.768; zero
r=0.767).
Every sweep row stores its overlap count. Only 1 of four
local TextGrid/audio windows met the predeclared informativeness rule, so no
drift slope was fit. TextGrid anchors audio, not MRI. Global/ROI/contour motion
peaks remain multimodal and are not accepted as clocks. Consequently content
evidence cannot bound audio-to-MRI drift through the unverified transcode.

## Deterministic conditional mapping

Conditional on candidate offset 0, MRI frame `i` uses its exact presentation
timestamp `t_i`; decoded sample index is
`floor(t_i * 22050 + 0.5)`. HuBERT is resampled to 16 kHz and feature
`j` is centered at `(j*320 + 199.5)/16000`. Supervised extraction now returns
and stores exact supported MRI frame indices so targets use the identical slice.
For this utterance frames 0–1 are outside centered support and frames
2–2744 are retained. Audio-only inference
uses the same deterministic center/support rule.

## Gate 2: NO-GO

Decoder/PTS mapping, AAC priming/discard, and HuBERT boundaries are resolved.
Gate 2 nevertheless remains **NO-GO** because the mounted derivative's release
chain is unverified and no informative MRI-content anchors bound relative drift.
Offset 0 is a reproducible candidate, not an accepted alignment. Do not run the
tiny overfit or full AAI training until an authenticated source/manifest or an
independent audio–MRI synchronization anchor closes this gap.
