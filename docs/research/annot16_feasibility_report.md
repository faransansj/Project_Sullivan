# Annot-16 acquisition and direct-contour feasibility report

**Gate decision: GO**  
**Scope:** data acquisition, one-speaker/one-utterance canonical conversion, coordinate/timing checks, and overlay validation only. No model was trained.

## Repository state at start

- Branch: `research/aai-phase0-phase1`
- HEAD: `2e30f04cae23d1aa8aac339cf5794da46755447f`
- Working tree: clean

## Source and provenance

Annot-16 was downloaded from the official Zenodo record:

- Title: **75-Speaker Annot-16**
- Record: [Zenodo 18931763](https://zenodo.org/records/18931763)
- Concept record: 18931762
- Publication date: 2025-08-17
- Record revision observed: 4; no separate semantic dataset version is declared
- DOI/citation: Shi et al., *75-Speaker Annot-16: A benchmark dataset for speech articulatory rt-MRI annotation with articulator contours and phonetic alignment*, Interspeech 2025, pp. 2175–2179, DOI `10.21437/Interspeech.2025-2394`
- Authors: Xuan Shi, Yubin Zhang, Yijing Lu, Tiantian Feng, Marcus Ma, Asterios Toutios, Haley Hsu, Louis Goldstein, Shrikanth Narayanan
- Derived from the USC 75-Speaker database, DOI `10.6084/m9.figshare.13725546`
- File: `75SpeakerAnnot16.zip`
- Size: 3,509,265,130 bytes
- Zenodo checksum: `md5:79a929fce5492e2d28fe0d1577cda091`
- Locally verified checksum: identical
- Access: open

### License caution

Zenodo metadata declares **CC BY 4.0**. The archive's own `README.md` also states
“No Redistribution” and limits access to individual research use. These terms are
not equivalent. This work follows the more restrictive bundled terms: raw data is
stored under ignored `data/annot16/`, is not committed, and must not be redistributed.
Legal/license clarification should be requested from the listed USC contacts before
publishing derived data.

## Archive inventory

The archive contains 2,898 entries and expands to approximately 5.2 GiB.

```text
75SpeakerAnnot16/
  README.md
  subNNN/
    track/*.mat
    alignment/*.TextGrid
  hand_ground_truth/
    extracted_frames_jpg/*.jpg
    ground_truth_json/*.json
    mriframes_grountruth_plots/*.png
    plot_mriwithgt.py
```

Observed inventory:

- 16 speakers
- 460 dense semi-automatic contour MAT utterances
- 1,199,966 dense track frames (`trackdata` shape summed with `scipy.io.whosmat`)
- 448 phonetic TextGrid alignment files
- 160 handmade ground-truth JSON frames, ten per speaker
- 160 corresponding 84×84 MRI JPG frames
- 160 archive-provided contour overlay PNGs

Handmade ground truth has nine named structures:

- epiglottis
- tongue
- lower lip
- chin
- arytenoid
- pharyngeal wall
- hard palate
- velum
- upper lip

Dense MAT files contain one `trackdata` entry per MRI frame with `frameNo`,
`template`, and region/segment vertices. The dense region-ID-to-articulator mapping
was not made part of the minimal adapter because the archive does not provide a
machine-readable label table. This Gate validates the explicitly named handmade
JSON path. A dense adapter must preserve the R1/R2/R3 mapping documented by the
paper and verify it separately.

Machine-readable inventory: `artifacts/research/annot16_validation/inventory.json`.

## Mapping to the repository

Annot-16 speakers:

```text
sub009 sub013 sub022 sub023 sub028 sub030 sub034 sub043
sub047 sub061 sub064 sub068 sub070 sub071 sub072 sub074
```

Exact IDs present in `data/processed/aligned/`:

```text
sub009 sub022 sub028 sub061
```

The remaining 12 Annot-16 speakers are missing from this repository's aligned
metadata. Eleven repository speakers are not in Annot-16. No fuzzy or inferred
speaker mapping was used.

### Selected validation sample

- Speaker: `sub061`
- Utterance: `sub061_2drt_17_topic1`
- Dense annotation: `sub061/track/sub061_2drt_17_topic1_track.mat`
- Handmade annotations: ten `ground_truth_json/sub061_2drt_17_topic1_track_frame-*.json`
- MRI sources: matching `extracted_frames_jpg/sub061_2drt_17_topic1_video.mp4_frame-*.jpg`
- Repository metadata: `data/processed/aligned/sub061/sub061_2drt_17_topic1_video_metadata.json`
- Repository metadata original shape: `[2745,84,84]`, matching MAT `trackdata` length 2745
- Source frame numbers inspected: 530, 1005, 1074, 1269, 1519, 1638, 2430, 2516, 2649, 2659

Filename, speaker, utterance, frame count, and individual frame identifiers agree
exactly. No identifier rewrite beyond removing `_track`/`_video.mp4` suffixes is
needed.

The repository metadata's historical audio-motion alignment is marked invalid
(correlation 0.226), but this does not affect MRI-to-contour frame correspondence:
the archive supplies each exact annotated MRI frame. It does mean future audio AAI
work must resolve alignment before training.

## Coordinate system and transform

The archive's official plotting code overlays handmade JSON coordinates directly
as `coordinates[:,0]` and `coordinates[:,1]` on the 84×84 source JPG. Reproducing
that operation yielded the same placement as the supplied official plots.

- Origin: top-left
- x direction: increasing right
- y direction: increasing down
- Units: source-image pixels
- Source image: 84×84
- Crop offset: none observed or required
- Resize before coordinate overlay: none
- Flip: none
- Rotation: none
- Canonical transform: identity

The dense MAT example uses centered vertices and explicitly converts with
`x=v[:,0]+width/2`, `y=-v[:,1]+height/2`; that transform applies to dense MAT only,
not handmade JSON.

Canonical conversion is implemented by
`src/research/annot16.py:Annot16GroundTruthAdapter`. It preserves source provenance,
uses zero-based canonical frame indices, retains ordered points and validity masks,
and marks hard palate static.

## Temporal correspondence

MAT `trackdata.frameNo` is 1-based and runs 1…2745 for the selected utterance.
Handmade filename frame 530 matches `trackdata[529].frameNo == 530` and the supplied
MRI JPG frame 530. Canonical conversion therefore uses:

```text
frame_index = source_frame_number - 1
timestamp_seconds = frame_index / 83.28
```

83.28 fps comes from the USC 75-Speaker acquisition description, not array-length
interpolation. There are no explicit timestamps in handmade JSON; timestamps are
derived deterministically from verified frame number and acquisition rate.

## Physical spacing

- Status: available at acquisition level
- Value: 2.4 mm/pixel in x and y
- Source: Lim et al., *A multispeaker dataset of raw and reconstructed speech production real-time MRI video and 3D volumetric images*, Scientific Data 8, 187 (2021), describing 84×84, 2.4×2.4 mm, 83.28 fps dynamic reconstruction

The archive JSON itself does not duplicate spacing. The adapter therefore leaves
spacing unset by default; callers must explicitly supply verified `(2.4,2.4)`.
This prevents accidental mm labelling for transformed or differently reconstructed
images.

## Overlay validation

Ten representative frames spanning early, middle, and late portions of the
selected utterance were converted and rendered. All nine structures were present.
Canonical overlays were visually compared against the archive-provided plots.

Result:

- all coordinates lie within 84×84 image bounds;
- tongue, lips, palate, velum, pharyngeal wall, epiglottis, arytenoid and chin
  follow plausible visible anatomical boundaries;
- no x/y inversion, crop offset, resize, reflection or rotation was required;
- canonical and official overlays agree in placement.

Artifacts:

```text
artifacts/research/annot16_validation/
  inventory.json
  coordinate_transform.json
  sample_manifest.json
  validation_report.json
  manual_overlay_review.json
  overlay/*.png
```

The generation command is reproducible through
`scripts/validate_annot16_feasibility.py`.

## Gate 1 decision

### GO

All Gate 1 feasibility requirements were demonstrated for the official handmade
contour format:

- provenance and checksum verified;
- restrictive-use license caveat recorded;
- speaker/utterance/frame IDs interpreted without guessing;
- exact MRI frame correspondence established;
- nine articulators have explicit identities and ordered coordinates;
- canonical conversion succeeds with validity masks and provenance;
- orientation and identity transform verified against official plotting code;
- deterministic frame/timestamp correspondence established;
- ten overlays are anatomically plausible and match supplied reference plots;
- physical spacing is sourced and opt-in.

### Remaining uncertainties

GO means direct-contour representation is feasible; it does not authorize formal
AAI experiments yet. Before dense training data can be used:

1. implement and validate the MAT R1/R2/R3 label adapter;
2. acquire the corresponding USC 75-Speaker videos/audio for all selected speakers;
3. resolve the selected repository utterance's failed audio-motion alignment;
4. obtain license clarification before distributing any derived annotation data.

Per the task stop condition, no HuBERT, BiLSTM, PCA, Conformer, or Transformer
training was started.
