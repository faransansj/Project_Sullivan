# Direct-contour dataset contract

**Status:** schema plus validated Annot-16 handmade-ground-truth adapter. See [`../research/annot16_feasibility_report.md`](../research/annot16_feasibility_report.md). Raw Annot-16 data remains ignored and is not redistributed.

## Canonical record

Each annotated MRI frame is represented as one JSON record consumed by
`src.research.contours.JsonContourLoader`:

```json
{
  "sample_id": "dataset:speaker:utterance:000123",
  "speaker_id": "speaker",
  "utterance_id": "utterance",
  "frame_index": 123,
  "timestamp": 2.46,
  "audio_path": "relative/audio.wav",
  "mri_path": "relative/frame.png",
  "coordinate_convention": "image_xy_x_anterior_y_inferior",
  "pixel_spacing": {
    "x_mm_per_pixel": 1.0,
    "y_mm_per_pixel": 1.0
  },
  "articulators": {
    "tongue": {
      "coordinates": [[12.5, 30.0], [13.0, 29.5]],
      "valid_mask": [true, true],
      "is_static": false
    }
  }
}
```

The top-level JSON document is `{ "samples": [...] }`. Paths remain relative to
the dataset root. Original annotation files must be retained; conversion creates
a derived canonical artifact and records its source version.

## Required invariants

- IDs are non-empty and `sample_id` is globally unique.
- Coordinates are ordered `(x, y)` image coordinates with shape `[P, 2]`.
- `valid_mask[P]` excludes missing or unreliable points from loss and metrics.
- Articulator names and original point counts come from source metadata; they are
  not inferred from Annot-16 literature.
- Point order and anterior/posterior direction are explicitly documented by the
  source adapter. Reversal is never guessed silently.
- Crop, resize, rotation, and reflection transforms are recorded and applied once.
- Pixel spacing is optional. Without verified spacing, results are reported in
  pixels or normalized coordinates, never millimetres.
- Static structures such as palate are marked `is_static` and excluded from
  temporal loss when appropriate.

## Canonical training representation

Ordered polylines are arc-length resampled per articulator to configured point
counts. A batch uses:

```text
coordinates  [B, T, A, P, 2] float32
valid_mask   [B, T, A, P]    bool
frame_mask   [B, T]          bool
timestamps   [B, T]          float64
```

If articulators require different point counts, adapters either pad to `P_max`
with `valid_mask=false` or retain separate tensors before collation. Missing
frames and missing articulators remain masked; they are not filled with a mean
shape for training.

## Coordinate normalization candidates

1. **Image-relative:** divide x/y by verified image width/height. This preserves
   anatomy but remains sensitive to crop and head placement.
2. **Anatomy-relative:** translate/rotate using stable palate or dental landmarks
   and optionally normalize scale. This can improve cross-speaker alignment but
   may remove genuine anatomy.

Both raw image coordinates and transform metadata must be retained so evaluation
can return to physical/image space. Normalization is fitted on training speakers
only and persisted with manifest/config hashes.

## Source-adapter acceptance gate

`src/research/annot16.py` satisfies this gate for the explicitly named handmade JSON format and matching 84×84 MRI JPG frames. The dense MAT R1/R2/R3 format still requires a separate label adapter and overlay check before use as training supervision.
