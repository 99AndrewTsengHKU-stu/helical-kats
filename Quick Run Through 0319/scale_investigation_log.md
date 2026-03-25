# Scale Investigation Log — 2026-03-21

## Problem
Katsevich reconstruction of DICOM data is ~1.41x too small compared to AAPM L067 ground truth.
MSE optimal scale: resampled=1.4131, raw=1.4264.

## Geometry Parameters
- SOD = 595.0 mm (source tags: 0x0018,0x9402 and 0x7031,0x1003)
- SDD = 1085.6 mm (source tags: 0x0018,0x1110 and 0x7031,0x1031)
- Magnification = SDD/SOD = 1.824
- Detector: 64 rows × 736 cols
- Detector extent (0x7031,0x1033): (369.625, 32.5) mm — "at isocenter"
- arc_psize_iso = 369.625/736 = 0.5022 mm
- delta_gamma = 0.5022/595 = 0.000844 rad
- flat_psize_cols (after equi-angular resample) = 0.9469 mm (at SDD)
- det_psize_rows = 0.9265 mm (at SDD)
- GT: PixelSpacing=0.664mm, ReconstructionDiameter=340mm, DataCollectionDiameter=500mm
- Volume: 512×512, delta_x=0.664mm, FOV=340mm

## Eliminated Causes

### 1. CuPy vs ASTRA backprojection
Both give identical results (ratio 0.9994, correlation 0.999999).
→ Not a kernel bug.

### 2. Katsevich pipeline itself
Phantom test with DICOM-like geometry (SOD=595, SDD=1085.6, psize=0.944):
- Position offset <1px
- Radius ratio (recon/GT) = 0.945-0.956 (close to 1.0, typical for Katsevich)
→ Pipeline is correct. Bug is in DICOM data interpretation.

### 3. Equi-angular resampling
Tested 4 configurations:
- A: standard resampled, psize=0.947
- B: raw + magnified_arc_psize, psize=0.916
- C: raw + center_flat_psize, psize=0.916
- D: raw + iso_arc_psize, psize=0.502
ALL give same body size (edge r90 ≈ 219-222 px).
MSE optimal: resampled=1.4131, raw=1.4264 (difference 0.013).
→ Resampling is NOT the cause.

### 4. Pixel size (psize)
Changing psize from 0.502 to 0.947 has NO significant effect on reconstruction scale.
Mathematical proof: Katsevich pipeline has approximate scale invariance w.r.t. psize because:
- psize appears in both the filter (differentiation, rebinning) and backprojection
- Effects cancel out: filter compensates for backprojection geometry changes
- psize is a discretization parameter; spatial scale is determined by delta_x and SOD/SDD
→ psize is NOT the cause.

## Key Observations

### Suspicious ratio
1.41 ≈ psize_avg / voxel_size = 0.937 / 0.664 = 1.411

But since psize doesn't affect scale, this ratio must be a COINCIDENCE or indicate
that the bug involves a confusion between psize and delta_x somewhere in the
data interpretation, NOT in the pipeline.

### Projection width diagnostic (new finding)
Measured body width on a DICOM projection at angle 72°:
- Actual body width on detector: 486 pixels
- Predicted width (using SOD/SDD/delta_gamma geometry): ~618 pixels
- Ratio: 486/618 = 0.787

The body appears ~21% NARROWER on the detector than the geometry predicts.
This suggests the DICOM geometry tags (SOD/SDD/detector_extents) don't correctly
describe the projection data.

Caveat: GT body size measurement is imprecise because:
- Body fills almost entire GT image width (511 out of 512 pixels = truncated)
- Different z-positions and projection angles affect body extent
- Body shape is non-circular

## ROOT CAUSE FOUND (2026-03-22)

**Tag (0x7031,0x1033) is NOT "detector extent in mm". It is `DetectorCentralElement` —
the pixel INDEX (col, row) of the central detector element.**

Per DICOM-CT-PD User Manual v12:
- `(7031,1033)` DetectorCentralElement: (Column X, Row Y) index of the detector
  element aligning with the isocenter and the detector's focal center.
- Values `(369.625, 32.5)` are pixel coordinates, NOT millimeters.

The correct detector pixel size comes from:
- `(7029,1002)` DetectorElementTransverseSpacing = **1.2858 mm** (at detector)
- `(7029,1006)` DetectorElementAxialSpacing = **1.0947 mm** (at detector)
- `(7029,100B)` DetectorShape = **CYLINDRICAL**

### Correct geometry computation
For a CYLINDRICAL detector:
- `delta_gamma = psize_col_at_det / SDD = 1.2858 / 1085.6 = 1.184e-3 rad`
- Center column = 369.625 (from tag, not (736-1)/2 = 367.5)

### Verification
| Parameter | Wrong (old) | Correct (fixed) | Ratio |
|-----------|-------------|-----------------|-------|
| delta_gamma | 8.44e-4 rad | 1.184e-3 rad | **1.403** |
| flat_psize | 0.947 mm | 1.374 mm | 1.451 |
| DCD | 363.7 mm | **502.4 mm** | 1.381 |
| center_col | 367.5 | 369.625 | +2.125 px |

- DCD from corrected geometry (502.4 mm) matches GT DataCollectionDiameter tag (500 mm) ✓
- Implied body diameter from raw projection: wrong=242.9mm (25.5% error), correct=338.5mm (3.8% error) ✓
- delta_gamma ratio 1.403 ≈ observed scale factor 1.41 ✓

### Fix applied to `pykatsevich/dicom.py`
- Read pixel spacing from (7029,1002) and (7029,1006)
- Read detector shape from (7029,100B)
- Use (7031,1033) as central element INDEX (not extent)
- Compute delta_gamma = psize_at_det / SDD for CYLINDRICAL detector

## File Index
- `test_phantom_dicom_geom.py` — Phantom test proving pipeline works
- `test_skip_resample.py` — Skip resampling test (4 configurations)
- `test_scale_raw_vs_resamp.py` — MSE optimal scale for raw vs resampled
- `diagnose_projection_width.py` — Quick projection width measurement
- `diagnose_geometry_fp.py` — Definitive FP(GT) vs DICOM comparison
- `find_optimal_scale.py` — MSE scale optimization vs GT
- `compare_scale_gt.py` — Visual scale comparison with GT
