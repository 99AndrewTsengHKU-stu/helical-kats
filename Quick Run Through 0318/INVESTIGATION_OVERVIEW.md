# Helical Katsevich Flicker Investigation — Complete Overview

Date: 2026-03-18
Covers: All work from 0228 → 0304 → 0313 → 0318

---

## The Problem

Reconstruction has **periodic flickering artifacts** across z-slices:

- Flicker metric: **0.229** (ours) vs **0.040** (GT) → **5.7x worse than ground truth**
- Couch (rigid object) drifts **92px** across 560 slices — should be zero
- Visible as intensity oscillation in single-pixel traces across slices
- NOT at chunk boundaries — present throughout the entire volume

## Current Pipeline

```
DICOM (48590 projs, 64×736 det)
  → negate_angles (CW→CCW)
  → angle_offset -π/2 (DICOM Y-axis → code X-axis)
  → flip_rows (detector z-axis correction)
  → per-chunk:
      z-cull → Diff → FwdRebin → Hilbert → BackRebin → TD weight → CuPy BP
  → stitch chunks → 512×512×560 volume
```

Key parameters:

- SOD=595mm, SDD=1085.6mm
- pitch_mm_per_rad = 3.656 (abs), progress_per_turn = 22.97mm
- VOXEL_SIZE_XY=0.664mm, VOXEL_SIZE_Z=0.800mm
- psize_col=0.947mm, psize_row=0.927mm (separate, not averaged)
- TD smoothing = 0.025

---

## What Has Been Tested — With Results

### Phase 1 (0228): Basic Setup

| Test | Result |
|------|--------|
| DICOM metadata extraction (SOD/SDD/angles) | ✅ Correct via private tags |
| Single-slice reconstruction | ✅ Anatomy visible, proves pipeline works |

### Phase 2 (0302): Pixel Size Fix

| Test | Result |
|------|--------|
| Detector pixel size magnification (iso→physical, ×1.824) | ✅ Significant improvement |
| Body still 9% undersized vs GT | ⚠️ Known, not yet resolved |

### Phase 3 (0304): Full Volume + Flicker Analysis

| Test | Result | Evidence |
|------|--------|----------|
| Full 560-slice reconstruction | ✅ Complete in ~50min | L067_rec_560.npy |
| CuPy vs ASTRA backprojection | ✅ Match < 1e-5 | validate_cupy_bp.png |
| **Chunk boundary blending** (single/overlap/hard) | ✅ **All identical** — not the cause | overlap_compare.png: flicker=0.0000699 for all 3 |
| **TD smoothing sweep** [0.025, 0.05, 0.1, 0.2] | ❌ **Negligible effect** | td_smoothing_compare.png: all ≈0.000029 |
| FFT of flicker signal | No dominant frequency, broadband | flicker_fft_detail.png |
| Flicker vs GT baseline | **5.7× higher** — confirmed artifact | gt_flicker_baseline.py |

### Phase 4 (0313): Angular Geometry

| Test | Result | Evidence |
|------|--------|----------|
| Angular uniformity (jitter) | ✅ std/mean < 0.1% — not the cause | angular_uniformity.png |
| Angle offset -π/2 replaces rot90 | ✅ Max diff < 0.00013 | correction_comparison.png |
| cos/sin precision with large angles | ✅ Error < 1e-6 — not the cause | trace_filter_pipeline.py |
| Full vs reduced (mod 2π) angles in BP | ✅ Identical output | filter_pipeline_trace.png panel 8 |
| Filter pipeline NaN/Inf check | ✅ Clean throughout | filter_pipeline_trace.png |
| TD energy retention per projection | ✅ mean 98.5%, no zeros | filter_pipeline_trace.png panel 5 |
| VOXEL_SIZE_Z fix (0.664→0.800) | ✅ Correct from GT DICOM | verify_fixes.png |
| Separate psize_col/psize_row | ✅ Applied (0.947/0.927) | verify_fixes.png |
| **Couch stability** | ❌ **92px drift, std=29px** | couch_stability.png |
| **TD smoothing sweep (with fixes)** [0.025–0.1] | ❌ **Flicker INCREASES with larger TD** | td_smoothing_flicker.png: 0.229→0.378 |

### Phase 5 (0318): Helix Direction Hypothesis

| Test | Result | Evidence |
|------|--------|----------|
| Helix handedness after transforms | ✅ **RIGHT-HANDED** (dz/ds = +3.815) | diagnose_helix_cpu_only.py output |
| TD window position on detector | ✅ Centered, symmetric | helix_td_diagnostic.png |
| Signed pitch → TD mask empty | Confirmed: formula requires abs(pitch) | BONUS section output |
| pitch_mm_per_rad consistency | ✅ 3.656 correct (3.815 was sampling artifact from first 100 projs) | Output analysis |

---

## What Has Been RULED OUT

1. **Chunk boundaries** — identical results with single-chunk, overlap+feather, hard boundary
2. **TD smoothing value** — 4 values tested (0.025–0.2), negligible effect on small-scale flicker; larger TD actually worsens it
3. **CuPy kernel bug** — matches ASTRA CPU backprojection within float32
4. **Angular jitter** — std/mean < 0.1%
5. **cos/sin precision** — large angle values cause < 1e-6 error
6. **Helix direction mismatch** — confirmed RIGHT-HANDED, matches code assumption
7. **Pitch numerical error** — 3.656 mm/rad is consistent across all projections
8. **flip_rows coordinate mismatch** — mathematical derivation shows row_coords aligns correctly after flip

## What Is STILL BROKEN — With Evidence

·

### 1. Couch Drift: 92px across volume

**couch_stability.png** shows:

- Couch row position oscillates wildly (300–400px range)
- Multiple sharp drops (at z≈130, 200, 270, 400)
- This is the **strongest evidence of geometry misalignment** per the advisor

### 2. Flicker: 5.7× GT baseline

**flicker_detail.png** shows:

- Slice 118 vs 125: significant intensity difference (diff range ±0.039)
- Single-pixel traces show non-smooth oscillation across z
- Body center and body edge pixels both affected

### 3. TD smoothing INCREASES flicker at larger values

**td_smoothing_flicker.png** shows:

- TD=0.025: flicker=0.229
- TD=0.050: flicker=0.266
- TD=0.075: flicker=0.345
- TD=0.100: flicker=0.378

This is **backwards** — larger smoothing should reduce boundary effects. The fact that it worsens flicker suggests the TD window boundary positions themselves are wrong: wider smoothing extends the wrong-weight region further.

---

## Unchecked Items from Advisor Checklist

From angular_correction_checklist.md:

- [ ] **#4**: Check that flip operations haven't swapped the effective rotation direction
- [ ] **#6**: Check if wrong angular assignment causes projections to accumulate at incorrect spatial positions
- [ ] **#6**: Investigate whether the T-D window computation matches the corrected angles
- [ ] **#7**: Couch stability test — track couch ROI, quantify drift (DONE but drift=92px, NOT RESOLVED)

---

## Open Hypotheses (Not Yet Tested — Need GPU)

### A. TD weighting itself causes flicker

Test: reconstruct WITH vs WITHOUT `sino_weight_td()`. If flicker drops without TD, the TD mask shape/position is wrong even though it looks centered.

Script ready: `diagnose_helix_direction.py` Test 3

### B. Backprojection v-coordinate mapping error

The CuPy kernel computes:

```c
float v = SDD/psize_row * (Z - src_z[p]) / L + v_center;
```

If `psize_row` in the kernel (from `conf['pixel_height']`) differs from what the filter assumed, every projection samples a slightly wrong detector row. This would cause systematic per-slice intensity variation.

### C. projs_per_turn scaling error

`inv_scale = SOD² / projs_per_turn` in CuPy kernel. If `projs_per_turn` is computed from a chunk's angle range (which varies per chunk), the backprojection amplitude could differ between chunks.

### D. The derivative combination direction

The Diff step computes:

```
d_proj + d_col*(u²+D²)/D + d_row*u*w/D
```

If `d_row` (vertical derivative) has the wrong sign relative to the Katsevich formula's convention, the filtered projections would have incorrect z-dependent weights. This wouldn't show up as a simple image flip — it would manifest as flicker because the error is projection-dependent.

### E. Forward/backward rebin direction

`fwd_rebin_row` maps rebin coordinates to detector rows. The mapping assumes a specific monotonicity (rebin row increases → detector row increases). If the actual data has the opposite convention after flip_rows, the interpolation reads from wrong neighborhoods — not completely wrong (the derivative analysis cancels double negation), but the rebinning is a nonlinear coordinate transform that may not cancel.

---

## Recommended Next Steps (Priority Order)

1. **Test A** — WITH vs WITHOUT TD (most informative, 1 GPU run)
2. **Test D** — Negate d_row term in Diff, check if flicker changes
3. **Test E** — Skip rebin entirely (just Diff → Hilbert → TD → BP), check quality
4. **Couch tracking** on Test A/D results to see if drift reduces
5. Only after isolating the filter stage that causes flicker → fix that stage

---

## File Locations

### Result images (all verified, with actual data)

| File | Content |
|------|---------|
| `0304/flicker_analysis.png` | 560-slice ROI mean + flicker + zoom |
| `0304/flicker_detail.png` | Slice 118 vs 125 diff + pixel traces |
| `0304/overlap_compare.png` | 3 chunk methods — identical |
| `0304/td_smoothing_compare.png` | Phase 3 TD sweep (small-scale metric) |
| `0313/couch_stability.png` | **92px drift** across volume |
| `0313/td_smoothing_flicker.png` | Phase 4 TD sweep (**flicker increases with TD**) |
| `0313/filter_pipeline_trace.png` | 8-panel filter intermediate outputs |
| `0313/verify_fixes.png` | VOXEL_SIZE_Z + psize fix validation |
| `0318/helix_td_diagnostic.png` | TD window + helix trajectory (right-handed) |

### Scripts (ready to run, need GPU)

| File | Purpose |
|------|---------|
| `0318/diagnose_helix_direction.py` | Tests 3–5: TD on/off, negate z, no flip |
| `0313/run_L067_fixed.py` | Full recon with all fixes applied |
