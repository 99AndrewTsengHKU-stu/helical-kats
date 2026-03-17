# Katsevich Angular Correction Checklist

Source: 0313 meeting transcript analysis (advisor discussion on periodic artifacts)

---

## Background / Observed Symptoms

- Reconstruction has a **defocused, rotating artifact** that flickers periodically across slices
- Previous attempts (flip, negate, manual direction alignment) reduced but did not eliminate the issue
- The **couch** (a rigid object that should remain stable) visibly shifts across slices
  - This is the strongest evidence of geometry misalignment — patient anatomy varies by position, but couch must not move
- Artifact is **periodic**, suggesting a systematic angular mismatch rather than random noise

## Advisor's Diagnosis

> "I don't think it's the Katsevich algorithm itself. I think it's more like the angular part or the weighting part in the reconstruction."

> "Your projection data's geometric parameters and what you set in the code — how they align. If they truly match, I don't think the algorithm itself has a problem."

Key conclusion: **prioritize data convention vs. code convention mismatch**, not the algorithm.

---

## Code Audit Findings (2026-03-13)

### Convention Summary

| Layer | Convention | Source |
|-------|-----------|--------|
| **ASTRA / geometry.py** | CCW from X-axis (standard math) | geometry.py:39-41 docstring |
| **Katsevich weighting** | `L = R - x·cos(θ) - y·sin(θ)` | filter.py:471-477, backproject_cupy.py:44-52 |
| **DICOM tag (0x7031,0x1001)** | Source angle in radians, direction **undocumented** | dicom.py:38 |
| **L067 working config** | `negate_angles=True`, `flip_rows=True`, post-recon `rot90(k=1)` | run_L067.py:270-276, 364 |
| **delta_s** | Signed; negative = CW scan | run_dicom_recon.py:163-177 |

### Key Implications

1. **L067 DICOM angles are CW** — negation required to match ASTRA's CCW convention
2. **Detector V-axis is flipped** — `flip_rows=True` corrects detector z-axis orientation
3. **Post-recon rot90** — XY plane has a 90° rotational offset (start angle or axis definition?)
4. **Differentiation handles CW correctly** — `delta_s < 0` inverts derivative sign as needed

---

## Checklist

### 1. Angle Definition Audit
- [x] What angle convention does the code / ASTRA expect?
  - **CCW rotation from X-axis** (geometry.py:39-41)
- [x] What is the angle definition in the DICOM data?
  - **(0x7031,0x1001)** stores source angle in radians; direction not documented in tag
  - **Empirically CW** — negation is required for correct reconstruction (run_L067.py:270)
- [x] Are they the same? If not, what is the mapping?
  - **No. DICOM = CW, Code = CCW. Mapping: θ_code = -θ_dicom**

### 2. Angle Sign / Direction
- [x] Is the rotation direction (CW vs CCW) consistent between data and code?
  - **No.** DICOM angles decrease (CW), code expects CCW (increasing)
- [x] Does negating the angle array change the artifact pattern?
  - **Yes.** `negate_angles=True` is part of the confirmed best configuration
- [x] Check: does the artifact "rotate" in the opposite direction after negation?
  - Confirmed via auto-focus: negation is required, produces correct orientation

### 3. Start Angle / Offset
- [x] What is the actual starting angle in the DICOM trajectory?
  - L067 starts ~268 deg (4.68 rad); pipeline uses absolute angles after override
- [x] Does the code assume a specific start angle (e.g., 0)?
  - **No.** run_dicom_recon.py replaces `source_pos` with actual DICOM angles (line 163)
  - initialize.py generates synthetic centered angles, but these get overwritten
- [x] Is there a constant angular offset between data and code?
  - **YES: -pi/2 (-90 deg).** DICOM defines 0 deg at the Y-axis; code expects 0 deg at X-axis.
  - **Verified (2026-03-13):** `angles -= pi/2` produces images identical to post-recon rot90
    (max pixel diff < 0.00013, see correction_comparison.png and correction_diff.png)
  - **Fix:** apply `angle_offset = -pi/2` in the pipeline, remove rot90

### 4. Flip / Transpose Consistency
- [x] When projection data was flipped (rows/cols), was the angle array updated accordingly?
  - `flip_rows` flips detector z-axis; this does NOT affect the in-plane angle → **OK**
  - `flip_cols` would flip fan direction → equivalent to negating fan angle → **not used for L067**
- [x] Are detector row/column axes consistent with the assumed geometry after any flips?
  - dicom.py:251 transposes pixel_array (`.T`) to match (rows=z, cols=fan) convention
  - `flip_rows=True` further corrects z-direction → **consistent after both transforms**
- [ ] Check that flip operations haven't swapped the effective rotation direction
  - *Transpose + flip_rows should NOT affect rotation direction — but verify by checking that couch is stable*

### 5. Projection-to-Angle Mapping
- [x] Is each projection mapped to the correct angle? (not off-by-one, not shuffled)
  - dicom.py sorts by InstanceNumber (line 143), then reads angles in that order
  - run_dicom_recon.py replaces source_pos with these sorted angles → **consistent**
- [x] Are projections sorted by angle? Does the code assume sorted input?
  - Sorted by InstanceNumber; angles are monotonic after unwrapping → **yes**
- [x] For decimated data: are the angles decimated in the same way as the projections?
  - run_dicom_recon.py:148-151: `sino = sino[::decimate]`, `angles = angles[::decimate]` → **consistent**

### 6. Accumulation / Weighting Verification
- [x] Verify Katsevich weight function uses the same angle convention as backprojection
  - Both use `L = R - x·cos(θ) - y·sin(θ)` with direct angle values → **consistent**
- [ ] Check if wrong angular assignment causes projections to accumulate at incorrect spatial positions
  - *The negate_angles fix handles the CW→CCW conversion, but residual periodic error may indicate:*
    - *Non-uniform angular spacing not correctly handled*
    - *Or: angle values are correct per-view, but weighting window (T-D bounds) uses wrong angle range*
- [ ] Investigate whether the Tam-Danielsson window computation matches the corrected angles
  - *filter.py T-D window logic should be checked against negated angles*

### 7. Couch Stability Test (Sanity Check)
- [ ] Reconstruct a few slices and track the couch position across z
- [ ] Couch should remain static — any drift indicates residual geometry error
- [ ] This is the single best visual diagnostic for angular alignment
- **Action: write a script that extracts couch ROI across ~50 slices and plots position vs z**

### 8. Modulo / Wrap-around
- [x] Are angles wrapped to [0, 2pi) or [-pi, pi)?
  - dicom.py:146 uses `np.unwrap()` → continuous (no wrapping) → **safe**
- [x] For helical: is the angular range correctly accounting for multi-turn coverage?
  - Unwrapped angles span the full helical range → **yes**

---

## Open Questions (Priority Order)

### A. Why rot90 post-reconstruction? -- RESOLVED
DICOM 0 deg is at the Y-axis (not X-axis), creating a 90 deg offset.
**Fix verified:** `angles -= pi/2` eliminates need for rot90. Max pixel diff < 0.00013.

### B. Is the residual flicker from non-uniform angular steps?
After negate + flip_rows, some periodic artifact may persist if:
- DICOM angles have small non-uniformities (jitter)
- The differentiation `(proj[i+1] - proj[i]) / (4·delta_s)` uses a constant delta_s instead of per-view differences

**Test:** check `np.std(np.diff(angles))` relative to `np.mean(np.diff(angles))`.

### C. Detector pixel size: rows vs cols asymmetry
- dicom.py:277 averages row and col pixel sizes: `np.mean([det_pixel_size_cols, det_pixel_size_rows])`
- If these differ significantly, using the mean introduces geometry error in one axis
- **Check:** print both values for L067 and assess impact

---

## Decision Principle

From advisor:
> "ASTRA is a mature, well-maintained toolbox. You've already reproduced working examples. The problem is the parametric discrepancy between data and what you use."

**Order of investigation:**
1. ~~Angle definition & sign~~ → **Resolved: CW→CCW via negation**
2. Start angle offset → **Suspected: rot90 symptom suggests ~90° offset**
3. Flip/transpose side effects → **Partially verified**
4. Weighting / T-D window with corrected angles → **Needs check**
5. Only after all above are verified — consider algorithm-level issues

---

## Task Summary

This checklist was extracted from the 0313 advisor meeting. The task was named
**"Katsevich angular correction"** at the end of the session as a priority item
before moving on to primitive sinograms.

### Next Steps
1. **Test 90° angle offset** — does `angles += π/2` eliminate the need for post-recon rot90?
2. **Couch stability test** — write and run a couch-tracking diagnostic
3. **Check angular uniformity** — quantify jitter in DICOM angle spacing
4. **Review T-D window** — verify Tam-Danielsson bounds use negated angles correctly
