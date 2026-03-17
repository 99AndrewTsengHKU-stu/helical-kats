"""
Trace angle variable through the entire Katsevich pipeline.

Checks two potential issues:
  1. source_pos override: initialize.py generates synthetic centered angles,
     but run scripts override with DICOM angles. Are dependent params
     (s_min, s_max, s_len, projs_per_turn, delta_s) still consistent?
  2. z-mapping dual-track: flat_backproject_chunk uses source_pos * progress_per_radian
     for z, while CuPy kernel uses ASTRA views src_z. Do they agree?

Usage:
  ~/anaconda3/envs/MNIST/python.exe "Quick Run Through 0313/trace_angle_pipeline.py"
"""
import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Quick Run Through 0304"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

import numpy as np

from pykatsevich import load_dicom_projections
from pykatsevich.geometry import astra_helical_views
from pykatsevich.initialize import create_configuration

DICOM_DIR = r"D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD"
ROWS = 512
COLS = 512
SLICES = 560
VOXEL_SIZE = 0.664
CHUNK_Z = 64

print("=" * 70)
print("ANGLE PIPELINE TRACE")
print("=" * 70)

# ---- Step 1: Load DICOM ----
print("\n[Step 1] Loading DICOM projections...")
sino, meta = load_dicom_projections(DICOM_DIR)
angles_raw = meta["angles_rad"].copy()
print(f"  Raw angles: {len(angles_raw)} views")
print(f"  Range: [{angles_raw[0]:.4f}, {angles_raw[-1]:.4f}] rad")
print(f"         [{np.degrees(angles_raw[0]):.2f}, {np.degrees(angles_raw[-1]):.2f}] deg")
print(f"  Total span: {abs(angles_raw[-1] - angles_raw[0]):.4f} rad = {abs(angles_raw[-1] - angles_raw[0])/(2*np.pi):.3f} turns")
print(f"  Mean step: {np.mean(np.diff(angles_raw)):.8f} rad ({'CW' if np.mean(np.diff(angles_raw)) < 0 else 'CCW'})")
print(f"  pitch_mm_per_rad_signed: {meta['pitch_mm_per_rad_signed']:.6f}")
print(f"  pitch_mm_per_angle: {meta['pitch_mm_per_angle']:.6f}")
print(f"  table_positions_mm range: [{meta['table_positions_mm'].min():.2f}, {meta['table_positions_mm'].max():.2f}]")

# ---- Step 2: User preprocessing (negate + offset) ----
print("\n[Step 2] User preprocessing: negate + offset(-pi/2)")
angles_full = angles_raw.copy()
angles_full = -angles_full             # negate CW -> CCW
angles_full = angles_full + (-np.pi/2) # offset
print(f"  After negate+offset:")
print(f"  Range: [{angles_full[0]:.4f}, {angles_full[-1]:.4f}] rad")
print(f"         [{np.degrees(angles_full[0]):.2f}, {np.degrees(angles_full[-1]):.2f}] deg")
print(f"  Mean step: {np.mean(np.diff(angles_full)):.8f} rad ({'CW' if np.mean(np.diff(angles_full)) < 0 else 'CCW'})")
print(f"  Total span: {abs(angles_full[-1] - angles_full[0]):.4f} rad = {abs(angles_full[-1] - angles_full[0])/(2*np.pi):.3f} turns")

# ---- Step 3: Build scan_geom ----
print("\n[Step 3] Build scan_geom_full")
scan_geom_full = copy.deepcopy(meta["scan_geometry"])
pitch_used = abs(meta["pitch_mm_per_rad_signed"])
scan_geom_full["helix"]["pitch_mm_rad"] = float(pitch_used)
scan_geom_full["helix"]["angles_range"] = float(abs(angles_full[-1] - angles_full[0]))
print(f"  pitch_mm_rad: {scan_geom_full['helix']['pitch_mm_rad']:.6f}")
print(f"  angles_range: {scan_geom_full['helix']['angles_range']:.4f} rad")
print(f"  angles_count: {scan_geom_full['helix']['angles_count']}")

# ---- Step 4: Build vertical shifts ----
print("\n[Step 4] Vertical shifts (z_shift_full)")
z_shift_full = meta["table_positions_mm"] - meta["table_positions_mm"].mean()
print(f"  z_shift range: [{z_shift_full.min():.2f}, {z_shift_full.max():.2f}] mm")
print(f"  z_shift span: {z_shift_full.max() - z_shift_full.min():.2f} mm")

# ---- Step 5: ASTRA views ----
print("\n[Step 5] astra_helical_views()")
views_full = astra_helical_views(
    scan_geom_full["SOD"], scan_geom_full["SDD"],
    scan_geom_full["detector"]["detector psize"],
    angles_full, meta["pitch_mm_per_angle"],
    vertical_shifts=z_shift_full,
)
src_x = views_full[:, 0]
src_y = views_full[:, 1]
src_z_astra = views_full[:, 2]
print(f"  Source X: [{src_x.min():.2f}, {src_x.max():.2f}]")
print(f"  Source Y: [{src_y.min():.2f}, {src_y.max():.2f}]")
print(f"  Source Z (from ASTRA views): [{src_z_astra.min():.2f}, {src_z_astra.max():.2f}]")
# Reconstruct angles from views
angles_from_views = np.arctan2(src_y, src_x)
print(f"  Angles recovered from views (atan2(y,x)): [{angles_from_views[0]:.4f}, {angles_from_views[-1]:.4f}]")
print(f"  Match input angles_full? max|diff| = {np.max(np.abs(angles_from_views - angles_full)):.8f} rad")

# ==================================================================
# Now simulate a CHUNK to check create_configuration behavior
# ==================================================================
print("\n" + "=" * 70)
print("CHUNK-LEVEL ANALYSIS (middle chunk)")
print("=" * 70)

total_half_z = SLICES * VOXEL_SIZE * 0.5
mid_chunk = SLICES // 2 // CHUNK_Z
z_start = mid_chunk * CHUNK_Z
z_end = min(z_start + CHUNK_Z, SLICES)
chunk_slices = z_end - z_start
chunk_z_min = -total_half_z + z_start * VOXEL_SIZE
chunk_z_max = -total_half_z + z_end * VOXEL_SIZE
print(f"\n  Chunk: slices [{z_start}, {z_end}), z=[{chunk_z_min:.2f}, {chunk_z_max:.2f}] mm")

# Simulate z-culling (simplified)
SOD = scan_geom_full["SOD"]
SDD = scan_geom_full["SDD"]
det_rows = scan_geom_full["detector"]["detector rows"]
psize = scan_geom_full["detector"]["detector psize"]
cone_half_z = 0.5 * det_rows * psize * (SOD / SDD)
margin_z = cone_half_z
z_lo = src_z_astra - cone_half_z - margin_z
z_hi = src_z_astra + cone_half_z + margin_z
mask = (z_hi >= chunk_z_min) & (z_lo <= chunk_z_max)
keep_idx = np.where(mask)[0]
print(f"  Z-culled: {len(angles_full)} -> {len(keep_idx)} projections")

angles_chunk = angles_full[keep_idx].copy()
views_chunk = views_full[keep_idx]

# ---- Step 6: create_configuration for chunk ----
print("\n[Step 6] create_configuration() -- BEFORE source_pos override")
import astra

scan_geom_chunk = copy.deepcopy(scan_geom_full)
scan_geom_chunk["helix"]["angles_range"] = float(abs(angles_chunk[-1] - angles_chunk[0]))
scan_geom_chunk["helix"]["angles_count"] = len(angles_chunk)

half_x = COLS * VOXEL_SIZE * 0.5
half_y = ROWS * VOXEL_SIZE * 0.5
vol_geom_chunk = astra.create_vol_geom(
    ROWS, COLS, chunk_slices,
    -half_x, half_x, -half_y, half_y, chunk_z_min, chunk_z_max,
)
conf = create_configuration(scan_geom_chunk, vol_geom_chunk)

print(f"  conf['s_min']:          {conf['s_min']:.6f} rad")
print(f"  conf['s_max']:          {conf['s_max']:.6f} rad")
print(f"  conf['s_len']:          {conf['s_len']:.6f} rad ({conf['s_len']/(2*np.pi):.3f} turns)")
print(f"  conf['total_projs']:    {conf['total_projs']}")
print(f"  conf['projs_per_turn']: {conf['projs_per_turn']:.2f}")
print(f"  conf['delta_s']:        {conf['delta_s']:.8f} rad")
print(f"  conf['progress_per_radian']: {conf['progress_per_radian']:.6f} mm/rad")
print(f"  conf['source_pos'] (synthetic):")
print(f"    range: [{conf['source_pos'].min():.6f}, {conf['source_pos'].max():.6f}] rad")
print(f"    center: {conf['source_pos'].mean():.6f} rad")

print(f"\n  Actual DICOM angles for this chunk:")
print(f"    range: [{angles_chunk.min():.6f}, {angles_chunk.max():.6f}] rad")
print(f"    center: {angles_chunk.mean():.6f} rad")
print(f"    actual delta_s: {np.mean(np.diff(angles_chunk)):.8f} rad")

# ---- Step 7: Override source_pos ----
print("\n[Step 7] Override source_pos with DICOM angles")
conf['source_pos'] = angles_chunk.astype(np.float32)
conf['delta_s'] = float(np.mean(np.diff(angles_chunk)))
print(f"  conf['source_pos'] now: [{conf['source_pos'].min():.6f}, {conf['source_pos'].max():.6f}]")
print(f"  conf['delta_s'] now:    {conf['delta_s']:.8f} rad")
print(f"  BUT conf['projs_per_turn'] still: {conf['projs_per_turn']:.2f}")
print(f"  BUT conf['s_len'] still:          {conf['s_len']:.6f}")

# ---- Issue 1: projs_per_turn consistency ----
print("\n" + "=" * 70)
print("ISSUE 1: projs_per_turn consistency")
print("=" * 70)
# projs_per_turn was computed as: total_projs / s_len * 2pi
# where s_len = angles_range (from scan_geom_chunk)
# After override, the actual angles span the same range, so s_len should match
actual_s_len = abs(angles_chunk[-1] - angles_chunk[0])
expected_ppt = len(angles_chunk) / actual_s_len * 2 * np.pi
print(f"  scan_geom angles_range: {scan_geom_chunk['helix']['angles_range']:.6f}")
print(f"  actual chunk angle span: {actual_s_len:.6f}")
print(f"  diff: {abs(scan_geom_chunk['helix']['angles_range'] - actual_s_len):.10f}")
print(f"  conf projs_per_turn: {conf['projs_per_turn']:.4f}")
print(f"  expected from actual: {expected_ppt:.4f}")
print(f"  diff: {abs(conf['projs_per_turn'] - expected_ppt):.6f}")

# Effect on CuPy kernel scaling:
inv_scale_conf = SOD**2 / conf['projs_per_turn']
inv_scale_actual = SOD**2 / expected_ppt
print(f"\n  CuPy inv_scale (from conf): {inv_scale_conf:.4f}")
print(f"  CuPy inv_scale (from actual): {inv_scale_actual:.4f}")
print(f"  Relative error: {abs(inv_scale_conf - inv_scale_actual) / inv_scale_actual * 100:.6f}%")

# ---- Issue 2: z-mapping dual track ----
print("\n" + "=" * 70)
print("ISSUE 2: z-mapping dual track")
print("=" * 70)

src_z_chunk_astra = views_chunk[:, 2]
progress_per_radian = conf['progress_per_radian']

# Method A: flat_backproject_chunk uses source_pos * progress_per_radian
# This is the z of the source at angle source_pos[i]
z_from_source_pos = conf['source_pos'] * progress_per_radian
print(f"  Method A: source_pos * progress_per_radian")
print(f"    z range: [{z_from_source_pos.min():.2f}, {z_from_source_pos.max():.2f}] mm")
print(f"    z span: {z_from_source_pos.max() - z_from_source_pos.min():.2f} mm")
print(f"    z center: {z_from_source_pos.mean():.2f} mm")

# Method B: ASTRA views src_z (from astra_helical_views using vertical_shifts)
print(f"\n  Method B: ASTRA views src_z")
print(f"    z range: [{src_z_chunk_astra.min():.2f}, {src_z_chunk_astra.max():.2f}] mm")
print(f"    z span: {src_z_chunk_astra.max() - src_z_chunk_astra.min():.2f} mm")
print(f"    z center: {src_z_chunk_astra.mean():.2f} mm")

# Direct comparison
z_diff = z_from_source_pos - src_z_chunk_astra
print(f"\n  Difference (A - B):")
print(f"    min: {z_diff.min():.4f} mm")
print(f"    max: {z_diff.max():.4f} mm")
print(f"    mean: {z_diff.mean():.4f} mm")
print(f"    std: {z_diff.std():.4f} mm")
print(f"    max|diff|: {np.abs(z_diff).max():.4f} mm")

if np.abs(z_diff).max() > 1.0:
    print(f"    [FAIL] Significant z-mapping mismatch! max|diff| = {np.abs(z_diff).max():.2f} mm")
    print(f"    This means flat_backproject_chunk and CuPy kernel see DIFFERENT source z-positions!")
elif np.abs(z_diff).max() > 0.1:
    print(f"    [!] Moderate z-mapping difference: {np.abs(z_diff).max():.4f} mm")
else:
    print(f"    [OK] z-mapping is consistent (max diff < 0.1 mm)")

# Check correlation
corr = np.corrcoef(z_from_source_pos, src_z_chunk_astra)[0, 1]
print(f"    Correlation: {corr:.8f}")

# Check if the relationship is linear but offset/scaled
if len(z_from_source_pos) > 10:
    slope, intercept = np.polyfit(src_z_chunk_astra, z_from_source_pos, 1)
    print(f"    Linear fit: z_A = {slope:.6f} * z_B + {intercept:.4f}")
    if abs(slope - 1.0) > 0.01:
        print(f"    [FAIL] Slope != 1 -- z scales are different!")
    if abs(intercept) > 1.0:
        print(f"    [FAIL] Intercept != 0 -- z origins are offset by {intercept:.2f} mm!")

# ---- Detailed: what makes up ASTRA src_z vs source_pos*ppr ----
print("\n  Breakdown:")
print(f"    ASTRA src_z comes from: vertical_shifts (table_pos - mean(table_pos))")
print(f"    source_pos*ppr comes from: angle_value * pitch_mm_per_rad")
print(f"    These are fundamentally different if angles are not centered at 0!")
print(f"    angles_chunk center: {angles_chunk.mean():.4f} rad = {np.degrees(angles_chunk.mean()):.2f} deg")
print(f"    z_shift_full center: 0 (by construction, mean-subtracted)")
print(f"    source_pos*ppr at center angle: {angles_chunk.mean() * progress_per_radian:.2f} mm")
print(f"    z_shift at center index: {z_shift_full[keep_idx].mean():.2f} mm")

# ---- Also check what the original code (without override) would compute ----
print("\n" + "=" * 70)
print("COMPARISON: synthetic source_pos vs overridden source_pos")
print("=" * 70)
conf_orig = create_configuration(scan_geom_chunk, vol_geom_chunk)
print(f"  Synthetic source_pos: [{conf_orig['source_pos'].min():.6f}, {conf_orig['source_pos'].max():.6f}]")
print(f"  Overridden (DICOM):   [{angles_chunk.min():.6f}, {angles_chunk.max():.6f}]")
print(f"  Synthetic center: {conf_orig['source_pos'].mean():.6f}")
print(f"  DICOM center: {angles_chunk.mean():.6f}")

z_synth = conf_orig['source_pos'] * progress_per_radian
z_dicom = angles_chunk * progress_per_radian
print(f"\n  z from synthetic * ppr: [{z_synth.min():.2f}, {z_synth.max():.2f}], center={z_synth.mean():.2f}")
print(f"  z from DICOM * ppr:    [{z_dicom.min():.2f}, {z_dicom.max():.2f}], center={z_dicom.mean():.2f}")
print(f"  z from ASTRA views:    [{src_z_chunk_astra.min():.2f}, {src_z_chunk_astra.max():.2f}], center={src_z_chunk_astra.mean():.2f}")

print(f"\n  Volume z-range: [{chunk_z_min:.2f}, {chunk_z_max:.2f}] mm (center={0.5*(chunk_z_min+chunk_z_max):.2f})")

# Are the z-ranges even overlapping?
vol_center = 0.5 * (chunk_z_min + chunk_z_max)
print(f"\n  Source z center vs volume center:")
print(f"    ASTRA src_z center:     {src_z_chunk_astra.mean():.2f} mm")
print(f"    synthetic*ppr center:   {z_synth.mean():.2f} mm")
print(f"    DICOM*ppr center:       {z_dicom.mean():.2f} mm")
print(f"    Volume center:          {vol_center:.2f} mm")
print(f"    DICOM*ppr - vol_center: {z_dicom.mean() - vol_center:.2f} mm  <-- offset in flat_backproject_chunk!")

# ---- Effect on T-D window z_coord_mins/maxs ----
print("\n" + "=" * 70)
print("EFFECT ON T-D z-boundary (filter.py line 534)")
print("=" * 70)
print("  z_coord_mins[p] = source_pos[p] * progress_per_radian + proj_row_min * scale / SDD")
print("  With DICOM angles, source_pos contains large absolute values,")
print("  not centered values. Let's check a few projections:\n")

# Pick first, middle, last projection in chunk
for label, idx in [("first", 0), ("middle", len(angles_chunk)//2), ("last", len(angles_chunk)-1)]:
    sp_synth = conf_orig['source_pos'][idx] if idx < len(conf_orig['source_pos']) else float('nan')
    sp_dicom = angles_chunk[idx]
    sz_astra = src_z_chunk_astra[idx]

    z_base_synth = sp_synth * progress_per_radian
    z_base_dicom = sp_dicom * progress_per_radian

    print(f"  [{label}] proj {idx}:")
    print(f"    synthetic source_pos: {sp_synth:.6f} rad -> z_base = {z_base_synth:.2f} mm")
    print(f"    DICOM source_pos:     {sp_dicom:.6f} rad -> z_base = {z_base_dicom:.2f} mm")
    print(f"    ASTRA src_z:          {sz_astra:.2f} mm")
    print(f"    DICOM z_base - ASTRA: {z_base_dicom - sz_astra:.2f} mm")
    print()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
The flat_backproject_chunk code computes z-coordinates as:
  z = source_pos * progress_per_radian + (row_correction)

When source_pos is overridden with DICOM angles (e.g., ~-6.27 rad for negated L067),
this produces z values far from the actual volume z-range.

The CuPy kernel does NOT have this problem because it uses ASTRA views' src_z
(which comes from table_positions_mm, correctly centered).

HOWEVER: the CuPy kernel path does NOT use source_pos for z at all.
The source_pos values are only used for:
  1. cos/sin in the projection formula (in-plane geometry) -- CORRECT after offset
  2. projs_per_turn scaling -- check if consistent

The problematic path is flat_backproject_chunk (filter.py), which is the
ORIGINAL backprojection. If the CuPy kernel is used instead, Issue 2 does
NOT affect backprojection.

BUT: Does any OTHER part of the pipeline use source_pos for z-mapping?
- differentiate(): uses delta_s only (Step 6) -- OK
- sino_weight_td(): uses pre-computed T-D mask only -- OK
- The T-D z_coord_mins/maxs in flat_backproject_chunk are only in that function

So for CuPy kernel path: Issue 2 is NOT a problem.
For flat_backproject_chunk path: Issue 2 IS a problem.
""")

print("Done.")
