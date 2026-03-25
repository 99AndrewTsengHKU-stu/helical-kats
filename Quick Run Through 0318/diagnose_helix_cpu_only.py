"""
CPU-only diagnostic: Check helix handedness and visualize TD window.
No GPU/reconstruction needed.
"""
import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Quick Run Through 0304"))

import numpy as np
from matplotlib import pyplot as plt
from time import time

from pykatsevich import load_dicom_projections
from pykatsevich.geometry import astra_helical_views
from pykatsevich.initialize import create_configuration

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Settings ──────────────────────────────────────────────────────────
DICOM_DIR = r"D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD"
ROWS = 512
COLS = 512
SLICES = 560
VOXEL_SIZE_XY = 0.664
VOXEL_SIZE_Z = 0.8

TEST_START = 276
TEST_END = 286
TEST_SLICES = TEST_END - TEST_START

# ══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════
print("Loading DICOM...")
t0 = time()
sino, meta = load_dicom_projections(DICOM_DIR)
print(f"Loaded {sino.shape[0]} projections in {time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════
# TEST 1: HELIX HANDEDNESS AFTER TRANSFORMS
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("TEST 1: Helix handedness after transforms")
print(f"{'='*60}")

angles_raw = meta["angles_rad"].copy()
table_raw = meta["table_positions_mm"].copy() if meta["table_positions_mm"] is not None else None

print(f"\n  Raw DICOM angles:")
print(f"    First 5: {angles_raw[:5]}")
print(f"    diff[0:4]: {np.diff(angles_raw[:5])}")
raw_dir = np.mean(np.diff(angles_raw[:20]))
print(f"    Mean step: {raw_dir:.6f} rad")
print(f"    Direction: {'CW (decreasing)' if raw_dir < 0 else 'CCW (increasing)'}")

if table_raw is not None:
    print(f"\n  Raw DICOM table positions:")
    print(f"    First 5: {table_raw[:5]}")
    print(f"    diff[0:4]: {np.diff(table_raw[:5])}")
    table_dir = np.mean(np.diff(table_raw[:20]))
    print(f"    Mean step: {table_dir:.6f} mm")
    print(f"    Direction: {'decreasing (into gantry)' if table_dir < 0 else 'increasing'}")

# After transforms: negate + offset
angles_transformed = -angles_raw - np.pi / 2
z_shift_centered = table_raw - table_raw.mean() if table_raw is not None else np.zeros(len(angles_raw))

print(f"\n  After negate + offset(-pi/2):")
print(f"    First 5 angles: {angles_transformed[:5]}")
print(f"    diff[0:4]: {np.diff(angles_transformed[:5])}")
ang_dir = np.mean(np.diff(angles_transformed[:20]))
print(f"    Mean step: {ang_dir:.6f} rad")
print(f"    Direction: {'CCW (increasing)' if ang_dir > 0 else 'CW (decreasing)'}")

# Effective pitch: dz/ds
ds = np.diff(angles_transformed[:100])
dz = np.diff(z_shift_centered[:100])
eff_pitch = np.mean(dz / ds)
print(f"\n  Effective pitch after transforms:")
print(f"    dz/ds = {eff_pitch:.6f} mm/rad")
print(f"    Helix: {'RIGHT-HANDED (z up as s increases)' if eff_pitch > 0 else 'LEFT-HANDED (z down as s increases)'}")

# What the code uses
pitch_signed = meta["pitch_mm_per_rad_signed"]
pitch_used = abs(pitch_signed)
print(f"\n  Code parameters:")
print(f"    pitch_mm_per_rad_signed = {pitch_signed:.6f}")
print(f"    abs(pitch) used         = {pitch_used:.6f}")
print(f"    progress_per_turn       = {pitch_used * 2 * np.pi:.4f} mm")

if eff_pitch < 0:
    print(f"\n  >>> MISMATCH DETECTED <<<")
    print(f"  Actual helix: LEFT-HANDED (dz/ds = {eff_pitch:.4f})")
    print(f"  Code assumes: RIGHT-HANDED (abs pitch = +{pitch_used:.4f})")
    print(f"  TD window and rebinning use positive pitch formulas.")
    print(f"  This means TD selects the COMPLEMENTARY redundancy region!")
else:
    print(f"\n  [OK] Helix is right-handed, matches code assumption.")


# ══════════════════════════════════════════════════════════════════════
# TEST 2: TD WINDOW VISUALIZATION
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("TEST 2: TD window visualization")
print(f"{'='*60}")

# Build geometry (CPU only, no ASTRA volume needed for visualization)
scan_geom_full = copy.deepcopy(meta["scan_geometry"])
scan_geom_full["helix"]["pitch_mm_rad"] = float(pitch_used)

angles_full = angles_transformed.copy()
scan_geom_full["helix"]["angles_range"] = float(abs(angles_full[-1] - angles_full[0]))
scan_geom_full["helix"]["angles_count"] = len(angles_full)

det_rows_n = scan_geom_full["detector"]["detector rows"]
det_cols_n = scan_geom_full["detector"]["detector cols"]

# Create a fake vol_geom dict (just need the structure for create_configuration)
total_half_z = SLICES * VOXEL_SIZE_Z * 0.5
chunk_z_min = -total_half_z + TEST_START * VOXEL_SIZE_Z
chunk_z_max = -total_half_z + TEST_END * VOXEL_SIZE_Z
half_x = COLS * VOXEL_SIZE_XY * 0.5
half_y = ROWS * VOXEL_SIZE_XY * 0.5

# Manually build vol_geom dict (avoid astra import)
vol_geom = {
    'GridColCount': COLS,
    'GridRowCount': ROWS,
    'GridSliceCount': TEST_SLICES,
    'option': {
        'WindowMinX': -half_x, 'WindowMaxX': half_x,
        'WindowMinY': -half_y, 'WindowMaxY': half_y,
        'WindowMinZ': chunk_z_min, 'WindowMaxZ': chunk_z_max,
    }
}

conf = create_configuration(scan_geom_full, vol_geom)

# Print TD window info
row_coords = conf['row_coords'][:-1]  # drop extended element
col_coords = conf['col_coords'][:-1]
pix_h = conf['pixel_height']

print(f"\n  Detector geometry:")
print(f"    {det_rows_n} rows x {det_cols_n} cols")
print(f"    pixel_height (z) = {pix_h:.6f} mm")
print(f"    pixel_span (fan) = {conf['pixel_span']:.6f} mm")
print(f"    row_coords range: [{row_coords[0]:.4f}, {row_coords[-1]:.4f}] mm")
print(f"    col_coords range: [{col_coords[0]:.4f}, {col_coords[-1]:.4f}] mm")

mid_col = len(conf['proj_row_mins']) // 2
print(f"\n  TD window bounds (center column):")
print(f"    w_bottom = {conf['proj_row_mins'][mid_col]:.4f} mm")
print(f"    w_top    = {conf['proj_row_maxs'][mid_col]:.4f} mm")

td_height = conf['proj_row_maxs'][mid_col] - conf['proj_row_mins'][mid_col]
det_height = row_coords[-1] - row_coords[0]
print(f"\n  TD window height (center): {td_height:.4f} mm")
print(f"  Detector height:           {det_height:.4f} mm")
print(f"  Ratio (TD/detector):       {td_height / det_height:.4f}")

if td_height > det_height:
    print(f"  [!] TD window LARGER than detector — entire detector passes")
elif td_height / det_height < 0.5:
    print(f"  [!] TD window covers < 50% of detector rows")
else:
    print(f"  [OK] TD window covers {100*td_height/det_height:.0f}% of detector")

# Compute TD mask
W, U = np.meshgrid(row_coords, col_coords, indexing='ij')
w_bottom = np.reshape(conf['proj_row_mins'][:-1], (1, -1))
w_top = np.reshape(conf['proj_row_maxs'][:-1], (1, -1))

a = conf['T-D smoothing']
dw = det_rows_n * pix_h

mask = np.zeros_like(W)
W_top_high = np.repeat(w_top + a * dw, W.shape[0], axis=0)
W_top_low = np.repeat(w_top - a * dw, W.shape[0], axis=0)
W_bottom_high = np.repeat(w_bottom + a * dw, W.shape[0], axis=0)
W_bottom_low = np.repeat(w_bottom - a * dw, W.shape[0], axis=0)

mask[(W_top_low < W) & (W < W_top_high)] = ((W_top_high - W) / (2 * a * dw))[(W_top_low < W) & (W < W_top_high)]
mask[(W_bottom_high < W) & (W < W_top_low)] = 1
mask[(W_bottom_low < W) & (W < W_bottom_high)] = ((W - W_bottom_low) / (2 * a * dw))[(W_bottom_low < W) & (W < W_bottom_high)]

print(f"\n  TD mask statistics:")
print(f"    Shape: {mask.shape}")
print(f"    Fraction == 1 (full weight): {np.mean(mask == 1):.4f}")
print(f"    Fraction == 0 (zero weight): {np.mean(mask == 0):.4f}")
print(f"    Fraction in (0,1) (ramp):    {np.mean((mask > 0) & (mask < 1)):.4f}")

# ── Figure: 4 panels ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel 1: TD mask in physical coordinates
im0 = axes[0, 0].imshow(
    mask, cmap='hot', aspect='auto',
    extent=[col_coords[0], col_coords[-1], row_coords[-1], row_coords[0]])
axes[0, 0].set_title(f"TD mask (smoothing={a})\nw_bottom to w_top in physical mm")
axes[0, 0].set_xlabel("u (column coord, mm)")
axes[0, 0].set_ylabel("w (row coord, mm)")
axes[0, 0].axhline(0, color='cyan', ls='--', alpha=0.5, label='w=0')
axes[0, 0].legend()
plt.colorbar(im0, ax=axes[0, 0])

# Panel 2: TD bounds overlaid on a sinogram frame
sino_flipped = sino[:, ::-1, :]  # flip_rows as code does
mid_proj = sino.shape[0] // 2
frame = sino_flipped[mid_proj]
im1 = axes[0, 1].imshow(frame, cmap='gray', aspect='auto',
                          extent=[0, det_cols_n, det_rows_n, 0])
# Convert TD bounds from mm to row indices
center_row = (det_rows_n - 1) / 2.0
for bound, color, label in [(conf['proj_row_mins'], 'red', 'w_bottom'),
                              (conf['proj_row_maxs'], 'lime', 'w_top')]:
    row_idx = bound[:-1] / pix_h + center_row
    axes[0, 1].plot(np.arange(det_cols_n), row_idx, color=color, lw=1.5, label=label)
axes[0, 1].axhline(center_row, color='cyan', ls='--', alpha=0.5, label='center')
axes[0, 1].set_title(f"Sinogram frame #{mid_proj} (after flip_rows)\nwith TD bounds")
axes[0, 1].set_xlabel("Column index")
axes[0, 1].set_ylabel("Row index")
axes[0, 1].legend(fontsize=8)

# Panel 3: Source z trajectory
z_shift = meta["table_positions_mm"] - meta["table_positions_mm"].mean()
n_show = min(4000, len(angles_full))
axes[1, 0].plot(angles_full[:n_show], z_shift[:n_show], 'b-', lw=0.3)
axes[1, 0].set_xlabel("Angle (rad, after negate+offset)")
axes[1, 0].set_ylabel("Source z (mm, centered)")
axes[1, 0].set_title(f"Helix trajectory (first {n_show} projs)\n"
                       f"dz/ds = {eff_pitch:.4f} mm/rad  "
                       f"({'LEFT-HANDED' if eff_pitch < 0 else 'RIGHT-HANDED'})")
axes[1, 0].axhline(0, color='gray', ls='--', alpha=0.3)
# Mark the test chunk z-range
axes[1, 0].axhspan(chunk_z_min, chunk_z_max, color='orange', alpha=0.15,
                     label=f'test chunk [{chunk_z_min:.0f}, {chunk_z_max:.0f}] mm')
axes[1, 0].legend(fontsize=8)

# Panel 4: Cross-section through TD mask at center column
center_col_idx = det_cols_n // 2
mask_col_slice = mask[:, center_col_idx]
axes[1, 1].plot(row_coords, mask_col_slice, 'r-', lw=2, label='TD weight')
axes[1, 1].axvline(conf['proj_row_mins'][center_col_idx], color='blue', ls='--',
                     label=f"w_bottom={conf['proj_row_mins'][center_col_idx]:.2f}")
axes[1, 1].axvline(conf['proj_row_maxs'][center_col_idx], color='green', ls='--',
                     label=f"w_top={conf['proj_row_maxs'][center_col_idx]:.2f}")
axes[1, 1].axvline(0, color='gray', ls=':', alpha=0.5)
axes[1, 1].set_xlabel("w (row coord, mm)")
axes[1, 1].set_ylabel("TD weight")
axes[1, 1].set_title("TD weight profile (center column)")
axes[1, 1].legend(fontsize=8)
axes[1, 1].set_xlim(row_coords[0], row_coords[-1])
axes[1, 1].set_ylim(-0.05, 1.1)

plt.suptitle("Helix Direction & TD Window Diagnostic (CPU only)", fontsize=14, y=1.01)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "helix_td_diagnostic.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\n  Saved -> {out_path}")

# ── Also check: what if we use SIGNED pitch? ─────────────────────────
print(f"\n{'='*60}")
print("BONUS: TD window with SIGNED pitch")
print(f"{'='*60}")

scan_geom_signed = copy.deepcopy(scan_geom_full)
scan_geom_signed["helix"]["pitch_mm_rad"] = float(pitch_signed)  # SIGNED, not abs

conf_signed = create_configuration(scan_geom_signed, vol_geom)

print(f"  With signed pitch ({pitch_signed:.6f}):")
print(f"    progress_per_turn = {conf_signed['progress_per_turn']:.4f} mm")
print(f"    w_bottom (center) = {conf_signed['proj_row_mins'][mid_col]:.4f} mm")
print(f"    w_top (center)    = {conf_signed['proj_row_maxs'][mid_col]:.4f} mm")

if conf_signed['proj_row_mins'][mid_col] > conf_signed['proj_row_maxs'][mid_col]:
    print(f"    [!] w_bottom > w_top → TD mask would be EMPTY!")
    print(f"    This confirms the formula requires positive pitch.")
    print(f"    Using abs(pitch) is necessary, but the helix direction")
    print(f"    mismatch may still cause issues in the rebin/filter steps.")
else:
    print(f"    [OK] Bounds are ordered correctly.")

# Check rebin_scale with both pitches
rs_abs = (conf['scan_diameter'] * conf['progress_per_turn']) / (2 * np.pi * conf['scan_radius'])
rs_signed = (conf_signed['scan_diameter'] * conf_signed['progress_per_turn']) / (2 * np.pi * conf_signed['scan_radius'])
print(f"\n  Rebin scale comparison:")
print(f"    abs pitch:    rebin_scale = {rs_abs:.6f}")
print(f"    signed pitch: rebin_scale = {rs_signed:.6f}")
if rs_signed < 0:
    print(f"    [!] Signed pitch gives NEGATIVE rebin_scale")
    print(f"    Forward rebin would map to negative row indices → broken")

print(f"\n{'='*60}")
print("CONCLUSION")
print(f"{'='*60}")
print(f"  Effective dz/ds = {eff_pitch:.4f} mm/rad")
if eff_pitch < 0:
    print(f"  The helix is LEFT-HANDED after transforms.")
    print(f"  The filter pipeline (TD, rebin) assumes RIGHT-HANDED.")
    print(f"  abs(pitch) prevents empty TD mask but may give wrong region.")
    print(f"")
    print(f"  SUGGESTED FIX OPTIONS:")
    print(f"    A) Negate z_shift_full to make helix right-handed")
    print(f"       (also negate psize_row or flip v-vector in backprojection)")
    print(f"    B) Reverse projection order so angle decreases while z increases")
    print(f"    C) Modify TD/rebin formulas to handle negative pitch")
else:
    print(f"  The helix is RIGHT-HANDED — matches code assumption.")
    print(f"  Flicker root cause is likely elsewhere.")

print("\nDone.")
