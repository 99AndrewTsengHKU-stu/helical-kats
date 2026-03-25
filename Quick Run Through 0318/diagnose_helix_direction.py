"""
Diagnose helix direction mismatch as root cause of flicker.

Hypothesis: After negate_angles, source z DECREASES as angle increases
(left-handed helix), but abs(pitch) forces the TD window to assume a
right-handed helix. This gives a mirrored TD window that causes flicker.

Tests:
  1. Check helix handedness after all transforms
  2. Visualize TD window vs actual pi-line projections on the detector
  3. Compare reconstruction with signed vs abs(pitch)
  4. Test flicker with/without TD weighting to isolate TD as the cause
"""
import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Quick Run Through 0304"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from backproject_safe import ensure_astra_cuda_init
ensure_astra_cuda_init()

import numpy as np
from matplotlib import pyplot as plt
from time import time
import astra

from pykatsevich import load_dicom_projections
from pykatsevich.geometry import astra_helical_views
from pykatsevich.initialize import create_configuration
from pykatsevich.filter import filter_katsevich, sino_weight_td
from backproject_cupy import backproject_cupy

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Settings ──────────────────────────────────────────────────────────
DICOM_DIR = r"D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD"
ROWS = 512
COLS = 512
SLICES = 560
VOXEL_SIZE_XY = 0.664
VOXEL_SIZE_Z = 0.8

# Test slab for flicker comparison
TEST_START = 276
TEST_END = 286
TEST_SLICES = TEST_END - TEST_START


def z_cull_indices(views, SOD, SDD, det_rows, pixel_size, vol_z_min, vol_z_max,
                   margin_turns=1.0, pitch_mm_per_angle=None):
    source_z = views[:, 2]
    cone_half_z = 0.5 * det_rows * pixel_size * (SOD / SDD)
    if pitch_mm_per_angle is not None and len(views) > 1:
        angle_step = abs(np.mean(np.diff(np.arctan2(views[:10, 1], views[:10, 0]))))
        projs_per_turn = 2 * np.pi / max(angle_step, 1e-12)
        margin_z = margin_turns * projs_per_turn * abs(pitch_mm_per_angle)
    else:
        margin_z = cone_half_z
    z_lo = source_z - cone_half_z - margin_z
    z_hi = source_z + cone_half_z + margin_z
    mask = (z_hi >= vol_z_min) & (z_lo <= vol_z_max)
    return np.where(mask)[0]


def flicker_metric(vol):
    n_slices = vol.shape[2]
    metrics = []
    for i in range(n_slices - 1):
        s0 = vol[:, :, i]
        s1 = vol[:, :, i + 1]
        diff = np.mean(np.abs(s1 - s0))
        base = 0.5 * (np.mean(np.abs(s0)) + np.mean(np.abs(s1)))
        metrics.append(diff / max(base, 1e-12))
    return np.array(metrics)


def reconstruct_chunk(sino_work, angles_full, z_shift_full, views_full,
                      scan_geom_full, meta, chunk_z_min, chunk_z_max,
                      chunk_slices, td_smoothing=0.025, skip_td=False):
    """Reconstruct a z-chunk. skip_td=True applies no TD weighting."""
    det_rows_n = scan_geom_full["detector"]["detector rows"]
    det_cols_n = scan_geom_full["detector"]["detector cols"]
    psize = scan_geom_full["detector"]["detector psize"]

    keep_idx = z_cull_indices(
        views_full, scan_geom_full["SOD"], scan_geom_full["SDD"],
        det_rows_n, psize, chunk_z_min, chunk_z_max,
        margin_turns=1.0, pitch_mm_per_angle=meta["pitch_mm_per_angle"],
    )

    if len(keep_idx) == 0:
        return np.zeros((ROWS, COLS, chunk_slices), dtype=np.float32)

    sino_chunk = sino_work[keep_idx].copy()
    angles_chunk = angles_full[keep_idx].copy()
    views_chunk = views_full[keep_idx]

    scan_geom_chunk = copy.deepcopy(scan_geom_full)
    scan_geom_chunk["helix"]["angles_range"] = float(abs(angles_chunk[-1] - angles_chunk[0]))
    scan_geom_chunk["helix"]["angles_count"] = len(angles_chunk)

    proj_geom_chunk = astra.create_proj_geom("cone_vec", det_rows_n, det_cols_n, views_chunk)

    half_x = COLS * VOXEL_SIZE_XY * 0.5
    half_y = ROWS * VOXEL_SIZE_XY * 0.5
    vol_geom_chunk = astra.create_vol_geom(
        ROWS, COLS, chunk_slices,
        -half_x, half_x, -half_y, half_y, chunk_z_min, chunk_z_max,
    )

    conf_chunk = create_configuration(scan_geom_chunk, vol_geom_chunk)
    conf_chunk['source_pos'] = angles_chunk.astype(np.float32)
    conf_chunk['delta_s'] = float(np.mean(np.diff(angles_chunk)))
    conf_chunk['T-D smoothing'] = td_smoothing

    sino_f32 = np.asarray(sino_chunk, dtype=np.float32, order="C")
    filtered = filter_katsevich(
        sino_f32, conf_chunk,
        {"Diff": {"Print time": False}, "FwdRebin": {"Print time": False},
         "BackRebin": {"Print time": False}},
    )

    if skip_td:
        sino_final = filtered
    else:
        sino_final = sino_weight_td(filtered, conf_chunk, False)

    rec_chunk = backproject_cupy(sino_final, conf_chunk, vol_geom_chunk, proj_geom_chunk, tqdm_bar=False)
    return rec_chunk


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
print(f"    Direction: {'CW (decreasing)' if np.mean(np.diff(angles_raw[:20])) < 0 else 'CCW (increasing)'}")

if table_raw is not None:
    print(f"\n  Raw DICOM table positions:")
    print(f"    First 5: {table_raw[:5]}")
    print(f"    diff[0:4]: {np.diff(table_raw[:5])}")
    print(f"    Direction: {'decreasing (into gantry)' if np.mean(np.diff(table_raw[:20])) < 0 else 'increasing'}")

# After transforms
angles_neg = -angles_raw  # negate_angles
angles_neg_off = angles_neg - np.pi / 2  # angle_offset

z_shift_centered = table_raw - table_raw.mean() if table_raw is not None else np.zeros(len(angles_raw))

print(f"\n  After negate + offset:")
print(f"    First 5 angles: {angles_neg_off[:5]}")
print(f"    diff[0:4]: {np.diff(angles_neg_off[:5])}")
print(f"    Direction: {'CCW (increasing)' if np.mean(np.diff(angles_neg_off[:20])) > 0 else 'CW (decreasing)'}")

# Effective pitch: dz/ds
ds = np.diff(angles_neg_off[:100])
dz = np.diff(z_shift_centered[:100])
eff_pitch_per_rad = np.mean(dz / ds)
print(f"\n  Effective pitch (dz/ds):")
print(f"    dz/ds = {eff_pitch_per_rad:.6f} mm/rad")
print(f"    {'RIGHT-HANDED (z increases with s)' if eff_pitch_per_rad > 0 else 'LEFT-HANDED (z decreases with s)'}")

# What the code uses
pitch_signed = meta["pitch_mm_per_rad_signed"]
pitch_used = abs(pitch_signed)
print(f"\n  Code uses:")
print(f"    pitch_mm_per_rad_signed = {pitch_signed:.6f}")
print(f"    abs(pitch) = {pitch_used:.6f} (ALWAYS POSITIVE)")
print(f"    progress_per_turn = {pitch_used * 2 * np.pi:.4f} mm")

if (eff_pitch_per_rad > 0) != (pitch_used > 0):
    print(f"\n  [!] MISMATCH: actual helix is {'right' if eff_pitch_per_rad > 0 else 'left'}-handed")
    print(f"      but code assumes right-handed (positive pitch)")
elif eff_pitch_per_rad < 0:
    print(f"\n  [!] WARNING: actual helix is left-handed (dz/ds < 0)")
    print(f"      abs(pitch) forces right-handed TD window")
    print(f"      This could cause TD window to select the complementary region!")
else:
    print(f"\n  [OK] Helix is right-handed, matches code assumption")


# ══════════════════════════════════════════════════════════════════════
# TEST 2: TD WINDOW VISUALIZATION
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("TEST 2: TD window visualization")
print(f"{'='*60}")

# Setup full geometry
angles_full = -meta["angles_rad"].copy() - np.pi / 2
scan_geom_full = copy.deepcopy(meta["scan_geometry"])
z_shift_full = meta["table_positions_mm"] - meta["table_positions_mm"].mean() if meta["table_positions_mm"] is not None else np.zeros(len(angles_full))
sino_work = sino[:, ::-1, :].copy()

scan_geom_full["helix"]["pitch_mm_rad"] = float(abs(meta["pitch_mm_per_rad_signed"]))
scan_geom_full["helix"]["angles_range"] = float(abs(angles_full[-1] - angles_full[0]))

psize_cols = scan_geom_full["detector"].get("detector psize cols", scan_geom_full["detector"]["detector psize"])
psize_rows = scan_geom_full["detector"].get("detector psize rows", scan_geom_full["detector"]["detector psize"])

views_full = astra_helical_views(
    scan_geom_full["SOD"], scan_geom_full["SDD"],
    scan_geom_full["detector"]["detector psize"],
    angles_full, meta["pitch_mm_per_angle"],
    vertical_shifts=z_shift_full,
    pixel_size_col=psize_cols, pixel_size_row=psize_rows,
)

# Create a dummy configuration to get TD window params
total_half_z = SLICES * VOXEL_SIZE_Z * 0.5
chunk_z_min = -total_half_z + TEST_START * VOXEL_SIZE_Z
chunk_z_max = -total_half_z + TEST_END * VOXEL_SIZE_Z

det_rows_n = scan_geom_full["detector"]["detector rows"]
det_cols_n = scan_geom_full["detector"]["detector cols"]
psize = scan_geom_full["detector"]["detector psize"]

keep_idx = z_cull_indices(
    views_full, scan_geom_full["SOD"], scan_geom_full["SDD"],
    det_rows_n, psize, chunk_z_min, chunk_z_max,
    margin_turns=1.0, pitch_mm_per_angle=meta["pitch_mm_per_angle"],
)
angles_chunk = angles_full[keep_idx]
views_chunk = views_full[keep_idx]

scan_geom_chunk = copy.deepcopy(scan_geom_full)
scan_geom_chunk["helix"]["angles_range"] = float(abs(angles_chunk[-1] - angles_chunk[0]))
scan_geom_chunk["helix"]["angles_count"] = len(angles_chunk)

half_x = COLS * VOXEL_SIZE_XY * 0.5
half_y = ROWS * VOXEL_SIZE_XY * 0.5
vol_geom_chunk = astra.create_vol_geom(
    ROWS, COLS, TEST_SLICES,
    -half_x, half_x, -half_y, half_y, chunk_z_min, chunk_z_max,
)

conf = create_configuration(scan_geom_chunk, vol_geom_chunk)

# Print TD window info
row_coords = conf['row_coords'][:-1]
col_coords = conf['col_coords'][:-1]
print(f"\n  row_coords range: [{row_coords[0]:.4f}, {row_coords[-1]:.4f}] mm")
print(f"  col_coords range: [{col_coords[0]:.4f}, {col_coords[-1]:.4f}] mm")
print(f"  proj_row_mins center: {conf['proj_row_mins'][len(conf['proj_row_mins'])//2]:.4f} mm")
print(f"  proj_row_maxs center: {conf['proj_row_maxs'][len(conf['proj_row_maxs'])//2]:.4f} mm")
print(f"  Total row extent: {row_coords[-1] - row_coords[0]:.4f} mm")
print(f"  TD window height (center col): {conf['proj_row_maxs'][len(conf['proj_row_maxs'])//2] - conf['proj_row_mins'][len(conf['proj_row_mins'])//2]:.4f} mm")

# Ratio of window to detector
td_height = conf['proj_row_maxs'][len(conf['proj_row_maxs'])//2] - conf['proj_row_mins'][len(conf['proj_row_mins'])//2]
det_height = row_coords[-1] - row_coords[0]
ratio = td_height / det_height
print(f"  TD window / detector height: {ratio:.4f}")
if ratio > 1.0:
    print(f"  [!] TD window is LARGER than detector — all data passes through")
elif ratio < 0.5:
    print(f"  [!] TD window covers less than half the detector")

# Visualize TD mask
W, U = np.meshgrid(row_coords, col_coords, indexing='ij')
w_bottom = np.reshape(conf['proj_row_mins'][:-1], (1, -1))
w_top = np.reshape(conf['proj_row_maxs'][:-1], (1, -1))

a = conf['T-D smoothing']
dw = det_rows_n * conf['pixel_height']

mask = np.zeros_like(W)
W_top_high = np.repeat(w_top + a * dw, W.shape[0], axis=0)
W_top_low = np.repeat(w_top - a * dw, W.shape[0], axis=0)
W_bottom_high = np.repeat(w_bottom + a * dw, W.shape[0], axis=0)
W_bottom_low = np.repeat(w_bottom - a * dw, W.shape[0], axis=0)

mask[(W_top_low < W) & (W < W_top_high)] = ((W_top_high - W) / (2 * a * dw))[(W_top_low < W) & (W < W_top_high)]
mask[(W_bottom_high < W) & (W < W_top_low)] = 1
mask[(W_bottom_low < W) & (W < W_bottom_high)] = ((W - W_bottom_low) / (2 * a * dw))[(W_bottom_low < W) & (W < W_bottom_high)]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# TD mask
im0 = axes[0].imshow(mask, cmap='hot', aspect='auto',
                      extent=[col_coords[0], col_coords[-1], row_coords[-1], row_coords[0]])
axes[0].set_title(f"TD mask (a={a})")
axes[0].set_xlabel("u (col coords, mm)")
axes[0].set_ylabel("w (row coords, mm)")
axes[0].axhline(0, color='cyan', linestyle='--', alpha=0.5, label='detector center')
plt.colorbar(im0, ax=axes[0])

# TD bounds overlay on sinogram
mid_proj = len(keep_idx) // 2
sino_frame = sino_work[keep_idx[mid_proj]]
im1 = axes[1].imshow(sino_frame, cmap='gray', aspect='auto',
                      extent=[0, det_cols_n, det_rows_n, 0])
# Overlay TD bounds (convert from mm to row/col indices)
center_row = (det_rows_n - 1) / 2.0
center_col = (det_cols_n - 1) / 2.0
for bound_arr, color, label in [(conf['proj_row_mins'], 'red', 'w_bottom'),
                                  (conf['proj_row_maxs'], 'lime', 'w_top')]:
    bound_row_idx = bound_arr[:-1] / conf['pixel_height'] + center_row
    axes[1].plot(np.arange(det_cols_n), bound_row_idx, color=color, linewidth=1.5, label=label)
axes[1].set_title(f"Sinogram frame #{mid_proj} + TD bounds")
axes[1].set_xlabel("Column index")
axes[1].set_ylabel("Row index")
axes[1].legend(fontsize=8)

# Actual source z trajectory vs angle
axes[2].plot(angles_full[:2000], z_shift_full[:2000], 'b-', linewidth=0.5)
axes[2].set_xlabel("Angle (rad, after negate+offset)")
axes[2].set_ylabel("Source z (mm)")
axes[2].set_title(f"Helix trajectory (first 2000 projs)\ndz/ds = {eff_pitch_per_rad:.4f} mm/rad")
axes[2].axhline(0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "helix_direction_diagnostic.png")
plt.savefig(out_path, dpi=150)
print(f"\n  Saved -> {out_path}")


# ══════════════════════════════════════════════════════════════════════
# TEST 3: FLICKER WITH vs WITHOUT TD WEIGHTING
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("TEST 3: Flicker WITH vs WITHOUT TD weighting")
print(f"{'='*60}")

results = {}
for label, skip_td in [("with_TD", False), ("without_TD", True)]:
    print(f"\n  [{label}] Reconstructing slices {TEST_START}-{TEST_END}...")
    t0 = time()
    rec = reconstruct_chunk(
        sino_work, angles_full, z_shift_full, views_full,
        scan_geom_full, meta, chunk_z_min, chunk_z_max,
        TEST_SLICES, td_smoothing=0.025, skip_td=skip_td,
    )
    elapsed = time() - t0
    fm = flicker_metric(rec)
    print(f"    Done in {elapsed:.1f}s, range=[{rec.min():.4f}, {rec.max():.4f}]")
    print(f"    Flicker mean: {fm.mean():.6f}, max: {fm.max():.6f}")
    results[label] = {"rec": rec, "flicker": fm}

fm_with = results["with_TD"]["flicker"].mean()
fm_without = results["without_TD"]["flicker"].mean()
print(f"\n  Comparison:")
print(f"    With TD:    {fm_with:.6f}")
print(f"    Without TD: {fm_without:.6f}")
if fm_with > fm_without * 1.2:
    print(f"    [!] TD weighting INCREASES flicker by {fm_with/fm_without:.2f}x")
    print(f"    → TD window is likely misconfigured (wrong helix direction?)")
elif fm_without > fm_with * 1.2:
    print(f"    [OK] TD weighting REDUCES flicker by {fm_without/fm_with:.2f}x (expected)")
else:
    print(f"    [~] TD weighting has minimal effect on flicker")


# ══════════════════════════════════════════════════════════════════════
# TEST 4: NEGATE Z-SHIFT TO REVERSE HELIX DIRECTION
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("TEST 4: Reverse helix direction (negate z_shift)")
print(f"{'='*60}")
print("  If helix is left-handed, negating z_shift makes it right-handed,")
print("  matching the positive-pitch assumption in the filter pipeline.")

# Rebuild views with negated z_shift
z_shift_negated = -z_shift_full
views_neg = astra_helical_views(
    scan_geom_full["SOD"], scan_geom_full["SDD"],
    scan_geom_full["detector"]["detector psize"],
    angles_full, meta["pitch_mm_per_angle"],
    vertical_shifts=z_shift_negated,
    pixel_size_col=psize_cols, pixel_size_row=psize_rows,
)

# Check new helix direction
ds_check = np.diff(angles_full[:100])
dz_check = np.diff(z_shift_negated[:100])
eff_pitch_neg = np.mean(dz_check / ds_check)
print(f"\n  After negate z_shift: dz/ds = {eff_pitch_neg:.6f} mm/rad")
print(f"  {'RIGHT-HANDED' if eff_pitch_neg > 0 else 'LEFT-HANDED'}")

print(f"\n  Reconstructing with negated z_shift...")
t0 = time()
rec_neg = reconstruct_chunk(
    sino_work, angles_full, z_shift_negated, views_neg,
    scan_geom_full, meta, chunk_z_min, chunk_z_max,
    TEST_SLICES, td_smoothing=0.025, skip_td=False,
)
elapsed = time() - t0
fm_neg = flicker_metric(rec_neg)
print(f"  Done in {elapsed:.1f}s, range=[{rec_neg.min():.4f}, {rec_neg.max():.4f}]")
print(f"  Flicker mean: {fm_neg.mean():.6f}, max: {fm_neg.max():.6f}")

print(f"\n  Comparison:")
print(f"    Original z_shift + TD:   {fm_with:.6f}")
print(f"    Negated z_shift + TD:    {fm_neg.mean():.6f}")
if fm_neg.mean() < fm_with * 0.5:
    print(f"    [!] Negating z_shift HALVES flicker!")
    print(f"    → Helix direction mismatch WAS the root cause")
else:
    print(f"    [~] Negating z_shift doesn't significantly reduce flicker")


# ══════════════════════════════════════════════════════════════════════
# TEST 5: NO FLIP_ROWS (to check if flip is truly needed)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("TEST 5: Without flip_rows (original DICOM row order)")
print(f"{'='*60}")

sino_noflip = sino.copy()  # no flip_rows
# Same geometry
print(f"  Reconstructing WITHOUT flip_rows...")
t0 = time()
rec_noflip = reconstruct_chunk(
    sino_noflip, angles_full, z_shift_full, views_full,
    scan_geom_full, meta, chunk_z_min, chunk_z_max,
    TEST_SLICES, td_smoothing=0.025, skip_td=False,
)
elapsed = time() - t0
fm_noflip = flicker_metric(rec_noflip)
print(f"  Done in {elapsed:.1f}s, range=[{rec_noflip.min():.4f}, {rec_noflip.max():.4f}]")
print(f"  Flicker mean: {fm_noflip.mean():.6f}")

print(f"\n  Comparison:")
print(f"    With flip_rows:    {fm_with:.6f}")
print(f"    Without flip_rows: {fm_noflip.mean():.6f}")


# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"  {'Test':<35} {'Flicker':>10}")
print(f"  {'-'*45}")
print(f"  {'Baseline (with TD, flip_rows)':<35} {fm_with:10.6f}")
print(f"  {'Without TD weighting':<35} {fm_without:10.6f}")
print(f"  {'Negated z_shift + TD':<35} {fm_neg.mean():10.6f}")
print(f"  {'No flip_rows + TD':<35} {fm_noflip.mean():10.6f}")
print(f"  {'GT baseline (hardcoded)':<35} {'~0.040':>10}")

# Save comparison figure
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
titles = ["Baseline (flip+TD)", "No TD", "Negated z_shift", "No flip_rows"]
recs = [results["with_TD"]["rec"], results["without_TD"]["rec"], rec_neg, rec_noflip]
fms = [results["with_TD"]["flicker"], results["without_TD"]["flicker"], fm_neg, fm_noflip]

for col, (title, rec, fm) in enumerate(zip(titles, recs, fms)):
    mid = rec.shape[2] // 2
    vmin, vmax = np.percentile(rec[:, :, mid], [1, 99])
    axes[0, col].imshow(rec[:, :, mid], cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, col].set_title(f"{title}\nslice {TEST_START + mid}")
    axes[0, col].axis('off')

    if rec.shape[2] > mid + 1:
        diff = rec[:, :, mid + 1] - rec[:, :, mid]
        vd = max(np.percentile(np.abs(diff), 99), 1e-10)
        axes[1, col].imshow(diff, cmap='RdBu', vmin=-vd, vmax=vd)
        axes[1, col].set_title(f"Diff(s{mid+1}-s{mid}), flicker={fm.mean():.5f}")
        axes[1, col].axis('off')

plt.suptitle("Flicker Diagnostic: Helix Direction Analysis", fontsize=14)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "helix_direction_comparison.png")
plt.savefig(out_path, dpi=150)
print(f"\n  Saved -> {out_path}")

print("\nDone.")
