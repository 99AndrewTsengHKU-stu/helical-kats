"""
Phase 5D: Perfect helix vs DICOM table positions.

The filter pipeline assumes a perfect helix (z = pitch * angle), but
backprojection uses actual DICOM table positions via vertical_shifts.
If the table motion is non-uniform, these two z-models disagree, causing
each projection's filtered data to be backprojected at a slightly wrong z.

This test:
  A) Baseline: reconstruct with DICOM table positions (as-is)
  B) Perfect helix: replace vertical_shifts with angle-derived linear z
  C) Also visualize the table position deviation from perfect helix
"""
import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Quick Run Through 0304"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from backproject_safe import ensure_astra_cuda_init
ensure_astra_cuda_init()

import numpy as np
import matplotlib
matplotlib.use("Agg")
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

TEST_START = 270
TEST_END = 290
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
    n = vol.shape[2]
    m = []
    for i in range(n - 1):
        s0 = vol[:, :, i].astype(np.float64)
        s1 = vol[:, :, i + 1].astype(np.float64)
        denom = 0.5 * (np.mean(np.abs(s0)) + np.mean(np.abs(s1)))
        m.append(np.mean(np.abs(s1 - s0)) / max(denom, 1e-12))
    return np.array(m)


def reconstruct_chunk(sino_chunk, angles_chunk, views_chunk,
                      scan_geom_chunk, chunk_z_min, chunk_z_max,
                      chunk_slices, label=""):
    det_rows_n = scan_geom_chunk["detector"]["detector rows"]
    det_cols_n = scan_geom_chunk["detector"]["detector cols"]

    half_x = COLS * VOXEL_SIZE_XY * 0.5
    half_y = ROWS * VOXEL_SIZE_XY * 0.5
    vol_geom = astra.create_vol_geom(
        ROWS, COLS, chunk_slices,
        -half_x, half_x, -half_y, half_y, chunk_z_min, chunk_z_max,
    )
    proj_geom = astra.create_proj_geom("cone_vec", det_rows_n, det_cols_n, views_chunk)

    conf = create_configuration(scan_geom_chunk, vol_geom)
    conf['source_pos'] = angles_chunk.astype(np.float32)
    conf['delta_s'] = float(np.mean(np.diff(angles_chunk)))

    sino_f32 = np.asarray(sino_chunk, dtype=np.float32, order="C")
    filtered = filter_katsevich(
        sino_f32, conf,
        {"Diff": {"Print time": False}, "FwdRebin": {"Print time": False},
         "BackRebin": {"Print time": False}},
    )
    sino_td = sino_weight_td(filtered, conf, False)
    rec = backproject_cupy(sino_td, conf, vol_geom, proj_geom, tqdm_bar=False)
    return rec


# ══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════
print("Loading DICOM...", flush=True)
t0 = time()
sino, meta = load_dicom_projections(DICOM_DIR)
print(f"Loaded {sino.shape[0]} projections in {time()-t0:.1f}s", flush=True)

# Standard transforms
angles_full = -meta["angles_rad"].copy() - np.pi / 2
z_shift_dicom = meta["table_positions_mm"] - meta["table_positions_mm"].mean()
sino_work = sino[:, ::-1, :].copy()  # flip_rows

scan_geom_full = copy.deepcopy(meta["scan_geometry"])
pitch_abs = float(abs(meta["pitch_mm_per_rad_signed"]))
scan_geom_full["helix"]["pitch_mm_rad"] = pitch_abs
scan_geom_full["helix"]["angles_range"] = float(abs(angles_full[-1] - angles_full[0]))

psize_cols = scan_geom_full["detector"].get("detector psize cols", scan_geom_full["detector"]["detector psize"])
psize_rows = scan_geom_full["detector"].get("detector psize rows", scan_geom_full["detector"]["detector psize"])
det_rows_n = scan_geom_full["detector"]["detector rows"]
psize = scan_geom_full["detector"]["detector psize"]

# ══════════════════════════════════════════════════════════════════════
# ANALYZE TABLE POSITION DEVIATION
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("Table position analysis: DICOM vs perfect helix")
print(f"{'='*60}")

# Perfect helix: z = angle * pitch_per_radian, centered same as DICOM
z_perfect = angles_full * pitch_abs
z_perfect = z_perfect - z_perfect.mean()  # center at 0 like DICOM

deviation = z_shift_dicom - z_perfect
print(f"  DICOM z-shift range: [{z_shift_dicom.min():.3f}, {z_shift_dicom.max():.3f}] mm")
print(f"  Perfect helix range: [{z_perfect.min():.3f}, {z_perfect.max():.3f}] mm")
print(f"  Deviation: mean={deviation.mean():.6f}, std={deviation.std():.6f}, "
      f"max|dev|={np.abs(deviation).max():.6f} mm")
print(f"  Max deviation / voxel_z = {np.abs(deviation).max() / VOXEL_SIZE_Z:.4f} voxels")

# ══════════════════════════════════════════════════════════════════════
# BUILD VIEWS: DICOM (baseline) and PERFECT HELIX
# ══════════════════════════════════════════════════════════════════════
views_dicom = astra_helical_views(
    scan_geom_full["SOD"], scan_geom_full["SDD"],
    scan_geom_full["detector"]["detector psize"],
    angles_full, meta["pitch_mm_per_angle"],
    vertical_shifts=z_shift_dicom,
    pixel_size_col=psize_cols, pixel_size_row=psize_rows,
)

views_perfect = astra_helical_views(
    scan_geom_full["SOD"], scan_geom_full["SDD"],
    scan_geom_full["detector"]["detector psize"],
    angles_full, meta["pitch_mm_per_angle"],
    vertical_shifts=z_perfect,
    pixel_size_col=psize_cols, pixel_size_row=psize_rows,
)

# Chunk setup
total_half_z = SLICES * VOXEL_SIZE_Z * 0.5
chunk_z_min = -total_half_z + TEST_START * VOXEL_SIZE_Z
chunk_z_max = -total_half_z + TEST_END * VOXEL_SIZE_Z

# Z-cull for both (use DICOM views for culling both since they're nearly identical)
keep_idx = z_cull_indices(
    views_dicom, scan_geom_full["SOD"], scan_geom_full["SDD"],
    det_rows_n, psize, chunk_z_min, chunk_z_max,
    margin_turns=1.0, pitch_mm_per_angle=meta["pitch_mm_per_angle"],
)

sino_chunk = sino_work[keep_idx].copy()
angles_chunk = angles_full[keep_idx].copy()

scan_geom_chunk = copy.deepcopy(scan_geom_full)
scan_geom_chunk["helix"]["angles_range"] = float(abs(angles_chunk[-1] - angles_chunk[0]))
scan_geom_chunk["helix"]["angles_count"] = len(angles_chunk)

print(f"\nChunk: slices [{TEST_START}:{TEST_END}], z=[{chunk_z_min:.1f}, {chunk_z_max:.1f}] mm")
print(f"  {len(keep_idx)} projections after z-cull", flush=True)

# ══════════════════════════════════════════════════════════════════════
# TEST A: BASELINE (DICOM table positions)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("TEST A: Baseline (DICOM table positions)")
print(f"{'='*60}", flush=True)

views_chunk_dicom = views_dicom[keep_idx]
t0 = time()
rec_A = reconstruct_chunk(
    sino_chunk, angles_chunk, views_chunk_dicom,
    scan_geom_chunk, chunk_z_min, chunk_z_max, TEST_SLICES,
)
elapsed = time() - t0
fm_A = flicker_metric(rec_A)
print(f"  Done in {elapsed:.1f}s, range=[{rec_A.min():.4f}, {rec_A.max():.4f}]")
print(f"  Flicker mean: {fm_A.mean():.6f}, max: {fm_A.max():.6f}", flush=True)

# ══════════════════════════════════════════════════════════════════════
# TEST B: PERFECT HELIX (angle-derived z positions)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("TEST B: Perfect helix (angle * pitch)")
print(f"{'='*60}", flush=True)

views_chunk_perfect = views_perfect[keep_idx]
t0 = time()
rec_B = reconstruct_chunk(
    sino_chunk, angles_chunk, views_chunk_perfect,
    scan_geom_chunk, chunk_z_min, chunk_z_max, TEST_SLICES,
)
elapsed = time() - t0
fm_B = flicker_metric(rec_B)
print(f"  Done in {elapsed:.1f}s, range=[{rec_B.min():.4f}, {rec_B.max():.4f}]")
print(f"  Flicker mean: {fm_B.mean():.6f}, max: {fm_B.max():.6f}", flush=True)

# ══════════════════════════════════════════════════════════════════════
# TEST C: z-cull margin=2.0 (more projections)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("TEST C: Increased z-cull margin (margin_turns=2.0)")
print(f"{'='*60}", flush=True)

keep_idx_wide = z_cull_indices(
    views_dicom, scan_geom_full["SOD"], scan_geom_full["SDD"],
    det_rows_n, psize, chunk_z_min, chunk_z_max,
    margin_turns=2.0, pitch_mm_per_angle=meta["pitch_mm_per_angle"],
)
print(f"  margin=1.0: {len(keep_idx)} projs, margin=2.0: {len(keep_idx_wide)} projs")

sino_chunk_wide = sino_work[keep_idx_wide].copy()
angles_chunk_wide = angles_full[keep_idx_wide].copy()
views_chunk_wide = views_dicom[keep_idx_wide]

scan_geom_wide = copy.deepcopy(scan_geom_full)
scan_geom_wide["helix"]["angles_range"] = float(abs(angles_chunk_wide[-1] - angles_chunk_wide[0]))
scan_geom_wide["helix"]["angles_count"] = len(angles_chunk_wide)

t0 = time()
rec_C = reconstruct_chunk(
    sino_chunk_wide, angles_chunk_wide, views_chunk_wide,
    scan_geom_wide, chunk_z_min, chunk_z_max, TEST_SLICES,
)
elapsed = time() - t0
fm_C = flicker_metric(rec_C)
print(f"  Done in {elapsed:.1f}s, range=[{rec_C.min():.4f}, {rec_C.max():.4f}]")
print(f"  Flicker mean: {fm_C.mean():.6f}, max: {fm_C.max():.6f}", flush=True)

# ══════════════════════════════════════════════════════════════════════
# SUMMARY & PLOT
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"  {'Test':<45} {'Flicker':>10}")
print(f"  {'-'*55}")
print(f"  {'A: Baseline (DICOM table positions)':<45} {fm_A.mean():.6f}")
print(f"  {'B: Perfect helix (angle * pitch)':<45} {fm_B.mean():.6f}")
print(f"  {'C: Wider z-cull (margin=2.0)':<45} {fm_C.mean():.6f}")
print(f"  {'GT baseline':<45} {'~0.040':>10}")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Row 1: center slice from each test
center = TEST_SLICES // 2
vmin = min(rec_A[:,:,center].min(), rec_B[:,:,center].min())
vmax = max(rec_A[:,:,center].max(), rec_B[:,:,center].max())
for ax, rec, label, fm in [
    (axes[0, 0], rec_A, f"A: DICOM table (fl={fm_A.mean():.4f})", fm_A),
    (axes[0, 1], rec_B, f"B: Perfect helix (fl={fm_B.mean():.4f})", fm_B),
    (axes[0, 2], rec_C, f"C: margin=2.0 (fl={fm_C.mean():.4f})", fm_C),
]:
    ax.imshow(rec[:, :, center], cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(label, fontsize=10)
    ax.axis('off')

# Row 2: slice diffs + table deviation
for ax, rec, label in [
    (axes[1, 0], rec_A, "A diff"),
    (axes[1, 1], rec_B, "B diff"),
    (axes[1, 2], rec_C, "C diff"),
]:
    d = rec[:, :, center].astype(np.float64) - rec[:, :, center - 1].astype(np.float64)
    vlim = max(abs(d.min()), abs(d.max()), 1e-6)
    ax.imshow(d, cmap='RdBu', vmin=-vlim, vmax=vlim)
    ax.set_title(f"{label}: slice[{center}]-slice[{center-1}]", fontsize=9)
    ax.axis('off')

plt.suptitle("Phase 5D: Perfect Helix vs DICOM Table Positions", fontsize=13)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "perfect_helix_comparison.png")
plt.savefig(out_path, dpi=150)
print(f"\nSaved -> {out_path}")

# Also plot table deviation
fig2, ax2 = plt.subplots(1, 1, figsize=(14, 4))
ax2.plot(deviation, linewidth=0.5)
ax2.set_xlabel("Projection index")
ax2.set_ylabel("DICOM - perfect helix (mm)")
ax2.set_title(f"Table position deviation | std={deviation.std():.4f}mm, max={np.abs(deviation).max():.4f}mm")
ax2.axhline(0, color='r', linestyle='--', linewidth=0.5)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
out_path2 = os.path.join(OUT_DIR, "table_deviation.png")
plt.savefig(out_path2, dpi=150)
print(f"Saved -> {out_path2}")

print("\nDone.", flush=True)
