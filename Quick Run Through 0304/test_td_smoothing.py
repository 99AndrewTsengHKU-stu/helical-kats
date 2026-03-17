"""
Quick test: compare T-D smoothing values on a small z-slab (10 slices)
around the volume center to see the effect on cross-slice flickering.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from backproject_safe import ensure_astra_cuda_init
ensure_astra_cuda_init()

import copy
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

# ── Settings ────────────────────────────────────────────────────────────
DICOM_DIR = r"D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD"
ROWS      = 512
COLS      = 512
VOXEL_SIZE = 0.664

# Test slab: 10 slices around center
TEST_SLICES = 10
CENTER_SLICE = 280   # center of full 560-slice volume
Z_START = CENTER_SLICE - TEST_SLICES // 2   # 275
Z_END   = CENTER_SLICE + TEST_SLICES // 2   # 285

TOTAL_SLICES = 560

# Smoothing values to compare
TD_SMOOTHING_VALUES = [0.025, 0.05, 0.1, 0.2]

# ── Load DICOM ──────────────────────────────────────────────────────────
print(f"Loading DICOM projections from {DICOM_DIR} ...")
t0 = time()
sino, meta = load_dicom_projections(DICOM_DIR)
print(f"Loaded {sino.shape[0]} projections in {time()-t0:.1f}s")

# ── Prepare geometry (same as run_L067.py) ──────────────────────────────
angles_full = -meta["angles_rad"].copy()  # negate_angles
scan_geom = copy.deepcopy(meta["scan_geometry"])

if meta["table_positions_mm"] is not None:
    z_shift_full = meta["table_positions_mm"] - meta["table_positions_mm"].mean()
else:
    z_shift_full = np.zeros(len(angles_full), dtype=np.float32)

sino_work = sino[:, ::-1, :].copy()  # flip_rows

pitch_used = abs(meta["pitch_mm_per_rad_signed"])
scan_geom["helix"]["pitch_mm_rad"] = float(pitch_used)
scan_geom["helix"]["angles_range"] = float(abs(angles_full[-1] - angles_full[0]))

views_full = astra_helical_views(
    scan_geom["SOD"], scan_geom["SDD"],
    scan_geom["detector"]["detector psize"],
    angles_full, meta["pitch_mm_per_angle"],
    vertical_shifts=z_shift_full,
)

det_rows_n = scan_geom["detector"]["detector rows"]
det_cols_n = scan_geom["detector"]["detector cols"]
psize = scan_geom["detector"]["detector psize"]

# ── Z-cull for this slab ────────────────────────────────────────────────
total_half_z = TOTAL_SLICES * VOXEL_SIZE * 0.5
chunk_z_min = -total_half_z + Z_START * VOXEL_SIZE
chunk_z_max = -total_half_z + Z_END * VOXEL_SIZE

def z_cull_indices(views, SOD, SDD, det_rows, pixel_size, vol_z_min, vol_z_max, margin_turns=1.0, pitch_mm_per_angle=None):
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

keep_idx = z_cull_indices(
    views_full, scan_geom["SOD"], scan_geom["SDD"],
    det_rows_n, psize, chunk_z_min, chunk_z_max,
    margin_turns=1.0, pitch_mm_per_angle=meta["pitch_mm_per_angle"],
)
print(f"Z-cull: {len(angles_full)} -> {len(keep_idx)} projections")

sino_chunk = sino_work[keep_idx].copy()
angles_chunk = angles_full[keep_idx].copy()
views_chunk = views_full[keep_idx]

scan_geom_chunk = copy.deepcopy(scan_geom)
scan_geom_chunk["helix"]["angles_range"] = float(abs(angles_chunk[-1] - angles_chunk[0]))
scan_geom_chunk["helix"]["angles_count"] = len(angles_chunk)

proj_geom = astra.create_proj_geom("cone_vec", det_rows_n, det_cols_n, views_chunk)

half_x = COLS * VOXEL_SIZE * 0.5
half_y = ROWS * VOXEL_SIZE * 0.5
vol_geom = astra.create_vol_geom(
    ROWS, COLS, TEST_SLICES,
    -half_x, half_x, -half_y, half_y, chunk_z_min, chunk_z_max,
)

# ── Test each smoothing value ───────────────────────────────────────────
results = {}

for td_val in TD_SMOOTHING_VALUES:
    print(f"\n{'='*50}")
    print(f"T-D smoothing = {td_val}")
    print(f"{'='*50}")

    conf = create_configuration(scan_geom_chunk, vol_geom)
    conf['source_pos'] = angles_chunk.astype(np.float32)
    conf['delta_s'] = float(np.mean(np.diff(angles_chunk)))

    # Override T-D smoothing
    conf['T-D smoothing'] = td_val

    # Filter
    t0 = time()
    sino_f32 = np.asarray(sino_chunk, dtype=np.float32, order="C")
    filtered = filter_katsevich(
        sino_f32, conf,
        {"Diff": {"Print time": False}, "FwdRebin": {"Print time": False}, "BackRebin": {"Print time": False}},
    )
    sino_td = sino_weight_td(filtered, conf, False)
    t_filt = time() - t0

    # Backproject
    t0 = time()
    rec = backproject_cupy(sino_td, conf, vol_geom, proj_geom, tqdm_bar=False)
    t_bp = time() - t0

    # CCW 90° rotation
    rec = np.rot90(rec, k=1, axes=(0, 1))

    print(f"  Filter: {t_filt:.1f}s  BP: {t_bp:.1f}s")
    print(f"  Range: [{rec.min():.5f}, {rec.max():.5f}]")

    # Measure per-slice mean in center ROI
    cx, cy = ROWS // 2, COLS // 2
    roi_r = 50
    means = []
    for s in range(TEST_SLICES):
        roi = rec[cx-roi_r:cx+roi_r, cy-roi_r:cy+roi_r, s]
        means.append(roi.mean())
    means = np.array(means)
    flicker = np.std(np.diff(means))
    print(f"  Center ROI slice-means: {means.min():.5f} - {means.max():.5f}")
    print(f"  Flicker (std of diff): {flicker:.6f}")

    results[td_val] = {
        "rec": rec,
        "means": means,
        "flicker": flicker,
    }

# ── Visualize ───────────────────────────────────────────────────────────
n_vals = len(TD_SMOOTHING_VALUES)
fig, axes = plt.subplots(3, n_vals, figsize=(5 * n_vals, 14))

for i, td_val in enumerate(TD_SMOOTHING_VALUES):
    rec = results[td_val]["rec"]
    means = results[td_val]["means"]
    flicker = results[td_val]["flicker"]

    # Row 1: middle slice
    mid = TEST_SLICES // 2
    vmin, vmax = np.percentile(rec[:, :, mid], [1, 99])
    axes[0, i].imshow(rec[:, :, mid], cmap="gray", vmin=vmin, vmax=vmax)
    axes[0, i].set_title(f"TD={td_val}\nSlice {Z_START+mid}")
    axes[0, i].axis("off")

    # Row 2: difference between consecutive slices
    diff = rec[:, :, mid] - rec[:, :, mid-1]
    dlim = max(abs(diff.min()), abs(diff.max()), 1e-6)
    axes[1, i].imshow(diff, cmap="RdBu_r", vmin=-dlim, vmax=dlim)
    axes[1, i].set_title(f"Diff(s{mid}-s{mid-1})\nmax={dlim:.5f}")
    axes[1, i].axis("off")

    # Row 3: per-slice center ROI mean
    axes[2, i].plot(range(Z_START, Z_START + TEST_SLICES), means, "o-")
    axes[2, i].set_title(f"Center ROI mean\nflicker={flicker:.6f}")
    axes[2, i].set_xlabel("Slice")
    axes[2, i].set_ylabel("Mean")

fig.suptitle("T-D Smoothing Comparison (10 slices around center)", fontsize=14)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "td_smoothing_compare.png")
plt.savefig(out_path, dpi=150)
print(f"\nSaved -> {out_path}")
plt.close()

# Print summary
print(f"\n{'='*50}")
print("SUMMARY")
print(f"{'='*50}")
for td_val in TD_SMOOTHING_VALUES:
    f = results[td_val]["flicker"]
    print(f"  TD={td_val:5.3f}  flicker={f:.6f}")
print(f"\nBest: TD={min(results, key=lambda k: results[k]['flicker'])}")
