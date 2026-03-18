"""
Phase 2: Diagnose flickering and test T-D smoothing values.

Reconstructs 10 consecutive slices (276-285, near center) with different
T-D smoothing values to find the optimal setting that minimizes flicker.

Flicker metric: mean absolute difference between adjacent slices,
normalized by mean intensity. Lower = less flicker.
"""
import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Quick Run Through 0304"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from backproject_safe import ensure_astra_cuda_init
ensure_astra_cuda_init()

import numpy as np
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
VOXEL_SIZE_XY = 0.664   # mm
VOXEL_SIZE_Z  = 0.8     # mm — from GT DICOM

# Test slices: 10 consecutive near center
TEST_START = 276
TEST_END   = 286
TEST_SLICES = TEST_END - TEST_START

# T-D smoothing values to test
TD_SMOOTHING_VALUES = [0.025, 0.05, 0.075, 0.10]


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


def reconstruct_chunk(sino_work, angles_full, z_shift_full, views_full,
                      scan_geom_full, meta, chunk_z_min, chunk_z_max,
                      chunk_slices, td_smoothing=0.025):
    """Reconstruct a z-chunk with given T-D smoothing."""
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
    conf_chunk['T-D smoothing'] = td_smoothing  # override!

    sino_f32 = np.asarray(sino_chunk, dtype=np.float32, order="C")
    filtered = filter_katsevich(
        sino_f32, conf_chunk,
        {"Diff": {"Print time": False}, "FwdRebin": {"Print time": False},
         "BackRebin": {"Print time": False}},
    )
    sino_td = sino_weight_td(filtered, conf_chunk, False)

    rec_chunk = backproject_cupy(sino_td, conf_chunk, vol_geom_chunk, proj_geom_chunk, tqdm_bar=False)
    return rec_chunk


def flicker_metric(vol):
    """
    Compute flicker metric: mean of abs(slice[i+1] - slice[i]) / mean(abs(slice[i])).
    Returns per-pair values and overall mean.
    """
    n_slices = vol.shape[2]
    metrics = []
    for i in range(n_slices - 1):
        s0 = vol[:, :, i]
        s1 = vol[:, :, i + 1]
        diff = np.mean(np.abs(s1 - s0))
        base = 0.5 * (np.mean(np.abs(s0)) + np.mean(np.abs(s1)))
        metrics.append(diff / max(base, 1e-12))
    return np.array(metrics)


# ── Main ──────────────────────────────────────────────────────────────
print("Loading DICOM...")
t0 = time()
sino, meta = load_dicom_projections(DICOM_DIR)
print(f"Loaded {sino.shape[0]} projections in {time()-t0:.1f}s")

# Prepare geometry (same as run_L067.py)
total_half_z = SLICES * VOXEL_SIZE_Z * 0.5

angles_full = meta["angles_rad"].copy()
scan_geom_full = copy.deepcopy(meta["scan_geometry"])

if meta["table_positions_mm"] is not None:
    z_shift_full = meta["table_positions_mm"] - meta["table_positions_mm"].mean()
else:
    z_shift_full = np.zeros(len(angles_full), dtype=np.float32)

# Apply angle transforms: negate + offset
angles_full = -angles_full - np.pi/2

sino_work = sino[:, ::-1, :].copy()  # flip_rows

pitch_signed = meta["pitch_mm_per_rad_signed"]
pitch_used = abs(pitch_signed)
scan_geom_full["helix"]["pitch_mm_rad"] = float(pitch_used)
scan_geom_full["helix"]["angles_range"] = float(abs(angles_full[-1] - angles_full[0]))

psize_cols = scan_geom_full["detector"].get("detector psize cols",
                scan_geom_full["detector"]["detector psize"])
psize_rows = scan_geom_full["detector"].get("detector psize rows",
                scan_geom_full["detector"]["detector psize"])

views_full = astra_helical_views(
    scan_geom_full["SOD"],
    scan_geom_full["SDD"],
    scan_geom_full["detector"]["detector psize"],
    angles_full,
    meta["pitch_mm_per_angle"],
    vertical_shifts=z_shift_full,
    pixel_size_col=psize_cols,
    pixel_size_row=psize_rows,
)

# Chunk z-range for test slices
chunk_z_min = -total_half_z + TEST_START * VOXEL_SIZE_Z
chunk_z_max = -total_half_z + TEST_END * VOXEL_SIZE_Z

print(f"\nTest chunk: slices [{TEST_START}:{TEST_END}], z=[{chunk_z_min:.1f}, {chunk_z_max:.1f}] mm")
print(f"VOXEL_SIZE_Z = {VOXEL_SIZE_Z} mm")

# ── Test each T-D smoothing value ─────────────────────────────────────
results = {}
for td_val in TD_SMOOTHING_VALUES:
    print(f"\n{'='*60}")
    print(f"T-D smoothing = {td_val}")
    print(f"{'='*60}")
    t0 = time()
    rec = reconstruct_chunk(
        sino_work, angles_full, z_shift_full, views_full,
        scan_geom_full, meta, chunk_z_min, chunk_z_max,
        TEST_SLICES, td_smoothing=td_val,
    )
    elapsed = time() - t0
    print(f"  Done in {elapsed:.1f}s, range=[{rec.min():.4f}, {rec.max():.4f}]")

    fm = flicker_metric(rec)
    print(f"  Flicker metric (per-pair): {fm}")
    print(f"  Flicker metric (mean): {fm.mean():.6f}")
    print(f"  Flicker metric (max):  {fm.max():.6f}")

    results[td_val] = {"rec": rec, "flicker": fm, "time": elapsed}

# ── Summary ───────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY: T-D smoothing vs flicker")
print(f"{'='*60}")
print(f"{'Smoothing':>10} {'Mean flicker':>14} {'Max flicker':>14} {'Time (s)':>10}")
for td_val in TD_SMOOTHING_VALUES:
    r = results[td_val]
    fm = r["flicker"]
    print(f"{td_val:10.3f} {fm.mean():14.6f} {fm.max():14.6f} {r['time']:10.1f}")

best_td = min(results, key=lambda k: results[k]["flicker"].mean())
print(f"\nBest smoothing: {best_td} (lowest mean flicker)")

# ── Save comparison plot ──────────────────────────────────────────────
try:
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(len(TD_SMOOTHING_VALUES), 4, figsize=(16, 4*len(TD_SMOOTHING_VALUES)))
    if len(TD_SMOOTHING_VALUES) == 1:
        axes = axes[np.newaxis, :]

    for row, td_val in enumerate(TD_SMOOTHING_VALUES):
        rec = results[td_val]["rec"]
        fm = results[td_val]["flicker"]

        # Show slice 0, slice 4, slice 9
        for col, si in enumerate([0, 4, 9]):
            if si < rec.shape[2]:
                vmin, vmax = np.percentile(rec[:, :, si], [1, 99])
                axes[row, col].imshow(rec[:, :, si], cmap='gray', vmin=vmin, vmax=vmax)
                axes[row, col].set_title(f"TD={td_val}, slice {TEST_START+si}")
                axes[row, col].axis('off')

        # Show diff between slice 4 and 5
        if rec.shape[2] > 5:
            diff = rec[:, :, 5] - rec[:, :, 4]
            vd = np.percentile(np.abs(diff), 99)
            axes[row, 3].imshow(diff, cmap='RdBu', vmin=-vd, vmax=vd)
            axes[row, 3].set_title(f"TD={td_val}, diff(s5-s4), flicker={fm.mean():.5f}")
            axes[row, 3].axis('off')

    plt.suptitle(f"T-D Smoothing vs Flicker (VOXEL_SIZE_Z={VOXEL_SIZE_Z}mm)", fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "td_smoothing_flicker.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved -> {out_path}")
except Exception as e:
    print(f"Plot failed: {e}")

print("\nDone.")
