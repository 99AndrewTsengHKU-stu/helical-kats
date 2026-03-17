"""
L067 reconstruction with angular correction applied.

Changes from 0304/run_L067.py:
  - Added angle_offset = -pi/2 to correct DICOM 0-axis vs code X-axis mismatch
  - Removed post-recon rot90 (the offset makes it unnecessary)
  - Quick validation mode: reconstruct 3 representative chunks (top/mid/bottom)
    instead of all 560 slices, for fast A/B comparison

Usage:
  ~/anaconda3/envs/MNIST/python.exe "Quick Run Through 0313/run_L067_fixed.py"
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Quick Run Through 0304"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

# Initialize ASTRA CUDA context BEFORE CuPy
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

# ---- Settings ----
DICOM_DIR = r"D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD"
GT_DIR    = r"D:\AAPM-Data\L067\L067\full_1mm"

ROWS      = 512
COLS      = 512
SLICES    = 560
VOXEL_SIZE = 0.664
CHUNK_Z   = 64
DECIMATE  = 1


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
                      chunk_slices):
    """Reconstruct a single z-chunk."""
    det_rows_n = scan_geom_full["detector"]["detector rows"]
    det_cols_n = scan_geom_full["detector"]["detector cols"]
    psize = scan_geom_full["detector"]["detector psize"]

    keep_idx = z_cull_indices(
        views_full, scan_geom_full["SOD"], scan_geom_full["SDD"],
        det_rows_n, psize, chunk_z_min, chunk_z_max,
        margin_turns=1.0, pitch_mm_per_angle=meta["pitch_mm_per_angle"],
    )
    print(f"  [Z-cull] {len(angles_full)} -> {len(keep_idx)} projections "
          f"({100*len(keep_idx)/len(angles_full):.1f}%)")

    if len(keep_idx) == 0:
        return np.zeros((ROWS, COLS, chunk_slices), dtype=np.float32)

    sino_chunk = sino_work[keep_idx].copy()
    angles_chunk = angles_full[keep_idx].copy()
    views_chunk = views_full[keep_idx]

    scan_geom_chunk = copy.deepcopy(scan_geom_full)
    scan_geom_chunk["helix"]["angles_range"] = float(abs(angles_chunk[-1] - angles_chunk[0]))
    scan_geom_chunk["helix"]["angles_count"] = len(angles_chunk)

    proj_geom_chunk = astra.create_proj_geom("cone_vec", det_rows_n, det_cols_n, views_chunk)

    half_x = COLS * VOXEL_SIZE * 0.5
    half_y = ROWS * VOXEL_SIZE * 0.5
    vol_geom_chunk = astra.create_vol_geom(
        ROWS, COLS, chunk_slices,
        -half_x, half_x, -half_y, half_y, chunk_z_min, chunk_z_max,
    )

    conf_chunk = create_configuration(scan_geom_chunk, vol_geom_chunk)
    conf_chunk['source_pos'] = angles_chunk.astype(np.float32)
    conf_chunk['delta_s'] = float(np.mean(np.diff(angles_chunk)))

    sino_f32 = np.asarray(sino_chunk, dtype=np.float32, order="C")
    filtered = filter_katsevich(
        sino_f32, conf_chunk,
        {"Diff": {"Print time": False}, "FwdRebin": {"Print time": False},
         "BackRebin": {"Print time": False}},
    )
    sino_td = sino_weight_td(filtered, conf_chunk, False)
    rec_chunk = backproject_cupy(sino_td, conf_chunk, vol_geom_chunk, proj_geom_chunk, tqdm_bar=True)
    return rec_chunk


def prepare_geometry(sino, meta, negate_angles, flip_rows, angle_offset):
    """Apply transforms and build full views array."""
    angles_full = meta["angles_rad"].copy()
    scan_geom_full = copy.deepcopy(meta["scan_geometry"])

    z_shift_full = (meta["table_positions_mm"] - meta["table_positions_mm"].mean()
                    if meta["table_positions_mm"] is not None
                    else np.zeros(len(angles_full), dtype=np.float32))

    sino_work = sino.copy()

    if negate_angles:
        angles_full = -angles_full
    if angle_offset != 0:
        angles_full = angles_full + angle_offset
    if flip_rows:
        sino_work = sino_work[:, ::-1, :].copy()

    scan_geom_full["helix"]["pitch_mm_rad"] = float(abs(meta["pitch_mm_per_rad_signed"]))
    scan_geom_full["helix"]["angles_range"] = float(abs(angles_full[-1] - angles_full[0]))

    views_full = astra_helical_views(
        scan_geom_full["SOD"], scan_geom_full["SDD"],
        scan_geom_full["detector"]["detector psize"],
        angles_full, meta["pitch_mm_per_angle"],
        vertical_shifts=z_shift_full,
    )
    return sino_work, angles_full, z_shift_full, views_full, scan_geom_full


# =====================================================================
# Main
# =====================================================================
print(f"Loading DICOM projections from {DICOM_DIR} ...")
t0 = time()
sino, meta = load_dicom_projections(DICOM_DIR)
print(f"Loaded {sino.shape[0]} projections in {time()-t0:.1f}s")

if DECIMATE > 1:
    sino = sino[::DECIMATE].copy()
    meta["angles_rad"] = meta["angles_rad"][::DECIMATE]
    if meta["table_positions_mm"] is not None:
        meta["table_positions_mm"] = meta["table_positions_mm"][::DECIMATE]
    meta["scan_geometry"]["helix"]["angles_count"] = len(sino)
    meta["pitch_mm_per_angle"] *= DECIMATE
    print(f"Decimated to {len(sino)} projections")

# Pick 3 representative chunk positions: top (25%), middle (50%), bottom (75%)
total_half_z = SLICES * VOXEL_SIZE * 0.5
test_slices = [SLICES // 4, SLICES // 2, SLICES * 3 // 4]

# Two modes to compare
modes = {
    "OLD (negate+flip+rot90)": {
        "negate": True, "flip_rows": True, "offset": 0.0, "rot90": True,
    },
    "NEW (negate+flip+offset -pi/2)": {
        "negate": True, "flip_rows": True, "offset": -np.pi/2, "rot90": False,
    },
}

results = {}  # mode_name -> list of (slice_idx, image) pairs

for mode_name, mode_cfg in modes.items():
    print(f"\n{'='*60}")
    print(f"MODE: {mode_name}")
    print(f"{'='*60}")

    sino_work, angles_full, z_shift_full, views_full, scan_geom_full = prepare_geometry(
        sino, meta,
        negate_angles=mode_cfg["negate"],
        flip_rows=mode_cfg["flip_rows"],
        angle_offset=mode_cfg["offset"],
    )

    step = float(np.mean(np.diff(angles_full)))
    print(f"  angle_offset = {mode_cfg['offset']:.4f} rad ({np.degrees(mode_cfg['offset']):.1f} deg)")
    print(f"  delta_s = {step:.6f} rad  ({'CCW' if step > 0 else 'CW'})")
    print(f"  rot90 post-recon: {mode_cfg['rot90']}")

    slices_result = []
    for center_slice in test_slices:
        z_start = max(center_slice - 1, 0)
        z_end = min(center_slice + 1, SLICES)
        chunk_slices = z_end - z_start

        chunk_z_min = -total_half_z + z_start * VOXEL_SIZE
        chunk_z_max = -total_half_z + z_end * VOXEL_SIZE

        print(f"\n  --- Slice {center_slice} (z=[{chunk_z_min:.1f}, {chunk_z_max:.1f}]mm) ---")
        t0 = time()
        rec_chunk = reconstruct_chunk(
            sino_work, angles_full, z_shift_full, views_full,
            scan_geom_full, meta, chunk_z_min, chunk_z_max, chunk_slices,
        )
        dt = time() - t0

        if mode_cfg["rot90"]:
            rec_chunk = np.rot90(rec_chunk, k=1, axes=(0, 1))

        # Take the center slice of the 2-slice chunk
        mid = min(1, rec_chunk.shape[2] - 1)
        img = rec_chunk[:, :, mid]
        slices_result.append((center_slice, img))
        print(f"  Done in {dt:.1f}s  range=[{img.min():.4f}, {img.max():.4f}]")

    results[mode_name] = slices_result

# =====================================================================
# Plot comparison
# =====================================================================
n_slices = len(test_slices)
fig, axes = plt.subplots(2, n_slices, figsize=(5 * n_slices, 10))

for row_i, (mode_name, slices_result) in enumerate(results.items()):
    for col_i, (si, img) in enumerate(slices_result):
        ax = axes[row_i, col_i]
        vmin, vmax = np.percentile(img, [1, 99])
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"{mode_name}\nz={si}", fontsize=9)
        ax.axis("off")

        # Compute cross-correlation with OLD mode for NEW mode
        if row_i == 1:
            old_img = results[list(results.keys())[0]][col_i][1]
            old_norm = (old_img - old_img.mean()) / max(old_img.std(), 1e-12)
            new_norm = (img - img.mean()) / max(img.std(), 1e-12)
            corr = float(np.mean(old_norm * new_norm))
            ax.set_xlabel(f"corr={corr:.4f}", fontsize=9)

plt.suptitle("Angular Correction: OLD (rot90) vs NEW (offset -pi/2)", fontsize=13)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "correction_comparison.png")
plt.savefig(out_path, dpi=150)
plt.close()
print(f"\nSaved comparison -> {out_path}")

# Also save a diff image
fig2, axes2 = plt.subplots(1, n_slices, figsize=(5 * n_slices, 5))
for col_i in range(n_slices):
    old_img = results[list(results.keys())[0]][col_i][1]
    new_img = results[list(results.keys())[1]][col_i][1]
    diff = new_img - old_img
    vd = max(abs(np.percentile(diff, 1)), abs(np.percentile(diff, 99)))
    ax = axes2[col_i]
    ax.imshow(diff, cmap="RdBu_r", vmin=-vd, vmax=vd)
    ax.set_title(f"NEW - OLD, z={test_slices[col_i]}\nmax|diff|={np.abs(diff).max():.5f}", fontsize=9)
    ax.axis("off")

plt.suptitle("Pixel difference: offset(-pi/2) minus rot90", fontsize=13)
plt.tight_layout()
diff_path = os.path.join(OUT_DIR, "correction_diff.png")
plt.savefig(diff_path, dpi=150)
plt.close()
print(f"Saved diff -> {diff_path}")

print("\nDone.")
