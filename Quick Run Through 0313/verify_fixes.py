"""
Verify geometric fixes: VOXEL_SIZE_Z separation + detector pixel size separation.
Reconstructs 3 representative slices, measures couch drift.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Quick Run Through 0304"))
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
DICOM_DIR = r"D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD"

ROWS = 512
COLS = 512
SLICES = 560
VOXEL_SIZE_XY = 0.664
CHUNK_Z = 1  # single slice per chunk for targeted reconstruction

# Slices to reconstruct
TEST_SLICES = [140, 280, 420]

print("=" * 60)
print("VERIFICATION: geometric fixes (VOXEL_SIZE_Z + separate psize)")
print("=" * 60)

# Load DICOM
print("\nLoading DICOM...")
t0 = time()
sino, meta = load_dicom_projections(DICOM_DIR)
print(f"Loaded {sino.shape} in {time()-t0:.1f}s")

# Compute VOXEL_SIZE_Z from table
table_z_span = meta["table_positions_mm"].max() - meta["table_positions_mm"].min()
VOXEL_SIZE_Z = table_z_span / SLICES
print(f"  table z-span: {table_z_span:.2f} mm")
print(f"  VOXEL_SIZE_XY = {VOXEL_SIZE_XY:.4f} mm")
print(f"  VOXEL_SIZE_Z  = {VOXEL_SIZE_Z:.6f} mm (was 0.664)")

# Pixel sizes
sg = meta["scan_geometry"]
psize_cols = sg["detector"]["detector psize cols"]
psize_rows = sg["detector"]["detector psize rows"]
psize_avg  = sg["detector"]["detector psize"]
print(f"  psize_cols = {psize_cols:.6f}, psize_rows = {psize_rows:.6f} (was avg={psize_avg:.6f})")

# Prepare angles, sinogram
angles_full = meta["angles_rad"].copy()
scan_geom_full = copy.deepcopy(meta["scan_geometry"])
z_shift_full = meta["table_positions_mm"] - meta["table_positions_mm"].mean()

# Apply mode: negate + flip_rows + offset(-pi/2)
angles_full = -angles_full - np.pi/2
sino_work = sino[:, ::-1, :].copy()

pitch_signed = meta["pitch_mm_per_rad_signed"]
scan_geom_full["helix"]["pitch_mm_rad"] = float(abs(pitch_signed))
scan_geom_full["helix"]["angles_range"] = float(abs(angles_full[-1] - angles_full[0]))

# Build full views (with separate pixel sizes)
views_full = astra_helical_views(
    scan_geom_full["SOD"], scan_geom_full["SDD"],
    scan_geom_full["detector"]["detector psize"],
    angles_full, meta["pitch_mm_per_angle"],
    vertical_shifts=z_shift_full,
    pixel_size_col=psize_cols,
    pixel_size_row=psize_rows,
)

det_rows_n = sg["detector"]["detector rows"]
det_cols_n = sg["detector"]["detector cols"]
total_half_z = SLICES * VOXEL_SIZE_Z * 0.5

print(f"\n  Volume z-range: [{-total_half_z:.2f}, {total_half_z:.2f}] mm (NEW)")
print(f"  Source z-range: [{z_shift_full.min():.2f}, {z_shift_full.max():.2f}] mm")

# ── z-cull helper ──
def z_cull(views, SOD, SDD, det_rows, psize, z_lo, z_hi, pitch_per_angle):
    src_z = views[:, 2]
    cone_half = 0.5 * det_rows * psize * (SOD / SDD)
    astep = abs(np.mean(np.diff(np.arctan2(views[:10, 1], views[:10, 0]))))
    ppt = 2 * np.pi / max(astep, 1e-12)
    margin = 1.0 * ppt * abs(pitch_per_angle)
    lo = src_z - cone_half - margin
    hi = src_z + cone_half + margin
    return np.where((hi >= z_lo) & (lo <= z_hi))[0]

# ── Reconstruct single slices ──
results = {}
for si in TEST_SLICES:
    print(f"\n--- Slice {si}/{SLICES} ---")
    chunk_z_min = -total_half_z + si * VOXEL_SIZE_Z
    chunk_z_max = -total_half_z + (si + 1) * VOXEL_SIZE_Z
    print(f"  z=[{chunk_z_min:.2f}, {chunk_z_max:.2f}] mm")

    keep = z_cull(views_full, scan_geom_full["SOD"], scan_geom_full["SDD"],
                  det_rows_n, psize_avg, chunk_z_min, chunk_z_max,
                  meta["pitch_mm_per_angle"])
    print(f"  z-cull: {len(keep)} projections")

    if len(keep) == 0:
        print("  [SKIP] no projections")
        continue

    sino_chunk = sino_work[keep].copy()
    angles_chunk = angles_full[keep].copy()
    views_chunk = views_full[keep]

    sg_chunk = copy.deepcopy(scan_geom_full)
    sg_chunk["helix"]["angles_range"] = float(abs(angles_chunk[-1] - angles_chunk[0]))
    sg_chunk["helix"]["angles_count"] = len(angles_chunk)

    proj_geom = astra.create_proj_geom("cone_vec", det_rows_n, det_cols_n, views_chunk)

    half_x = COLS * VOXEL_SIZE_XY * 0.5
    half_y = ROWS * VOXEL_SIZE_XY * 0.5
    vol_geom = astra.create_vol_geom(ROWS, COLS, 1,
                                      -half_x, half_x, -half_y, half_y,
                                      chunk_z_min, chunk_z_max)

    conf = create_configuration(sg_chunk, vol_geom)
    conf['source_pos'] = angles_chunk.astype(np.float32)
    conf['delta_s'] = float(np.mean(np.diff(angles_chunk)))

    t0 = time()
    sino_f32 = np.asarray(sino_chunk, dtype=np.float32, order="C")
    filtered = filter_katsevich(sino_f32, conf,
        {"Diff": {"Print time": False}, "FwdRebin": {"Print time": False}, "BackRebin": {"Print time": False}})
    sino_td = sino_weight_td(filtered, conf, False)
    t_filt = time() - t0

    t0 = time()
    rec = backproject_cupy(sino_td, conf, vol_geom, proj_geom, tqdm_bar=False)
    t_bp = time() - t0

    img = rec[:, :, 0]
    results[si] = img
    print(f"  Filter: {t_filt:.1f}s  BP: {t_bp:.1f}s  range=[{img.min():.4f}, {img.max():.4f}]")

# ── Couch drift measurement ──
print("\n" + "=" * 60)
print("COUCH DRIFT MEASUREMENT")
print("=" * 60)

def find_couch_row(img):
    """Find the couch (brightest structure near bottom of image)."""
    col_sum = np.mean(img[img.shape[0]//2:, :], axis=1)
    return np.argmax(col_sum) + img.shape[0] // 2

slices_sorted = sorted(results.keys())
if len(slices_sorted) >= 2:
    rows_found = []
    for si in slices_sorted:
        r = find_couch_row(results[si])
        rows_found.append(r)
        print(f"  Slice {si}: couch row = {r}")

    drift = max(rows_found) - min(rows_found)
    print(f"\n  Couch drift: {drift} px (was 92 px before fixes)")
    if drift < 10:
        print("  [OK] Drift is small - geometric alignment improved!")
    else:
        print("  [!] Drift still significant - further investigation needed")

# ── Sharpness measurement ──
print("\n" + "=" * 60)
print("SHARPNESS MEASUREMENT")
print("=" * 60)
for si in slices_sorted:
    img = results[si]
    gy, gx = np.gradient(img)
    sharpness = np.mean(np.sqrt(gx**2 + gy**2))
    print(f"  Slice {si}: sharpness = {sharpness:.6f}")

# ── Save comparison plot ──
n = len(slices_sorted)
fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
if n == 1:
    axes = [axes]
for i, si in enumerate(slices_sorted):
    img = results[si]
    vmin, vmax = np.percentile(img, [1, 99])
    axes[i].imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    axes[i].set_title(f"z={si} (FIXED)")
    axes[i].axis("off")

fig.suptitle(f"Corrected: voxel_z={VOXEL_SIZE_Z:.4f}mm, psize_col={psize_cols:.4f}, psize_row={psize_rows:.4f}")
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "verify_fixes.png")
plt.savefig(out_path, dpi=150)
print(f"\nSaved -> {out_path}")
plt.close()

print("\nDone.")
