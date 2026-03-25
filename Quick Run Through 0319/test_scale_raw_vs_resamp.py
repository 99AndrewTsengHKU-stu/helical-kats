"""
Definitive test: Find optimal scale for BOTH resampled and raw DICOM data.
Uses MSE minimization (same as find_optimal_scale.py) for reliable comparison.

If raw and resampled give the SAME optimal scale → resampling is NOT the issue
If they differ → resampling IS involved
"""
import sys, os, copy, struct
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Quick Run Through 0304"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from backproject_safe import ensure_astra_cuda_init
ensure_astra_cuda_init()

import numpy as np
from time import time
from pathlib import Path
from scipy.ndimage import zoom, shift as ndshift
from scipy.optimize import minimize_scalar, minimize
import astra, pydicom, matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from pykatsevich import load_dicom_projections
from pykatsevich.geometry import astra_helical_views
from pykatsevich.initialize import create_configuration
from pykatsevich.filter import filter_katsevich, sino_weight_td
from backproject_cupy import backproject_cupy

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DICOM_DIR = r"D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD"
GT_DIR = Path(r"D:\AAPM-Data\L067\L067\full_1mm")
ROWS = COLS = 512
SLICES = 560
VOXEL_SIZE_XY = 0.664
VOXEL_SIZE_Z = 0.8
TARGET_SLICE = 280


def recon_one_slice(sino_work, angles_full, z_shift, sg, label=""):
    psize_cols = sg['detector'].get('detector psize cols', sg['detector']['detector psize'])
    psize_rows = sg['detector'].get('detector psize rows', sg['detector']['detector psize'])
    psize = sg['detector']['detector psize']
    det_rows_n = sg['detector']['detector rows']

    views = astra_helical_views(
        sg['SOD'], sg['SDD'], psize, angles_full,
        float(np.mean(np.diff(z_shift))),
        vertical_shifts=z_shift,
        pixel_size_col=psize_cols, pixel_size_row=psize_rows,
    )

    total_half_z = SLICES * VOXEL_SIZE_Z * 0.5
    z_c = -total_half_z + TARGET_SLICE * VOXEL_SIZE_Z
    z_min, z_max = z_c, z_c + VOXEL_SIZE_Z

    source_z = views[:, 2]
    cone_half_z = 0.5 * det_rows_n * psize * (sg['SOD'] / sg['SDD'])
    angle_step = abs(np.mean(np.diff(np.arctan2(views[:10, 1], views[:10, 0]))))
    projs_per_turn = 2 * np.pi / max(angle_step, 1e-12)
    margin_z = projs_per_turn * abs(np.mean(np.diff(z_shift)))
    keep = np.where(
        (source_z + cone_half_z + margin_z >= z_min) &
        (source_z - cone_half_z - margin_z <= z_max)
    )[0]

    sino_c = sino_work[keep].copy()
    angles_c = angles_full[keep].copy()
    views_c = views[keep]

    sg_c = copy.deepcopy(sg)
    pitch_abs = float(abs(np.mean(np.diff(z_shift)) / np.mean(np.diff(angles_full))))
    sg_c['helix']['pitch_mm_rad'] = pitch_abs
    sg_c['helix']['angles_range'] = float(abs(angles_c[-1] - angles_c[0]))
    sg_c['helix']['angles_count'] = len(angles_c)

    half_x = COLS * VOXEL_SIZE_XY * 0.5
    vol_geom = astra.create_vol_geom(ROWS, COLS, 1, -half_x, half_x, -half_x, half_x, z_min, z_max)
    proj_geom = astra.create_proj_geom('cone_vec', det_rows_n, sg['detector']['detector cols'], views_c)

    conf = create_configuration(sg_c, vol_geom)
    conf['source_pos'] = angles_c.astype(np.float32)
    conf['delta_s'] = float(np.mean(np.diff(angles_c)))

    print(f"  [{label}] {len(keep)} projs, psize_col={psize_cols:.4f}, filtering...", flush=True)
    t0 = time()
    filtered = filter_katsevich(
        np.asarray(sino_c, dtype=np.float32, order='C'), conf,
        {'Diff': {'Print time': False}, 'FwdRebin': {'Print time': False},
         'BackRebin': {'Print time': False}})
    sino_td = sino_weight_td(filtered, conf, False)
    rec = backproject_cupy(sino_td, conf, vol_geom, proj_geom, tqdm_bar=False)[:, :, 0]
    print(f"  [{label}] done in {time()-t0:.1f}s")
    return rec


def norm01(img):
    v0, v1 = np.percentile(img, [2, 98])
    return np.clip((img - v0) / max(v1 - v0, 1e-10), 0, 1).astype(np.float64)


def apply_scale(img, scale, output_shape):
    scaled = zoom(img, scale, order=1)
    out = np.zeros(output_shape, dtype=np.float64)
    y0 = (output_shape[0] - scaled.shape[0]) // 2
    x0 = (output_shape[1] - scaled.shape[1]) // 2
    sy0, sx0 = max(0, -y0), max(0, -x0)
    sy1 = min(scaled.shape[0], output_shape[0] - y0)
    sx1 = min(scaled.shape[1], output_shape[1] - x0)
    dy0, dx0 = max(0, y0), max(0, x0)
    dy1, dx1 = dy0 + (sy1 - sy0), dx0 + (sx1 - sx0)
    out[dy0:dy1, dx0:dx1] = scaled[sy0:sy1, sx0:sx1]
    return out


def find_optimal_scale(rec_n, gt_n, label=""):
    """Find optimal scale + shift using MSE."""
    def diff_at_scale(scale):
        scaled = apply_scale(rec_n, scale, gt_n.shape)
        mask = (scaled > 0.05) & (gt_n > 0.05)
        if mask.sum() < 1000:
            return 1e10
        return np.mean((scaled[mask] - gt_n[mask])**2)

    # Coarse sweep
    scales = np.linspace(0.7, 1.6, 46)
    losses = [diff_at_scale(s) for s in scales]
    best_idx = np.argmin(losses)
    coarse_best = scales[best_idx]

    # Fine search
    result = minimize_scalar(diff_at_scale, bounds=(coarse_best - 0.05, coarse_best + 0.05), method='bounded')
    optimal_scale = result.x

    # With translation
    def diff_sst(params):
        scale, dy, dx = params
        scaled = apply_scale(rec_n, scale, gt_n.shape)
        shifted = ndshift(scaled, [dy, dx], order=1)
        mask = (shifted > 0.05) & (gt_n > 0.05)
        if mask.sum() < 1000:
            return 1e10
        return np.mean((shifted[mask] - gt_n[mask])**2)

    res2 = minimize(diff_sst, [optimal_scale, 0, 0],
                    method='Nelder-Mead',
                    options={'xatol': 0.001, 'fatol': 1e-8, 'maxiter': 500})
    best_scale, best_dy, best_dx = res2.x

    print(f"  [{label}] scale={best_scale:.4f}, dy={best_dy:.1f}, dx={best_dx:.1f}, loss={res2.fun:.6f}")
    return best_scale, best_dy, best_dx, res2.fun


# ══════════════════════════════════════════════════════════════════════
# LOAD GT
# ══════════════════════════════════════════════════════════════════════
print("Loading GT...", flush=True)
gt_files = sorted(GT_DIR.glob("*.IMA")) or sorted(GT_DIR.glob("*.dcm"))
ds_gt = pydicom.dcmread(str(gt_files[TARGET_SLICE]))
slope = float(getattr(ds_gt, 'RescaleSlope', 1.0))
intercept = float(getattr(ds_gt, 'RescaleIntercept', 0.0))
gt = ds_gt.pixel_array.astype(np.float32) * slope + intercept
gt_n = norm01(gt)

# ══════════════════════════════════════════════════════════════════════
# LOAD RESAMPLED
# ══════════════════════════════════════════════════════════════════════
print("Loading DICOM (resampled)...", flush=True)
t0 = time()
sino_std, meta_std = load_dicom_projections(DICOM_DIR)
print(f"  Loaded in {time()-t0:.1f}s")
sg_std = meta_std['scan_geometry']
angles_std = -meta_std['angles_rad'].copy() - np.pi / 2
z_shift_std = meta_std['table_positions_mm'] - meta_std['table_positions_mm'].mean()
sino_std_work = sino_std[:, ::-1, :].copy()

# ══════════════════════════════════════════════════════════════════════
# LOAD RAW
# ══════════════════════════════════════════════════════════════════════
print("Loading DICOM (raw, no resample)...", flush=True)
t0 = time()
dicom_dir = Path(DICOM_DIR)
paths = sorted(p for p in dicom_dir.iterdir() if p.suffix.lower() == '.dcm')
first_ds = pydicom.dcmread(str(paths[0]), stop_before_pixels=True)
sod = float(struct.unpack('<f', bytes(first_ds[(0x7031, 0x1003)].value)[:4])[0])
sdd = float(struct.unpack('<f', bytes(first_ds[(0x7031, 0x1031)].value)[:4])[0])
det_extents = struct.unpack('<ff', bytes(first_ds[(0x7031, 0x1033)].value)[:8])
det_length_cols_mm, det_length_rows_mm = det_extents
det_rows = int(first_ds.Columns)
det_cols = int(first_ds.Rows)

arc_psize_cols = det_length_cols_mm / det_cols
arc_psize_rows = det_length_rows_mm / det_rows
mag_arc_cols = arc_psize_cols * sdd / sod
mag_arc_rows = arc_psize_rows * sdd / sod

meta_list = []
for path in paths:
    ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    instance = int(getattr(ds, "InstanceNumber", 0))
    meta_list.append((instance, path))
meta_list.sort(key=lambda x: x[0])

sino_raw = np.empty((len(meta_list), det_rows, det_cols), dtype=np.float32)
for idx, (_, path) in enumerate(meta_list):
    ds = pydicom.dcmread(str(path))
    pixels = ds.pixel_array.astype(np.float32)
    slope_v = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept_v = float(getattr(ds, 'RescaleIntercept', 0.0))
    sino_raw[idx] = (pixels * slope_v + intercept_v).T

print(f"  Loaded raw in {time()-t0:.1f}s")
sino_raw_work = sino_raw[:, ::-1, :].copy()

sg_raw = copy.deepcopy(sg_std)
sg_raw['detector']['detector psize cols'] = mag_arc_cols
sg_raw['detector']['detector psize rows'] = mag_arc_rows
sg_raw['detector']['detector psize'] = float(np.mean([mag_arc_cols, mag_arc_rows]))

# ══════════════════════════════════════════════════════════════════════
# RECONSTRUCT BOTH
# ══════════════════════════════════════════════════════════════════════
print("\n=== Reconstructing ===", flush=True)
rec_resamp = recon_one_slice(sino_std_work, angles_std, z_shift_std, sg_std, label="Resampled")
rec_raw = recon_one_slice(sino_raw_work, angles_std, z_shift_std, sg_raw, label="Raw")

# ══════════════════════════════════════════════════════════════════════
# FIND OPTIMAL SCALE FOR BOTH
# ══════════════════════════════════════════════════════════════════════
print("\n=== Finding optimal scale ===", flush=True)
rec_resamp_n = norm01(rec_resamp[:, ::-1])  # LR flip + normalize
rec_raw_n = norm01(rec_raw[:, ::-1])

scale_r, dy_r, dx_r, loss_r = find_optimal_scale(rec_resamp_n, gt_n, "Resampled")
scale_w, dy_w, dx_w, loss_w = find_optimal_scale(rec_raw_n, gt_n, "Raw")

print(f"\n{'='*50}")
print(f"RESULTS:")
print(f"  Resampled: scale={scale_r:.4f}, loss={loss_r:.6f}")
print(f"  Raw:       scale={scale_w:.4f}, loss={loss_w:.6f}")
print(f"  Difference: {abs(scale_r - scale_w):.4f}")
print(f"  psize_col resampled: {sg_std['detector']['detector psize cols']:.6f}")
print(f"  psize_col raw:       {mag_arc_cols:.6f}")
print(f"  Ratio:               {sg_std['detector']['detector psize cols']/mag_arc_cols:.4f}")
print(f"{'='*50}")

# ══════════════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Apply optimal transforms
rec_r_opt = ndshift(apply_scale(rec_resamp_n, scale_r, gt_n.shape), [dy_r, dx_r], order=1)
rec_w_opt = ndshift(apply_scale(rec_raw_n, scale_w, gt_n.shape), [dy_w, dx_w], order=1)

axes[0, 0].imshow(rec_resamp_n, cmap='gray'); axes[0, 0].set_title(f"Resampled (psize={sg_std['detector']['detector psize cols']:.3f})")
axes[0, 1].imshow(rec_raw_n, cmap='gray'); axes[0, 1].set_title(f"Raw (psize={mag_arc_cols:.3f})")
axes[0, 2].imshow(gt_n, cmap='gray'); axes[0, 2].set_title("GT")
axes[0, 3].axis('off')
axes[0, 3].text(0.05, 0.9, f"Optimal Scale:\n"
    f"  Resampled: {scale_r:.4f}\n"
    f"  Raw:       {scale_w:.4f}\n"
    f"  Delta:     {abs(scale_r-scale_w):.4f}\n\n"
    f"psize_col:\n"
    f"  Resampled: {sg_std['detector']['detector psize cols']:.4f}\n"
    f"  Raw:       {mag_arc_cols:.4f}\n"
    f"  Ratio:     {sg_std['detector']['detector psize cols']/mag_arc_cols:.4f}\n\n"
    f"voxel_size: {VOXEL_SIZE_XY}\n"
    f"psize/voxel:\n"
    f"  Resamp: {sg_std['detector']['detector psize cols']/VOXEL_SIZE_XY:.4f}\n"
    f"  Raw:    {mag_arc_cols/VOXEL_SIZE_XY:.4f}",
    fontsize=11, transform=axes[0, 3].transAxes, va='top', fontfamily='monospace')

diff_r = rec_r_opt - gt_n
diff_w = rec_w_opt - gt_n
vlim = max(abs(diff_r).max(), abs(diff_w).max(), 0.01) * 0.4
axes[1, 0].imshow(diff_r, cmap='RdBu', vmin=-vlim, vmax=vlim)
axes[1, 0].set_title(f"Diff resampled\nscale={scale_r:.3f}, MSE={np.mean(diff_r**2):.5f}")
axes[1, 1].imshow(diff_w, cmap='RdBu', vmin=-vlim, vmax=vlim)
axes[1, 1].set_title(f"Diff raw\nscale={scale_w:.3f}, MSE={np.mean(diff_w**2):.5f}")

# Edge overlays
from scipy.ndimage import sobel
def edges(img):
    return np.sqrt(sobel(img, 0)**2 + sobel(img, 1)**2)
e_gt = edges(gt_n)
e_r = edges(rec_r_opt)
e_w = edges(rec_w_opt)
eth = np.percentile(e_gt, 95)
overlay_r = np.stack([e_r > eth, e_gt > eth, np.zeros_like(gt_n)], axis=-1).astype(float)
overlay_w = np.stack([e_w > eth, e_gt > eth, np.zeros_like(gt_n)], axis=-1).astype(float)
axes[1, 2].imshow(overlay_r); axes[1, 2].set_title("Edges resampled+scaled")
axes[1, 3].imshow(overlay_w); axes[1, 3].set_title("Edges raw+scaled")

for ax in axes.flat:
    ax.axis('off')

plt.suptitle("Raw vs Resampled: Optimal Scale Comparison", fontsize=14)
plt.tight_layout()
out = os.path.join(OUT_DIR, "raw_vs_resamp_scale.png")
plt.savefig(out, dpi=150)
print(f"\nSaved -> {out}")
print("Done.", flush=True)
