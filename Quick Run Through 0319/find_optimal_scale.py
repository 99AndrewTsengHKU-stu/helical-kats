"""
Find optimal scale: flip recon LR, then sweep scale factors to minimize diff with GT.
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
from scipy.optimize import minimize_scalar
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

# ── Load & reconstruct 1 slice ─────────────────────────────────────
print("Loading DICOM...", flush=True)
t0 = time()
sino, meta = load_dicom_projections(DICOM_DIR)
print(f"Loaded in {time()-t0:.1f}s", flush=True)

sg = meta['scan_geometry']
angles_full = -meta['angles_rad'].copy() - np.pi / 2
z_shift = meta['table_positions_mm'] - meta['table_positions_mm'].mean()
sino_work = sino[:, ::-1, :].copy()

scan_geom = copy.deepcopy(sg)
pitch_abs = float(abs(meta['pitch_mm_per_rad_signed']))
scan_geom['helix']['pitch_mm_rad'] = pitch_abs
scan_geom['helix']['angles_range'] = float(abs(angles_full[-1] - angles_full[0]))

psize_cols = sg['detector'].get('detector psize cols', sg['detector']['detector psize'])
psize_rows = sg['detector'].get('detector psize rows', sg['detector']['detector psize'])
psize = sg['detector']['detector psize']
det_rows_n = sg['detector']['detector rows']

views = astra_helical_views(
    sg['SOD'], sg['SDD'], psize, angles_full, meta['pitch_mm_per_angle'],
    vertical_shifts=z_shift, pixel_size_col=psize_cols, pixel_size_row=psize_rows,
)

total_half_z = SLICES * VOXEL_SIZE_Z * 0.5
z_c = -total_half_z + TARGET_SLICE * VOXEL_SIZE_Z
z_min, z_max = z_c, z_c + VOXEL_SIZE_Z

source_z = views[:, 2]
cone_half_z = 0.5 * det_rows_n * psize * (sg['SOD'] / sg['SDD'])
angle_step = abs(np.mean(np.diff(np.arctan2(views[:10, 1], views[:10, 0]))))
projs_per_turn = 2 * np.pi / max(angle_step, 1e-12)
margin_z = projs_per_turn * abs(meta['pitch_mm_per_angle'])
keep = np.where((source_z + cone_half_z + margin_z >= z_min) &
                (source_z - cone_half_z - margin_z <= z_max))[0]

sino_c = sino_work[keep].copy()
angles_c = angles_full[keep].copy()
views_c = views[keep]

sg_c = copy.deepcopy(scan_geom)
sg_c['helix']['angles_range'] = float(abs(angles_c[-1] - angles_c[0]))
sg_c['helix']['angles_count'] = len(angles_c)

half_xy = COLS * VOXEL_SIZE_XY * 0.5
vol_geom = astra.create_vol_geom(ROWS, COLS, 1, -half_xy, half_xy, -half_xy, half_xy, z_min, z_max)
proj_geom = astra.create_proj_geom('cone_vec', det_rows_n, sg['detector']['detector cols'], views_c)

conf = create_configuration(sg_c, vol_geom)
conf['source_pos'] = angles_c.astype(np.float32)
conf['delta_s'] = float(np.mean(np.diff(angles_c)))

print(f"Reconstructing slice {TARGET_SLICE}...", flush=True)
t0 = time()
filtered = filter_katsevich(
    np.asarray(sino_c, dtype=np.float32, order='C'), conf,
    {'Diff': {'Print time': False}, 'FwdRebin': {'Print time': False}, 'BackRebin': {'Print time': False}})
sino_td = sino_weight_td(filtered, conf, False)
rec = backproject_cupy(sino_td, conf, vol_geom, proj_geom, tqdm_bar=False)[:, :, 0]
print(f"Done in {time()-t0:.1f}s", flush=True)

# ── Load GT ────────────────────────────────────────────────────────
gt_files = sorted(GT_DIR.glob("*.IMA")) or sorted(GT_DIR.glob("*.dcm"))
ds = pydicom.dcmread(str(gt_files[TARGET_SLICE]))
slope = float(getattr(ds, 'RescaleSlope', 1.0))
intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
gt = ds.pixel_array.astype(np.float32) * slope + intercept

# ── Normalize both to [0,1] for comparison ─────────────────────────
def norm01(img):
    v0, v1 = np.percentile(img, [2, 98])
    return np.clip((img - v0) / max(v1 - v0, 1e-10), 0, 1).astype(np.float64)

gt_n = norm01(gt)

# LR flip the recon
rec_flip = rec[:, ::-1]
rec_flip_n = norm01(rec_flip)

# ── Find optimal scale + translation ──────────────────────────────
def apply_scale(img, scale, output_shape):
    """Rescale img by `scale` and center-crop/pad to output_shape."""
    h, w = img.shape
    new_h, new_w = int(h * scale), int(w * scale)
    scaled = zoom(img, scale, order=1)
    out = np.zeros(output_shape, dtype=np.float64)
    # Center the scaled image
    y0 = (output_shape[0] - scaled.shape[0]) // 2
    x0 = (output_shape[1] - scaled.shape[1]) // 2
    # Crop region from scaled
    sy0 = max(0, -y0)
    sx0 = max(0, -x0)
    sy1 = min(scaled.shape[0], output_shape[0] - y0)
    sx1 = min(scaled.shape[1], output_shape[1] - x0)
    dy0 = max(0, y0)
    dx0 = max(0, x0)
    dy1 = dy0 + (sy1 - sy0)
    dx1 = dx0 + (sx1 - sx0)
    out[dy0:dy1, dx0:dx1] = scaled[sy0:sy1, sx0:sx1]
    return out


def diff_at_scale(scale):
    """Compute MSE between scaled recon and GT, using body mask."""
    scaled = apply_scale(rec_flip_n, scale, gt_n.shape)
    # Only compare within body region (non-zero in both)
    mask = (scaled > 0.05) & (gt_n > 0.05)
    if mask.sum() < 1000:
        return 1e10
    return np.mean((scaled[mask] - gt_n[mask])**2)


# Sweep scale factors
scales = np.linspace(0.7, 1.3, 61)
losses = [diff_at_scale(s) for s in scales]
best_idx = np.argmin(losses)
coarse_best = scales[best_idx]
print(f"\nCoarse sweep: best scale = {coarse_best:.4f}, loss = {losses[best_idx]:.6f}")

# Fine search around best
result = minimize_scalar(diff_at_scale, bounds=(coarse_best - 0.05, coarse_best + 0.05), method='bounded')
optimal_scale = result.x
optimal_loss = result.fun
print(f"Fine search:  best scale = {optimal_scale:.4f}, loss = {optimal_loss:.6f}")
print(f"  → Recon needs {(optimal_scale-1)*100:+.1f}% scaling to match GT")

# ── Also try with translation ──────────────────────────────────────
def diff_at_scale_shift(params):
    scale, dy, dx = params
    scaled = apply_scale(rec_flip_n, scale, gt_n.shape)
    shifted = ndshift(scaled, [dy, dx], order=1)
    mask = (shifted > 0.05) & (gt_n > 0.05)
    if mask.sum() < 1000:
        return 1e10
    return np.mean((shifted[mask] - gt_n[mask])**2)

from scipy.optimize import minimize
res2 = minimize(diff_at_scale_shift, [optimal_scale, 0, 0],
                method='Nelder-Mead',
                options={'xatol': 0.001, 'fatol': 1e-8, 'maxiter': 500})
best_scale, best_dy, best_dx = res2.x
print(f"\nWith translation: scale={best_scale:.4f}, dy={best_dy:.1f}px, dx={best_dx:.1f}px")
print(f"  loss = {res2.fun:.6f}")

# ── Plot ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

# Apply best transform
rec_best = apply_scale(rec_flip_n, best_scale, gt_n.shape)
rec_best = ndshift(rec_best, [best_dy, best_dx], order=1)

# Row 0: images
axes[0, 0].imshow(rec_flip_n, cmap='gray')
axes[0, 0].set_title("Recon (LR flipped, normalized)")
axes[0, 1].imshow(gt_n, cmap='gray')
axes[0, 1].set_title("GT (normalized)")
axes[0, 2].imshow(rec_best, cmap='gray')
axes[0, 2].set_title(f"Recon scaled {best_scale:.3f}\n+ shift ({best_dx:.1f}, {best_dy:.1f})")
axes[0, 3].imshow(gt_n, cmap='gray')
axes[0, 3].set_title("GT (for reference)")

# Row 1: diff maps
diff_raw = rec_flip_n - gt_n
diff_best = rec_best - gt_n
vlim = max(abs(diff_raw).max(), abs(diff_best).max(), 0.01) * 0.5

axes[1, 0].imshow(diff_raw, cmap='RdBu', vmin=-vlim, vmax=vlim)
axes[1, 0].set_title(f"Diff (no scale)\nMSE={np.mean(diff_raw**2):.5f}")
axes[1, 1].imshow(diff_best, cmap='RdBu', vmin=-vlim, vmax=vlim)
axes[1, 1].set_title(f"Diff (scale={best_scale:.3f}+shift)\nMSE={np.mean(diff_best**2):.5f}")

# Overlay: recon edges on GT
from scipy.ndimage import sobel
edges_raw = np.sqrt(sobel(rec_flip_n, 0)**2 + sobel(rec_flip_n, 1)**2)
edges_best = np.sqrt(sobel(rec_best, 0)**2 + sobel(rec_best, 1)**2)
edges_gt = np.sqrt(sobel(gt_n, 0)**2 + sobel(gt_n, 1)**2)

eth = np.percentile(edges_gt, 95)
overlay_raw = np.stack([edges_raw > eth, edges_gt > eth, np.zeros_like(gt_n)], axis=-1).astype(float)
overlay_best = np.stack([edges_best > eth, edges_gt > eth, np.zeros_like(gt_n)], axis=-1).astype(float)

axes[1, 2].imshow(overlay_raw)
axes[1, 2].set_title("Edges: Red=Recon(raw) Green=GT")
axes[1, 3].imshow(overlay_best)
axes[1, 3].set_title("Edges: Red=Recon(scaled) Green=GT")

# Row 2: scale sweep + profiles
axes[2, 0].plot(scales, losses, 'b.-')
axes[2, 0].axvline(best_scale, color='r', linestyle='--', label=f'best={best_scale:.3f}')
axes[2, 0].axvline(1.0, color='g', linestyle=':', label='scale=1.0')
axes[2, 0].set_xlabel("Scale factor")
axes[2, 0].set_ylabel("MSE (body region)")
axes[2, 0].set_title("Scale sweep")
axes[2, 0].legend()

# Horizontal profile comparison
mid = ROWS // 2
axes[2, 1].plot(rec_flip_n[mid, :], 'b-', alpha=0.7, label='Recon (flip, no scale)')
axes[2, 1].plot(rec_best[mid, :], 'r-', alpha=0.7, label=f'Recon (scale={best_scale:.3f})')
axes[2, 1].plot(gt_n[mid, :], 'g-', alpha=0.7, label='GT')
axes[2, 1].set_title(f"Horizontal profile row={mid}")
axes[2, 1].legend(fontsize=8)

# Vertical profile
mid_c = COLS // 2
axes[2, 2].plot(rec_flip_n[:, mid_c], 'b-', alpha=0.7, label='Recon (no scale)')
axes[2, 2].plot(rec_best[:, mid_c], 'r-', alpha=0.7, label=f'Recon (scaled)')
axes[2, 2].plot(gt_n[:, mid_c], 'g-', alpha=0.7, label='GT')
axes[2, 2].set_title(f"Vertical profile col={mid_c}")
axes[2, 2].legend(fontsize=8)

axes[2, 3].axis('off')
axes[2, 3].text(0.1, 0.7, f"Optimal scale: {best_scale:.4f}\n"
                f"Shift: dx={best_dx:.1f}, dy={best_dy:.1f} px\n"
                f"MSE raw: {np.mean(diff_raw**2):.5f}\n"
                f"MSE scaled: {np.mean(diff_best**2):.5f}\n"
                f"Improvement: {(1-np.mean(diff_best**2)/np.mean(diff_raw**2))*100:.1f}%",
                fontsize=14, transform=axes[2, 3].transAxes, verticalalignment='top',
                fontfamily='monospace')

for ax in axes.flat:
    ax.axis('off') if not ax.lines else None

plt.suptitle(f"Optimal Scale Search: Recon (LR flip) vs GT @ slice {TARGET_SLICE}", fontsize=14)
plt.tight_layout()
out = os.path.join(OUT_DIR, "optimal_scale_search.png")
plt.savefig(out, dpi=150)
print(f"\nSaved -> {out}")
print("Done.", flush=True)
