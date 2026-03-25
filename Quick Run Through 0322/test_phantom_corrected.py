"""
Phantom test with CORRECTED geometry to measure intrinsic Katsevich scale bias.
Uses flat_psize_cols=1.3737 and psize_row=1.0947 (from corrected dicom.py).
If phantom scale ≈ 1.0 → 2.4% on clinical data is NOT algorithmic.
If phantom scale ≈ 1.024 → 2.4% is intrinsic to the pipeline.
"""
import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Quick Run Through 0304"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from backproject_safe import ensure_astra_cuda_init
ensure_astra_cuda_init()

import numpy as np
from time import time
from scipy.optimize import minimize_scalar
import astra, matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from pykatsevich.geometry import astra_helical_views
from pykatsevich.initialize import create_configuration
from pykatsevich.filter import filter_katsevich, sino_weight_td
from backproject_cupy import backproject_cupy

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── CORRECTED geometry (matching dicom.py output) ────────────────────
SOD = 595.0
SDD = 1085.6
PSIZE_COL = 1.3737    # flat pixel size after equi-angular resampling (corrected)
PSIZE_ROW = 1.0947    # row spacing at detector (corrected)
DET_ROWS = 64
DET_COLS = 736

VOXEL_SIZE = 0.664
ROWS = COLS = 512
SLICES = 64
N_ANGLES = 2000
PITCH_PER_RAD = 2.0

ANGLE_STEP = 2 * np.pi / 1000
ANGLES = np.arange(N_ANGLES, dtype=np.float64) * ANGLE_STEP
Z_SHIFT = PITCH_PER_RAD * (ANGLES - ANGLES.mean())

# ── Create phantom: two spheres ──────────────────────────────────────
print("Creating phantom...", flush=True)
half_xy = COLS * VOXEL_SIZE * 0.5
half_z = SLICES * VOXEL_SIZE * 0.5

vol_geom_full = astra.create_vol_geom(
    ROWS, COLS, SLICES,
    -half_xy, half_xy, -half_xy, half_xy, -half_z, half_z
)

# Sphere 1: center (50, 30, 0)mm, radius 40mm
# Sphere 2: center (-60, -40, 0)mm, radius 25mm
X = np.linspace(-half_xy + VOXEL_SIZE/2, half_xy - VOXEL_SIZE/2, COLS)
Y = np.linspace(-half_xy + VOXEL_SIZE/2, half_xy - VOXEL_SIZE/2, ROWS)
Z = np.linspace(-half_z + VOXEL_SIZE/2, half_z - VOXEL_SIZE/2, SLICES)
XX, YY = np.meshgrid(X, Y)

phantom = np.zeros((SLICES, ROWS, COLS), dtype=np.float32)
for iz in range(SLICES):
    r1 = np.sqrt((XX - 50)**2 + (YY - 30)**2 + Z[iz]**2)
    r2 = np.sqrt((XX + 60)**2 + (YY + 40)**2 + Z[iz]**2)
    phantom[iz][r1 < 40] = 1.0
    phantom[iz][r2 < 25] = 0.5

print(f"  Non-zero voxels: {(phantom > 0).sum()}")

# ── Forward project ──────────────────────────────────────────────────
print("Forward projecting...", flush=True)
t0 = time()

views = astra_helical_views(
    SOD, SDD, (PSIZE_COL + PSIZE_ROW) / 2, ANGLES, PITCH_PER_RAD * ANGLE_STEP,
    vertical_shifts=Z_SHIFT,
    pixel_size_col=PSIZE_COL, pixel_size_row=PSIZE_ROW,
)

proj_geom = astra.create_proj_geom('cone_vec', DET_ROWS, DET_COLS, views)
proj_id = astra.data3d.create('-sino', proj_geom, 0)
vol_id = astra.data3d.create('-vol', vol_geom_full, phantom)

cfg_fp = astra.astra_dict('FP3D_CUDA')
cfg_fp['ProjectionDataId'] = proj_id
cfg_fp['VolumeDataId'] = vol_id
alg_fp = astra.algorithm.create(cfg_fp)
astra.algorithm.run(alg_fp)

sino = np.swapaxes(astra.data3d.get(proj_id), 0, 1)
sino = np.ascontiguousarray(sino, dtype=np.float32)
astra.algorithm.delete([alg_fp])
astra.data3d.delete([proj_id, vol_id])
print(f"  FP done in {time()-t0:.1f}s, shape={sino.shape}")

# ── Reconstruct center slice ─────────────────────────────────────────
TARGET_SLICE = SLICES // 2
z_c = -half_z + TARGET_SLICE * VOXEL_SIZE
z_min_sl, z_max_sl = z_c, z_c + VOXEL_SIZE

source_z = views[:, 2]
cone_half_z = 0.5 * DET_ROWS * PSIZE_ROW * (SOD / SDD)
projs_per_turn = 2 * np.pi / ANGLE_STEP
margin_z = projs_per_turn * PITCH_PER_RAD * ANGLE_STEP
keep = np.where(
    (source_z + cone_half_z + margin_z >= z_min_sl) &
    (source_z - cone_half_z - margin_z <= z_max_sl)
)[0]

sino_c = sino[keep].copy()
angles_c = ANGLES[keep].copy()
views_c = views[keep]

scan_geom = {
    'SOD': SOD, 'SDD': SDD,
    'detector': {
        'detector psize': float(np.mean([PSIZE_COL, PSIZE_ROW])),
        'detector psize cols': PSIZE_COL,
        'detector psize rows': PSIZE_ROW,
        'detector rows': DET_ROWS,
        'detector cols': DET_COLS,
    },
    'helix': {
        'angles_count': len(angles_c),
        'pitch_mm_rad': PITCH_PER_RAD,
        'angles_range': float(abs(angles_c[-1] - angles_c[0])),
    },
}

vol_geom_1 = astra.create_vol_geom(
    ROWS, COLS, 1,
    -half_xy, half_xy, -half_xy, half_xy, z_min_sl, z_max_sl
)
proj_geom_1 = astra.create_proj_geom('cone_vec', DET_ROWS, DET_COLS, views_c)

conf = create_configuration(scan_geom, vol_geom_1)
conf['source_pos'] = angles_c.astype(np.float32)
conf['delta_s'] = float(np.mean(np.diff(angles_c)))

print(f"\nReconstructing slice {TARGET_SLICE} with {len(keep)} projections...", flush=True)
t0 = time()
filtered = filter_katsevich(
    np.asarray(sino_c, dtype=np.float32, order='C'), conf,
    {'Diff': {'Print time': False}, 'FwdRebin': {'Print time': False},
     'BackRebin': {'Print time': False}}
)
sino_td = sino_weight_td(filtered, conf, False)
rec = backproject_cupy(sino_td, conf, vol_geom_1, proj_geom_1, tqdm_bar=False)[:, :, 0]
print(f"  Done in {time()-t0:.1f}s")

# ── Measure scale ────────────────────────────────────────────────────
phantom_slice = phantom[TARGET_SLICE]

# Normalize both
def norm01(img):
    v0, v1 = np.percentile(img, [2, 98])
    return np.clip((img - v0) / max(v1 - v0, 1e-10), 0, 1).astype(np.float64)

gt_n = norm01(phantom_slice)
rec_n = norm01(rec)

from scipy.ndimage import zoom as ndzoom

def apply_scale(img, scale, output_shape):
    scaled = ndzoom(img, scale, order=1)
    out = np.zeros(output_shape, dtype=np.float64)
    y0 = (output_shape[0] - scaled.shape[0]) // 2
    x0 = (output_shape[1] - scaled.shape[1]) // 2
    sy0, sx0 = max(0, -y0), max(0, -x0)
    sy1 = min(scaled.shape[0], output_shape[0] - y0)
    sx1 = min(scaled.shape[1], output_shape[1] - x0)
    dy0, dx0 = max(0, y0), max(0, x0)
    out[dy0:dy0+(sy1-sy0), dx0:dx0+(sx1-sx0)] = scaled[sy0:sy1, sx0:sx1]
    return out

def mse_at_scale(scale):
    scaled = apply_scale(rec_n, scale, gt_n.shape)
    mask = (scaled > 0.05) & (gt_n > 0.05)
    if mask.sum() < 1000:
        return 1e10
    return np.mean((scaled[mask] - gt_n[mask])**2)

# Coarse sweep
scales = np.linspace(0.9, 1.1, 41)
losses = [mse_at_scale(s) for s in scales]
coarse_best = scales[np.argmin(losses)]

# Fine search
result = minimize_scalar(mse_at_scale, bounds=(coarse_best - 0.03, coarse_best + 0.03), method='bounded')
opt_scale = result.x
print(f"\nMSE optimal scale: {opt_scale:.4f} ({(opt_scale-1)*100:+.1f}%)")
print(f"MSE at optimal: {result.fun:.6f}")

# Also measure sphere positions/radii
from scipy.ndimage import label, sobel

rec_mask = rec_n > 0.3
labeled, n_feat = label(rec_mask)
gt_centers = {
    'sphere1': ((50 + half_xy) / VOXEL_SIZE, (30 + half_xy) / VOXEL_SIZE, 40/VOXEL_SIZE),
    'sphere2': ((-60 + half_xy) / VOXEL_SIZE, (-40 + half_xy) / VOXEL_SIZE, 25/VOXEL_SIZE),
}

print(f"\nFound {n_feat} regions:")
for i in range(1, min(n_feat + 1, 5)):
    region = labeled == i
    ys, xs = np.where(region)
    if len(ys) < 50:
        continue
    cy, cx = ys.mean(), xs.mean()
    r90 = np.percentile(np.sqrt((ys - cy)**2 + (xs - cx)**2), 90)
    # Match to GT sphere
    d1 = np.sqrt((cx - gt_centers['sphere1'][0])**2 + (cy - gt_centers['sphere1'][1])**2)
    d2 = np.sqrt((cx - gt_centers['sphere2'][0])**2 + (cy - gt_centers['sphere2'][1])**2)
    name = 'sphere1' if d1 < d2 else 'sphere2'
    gt_cx, gt_cy, gt_r = gt_centers[name]
    print(f"  {name}: center=({cx:.1f},{cy:.1f}) vs GT({gt_cx:.1f},{gt_cy:.1f}), "
          f"offset=({cx-gt_cx:.1f},{cy-gt_cy:.1f})px, "
          f"r90={r90:.1f} vs GT_r={gt_r:.1f}, ratio={r90/gt_r:.4f}")

# ── Plot ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].imshow(phantom_slice, cmap='gray')
axes[0, 0].set_title("Phantom (GT)")
axes[0, 1].imshow(rec_n, cmap='gray')
axes[0, 1].set_title("Reconstruction (normalized)")

# Edges overlay
ep = np.sqrt(sobel(phantom_slice.astype(float), 0)**2 + sobel(phantom_slice.astype(float), 1)**2)
er = np.sqrt(sobel(rec_n, 0)**2 + sobel(rec_n, 1)**2)
eth_p = np.percentile(ep[ep > 0], 70) if (ep > 0).any() else 0.1
eth_r = np.percentile(er[er > 0], 90) if (er > 0).any() else 0.1
overlay = np.stack([er > eth_r, ep > eth_p, np.zeros_like(rec_n)], axis=-1).astype(float)
axes[0, 2].imshow(overlay)
axes[0, 2].set_title("Edges: Red=Recon, Green=Phantom")

# Scale sweep
axes[1, 0].plot(scales, losses, 'b.-')
axes[1, 0].axvline(opt_scale, color='r', ls='--', label=f'best={opt_scale:.4f}')
axes[1, 0].axvline(1.0, color='g', ls=':', label='1.0')
axes[1, 0].set_xlabel("Scale factor")
axes[1, 0].set_ylabel("MSE")
axes[1, 0].set_title("Scale sweep")
axes[1, 0].legend()

# Profiles through sphere 1
iy_s1 = int(gt_centers['sphere1'][1])
axes[1, 1].plot(phantom_slice[iy_s1, :], 'g-', alpha=0.7, label='Phantom')
axes[1, 1].plot(rec_n[iy_s1, :], 'r-', alpha=0.7, label='Recon')
axes[1, 1].set_title(f"Horizontal profile y={iy_s1}")
axes[1, 1].legend(fontsize=8)

# Summary
axes[1, 2].axis('off')
txt = (f"Corrected geometry:\n"
       f"  SOD={SOD}, SDD={SDD}\n"
       f"  psize_col={PSIZE_COL:.4f} (corrected flat)\n"
       f"  psize_row={PSIZE_ROW:.4f}\n"
       f"  voxel={VOXEL_SIZE}mm\n\n"
       f"MSE optimal scale: {opt_scale:.4f}\n"
       f"  → {(opt_scale-1)*100:+.2f}% deviation\n\n"
       f"Conclusion:\n")
if abs(opt_scale - 1.0) < 0.01:
    txt += "  Pipeline is geometrically correct.\n  2.4% on clinical data is NOT algorithmic."
elif abs(opt_scale - 1.024) < 0.01:
    txt += "  Pipeline has intrinsic ~2.4% bias.\n  Clinical 2.4% is algorithmic."
else:
    txt += f"  Pipeline bias = {(opt_scale-1)*100:+.1f}%\n  Partial explanation for clinical 2.4%."
axes[1, 2].text(0.05, 0.95, txt, fontsize=12, transform=axes[1, 2].transAxes,
                va='top', fontfamily='monospace')

plt.suptitle("Phantom Test: Katsevich Scale with Corrected Geometry", fontsize=14)
plt.tight_layout()
out = os.path.join(OUT_DIR, "phantom_corrected_test.png")
plt.savefig(out, dpi=150)
print(f"\nSaved -> {out}")
print("Done.", flush=True)
