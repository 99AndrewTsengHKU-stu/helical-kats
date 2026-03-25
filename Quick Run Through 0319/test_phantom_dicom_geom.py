"""
Diagnostic: Does the Katsevich pipeline produce correct scale when using
DICOM-like geometry (SOD=595, SDD=1085.6, psize=0.944) with a synthetic
phantom (no equi-angular resampling)?

If phantom scale is correct → bug is in DICOM data loading / equi-angular resampling
If phantom scale is wrong  → bug is in the Katsevich pipeline itself
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
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from pykatsevich.geometry import astra_helical_views
from pykatsevich.initialize import create_configuration
from pykatsevich.filter import filter_katsevich, sino_weight_td
from backproject_cupy import backproject_cupy

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Use DICOM-like geometry ─────────────────────────────────────────
SOD = 595.0
SDD = 1085.6
PSIZE_COL = 0.9437   # flat pixel size after equi-angular resampling
PSIZE_ROW = 0.9265
DET_ROWS = 64
DET_COLS = 736

VOXEL_SIZE = 0.664    # same as DICOM GT
ROWS = COLS = 512
SLICES = 64           # fewer slices for speed
N_ANGLES = 2000       # enough projections

PITCH_PER_RAD = 2.0   # moderate pitch
ANGLE_RANGE = N_ANGLES * 2 * np.pi / N_ANGLES  # just describe full range
# Actually compute properly
ANGLE_STEP = 2 * np.pi / 1000   # ~1000 projs per turn
ANGLES = np.arange(N_ANGLES, dtype=np.float64) * ANGLE_STEP
ANGLE_RANGE = ANGLES[-1] - ANGLES[0]
Z_SHIFT = PITCH_PER_RAD * (ANGLES - ANGLES.mean())

# ── Create phantom: sphere at known position ────────────────────────
print("Creating phantom...", flush=True)
half_xy = COLS * VOXEL_SIZE * 0.5
half_z = SLICES * VOXEL_SIZE * 0.5  # use same voxel size for z

vol_geom_full = astra.create_vol_geom(
    ROWS, COLS, SLICES,
    -half_xy, half_xy, -half_xy, half_xy, -half_z, half_z
)

# Sphere at (x=50mm, y=30mm, z=0mm) with radius 40mm
phantom = np.zeros((SLICES, ROWS, COLS), dtype=np.float32)
for iz in range(SLICES):
    z = -half_z + (iz + 0.5) * VOXEL_SIZE
    for iy in range(ROWS):
        y = -half_xy + (iy + 0.5) * VOXEL_SIZE
        for ix in range(COLS):
            x = -half_xy + (ix + 0.5) * VOXEL_SIZE
            r2 = (x - 50)**2 + (y - 30)**2 + z**2
            if r2 < 40**2:
                phantom[iz, iy, ix] = 1.0
            # Second sphere at (-60, -40, 0) radius 25
            r2b = (x + 60)**2 + (y + 40)**2 + z**2
            if r2b < 25**2:
                phantom[iz, iy, ix] = 0.5

print(f"  Phantom range: [{phantom.min():.2f}, {phantom.max():.2f}]")
print(f"  Non-zero voxels: {(phantom > 0).sum()}")

# ── Forward project with ASTRA ──────────────────────────────────────
print("Forward projecting...", flush=True)
t0 = time()

views = astra_helical_views(
    SOD, SDD, (PSIZE_COL + PSIZE_ROW) / 2, ANGLES, PITCH_PER_RAD * ANGLE_STEP,
    vertical_shifts=Z_SHIFT,
    pixel_size_col=PSIZE_COL, pixel_size_row=PSIZE_ROW,
)

proj_geom = astra.create_proj_geom('cone_vec', DET_ROWS, DET_COLS, views)

# Use ASTRA FP3D_CUDA for forward projection
proj_id = astra.data3d.create('-sino', proj_geom, 0)
vol_id = astra.data3d.create('-vol', vol_geom_full, phantom)

cfg_fp = astra.astra_dict('FP3D_CUDA')
cfg_fp['ProjectionDataId'] = proj_id
cfg_fp['VolumeDataId'] = vol_id
alg_fp = astra.algorithm.create(cfg_fp)
astra.algorithm.run(alg_fp)

sino = astra.data3d.get(proj_id)  # shape: (DET_ROWS, N_ANGLES, DET_COLS)
sino = np.swapaxes(sino, 0, 1)   # → (N_ANGLES, DET_ROWS, DET_COLS)
sino = np.ascontiguousarray(sino, dtype=np.float32)

astra.algorithm.delete([alg_fp])
astra.data3d.delete([proj_id, vol_id])
print(f"  FP done in {time()-t0:.1f}s, sino shape={sino.shape}, range=[{sino.min():.4f}, {sino.max():.4f}]")

# ── Reconstruct center slice ────────────────────────────────────────
TARGET_SLICE = SLICES // 2
z_c = -half_z + TARGET_SLICE * VOXEL_SIZE
z_min_sl, z_max_sl = z_c, z_c + VOXEL_SIZE

# Z-cull
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
print(f"  Done in {time()-t0:.1f}s, range=[{rec.min():.5f}, {rec.max():.5f}]")

# ── Compare positions ────────────────────────────────────────────────
# GT: sphere 1 center at (x=50, y=30), radius=40mm
# In pixel coords: ix = (50 + half_xy) / VOXEL_SIZE = (50+170)/0.664 = 331.3
#                   iy = (30 + half_xy) / VOXEL_SIZE = (30+170)/0.664 = 301.2
gt_center1_px = ((50 + half_xy) / VOXEL_SIZE, (30 + half_xy) / VOXEL_SIZE)
gt_radius1_px = 40.0 / VOXEL_SIZE

# GT: sphere 2 center at (x=-60, y=-40), radius=25mm
gt_center2_px = ((-60 + half_xy) / VOXEL_SIZE, (-40 + half_xy) / VOXEL_SIZE)
gt_radius2_px = 25.0 / VOXEL_SIZE

print(f"\nGT sphere 1: center=({gt_center1_px[0]:.1f}, {gt_center1_px[1]:.1f}) px, radius={gt_radius1_px:.1f} px")
print(f"GT sphere 2: center=({gt_center2_px[0]:.1f}, {gt_center2_px[1]:.1f}) px, radius={gt_radius2_px:.1f} px")

# Phantom center slice (for comparison)
phantom_slice = phantom[TARGET_SLICE]

# Measure sphere positions in reconstruction
from scipy.ndimage import sobel, label, center_of_mass

# Threshold reconstruction
rec_norm = rec.copy()
rec_norm -= rec_norm.min()
if rec_norm.max() > 0:
    rec_norm /= rec_norm.max()

threshold = 0.3
rec_mask = rec_norm > threshold
labeled, n_features = label(rec_mask)
print(f"\nFound {n_features} connected regions in reconstruction")

for feat_idx in range(1, min(n_features + 1, 5)):
    region = labeled == feat_idx
    ys, xs = np.where(region)
    if len(ys) < 50:
        continue
    cy, cx = ys.mean(), xs.mean()
    radii = np.sqrt((ys - cy)**2 + (xs - cx)**2)
    r90 = np.percentile(radii, 90)

    # Compare with GT
    d1 = np.sqrt((cx - gt_center1_px[0])**2 + (cy - gt_center1_px[1])**2)
    d2 = np.sqrt((cx - gt_center2_px[0])**2 + (cy - gt_center2_px[1])**2)
    if d1 < d2:
        gt_label = "sphere1"
        gt_cx, gt_cy = gt_center1_px
        gt_r = gt_radius1_px
    else:
        gt_label = "sphere2"
        gt_cx, gt_cy = gt_center2_px
        gt_r = gt_radius2_px

    scale_pos = np.sqrt((cx - COLS/2)**2 + (cy - ROWS/2)**2) / max(np.sqrt((gt_cx - COLS/2)**2 + (gt_cy - ROWS/2)**2), 1e-10)
    print(f"  Region {feat_idx} ({gt_label}): center=({cx:.1f}, {cy:.1f}), r90={r90:.1f} px")
    print(f"    GT: center=({gt_cx:.1f}, {gt_cy:.1f}), radius={gt_r:.1f} px")
    print(f"    Position offset: ({cx-gt_cx:.1f}, {cy-gt_cy:.1f}) px")
    print(f"    Radius ratio (recon/GT): {r90/gt_r:.4f}")
    print(f"    Position scale from center: {scale_pos:.4f}")

# ── Plot ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Row 0: phantom, reconstruction, overlay
axes[0, 0].imshow(phantom_slice, cmap='gray')
axes[0, 0].set_title(f"Phantom slice {TARGET_SLICE}")
# Mark sphere centers
axes[0, 0].plot(gt_center1_px[0], gt_center1_px[1], 'r+', markersize=15, markeredgewidth=2)
axes[0, 0].plot(gt_center2_px[0], gt_center2_px[1], 'r+', markersize=15, markeredgewidth=2)

v0, v1 = np.percentile(rec, [2, 98])
axes[0, 1].imshow(rec, cmap='gray', vmin=v0, vmax=v1)
axes[0, 1].set_title("Reconstruction")
# Mark GT positions
axes[0, 1].plot(gt_center1_px[0], gt_center1_px[1], 'r+', markersize=15, markeredgewidth=2, label='GT center')
axes[0, 1].plot(gt_center2_px[0], gt_center2_px[1], 'r+', markersize=15, markeredgewidth=2)
axes[0, 1].legend(fontsize=8)

# Overlay: phantom edges on reconstruction
edges_phantom = np.sqrt(sobel(phantom_slice.astype(float), 0)**2 + sobel(phantom_slice.astype(float), 1)**2)
edges_rec = np.sqrt(sobel(rec_norm, 0)**2 + sobel(rec_norm, 1)**2)
eth_p = np.percentile(edges_phantom[edges_phantom > 0], 70) if (edges_phantom > 0).any() else 0.1
eth_r = np.percentile(edges_rec[edges_rec > 0], 90) if (edges_rec > 0).any() else 0.1

overlay = np.stack([
    edges_rec > eth_r,       # Red = recon edges
    edges_phantom > eth_p,   # Green = phantom edges
    np.zeros_like(rec_norm)  # Blue
], axis=-1).astype(float)
axes[0, 2].imshow(overlay)
axes[0, 2].set_title("Edges: Red=Recon, Green=Phantom")

# Row 1: profiles
mid_y = ROWS // 2
axes[1, 0].plot(phantom_slice[int(gt_center1_px[1]), :], 'g-', alpha=0.7, label='Phantom')
axes[1, 0].plot(rec_norm[int(gt_center1_px[1]), :], 'r-', alpha=0.7, label='Recon (norm)')
axes[1, 0].axvline(gt_center1_px[0], color='g', ls='--', alpha=0.5, label='GT center')
axes[1, 0].set_title(f"Horizontal profile y={int(gt_center1_px[1])}")
axes[1, 0].legend(fontsize=8)

axes[1, 1].plot(phantom_slice[:, int(gt_center1_px[0])], 'g-', alpha=0.7, label='Phantom')
axes[1, 1].plot(rec_norm[:, int(gt_center1_px[0])], 'r-', alpha=0.7, label='Recon (norm)')
axes[1, 1].axvline(gt_center1_px[1], color='g', ls='--', alpha=0.5, label='GT center')
axes[1, 1].set_title(f"Vertical profile x={int(gt_center1_px[0])}")
axes[1, 1].legend(fontsize=8)

# Summary text
info_text = (
    f"Geometry:\n"
    f"  SOD={SOD}, SDD={SDD}\n"
    f"  psize_col={PSIZE_COL:.4f}, psize_row={PSIZE_ROW:.4f}\n"
    f"  voxel_size={VOXEL_SIZE}\n"
    f"  psize/voxel = {PSIZE_COL/VOXEL_SIZE:.4f}\n"
    f"  det: {DET_ROWS}x{DET_COLS}\n"
    f"  vol: {ROWS}x{COLS}x{SLICES}\n"
    f"  FOV: {COLS*VOXEL_SIZE:.1f}mm\n"
    f"\nSphere 1 GT: ({gt_center1_px[0]:.0f},{gt_center1_px[1]:.0f}) r={gt_radius1_px:.0f}px\n"
    f"Sphere 2 GT: ({gt_center2_px[0]:.0f},{gt_center2_px[1]:.0f}) r={gt_radius2_px:.0f}px\n"
)
axes[1, 2].axis('off')
axes[1, 2].text(0.05, 0.95, info_text, fontsize=11, transform=axes[1, 2].transAxes,
                verticalalignment='top', fontfamily='monospace')

plt.suptitle("Phantom Test with DICOM Geometry: Scale Diagnostic", fontsize=14)
plt.tight_layout()
out = os.path.join(OUT_DIR, "phantom_dicom_geom_test.png")
plt.savefig(out, dpi=150)
print(f"\nSaved -> {out}")
print("Done.", flush=True)
