"""
Validate the det_col_offset / det_row_offset fix in backproject_cupy.
Reconstructs 3 slices and compares MAE vs GT before/after the fix.
"""
import sys, os, copy, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Quick Run Through 0304"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from backproject_safe import ensure_astra_cuda_init
ensure_astra_cuda_init()

import numpy as np
from time import time
from pathlib import Path
import astra, pydicom, matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from pykatsevich import load_dicom_projections
from pykatsevich.geometry import astra_helical_views
from pykatsevich.initialize import create_configuration
from pykatsevich.filter_gpu import filter_katsevich_gpu as filter_katsevich, sino_weight_td_gpu_np as sino_weight_td
from backproject_cupy import backproject_cupy

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DICOM_DIR = r"D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD"
GT_DIR = Path(r"D:\AAPM-Data\L067\L067\full_1mm")

ROWS = COLS = 512
SLICES = 560
VOXEL_SIZE_XY = 0.6640625
VOXEL_SIZE_Z = 0.8
RECON_OFFSET_X = +11.0

# Test these 3 slices (spaced ~1 turn apart to see per-turn variation)
TEST_SLICES = [200, 229, 258]   # ~28.7 slice apart = 1 turn each

print("Loading DICOM...", flush=True)
t0 = time()
sino, meta = load_dicom_projections(DICOM_DIR)
print(f"Loaded in {time()-t0:.1f}s: {sino.shape}", flush=True)

sg = meta['scan_geometry']
angles_full = -meta['angles_rad'].copy() - np.pi / 2
ffs_z = meta.get('ffs_z_offsets_mm', np.zeros(len(meta['angles_rad']), dtype=np.float32))
z_shift = meta['table_positions_mm'] + ffs_z - (meta['table_positions_mm'] + ffs_z).mean()
sino_work = sino[:, ::-1, :].copy()
del sino; gc.collect()

scan_geom = copy.deepcopy(sg)
scan_geom['helix']['pitch_mm_rad'] = float(abs(meta['pitch_mm_per_rad_signed']))

psize_cols = sg['detector'].get('detector psize cols', sg['detector']['detector psize'])
psize_rows = sg['detector'].get('detector psize rows', sg['detector']['detector psize'])
psize = sg['detector']['detector psize']
det_rows_n = sg['detector']['detector rows']
det_col_offset = sg['detector'].get('detector_col_offset', 0.0)
det_row_offset = sg['detector'].get('detector_row_offset', 0.0)

print(f"det_col_offset = {det_col_offset:.4f} px")
print(f"det_row_offset = {det_row_offset:.4f} px")

views = astra_helical_views(
    sg['SOD'], sg['SDD'], psize, angles_full, meta['pitch_mm_per_angle'],
    vertical_shifts=z_shift, pixel_size_col=psize_cols, pixel_size_row=psize_rows,
    detector_col_offset=det_col_offset, detector_row_offset=det_row_offset,
)

source_z = views[:, 2]
cone_half_z = 0.5 * det_rows_n * psize * (sg['SOD'] / sg['SDD'])
angle_step = abs(np.mean(np.diff(np.arctan2(views[:10, 1], views[:10, 0]))))
projs_per_turn = 2 * np.pi / max(angle_step, 1e-12)
margin_z = projs_per_turn * abs(meta['pitch_mm_per_angle'])

half_xy = COLS * VOXEL_SIZE_XY * 0.5
total_half_z = SLICES * VOXEL_SIZE_Z * 0.5

gt_files = sorted(GT_DIR.glob("*.IMA")) or sorted(GT_DIR.glob("*.dcm"))
print(f"GT files: {len(gt_files)}")

def norm01(img):
    v0, v1 = np.percentile(img, [2, 98])
    return np.clip((img - v0) / max(v1 - v0, 1e-10), 0, 1)

def recon_slice(sl, with_offset):
    """Reconstruct a single slice. with_offset controls whether offsets are used."""
    z_min_c = -total_half_z + sl * VOXEL_SIZE_Z
    z_max_c = z_min_c + VOXEL_SIZE_Z

    keep = np.where(
        (source_z + cone_half_z + margin_z >= z_min_c) &
        (source_z - cone_half_z - margin_z <= z_max_c)
    )[0]

    sino_c = sino_work[keep].copy()
    angles_c = angles_full[keep].copy()
    views_c = views[keep]

    sg_c = copy.deepcopy(scan_geom)
    sg_c['helix']['angles_range'] = float(abs(angles_c[-1] - angles_c[0]))
    sg_c['helix']['angles_count'] = len(angles_c)

    vol_geom = astra.create_vol_geom(
        ROWS, COLS, 1,
        -half_xy + RECON_OFFSET_X, half_xy + RECON_OFFSET_X,
        -half_xy, half_xy,
        z_min_c, z_max_c,
    )
    proj_geom = astra.create_proj_geom('cone_vec', det_rows_n, sg['detector']['detector cols'], views_c)

    conf = create_configuration(sg_c, vol_geom)
    conf['source_pos'] = angles_c.astype(np.float32)
    conf['delta_s'] = float(np.mean(np.diff(angles_c)))

    # Control: zero out offsets to simulate old behavior
    if not with_offset:
        conf['detector_col_offset'] = 0.0
        conf['detector_row_offset'] = 0.0

    filtered = filter_katsevich(np.asarray(sino_c, dtype=np.float32, order='C'), conf)
    sino_td = sino_weight_td(filtered, conf)
    rec = backproject_cupy(sino_td, conf, vol_geom, proj_geom)
    return rec[:, :, 0]

# Load GT slices
gt_slices = {}
for sl in TEST_SLICES:
    ds = pydicom.dcmread(str(gt_files[sl]))
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    gt_slices[sl] = ds.pixel_array.astype(np.float32) * slope + intercept

# Reconstruct with and without fix
print("\nReconstructing slices WITHOUT fix (old behavior)...", flush=True)
recs_old = {}
for sl in TEST_SLICES:
    t1 = time()
    recs_old[sl] = recon_slice(sl, with_offset=False)
    print(f"  Slice {sl}: {time()-t1:.1f}s", flush=True)
    gc.collect()

print("\nReconstructing slices WITH fix...", flush=True)
recs_new = {}
for sl in TEST_SLICES:
    t1 = time()
    recs_new[sl] = recon_slice(sl, with_offset=True)
    print(f"  Slice {sl}: {time()-t1:.1f}s", flush=True)
    gc.collect()

# Compare
fig, axes = plt.subplots(len(TEST_SLICES), 5, figsize=(20, 4*len(TEST_SLICES)))
fig.suptitle("det_col/row_offset fix: OLD vs NEW vs GT", fontsize=13)

print("\n=== MAE Comparison ===")
for row, sl in enumerate(TEST_SLICES):
    gt = gt_slices[sl]
    rec_old = recs_old[sl][:, ::-1]   # LR flip to match GT
    rec_new = recs_new[sl][:, ::-1]

    gt_n  = norm01(gt)
    old_n = norm01(rec_old)
    new_n = norm01(rec_new)

    mae_old = float(np.mean(np.abs(old_n - gt_n)))
    mae_new = float(np.mean(np.abs(new_n - gt_n)))
    print(f"  Slice {sl}: OLD MAE={mae_old:.4f}  NEW MAE={mae_new:.4f}  Δ={mae_new-mae_old:+.4f}")

    axes[row, 0].imshow(old_n, cmap='gray'); axes[row, 0].set_title(f"OLD sl={sl}\nMAE={mae_old:.4f}"); axes[row, 0].axis('off')
    axes[row, 1].imshow(new_n, cmap='gray'); axes[row, 1].set_title(f"NEW sl={sl}\nMAE={mae_new:.4f}"); axes[row, 1].axis('off')
    axes[row, 2].imshow(gt_n,  cmap='gray'); axes[row, 2].set_title(f"GT sl={sl}"); axes[row, 2].axis('off')
    axes[row, 3].imshow(old_n - gt_n, cmap='RdBu', vmin=-0.2, vmax=0.2); axes[row, 3].set_title("OLD - GT"); axes[row, 3].axis('off')
    axes[row, 4].imshow(new_n - gt_n, cmap='RdBu', vmin=-0.2, vmax=0.2); axes[row, 4].set_title("NEW - GT"); axes[row, 4].axis('off')

plt.tight_layout()
out = os.path.join(OUT_DIR, "offset_fix_comparison.png")
plt.savefig(out, dpi=130)
print(f"\nSaved -> {out}")
print("Done.", flush=True)
