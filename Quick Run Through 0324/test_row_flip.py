"""
Diagnostic: reconstruct a single z-chunk with and without the detector row flip
to determine if sino[:, ::-1, :] is causing the Y-gradient artifact.
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
VOXEL_SIZE_XY = 0.6640625
VOXEL_SIZE_Z = 0.8
SLICES = 560
CHUNK_SIZE = 20
TEST_CHUNK = 14  # reconstruct slices 280-299 (middle of volume)

print("Loading DICOM projections...", flush=True)
t0 = time()
sino, meta = load_dicom_projections(DICOM_DIR)
t_load = time() - t0
print(f"Loaded in {t_load:.0f}s: {sino.shape}", flush=True)

sg = meta['scan_geometry']
angles_full = -meta['angles_rad'].copy() - np.pi / 2
z_shift = meta['table_positions_mm'] - meta['table_positions_mm'].mean()

psize_cols = sg['detector'].get('detector psize cols', sg['detector']['detector psize'])
psize_rows = sg['detector'].get('detector psize rows', sg['detector']['detector psize'])
psize = sg['detector']['detector psize']
det_rows_n = sg['detector']['detector rows']
det_col_offset = sg['detector'].get('detector_col_offset', 0.0)
det_row_offset = sg['detector'].get('detector_row_offset', 0.0)

views = astra_helical_views(
    sg['SOD'], sg['SDD'], psize, angles_full, meta['pitch_mm_per_angle'],
    vertical_shifts=z_shift, pixel_size_col=psize_cols, pixel_size_row=psize_rows,
    detector_col_offset=det_col_offset, detector_row_offset=det_row_offset,
)

RECON_OFFSET_X = +11.0
half_xy = COLS * VOXEL_SIZE_XY * 0.5
total_half_z = SLICES * VOXEL_SIZE_Z * 0.5

source_z = views[:, 2]
cone_half_z = 0.5 * det_rows_n * psize * (sg['SOD'] / sg['SDD'])
angle_step = abs(np.mean(np.diff(np.arctan2(views[:10, 1], views[:10, 0]))))
projs_per_turn = 2 * np.pi / max(angle_step, 1e-12)
margin_z = projs_per_turn * abs(meta['pitch_mm_per_angle'])

scan_geom = copy.deepcopy(sg)
pitch_abs = float(abs(meta['pitch_mm_per_rad_signed']))
scan_geom['helix']['pitch_mm_rad'] = pitch_abs

# Chunk geometry
sl_start = TEST_CHUNK * CHUNK_SIZE
sl_end = min(sl_start + CHUNK_SIZE, SLICES)
n_sl = sl_end - sl_start
z_min_chunk = -total_half_z + sl_start * VOXEL_SIZE_Z
z_max_chunk = -total_half_z + sl_end * VOXEL_SIZE_Z

keep = np.where(
    (source_z + cone_half_z + margin_z >= z_min_chunk) &
    (source_z - cone_half_z - margin_z <= z_max_chunk)
)[0]
print(f"Test chunk {TEST_CHUNK}: slices {sl_start}-{sl_end-1}, {len(keep)} projs")

angles_c = angles_full[keep].copy()
views_c = views[keep]

sg_c = copy.deepcopy(scan_geom)
sg_c['helix']['angles_range'] = float(abs(angles_c[-1] - angles_c[0]))
sg_c['helix']['angles_count'] = len(angles_c)

vol_geom = astra.create_vol_geom(
    ROWS, COLS, n_sl,
    -half_xy + RECON_OFFSET_X, half_xy + RECON_OFFSET_X,
    -half_xy, half_xy,
    z_min_chunk, z_max_chunk,
)
proj_geom = astra.create_proj_geom('cone_vec', det_rows_n, sg['detector']['detector cols'], views_c)

# Load GT for comparison
gt_files = sorted(GT_DIR.glob("*.IMA")) or sorted(GT_DIR.glob("*.dcm"))
gt_slices = []
for sl in range(sl_start, sl_end):
    ds = pydicom.dcmread(str(gt_files[sl]))
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    gt_slices.append(ds.pixel_array.astype(np.float32) * slope + intercept)

def norm01(img):
    v0, v1 = np.percentile(img, [2, 98])
    return np.clip((img - v0) / max(v1 - v0, 1e-10), 0, 1)

def reconstruct_chunk(sino_chunk, label):
    conf = create_configuration(sg_c, vol_geom)
    conf['source_pos'] = angles_c.astype(np.float32)
    conf['delta_s'] = float(np.mean(np.diff(angles_c)))

    t1 = time()
    filtered = filter_katsevich(
        np.asarray(sino_chunk, dtype=np.float32, order='C'), conf,
        {'Diff': {'Print time': False}, 'FwdRebin': {'Print time': False},
         'BackRebin': {'Print time': False}}
    )
    sino_td = sino_weight_td(filtered, conf, False)
    rec = backproject_cupy(sino_td, conf, vol_geom, proj_geom, tqdm_bar=False)
    print(f"  {label}: {time()-t1:.1f}s")

    import cupy as cp
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    return rec

# Test 1: WITH row flip (current behavior)
print("\nReconstructing WITH row flip...", flush=True)
sino_flipped = sino[keep][:, ::-1, :].copy()
rec_flipped = reconstruct_chunk(sino_flipped, "flipped")

# Test 2: WITHOUT row flip
print("Reconstructing WITHOUT row flip...", flush=True)
sino_noflip = sino[keep].copy()
rec_noflip = reconstruct_chunk(sino_noflip, "no-flip")

del sino
gc.collect()

# Compare
mid_sl = n_sl // 2  # middle slice of chunk
sl_global = sl_start + mid_sl

rec_f = rec_flipped[:, :, mid_sl][:, ::-1]  # LR flip to match GT
rec_n = rec_noflip[:, :, mid_sl][:, ::-1]
gt_img = gt_slices[mid_sl]

rec_f_n = norm01(rec_f)
rec_n_n = norm01(rec_n)
gt_n = norm01(gt_img)

diff_f = rec_f_n - gt_n
diff_n = rec_n_n - gt_n

mae_f = np.mean(np.abs(diff_f))
mae_n = np.mean(np.abs(diff_n))

print(f"\nSlice {sl_global}:")
print(f"  With flip:    MAE={mae_f:.4f}")
print(f"  Without flip: MAE={mae_n:.4f}")

# Row-averaged diff profiles (Y-gradient diagnostic)
row_profile_f = np.mean(diff_f, axis=1)
row_profile_n = np.mean(diff_n, axis=1)

fig, axes = plt.subplots(3, 4, figsize=(20, 15))

# Row 0: With flip
axes[0, 0].imshow(rec_f_n, cmap='gray')
axes[0, 0].set_title(f"Recon WITH flip (sl {sl_global})")
axes[0, 1].imshow(gt_n, cmap='gray')
axes[0, 1].set_title("GT")
axes[0, 2].imshow(diff_f, cmap='RdBu', vmin=-0.3, vmax=0.3)
axes[0, 2].set_title(f"Diff (flip) MAE={mae_f:.4f}")
axes[0, 3].plot(row_profile_f, 'r-')
axes[0, 3].axhline(0, color='k', ls='--', alpha=0.3)
axes[0, 3].set_title("Row-avg diff (flip)")
axes[0, 3].set_xlabel("Row (Y)")

# Row 1: Without flip
axes[1, 0].imshow(rec_n_n, cmap='gray')
axes[1, 0].set_title(f"Recon WITHOUT flip (sl {sl_global})")
axes[1, 1].imshow(gt_n, cmap='gray')
axes[1, 1].set_title("GT")
axes[1, 2].imshow(diff_n, cmap='RdBu', vmin=-0.3, vmax=0.3)
axes[1, 2].set_title(f"Diff (no-flip) MAE={mae_n:.4f}")
axes[1, 3].plot(row_profile_n, 'b-')
axes[1, 3].axhline(0, color='k', ls='--', alpha=0.3)
axes[1, 3].set_title("Row-avg diff (no-flip)")
axes[1, 3].set_xlabel("Row (Y)")

# Row 2: Overlay profiles + multi-slice Y-gradient check
axes[2, 0].plot(row_profile_f, 'r-', label='With flip')
axes[2, 0].plot(row_profile_n, 'b-', label='Without flip')
axes[2, 0].axhline(0, color='k', ls='--', alpha=0.3)
axes[2, 0].set_title("Row-avg diff overlay")
axes[2, 0].legend()

# Y-gradient across multiple slices (with flip)
row_profiles_f = []
row_profiles_n = []
for i in range(n_sl):
    rf = rec_flipped[:, :, i][:, ::-1]
    rn = rec_noflip[:, :, i][:, ::-1]
    gt_i = gt_slices[i]
    rf_norm = norm01(rf)
    rn_norm = norm01(rn)
    gt_i_norm = norm01(gt_i)
    row_profiles_f.append(np.mean(rf_norm - gt_i_norm, axis=1))
    row_profiles_n.append(np.mean(rn_norm - gt_i_norm, axis=1))

rp_f = np.array(row_profiles_f)  # (n_sl, 512)
rp_n = np.array(row_profiles_n)

axes[2, 1].imshow(rp_f.T, aspect='auto', cmap='RdBu', vmin=-0.15, vmax=0.15)
axes[2, 1].set_title("Y-gradient heatmap (with flip)")
axes[2, 1].set_xlabel("Slice (within chunk)")
axes[2, 1].set_ylabel("Row (Y)")

axes[2, 2].imshow(rp_n.T, aspect='auto', cmap='RdBu', vmin=-0.15, vmax=0.15)
axes[2, 2].set_title("Y-gradient heatmap (no flip)")
axes[2, 2].set_xlabel("Slice (within chunk)")
axes[2, 2].set_ylabel("Row (Y)")

# Grand average profiles
axes[2, 3].plot(np.mean(rp_f, axis=0), 'r-', label=f'Flip avg (range={np.ptp(np.mean(rp_f,axis=0)):.3f})')
axes[2, 3].plot(np.mean(rp_n, axis=0), 'b-', label=f'No-flip avg (range={np.ptp(np.mean(rp_n,axis=0)):.3f})')
axes[2, 3].axhline(0, color='k', ls='--', alpha=0.3)
axes[2, 3].set_title("Grand avg Y-gradient profile")
axes[2, 3].set_xlabel("Row (Y)")
axes[2, 3].legend(fontsize=8)

for ax in [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]:
    ax.axis('off')

plt.suptitle("Row Flip Diagnostic: Y-gradient artifact test", fontsize=14)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "test_row_flip.png")
plt.savefig(out_path, dpi=150)
print(f"\nSaved -> {out_path}")
print("Done.", flush=True)
