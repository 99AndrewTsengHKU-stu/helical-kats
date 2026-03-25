"""
Full volume Katsevich reconstruction with corrected geometry.
Loads DICOM once, reconstructs 560 slices in z-chunks.
Saves volume as .npy and generates comparison with GT.
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
VOXEL_SIZE_XY = 0.6640625  # exact GT PixelSpacing
VOXEL_SIZE_Z = 0.8
CHUNK_SIZE = 20  # slices per chunk

# ── Load DICOM ────────────────────────────────────────────────────────
print("Loading DICOM projections...", flush=True)
t0 = time()
sino, meta = load_dicom_projections(DICOM_DIR)
t_load = time() - t0
print(f"Loaded in {t_load:.0f}s: {sino.shape}", flush=True)

sg = meta['scan_geometry']
angles_full = -meta['angles_rad'].copy() - np.pi / 2
ffs_z = meta.get('ffs_z_offsets_mm', np.zeros(len(meta['angles_rad']), dtype=np.float32))
z_shift = meta['table_positions_mm'] + ffs_z - (meta['table_positions_mm'] + ffs_z).mean()
print(f"FFS z-offset: min={ffs_z.min():.3f}, max={ffs_z.max():.3f}, "
      f"unique={len(np.unique(np.round(ffs_z, 3)))} values", flush=True)
sino_work = sino[:, ::-1, :].copy()
del sino  # free 8.6 GB
gc.collect()

scan_geom = copy.deepcopy(sg)
pitch_abs = float(abs(meta['pitch_mm_per_rad_signed']))
scan_geom['helix']['pitch_mm_rad'] = pitch_abs

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

# ── Volume geometry ───────────────────────────────────────────────────
# GT ReconstructionTargetCenterPatient = (-10.668, -125.668)
# DataCollectionCenterPatient (isocenter) = (0.332, -125.668)
# Offset: ReconTarget - Isocenter = (-11.0, 0.0) mm
RECON_OFFSET_X = +11.0  # mm, shift volume center to match GT reconstruction center (sign due to LR flip)
RECON_OFFSET_Y = 0.0

half_xy = COLS * VOXEL_SIZE_XY * 0.5
total_half_z = SLICES * VOXEL_SIZE_Z * 0.5

# Precompute z-culling helpers
source_z = views[:, 2]
cone_half_z = 0.5 * det_rows_n * psize * (sg['SOD'] / sg['SDD'])
angle_step = abs(np.mean(np.diff(np.arctan2(views[:10, 1], views[:10, 0]))))
projs_per_turn = 2 * np.pi / max(angle_step, 1e-12)
margin_z = projs_per_turn * abs(meta['pitch_mm_per_angle'])

print(f"\nReconstruction: {ROWS}x{COLS}x{SLICES}, voxel=({VOXEL_SIZE_XY:.4f}, {VOXEL_SIZE_XY:.4f}, {VOXEL_SIZE_Z})mm")
print(f"psize_cols={psize_cols:.4f}, psize_rows={psize_rows:.4f}")
print(f"SOD={sg['SOD']}, SDD={sg['SDD']}")
print(f"Chunk size: {CHUNK_SIZE} slices", flush=True)

# ── Reconstruct in z-chunks ──────────────────────────────────────────
volume = np.zeros((ROWS, COLS, SLICES), dtype=np.float32)
n_chunks = (SLICES + CHUNK_SIZE - 1) // CHUNK_SIZE
t_start = time()

for chunk_idx in range(n_chunks):
    sl_start = chunk_idx * CHUNK_SIZE
    sl_end = min(sl_start + CHUNK_SIZE, SLICES)
    n_sl = sl_end - sl_start

    z_min_chunk = -total_half_z + sl_start * VOXEL_SIZE_Z
    z_max_chunk = -total_half_z + sl_end * VOXEL_SIZE_Z

    # Select projections that contribute to this z-range
    keep = np.where(
        (source_z + cone_half_z + margin_z >= z_min_chunk) &
        (source_z - cone_half_z - margin_z <= z_max_chunk)
    )[0]

    if len(keep) == 0:
        print(f"  Chunk {chunk_idx+1}/{n_chunks}: slices {sl_start}-{sl_end-1} — no projections, skipping")
        continue

    sino_c = sino_work[keep].copy()
    angles_c = angles_full[keep].copy()
    views_c = views[keep]

    # Config for this chunk
    sg_c = copy.deepcopy(scan_geom)
    sg_c['helix']['angles_range'] = float(abs(angles_c[-1] - angles_c[0]))
    sg_c['helix']['angles_count'] = len(angles_c)

    vol_geom = astra.create_vol_geom(
        ROWS, COLS, n_sl,
        -half_xy + RECON_OFFSET_X, half_xy + RECON_OFFSET_X,
        -half_xy + RECON_OFFSET_Y, half_xy + RECON_OFFSET_Y,
        z_min_chunk, z_max_chunk,
    )
    proj_geom = astra.create_proj_geom('cone_vec', det_rows_n, sg['detector']['detector cols'], views_c)

    conf = create_configuration(sg_c, vol_geom)
    conf['source_pos'] = angles_c.astype(np.float32)
    conf['delta_s'] = float(np.mean(np.diff(angles_c)))

    t_chunk = time()
    filtered = filter_katsevich(
        np.asarray(sino_c, dtype=np.float32, order='C'), conf,
        {'Diff': {'Print time': False}, 'FwdRebin': {'Print time': False},
         'BackRebin': {'Print time': False}}
    )
    sino_td = sino_weight_td(filtered, conf, False)
    rec_chunk = backproject_cupy(sino_td, conf, vol_geom, proj_geom, tqdm_bar=False)

    volume[:, :, sl_start:sl_end] = rec_chunk[:, :, :n_sl]

    elapsed = time() - t_start
    eta = elapsed / (chunk_idx + 1) * n_chunks - elapsed
    print(f"  Chunk {chunk_idx+1}/{n_chunks}: slices {sl_start}-{sl_end-1}, "
          f"{len(keep)} projs, {time()-t_chunk:.0f}s "
          f"[elapsed {elapsed/60:.0f}m, ETA {eta/60:.0f}m]", flush=True)

    del sino_c, filtered, sino_td, rec_chunk
    gc.collect()
    # Force free GPU memory pool to prevent OOM across chunks
    import cupy as cp
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

total_time = time() - t0
print(f"\nTotal time: {total_time/60:.1f} min (load {t_load/60:.1f}m + recon {(total_time-t_load)/60:.1f}m)")

# ── Save volume as TIFF stack ─────────────────────────────────────────
import tifffile
tiff_path = os.path.join(OUT_DIR, "full_recon_volume.tiff")
# volume is (Y, X, Z) → transpose to (Z, Y, X) for standard TIFF stack
vol_zyx = np.moveaxis(volume, 2, 0)  # (Z, Y, X)
# LR flip to match GT orientation
vol_zyx = vol_zyx[:, :, ::-1].copy()
tifffile.imwrite(tiff_path, vol_zyx, imagej=True,
                 metadata={'spacing': VOXEL_SIZE_Z, 'unit': 'mm'},
                 resolution=(1.0/VOXEL_SIZE_XY, 1.0/VOXEL_SIZE_XY, 'MILLIMETER'))
print(f"Saved TIFF stack ({vol_zyx.shape}) -> {tiff_path}")

# ── Comparison with GT ───────────────────────────────────────────────
gt_files = sorted(GT_DIR.glob("*.IMA")) or sorted(GT_DIR.glob("*.dcm"))
print(f"\nGT files: {len(gt_files)}")

compare_slices = [100, 200, 280, 350, 450]
fig, axes = plt.subplots(len(compare_slices), 4, figsize=(16, 4*len(compare_slices)))

for row, sl in enumerate(compare_slices):
    # Reconstruction
    rec_sl = volume[:, :, sl]
    rec_flip = rec_sl[:, ::-1]

    # GT
    ds = pydicom.dcmread(str(gt_files[sl]))
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    gt = ds.pixel_array.astype(np.float32) * slope + intercept

    # Normalize
    def norm01(img):
        v0, v1 = np.percentile(img, [2, 98])
        return np.clip((img - v0) / max(v1 - v0, 1e-10), 0, 1)

    rec_n = norm01(rec_flip)
    gt_n = norm01(gt)
    diff = rec_n - gt_n

    axes[row, 0].imshow(rec_n, cmap='gray')
    axes[row, 0].set_title(f"Recon slice {sl}")
    axes[row, 1].imshow(gt_n, cmap='gray')
    axes[row, 1].set_title(f"GT slice {sl}")
    axes[row, 2].imshow(diff, cmap='RdBu', vmin=-0.3, vmax=0.3)
    axes[row, 2].set_title(f"Diff MSE={np.mean(diff**2):.4f}")

    # Profile
    mid = ROWS // 2
    axes[row, 3].plot(rec_n[mid, :], 'r-', alpha=0.7, label='Recon')
    axes[row, 3].plot(gt_n[mid, :], 'g-', alpha=0.7, label='GT')
    axes[row, 3].set_title(f"Profile row={mid}")
    axes[row, 3].legend(fontsize=7)

    for ax in axes[row, :3]:
        ax.axis('off')

plt.suptitle("Full Katsevich Reconstruction vs GT (corrected geometry)", fontsize=14)
plt.tight_layout()
plot_path = os.path.join(OUT_DIR, "full_recon_comparison.png")
plt.savefig(plot_path, dpi=150)
print(f"Saved comparison -> {plot_path}")
print("Done.", flush=True)
