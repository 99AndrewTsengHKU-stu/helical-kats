"""
Test: skip equi-angular resampling entirely and reconstruct with raw DICOM data.

Hypothesis: The DICOM-CT-PD data may already be on a flat detector grid,
and the equi-angular resampling is distorting it.

We test several pixel size assumptions for the raw (un-resampled) data:
  A) Standard (with resampling, flat_psize=0.944) — baseline
  B) No resample + magnified arc psize (arc * SDD/SOD = 0.916)
  C) No resample + flat_psize_cols from resampling code (0.944)
  D) No resample + correct psize to match GT scale (back-calculated)
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
import astra, pydicom, matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from scipy.ndimage import sobel

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
    """Reconstruct a single slice and return it."""
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


def measure_body_edge(img, label=""):
    """Measure body edge radius using Sobel."""
    n = (img - np.percentile(img, 2)) / max(np.percentile(img, 98) - np.percentile(img, 2), 1e-10)
    n = np.clip(n, 0, 1)
    e = np.sqrt(sobel(n, 0)**2 + sobel(n, 1)**2)
    # Use a high threshold to get only strong edges
    thresh = np.percentile(e, 95)
    mask = e > thresh
    y, x = np.where(mask)
    if len(y) < 100:
        print(f"  {label}: too few edge pixels")
        return None, None, None
    cy, cx = y.mean(), x.mean()
    r = np.sqrt((y - cy)**2 + (x - cx)**2)
    r90 = np.percentile(r, 90)
    print(f"  {label}: edge r90={r90:.1f}px = {r90*VOXEL_SIZE_XY:.1f}mm")
    return r90, cx, cy


# ══════════════════════════════════════════════════════════════════════
# LOAD WITH RESAMPLING (standard)
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Loading DICOM WITH resampling (standard)...")
print("=" * 60, flush=True)
t0 = time()
sino_std, meta_std = load_dicom_projections(DICOM_DIR)
print(f"Loaded in {time()-t0:.1f}s, shape={sino_std.shape}")
sg_std = meta_std['scan_geometry']
psize_col_std = sg_std['detector']['detector psize cols']
psize_row_std = sg_std['detector']['detector psize rows']
print(f"  flat_psize_cols={psize_col_std:.6f}")
print(f"  det_psize_rows={psize_row_std:.6f}")

angles_std = -meta_std['angles_rad'].copy() - np.pi / 2
z_shift_std = meta_std['table_positions_mm'] - meta_std['table_positions_mm'].mean()
sino_std_work = sino_std[:, ::-1, :].copy()

# ══════════════════════════════════════════════════════════════════════
# LOAD WITHOUT RESAMPLING (raw)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Loading DICOM WITHOUT resampling (raw flat)...")
print("=" * 60, flush=True)

dicom_dir = Path(DICOM_DIR)
paths = sorted(p for p in dicom_dir.iterdir() if p.suffix.lower() == '.dcm')
first_ds = pydicom.dcmread(str(paths[0]), stop_before_pixels=True)

sod = float(struct.unpack('<f', bytes(first_ds[(0x7031, 0x1003)].value)[:4])[0])
sdd = float(struct.unpack('<f', bytes(first_ds[(0x7031, 0x1031)].value)[:4])[0])
det_extents = struct.unpack('<ff', bytes(first_ds[(0x7031, 0x1033)].value)[:8])
det_length_cols_mm, det_length_rows_mm = det_extents

det_rows_raw = int(first_ds.Columns)
det_cols_raw = int(first_ds.Rows)

print(f"  SOD={sod}, SDD={sdd}")
print(f"  det_extents = ({det_length_cols_mm}, {det_length_rows_mm}) mm")
print(f"  det: {det_rows_raw}x{det_cols_raw}")

# Compute various pixel size options
arc_psize_iso_cols = det_length_cols_mm / det_cols_raw     # arc at isocenter
arc_psize_iso_rows = det_length_rows_mm / det_rows_raw
magnified_arc_cols = arc_psize_iso_cols * sdd / sod        # arc projected to detector
magnified_arc_rows = arc_psize_iso_rows * sdd / sod
# Central flat pixel size
delta_gamma = arc_psize_iso_cols / sod
center_flat_cols = sdd * delta_gamma                        # flat psize at detector center

print(f"  arc_psize_iso_cols = {arc_psize_iso_cols:.6f} mm")
print(f"  magnified_arc_cols = {magnified_arc_cols:.6f} mm")
print(f"  center_flat_cols   = {center_flat_cols:.6f} mm")
print(f"  flat_psize (resamp)= {psize_col_std:.6f} mm")

# Load raw sinogram (no resampling)
t0 = time()
sino_raw = np.empty((len(paths), det_rows_raw, det_cols_raw), dtype=np.float32)
# Re-read ordered by instance number
meta_list = []
for path in paths:
    ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    instance = int(getattr(ds, "InstanceNumber", 0))
    meta_list.append((instance, path))
meta_list.sort(key=lambda x: x[0])

for idx, (_, path) in enumerate(meta_list):
    ds = pydicom.dcmread(str(path))
    pixels = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    pixels = pixels * slope + intercept
    sino_raw[idx] = pixels.T  # transpose: (Rows, Cols) -> (det_rows, det_cols)

print(f"Loaded raw in {time()-t0:.1f}s, shape={sino_raw.shape}")
sino_raw_work = sino_raw[:, ::-1, :].copy()

# ══════════════════════════════════════════════════════════════════════
# RECONSTRUCT
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Reconstructing with different data/psize combinations:")
print("=" * 60, flush=True)

# A: Standard (with resampling + flat psize)
rec_a = recon_one_slice(sino_std_work, angles_std, z_shift_std, sg_std, label="A: std resamp")

# B: Raw (no resample) + magnified arc psize
sg_b = copy.deepcopy(sg_std)
sg_b['detector']['detector psize cols'] = magnified_arc_cols
sg_b['detector']['detector psize rows'] = magnified_arc_rows
sg_b['detector']['detector psize'] = float(np.mean([magnified_arc_cols, magnified_arc_rows]))
rec_b = recon_one_slice(sino_raw_work, angles_std, z_shift_std, sg_b, label="B: raw+mag_arc")

# C: Raw (no resample) + center flat psize
sg_c = copy.deepcopy(sg_std)
sg_c['detector']['detector psize cols'] = center_flat_cols
sg_c['detector']['detector psize rows'] = magnified_arc_rows  # rows don't change much
sg_c['detector']['detector psize'] = float(np.mean([center_flat_cols, magnified_arc_rows]))
rec_c = recon_one_slice(sino_raw_work, angles_std, z_shift_std, sg_c, label="C: raw+center_flat")

# D: Raw (no resample) + arc psize at isocenter
sg_d = copy.deepcopy(sg_std)
sg_d['detector']['detector psize cols'] = arc_psize_iso_cols
sg_d['detector']['detector psize rows'] = arc_psize_iso_rows
sg_d['detector']['detector psize'] = float(np.mean([arc_psize_iso_cols, arc_psize_iso_rows]))
rec_d = recon_one_slice(sino_raw_work, angles_std, z_shift_std, sg_d, label="D: raw+iso_arc")

# ══════════════════════════════════════════════════════════════════════
# LOAD GT
# ══════════════════════════════════════════════════════════════════════
gt_files = sorted(GT_DIR.glob("*.IMA")) or sorted(GT_DIR.glob("*.dcm"))
ds_gt = pydicom.dcmread(str(gt_files[TARGET_SLICE]))
slope = float(getattr(ds_gt, 'RescaleSlope', 1.0))
intercept = float(getattr(ds_gt, 'RescaleIntercept', 0.0))
gt = ds_gt.pixel_array.astype(np.float32) * slope + intercept

# ══════════════════════════════════════════════════════════════════════
# MEASURE + COMPARE
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("BODY SIZE COMPARISON")
print("=" * 60)

r_gt, _, _ = measure_body_edge(gt, "GT")

results = []
for label, rec, psc in [
    ("A: std resamp", rec_a, psize_col_std),
    ("B: raw+mag_arc", rec_b, magnified_arc_cols),
    ("C: raw+center_flat", rec_c, center_flat_cols),
    ("D: raw+iso_arc", rec_d, arc_psize_iso_cols),
]:
    r, _, _ = measure_body_edge(rec[:, ::-1], f"{label} (flip)")
    if r is not None and r_gt is not None:
        ratio = r / r_gt
        print(f"    → recon/GT edge ratio = {ratio:.4f}, psize_col={psc:.4f}")
        results.append((label, rec, ratio, psc))

# ══════════════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 5, figsize=(25, 10))

def norm01(img):
    v0, v1 = np.percentile(img, [2, 98])
    return np.clip((img - v0) / max(v1 - v0, 1e-10), 0, 1)

gt_n = norm01(gt)

for col, (label, rec, ratio, psc) in enumerate(results):
    rec_flip = rec[:, ::-1]
    rec_n = norm01(rec_flip)
    axes[0, col].imshow(rec_n, cmap='gray')
    axes[0, col].set_title(f"{label}\npsize={psc:.4f}\nratio={ratio:.3f}", fontsize=9)
    axes[0, col].axis('off')

    # Diff with GT
    diff = rec_n - gt_n
    vlim = max(abs(diff).max() * 0.3, 0.01)
    axes[1, col].imshow(diff, cmap='RdBu', vmin=-vlim, vmax=vlim)
    axes[1, col].set_title(f"Diff MSE={np.mean(diff**2):.5f}", fontsize=9)
    axes[1, col].axis('off')

axes[0, 4].imshow(gt_n, cmap='gray')
axes[0, 4].set_title("GT", fontsize=9)
axes[0, 4].axis('off')
axes[1, 4].axis('off')

# Summary
txt = "PIXEL SIZE OPTIONS:\n"
txt += f"  arc_iso_cols   = {arc_psize_iso_cols:.6f} mm\n"
txt += f"  mag_arc_cols   = {magnified_arc_cols:.6f} mm\n"
txt += f"  center_flat    = {center_flat_cols:.6f} mm\n"
txt += f"  avg_flat(resamp)= {psize_col_std:.6f} mm\n"
txt += f"\n  voxel_size     = {VOXEL_SIZE_XY:.6f} mm\n"
txt += f"  psize/voxel ratios:\n"
for name, val in [("arc_iso", arc_psize_iso_cols), ("mag_arc", magnified_arc_cols),
                  ("center_flat", center_flat_cols), ("avg_flat", psize_col_std)]:
    txt += f"    {name}: {val/VOXEL_SIZE_XY:.4f}\n"
axes[1, 4].text(0.05, 0.95, txt, fontsize=9, transform=axes[1, 4].transAxes,
                verticalalignment='top', fontfamily='monospace')

plt.suptitle("Skip Equi-Angular Resampling Test", fontsize=13)
plt.tight_layout()
out = os.path.join(OUT_DIR, "skip_resample_test.png")
plt.savefig(out, dpi=150)
print(f"\nSaved -> {out}")
print("Done.", flush=True)
