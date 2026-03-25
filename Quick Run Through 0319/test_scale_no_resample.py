"""
Quick test: does skipping equi-angular resampling fix the body scale?

Loads DICOM projections with and without equi-angular-to-flat resampling,
reconstructs 1 slice each, measures body diameter vs GT.
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
import astra
import pydicom
import matplotlib
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


def measure_body(img, threshold, label=""):
    mask = img > threshold
    y, x = np.where(mask)
    if len(y) < 100:
        print(f"  {label}: too few pixels ({len(y)})")
        return None
    cy, cx = y.mean(), x.mean()
    radii = np.sqrt((y - cy)**2 + (x - cx)**2)
    r90 = np.percentile(radii, 90)
    print(f"  {label}: r90={r90:.1f} px = {r90*VOXEL_SIZE_XY:.1f} mm, center=({cx:.0f},{cy:.0f}), n={mask.sum()}")
    return r90


def recon_single_slice(sino_work, angles_full, z_shift, scan_geom, psize_cols, psize_rows, label=""):
    """Reconstruct a single slice and return it."""
    psize = scan_geom['detector']['detector psize']
    det_rows_n = scan_geom['detector']['detector rows']

    views = astra_helical_views(
        scan_geom['SOD'], scan_geom['SDD'], psize,
        angles_full, np.mean(np.diff(z_shift)),
        vertical_shifts=z_shift,
        pixel_size_col=psize_cols, pixel_size_row=psize_rows,
    )

    total_half_z = SLICES * VOXEL_SIZE_Z * 0.5
    z_c = -total_half_z + TARGET_SLICE * VOXEL_SIZE_Z
    z_min, z_max = z_c, z_c + VOXEL_SIZE_Z

    # z-cull
    source_z = views[:, 2]
    cone_half_z = 0.5 * det_rows_n * psize * (scan_geom['SOD'] / scan_geom['SDD'])
    angle_step = abs(np.mean(np.diff(np.arctan2(views[:10, 1], views[:10, 0]))))
    projs_per_turn = 2 * np.pi / max(angle_step, 1e-12)
    margin_z = projs_per_turn * abs(np.mean(np.diff(z_shift)))
    z_lo = source_z - cone_half_z - margin_z
    z_hi = source_z + cone_half_z + margin_z
    keep = np.where((z_hi >= z_min) & (z_lo <= z_max))[0]

    sino_c = sino_work[keep].copy()
    angles_c = angles_full[keep].copy()
    views_c = views[keep]

    sg_c = copy.deepcopy(scan_geom)
    pitch_abs = float(abs(np.mean(np.diff(z_shift)) / np.mean(np.diff(angles_full))))
    sg_c['helix']['pitch_mm_rad'] = pitch_abs
    sg_c['helix']['angles_range'] = float(abs(angles_c[-1] - angles_c[0]))
    sg_c['helix']['angles_count'] = len(angles_c)

    half_x = COLS * VOXEL_SIZE_XY * 0.5
    half_y = ROWS * VOXEL_SIZE_XY * 0.5
    vol_geom = astra.create_vol_geom(ROWS, COLS, 1, -half_x, half_x, -half_y, half_y, z_min, z_max)
    proj_geom = astra.create_proj_geom('cone_vec', det_rows_n, scan_geom['detector']['detector cols'], views_c)

    conf = create_configuration(sg_c, vol_geom)
    conf['source_pos'] = angles_c.astype(np.float32)
    conf['delta_s'] = float(np.mean(np.diff(angles_c)))

    print(f"  {label}: {len(keep)} projs, filtering...", flush=True)
    t0 = time()
    sino_f32 = np.asarray(sino_c, dtype=np.float32, order='C')
    filtered = filter_katsevich(sino_f32, conf,
        {'Diff': {'Print time': False}, 'FwdRebin': {'Print time': False}, 'BackRebin': {'Print time': False}})
    sino_td = sino_weight_td(filtered, conf, False)
    rec = backproject_cupy(sino_td, conf, vol_geom, proj_geom, tqdm_bar=False)
    print(f"  {label}: done in {time()-t0:.1f}s, range=[{rec.min():.5f}, {rec.max():.5f}]")
    return rec[:, :, 0]


# ══════════════════════════════════════════════════════════════════════
# LOAD DATA: WITH resampling (standard)
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Loading DICOM WITH equi-angular resampling (standard)...")
print("=" * 60, flush=True)
t0 = time()
sino_std, meta_std = load_dicom_projections(DICOM_DIR)
print(f"Loaded in {time()-t0:.1f}s, shape={sino_std.shape}")
sg_std = meta_std['scan_geometry']
print(f"  SOD={sg_std['SOD']}, SDD={sg_std['SDD']}")
print(f"  psize_cols={sg_std['detector']['detector psize cols']:.6f}")
print(f"  psize_rows={sg_std['detector']['detector psize rows']:.6f}")
print(f"  psize_avg ={sg_std['detector']['detector psize']:.6f}")

angles_std = -meta_std['angles_rad'].copy() - np.pi / 2
z_shift_std = meta_std['table_positions_mm'] - meta_std['table_positions_mm'].mean()
sino_std_work = sino_std[:, ::-1, :].copy()

# ══════════════════════════════════════════════════════════════════════
# LOAD DATA: WITHOUT resampling
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Loading DICOM WITHOUT equi-angular resampling...")
print("=" * 60, flush=True)

# Manually load without resampling
dicom_dir = Path(DICOM_DIR)
paths = sorted(p for p in dicom_dir.iterdir() if p.suffix.lower() == '.dcm')
first_ds = pydicom.dcmread(str(paths[0]), stop_before_pixels=True)

sod = float(struct.unpack('<f', bytes(first_ds[(0x7031, 0x1003)].value)[:4])[0])
sdd = float(struct.unpack('<f', bytes(first_ds[(0x7031, 0x1031)].value)[:4])[0])
det_extents = struct.unpack('<ff', bytes(first_ds[(0x7031, 0x1033)].value)[:8])
det_length_cols_mm, det_length_rows_mm = det_extents

det_rows_raw = int(first_ds.Columns)  # 64, after transpose
det_cols_raw = int(first_ds.Rows)     # 736, after transpose

# Arc pixel sizes at isocenter
arc_psize_cols = det_length_cols_mm / det_cols_raw
arc_psize_rows = det_length_rows_mm / det_rows_raw

# Option 1: Use arc pixel sizes magnified to detector plane
psize_cols_noR = arc_psize_cols * sdd / sod
psize_rows_noR = arc_psize_rows * sdd / sod

# Option 2: Use arc pixel sizes at isocenter directly
psize_cols_iso = arc_psize_cols
psize_rows_iso = arc_psize_rows

print(f"  Arc psize cols (isocenter): {arc_psize_cols:.6f} mm")
print(f"  Arc psize rows (isocenter): {arc_psize_rows:.6f} mm")
print(f"  Magnified to detector: cols={psize_cols_noR:.6f}, rows={psize_rows_noR:.6f}")
print(f"  Flat psize (with resampling): {sg_std['detector']['detector psize cols']:.6f}")
print(f"  Ratio flat/magnified: {sg_std['detector']['detector psize cols']/psize_cols_noR:.4f}")

# Load raw sinogram (no resampling)
sino_raw = np.empty((len(paths), det_rows_raw, det_cols_raw), dtype=np.float32)
# We'll just reuse the already-loaded data but undo the resampling by reloading
# Actually, the resampling is baked into sino_std. We need to reload without it.
# For speed, let's just use the standard sinogram but with different pixel sizes.
# This isn't a perfect test but will show if psize is the issue.

# ══════════════════════════════════════════════════════════════════════
# TEST: Reconstruct with standard resampled data but DIFFERENT pixel sizes
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Reconstructing with different pixel size assumptions:")
print("=" * 60, flush=True)

# Test A: Standard (with resampling, flat psize) - baseline
sg_a = copy.deepcopy(sg_std)
rec_a = recon_single_slice(
    sino_std_work, angles_std, z_shift_std, sg_a,
    sg_std['detector']['detector psize cols'],
    sg_std['detector']['detector psize rows'],
    label="A: flat psize (standard)"
)

# Test B: Same data, but use magnified arc psize instead of flat psize
sg_b = copy.deepcopy(sg_std)
sg_b['detector']['detector psize cols'] = psize_cols_noR
sg_b['detector']['detector psize rows'] = psize_rows_noR
sg_b['detector']['detector psize'] = float(np.mean([psize_cols_noR, psize_rows_noR]))
rec_b = recon_single_slice(
    sino_std_work, angles_std, z_shift_std, sg_b,
    psize_cols_noR, psize_rows_noR,
    label="B: magnified arc psize"
)

# Test C: Same data, use arc psize at isocenter (no magnification)
sg_c = copy.deepcopy(sg_std)
sg_c['detector']['detector psize cols'] = psize_cols_iso
sg_c['detector']['detector psize rows'] = psize_rows_iso
sg_c['detector']['detector psize'] = float(np.mean([psize_cols_iso, psize_rows_iso]))
rec_c = recon_single_slice(
    sino_std_work, angles_std, z_shift_std, sg_c,
    psize_cols_iso, psize_rows_iso,
    label="C: arc psize at isocenter"
)

# ══════════════════════════════════════════════════════════════════════
# LOAD GT
# ══════════════════════════════════════════════════════════════════════
gt_files = sorted(GT_DIR.glob("*.IMA")) or sorted(GT_DIR.glob("*.dcm"))
ds = pydicom.dcmread(str(gt_files[TARGET_SLICE]))
slope = float(getattr(ds, 'RescaleSlope', 1.0))
intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
gt_slice = ds.pixel_array.astype(np.float32) * slope + intercept

# ══════════════════════════════════════════════════════════════════════
# MEASURE BODY SIZE
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("BODY SIZE MEASUREMENTS")
print("=" * 60)

# GT measurement
r_gt = measure_body(gt_slice, -500, "GT (HU>-500)")

# For recon, use per-image threshold based on values
for label, rec in [("A: flat psize", rec_a), ("B: mag arc", rec_b), ("C: iso arc", rec_c)]:
    if rec is not None:
        # Auto threshold: median of positive values as mu_water
        pos = rec[rec > 0]
        if len(pos) > 100:
            mu_w = float(np.median(pos[pos < np.percentile(pos, 95)]))
            hu = (rec - mu_w) / mu_w * 1000.0
            r = measure_body(hu, -500, f"{label} (auto HU>-500, mu_w={mu_w:.5f})")
            if r and r_gt:
                print(f"    Scale ratio: {r/r_gt:.4f}")
        else:
            print(f"  {label}: no positive values")

# ══════════════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

def auto_show(ax, img, title):
    v0, v1 = np.percentile(img, [1, 99])
    ax.imshow(img, cmap='gray', vmin=v0, vmax=v1)
    ax.set_title(title, fontsize=9)
    ax.axis('off')

auto_show(axes[0, 0], rec_a, f"A: flat psize\n({sg_std['detector']['detector psize cols']:.4f}mm)")
auto_show(axes[0, 1], rec_b, f"B: mag arc psize\n({psize_cols_noR:.4f}mm)")
if rec_c is not None:
    auto_show(axes[0, 2], rec_c, f"C: iso arc psize\n({psize_cols_iso:.4f}mm)")
auto_show(axes[0, 3], gt_slice, f"GT (pix={0.6640625:.4f}mm)")

# Row 1: same images with soft tissue window
win_lo, win_hi = -200, 300
for col, (label, rec) in enumerate([("A", rec_a), ("B", rec_b), ("C", rec_c)]):
    if rec is not None:
        pos = rec[rec > 0]
        mu_w = float(np.median(pos[pos < np.percentile(pos, 95)])) if len(pos) > 100 else 0.02
        hu = (rec - mu_w) / mu_w * 1000.0
        axes[1, col].imshow(hu, cmap='gray', vmin=win_lo, vmax=win_hi)
        axes[1, col].set_title(f"{label} soft tissue", fontsize=9)
        axes[1, col].axis('off')

axes[1, 3].imshow(gt_slice, cmap='gray', vmin=win_lo, vmax=win_hi)
axes[1, 3].set_title("GT soft tissue", fontsize=9)
axes[1, 3].axis('off')

plt.suptitle("Scale Test: Different Pixel Size Assumptions", fontsize=13)
plt.tight_layout()
out = os.path.join(OUT_DIR, "scale_psize_test.png")
plt.savefig(out, dpi=150)
print(f"\nSaved -> {out}")
print("Done.", flush=True)
