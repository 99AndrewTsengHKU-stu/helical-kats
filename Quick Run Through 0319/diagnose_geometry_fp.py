"""
Definitive geometry test: forward-project GT volume at a specific angle,
then compare the FP sinogram profile with the actual DICOM projection at
the same angle.

If the profiles match in WIDTH → geometry is correct
If they disagree → geometry parameters are wrong and we can measure the error
"""
import sys, os, struct
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Quick Run Through 0304"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from backproject_safe import ensure_astra_cuda_init
ensure_astra_cuda_init()

import numpy as np
import pydicom
from pathlib import Path
from time import time
import astra
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DICOM_DIR = Path(r"D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD")
GT_DIR = Path(r"D:\AAPM-Data\L067\L067\full_1mm")

# ── Helper functions ──────────────────────────────────────────────────
def decode_f32(ds, tag, count=1):
    if tag not in ds:
        return None
    raw = bytes(ds[tag].value)
    vals = struct.unpack("<" + "f" * count, raw[:4*count])
    return vals if count > 1 else vals[0]


# ── Step 1: Load GT volume (subset of slices for speed) ──────────────
print("Loading GT volume...", flush=True)
gt_files = sorted(GT_DIR.glob("*.IMA")) or sorted(GT_DIR.glob("*.dcm"))
n_gt = len(gt_files)
print(f"  {n_gt} GT slices available")

# Load metadata from first and last slice to get z-range
ds_first = pydicom.dcmread(str(gt_files[0]))
ds_last = pydicom.dcmread(str(gt_files[-1]))
gt_pix_spacing = [float(x) for x in ds_first.PixelSpacing]
gt_z_first = float(getattr(ds_first, 'SliceLocation', 0))
gt_z_last = float(getattr(ds_last, 'SliceLocation', n_gt-1))
gt_recon_diam = float(getattr(ds_first, 'ReconstructionDiameter', 340))
gt_rows, gt_cols = ds_first.Rows, ds_first.Columns
gt_slope = float(getattr(ds_first, 'RescaleSlope', 1.0))
gt_intercept = float(getattr(ds_first, 'RescaleIntercept', 0.0))

print(f"  GT z-range: [{gt_z_first:.1f}, {gt_z_last:.1f}] mm")
print(f"  GT pixel spacing: {gt_pix_spacing}")
print(f"  GT ReconstructionDiameter: {gt_recon_diam}")
print(f"  GT image size: {gt_rows}x{gt_cols}")

# Load a subset of GT slices (say 40 slices around the middle)
gt_mid_idx = n_gt // 2
gt_start = gt_mid_idx - 20
gt_end = gt_mid_idx + 20
gt_n_load = gt_end - gt_start

gt_vol = np.empty((gt_rows, gt_cols, gt_n_load), dtype=np.float32)
gt_z_pos = np.empty(gt_n_load, dtype=np.float64)
for i, fidx in enumerate(range(gt_start, gt_end)):
    ds = pydicom.dcmread(str(gt_files[fidx]))
    gt_vol[:, :, i] = ds.pixel_array.astype(np.float32) * gt_slope + gt_intercept
    gt_z_pos[i] = float(getattr(ds, 'SliceLocation', fidx))

print(f"  Loaded GT slices [{gt_start}:{gt_end}], z=[{gt_z_pos[0]:.1f}, {gt_z_pos[-1]:.1f}] mm")

# Convert HU to linear attenuation coefficient (mu)
# mu = (HU/1000 + 1) * mu_water
mu_water = 0.019  # mm^-1
gt_mu = (gt_vol / 1000.0 + 1.0) * mu_water
gt_mu[gt_mu < 0] = 0  # clamp negative values

# ── Step 2: Read DICOM projection geometry ────────────────────────────
print("\nReading DICOM projection metadata...", flush=True)
dcm_files = sorted(DICOM_DIR.glob("*.dcm"))
n_dcm = len(dcm_files)
print(f"  {n_dcm} DICOM projections")

# Read geometry from one file
ds_dcm = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
sod = float(ds_dcm[(0x0018, 0x9402)].value) if (0x0018, 0x9402) in ds_dcm else decode_f32(ds_dcm, (0x7031, 0x1003))
sdd = float(ds_dcm[(0x0018, 0x1110)].value) if (0x0018, 0x1110) in ds_dcm else decode_f32(ds_dcm, (0x7031, 0x1031))
det_extents = decode_f32(ds_dcm, (0x7031, 0x1033), count=2)
det_rows = int(ds_dcm.Columns)  # z-direction
det_cols = int(ds_dcm.Rows)     # fan-direction

det_extent_fan = det_extents[0]
det_extent_z = det_extents[1]
arc_psize_iso = det_extent_fan / det_cols
delta_gamma = arc_psize_iso / sod

print(f"  SOD={sod}, SDD={sdd}")
print(f"  det: {det_rows}x{det_cols}, extent=({det_extent_fan}, {det_extent_z}) mm")
print(f"  arc_psize_iso={arc_psize_iso:.4f} mm, delta_gamma={delta_gamma:.6f} rad")

# ── Step 3: Find DICOM projections at the GT z-range ──────────────────
# Read angles and z-positions for a sample of DICOM files
print("\nScanning DICOM z-positions...", flush=True)
sample_step = max(1, n_dcm // 200)  # sample ~200 projections
sample_indices = range(0, n_dcm, sample_step)

dcm_angles = []
dcm_z_pos = []
dcm_instance = []
for idx in sample_indices:
    ds = pydicom.dcmread(str(dcm_files[idx]), stop_before_pixels=True)
    angle = decode_f32(ds, (0x7031, 0x1001))
    z_pos = decode_f32(ds, (0x7031, 0x1002))  # detector axial position
    inst = int(getattr(ds, 'InstanceNumber', 0))
    dcm_angles.append(angle)
    dcm_z_pos.append(z_pos)
    dcm_instance.append(inst)

dcm_angles = np.array(dcm_angles)
dcm_z_pos = np.array(dcm_z_pos)
print(f"  Sampled {len(dcm_angles)} projections")
print(f"  z-range: [{dcm_z_pos.min():.1f}, {dcm_z_pos.max():.1f}] mm")
print(f"  GT z-range: [{gt_z_pos[0]:.1f}, {gt_z_pos[-1]:.1f}] mm")

# Find a DICOM projection whose z-position is close to the GT center
gt_z_center = np.mean(gt_z_pos)
z_dist = np.abs(dcm_z_pos - gt_z_center)
best_sample = np.argmin(z_dist)
best_dcm_idx = list(sample_indices)[best_sample]
print(f"\n  Best match: DICOM file index {best_dcm_idx}, z={dcm_z_pos[best_sample]:.1f}, "
      f"GT center z={gt_z_center:.1f}, dist={z_dist[best_sample]:.1f} mm")
print(f"  angle = {dcm_angles[best_sample]:.4f} rad = {np.degrees(dcm_angles[best_sample]):.1f}°")

# ── Step 4: Forward project GT volume at the same angle ───────────────
view_angle = float(dcm_angles[best_sample])
view_z = float(dcm_z_pos[best_sample])

print(f"\nForward projecting GT at angle={np.degrees(view_angle):.1f}°, z={view_z:.1f} mm...", flush=True)

# Create volume geometry for the GT subset
half_xy = gt_recon_diam / 2.0
gt_z_spacing = abs(gt_z_pos[1] - gt_z_pos[0]) if len(gt_z_pos) > 1 else 1.0
gt_z_min = gt_z_pos[0] - gt_z_spacing / 2
gt_z_max = gt_z_pos[-1] + gt_z_spacing / 2

vol_geom = astra.create_vol_geom(
    gt_rows, gt_cols, gt_n_load,
    -half_xy, half_xy, -half_xy, half_xy, gt_z_min, gt_z_max
)

# Compute the flat detector pixel size (same as dicom.py)
j_center = (det_cols - 1) / 2.0
j_indices = np.arange(det_cols, dtype=np.float64)
gamma = (j_indices - j_center) * delta_gamma
u_flat = sdd * np.tan(gamma)
flat_psize_cols = float((u_flat[-1] - u_flat[0]) / (det_cols - 1))
arc_psize_rows = det_extent_z / det_rows
det_psize_rows = float(arc_psize_rows * sdd / sod)

print(f"  flat_psize_cols = {flat_psize_cols:.4f} mm")
print(f"  det_psize_rows = {det_psize_rows:.4f} mm")

# Create a single-view cone_vec geometry
from pykatsevich.geometry import astra_helical_views
view = astra_helical_views(
    sod, sdd, float(np.mean([flat_psize_cols, det_psize_rows])),
    np.array([view_angle]),
    0,  # no pitch for single view
    vertical_shifts=np.array([view_z]),
    pixel_size_col=flat_psize_cols,
    pixel_size_row=det_psize_rows,
)
proj_geom = astra.create_proj_geom('cone_vec', det_rows, det_cols, view)

# Forward project GT (mu values)
# Need to transpose GT from (rows, cols, slices) to ASTRA format (slices, rows, cols)
gt_mu_astra = np.moveaxis(gt_mu, 2, 0)  # → (slices, rows, cols) = (Z, Y, X)
gt_mu_astra = np.ascontiguousarray(gt_mu_astra, dtype=np.float32)

proj_id = astra.data3d.create('-sino', proj_geom, 0)
vol_id = astra.data3d.create('-vol', vol_geom, gt_mu_astra)

cfg_fp = astra.astra_dict('FP3D_CUDA')
cfg_fp['ProjectionDataId'] = proj_id
cfg_fp['VolumeDataId'] = vol_id
alg_fp = astra.algorithm.create(cfg_fp)
astra.algorithm.run(alg_fp)

fp_sino = astra.data3d.get(proj_id)  # (det_rows, 1, det_cols)
fp_proj = fp_sino[:, 0, :]  # single view, shape (det_rows, det_cols)

astra.algorithm.delete([alg_fp])
astra.data3d.delete([proj_id, vol_id])

print(f"  FP done, proj shape={fp_proj.shape}, range=[{fp_proj.min():.4f}, {fp_proj.max():.4f}]")

# ── Step 5: Load the actual DICOM projection ──────────────────────────
print(f"\nLoading DICOM projection {best_dcm_idx}...", flush=True)
ds_proj = pydicom.dcmread(str(dcm_files[best_dcm_idx]))
dcm_slope = float(getattr(ds_proj, 'RescaleSlope', 1.0))
dcm_intercept = float(getattr(ds_proj, 'RescaleIntercept', 0.0))
dcm_pixels = ds_proj.pixel_array.astype(np.float32) * dcm_slope + dcm_intercept
dcm_proj = dcm_pixels.T  # → (det_rows, det_cols)

print(f"  DICOM proj shape={dcm_proj.shape}, range=[{dcm_proj.min():.4f}, {dcm_proj.max():.4f}]")

# The DICOM data is equi-angular. Resample to flat for comparison.
from scipy.ndimage import map_coordinates
j_src = j_center + np.arctan(np.arange(det_cols, dtype=np.float64) * flat_psize_cols
        + u_flat[0]) / sdd / delta_gamma
# Alternatively: use the same resampling as dicom.py
u_target = u_flat[0] + np.arange(det_cols, dtype=np.float64) * flat_psize_cols
j_src = j_center + np.arctan(u_target / sdd) / delta_gamma

col_coords = np.broadcast_to(j_src[np.newaxis, :], (det_rows, det_cols))
row_coords = np.broadcast_to(
    np.arange(det_rows, dtype=np.float64)[:, np.newaxis], (det_rows, det_cols))
dcm_proj_flat = map_coordinates(
    dcm_proj.astype(np.float64), [row_coords, col_coords],
    order=3, mode='constant', cval=0.0).astype(np.float32)

print(f"  Resampled to flat: shape={dcm_proj_flat.shape}")

# ── Step 6: Compare profiles ─────────────────────────────────────────
mid_row = det_rows // 2
fp_profile = fp_proj[mid_row, :]
dcm_profile_raw = dcm_proj[mid_row, :]  # equi-angular
dcm_profile_flat = dcm_proj_flat[mid_row, :]  # resampled flat

# Normalize for shape comparison
def normalize(x):
    v0, v1 = np.percentile(x[x > 0] if (x > 0).any() else x, [5, 95])
    return np.clip((x - v0) / max(v1 - v0, 1e-10), 0, 1)

fp_n = normalize(fp_profile)
dcm_n = normalize(dcm_profile_flat)

# Find body edges in both
thresh = 0.1
fp_above = fp_n > thresh
dcm_above = dcm_n > thresh

fp_left = np.argmax(fp_above) if fp_above.any() else 0
fp_right = len(fp_above) - 1 - np.argmax(fp_above[::-1]) if fp_above.any() else len(fp_above)-1
dcm_left = np.argmax(dcm_above) if dcm_above.any() else 0
dcm_right = len(dcm_above) - 1 - np.argmax(dcm_above[::-1]) if dcm_above.any() else len(dcm_above)-1

fp_width = fp_right - fp_left
dcm_width = dcm_right - dcm_left
fp_center = (fp_left + fp_right) / 2
dcm_center = (dcm_left + dcm_right) / 2

print(f"\n{'='*60}")
print(f"PROFILE COMPARISON (flat detector, center row)")
print(f"{'='*60}")
print(f"  FP (GT):    left={fp_left}, right={fp_right}, width={fp_width}, center={fp_center:.1f}")
print(f"  DICOM:      left={dcm_left}, right={dcm_right}, width={dcm_width}, center={dcm_center:.1f}")
print(f"  Width ratio (DICOM/FP): {dcm_width/fp_width:.4f}" if fp_width > 0 else "  FP width = 0!")
print(f"  Center shift: {dcm_center - fp_center:.1f} pixels")

# Cross-correlation to find the best shift between profiles
from scipy.signal import correlate
corr = correlate(dcm_n, fp_n, mode='full')
best_shift = np.argmax(corr) - len(fp_n) + 1
print(f"  Cross-correlation best shift: {best_shift} pixels")

# Find optimal scale between FP and DICOM profiles
from scipy.optimize import minimize_scalar
def profile_mse(scale):
    from scipy.ndimage import zoom as zoom1d
    scaled = zoom1d(fp_n, scale, order=1)
    n = min(len(scaled), len(dcm_n))
    s_off = (len(scaled) - n) // 2
    d_off = (len(dcm_n) - n) // 2
    sp = scaled[s_off:s_off+n]
    dp = dcm_n[d_off:d_off+n]
    mask = (sp > 0.05) | (dp > 0.05)
    if mask.sum() < 10:
        return 1e10
    return np.mean((sp[mask] - dp[mask])**2)

result = minimize_scalar(profile_mse, bounds=(0.5, 2.0), method='bounded')
optimal_profile_scale = result.x
print(f"  Optimal FP-to-DICOM profile scale: {optimal_profile_scale:.4f}")
print(f"  (< 1 means FP profile is wider than DICOM → body wider in FP)")

# ── Step 7: Also compare the raw (equi-angular) profiles ─────────────
dcm_raw_n = normalize(dcm_profile_raw)
# The FP is on a flat detector; the raw DICOM is equi-angular
# To compare: either resample FP to equi-angular, or use the flat-resampled DICOM
# We already have dcm_profile_flat (resampled to flat) for comparison with FP

# ── Plot ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(16, 14))

# Row 0: Full projection images
v0_fp, v1_fp = np.percentile(fp_proj, [2, 98])
axes[0, 0].imshow(fp_proj, cmap='gray', vmin=v0_fp, vmax=v1_fp, aspect='auto')
axes[0, 0].set_title(f"Forward Projection of GT (angle={np.degrees(view_angle):.1f}°)")
axes[0, 0].axhline(mid_row, color='r', ls='--', alpha=0.5)

v0_d, v1_d = np.percentile(dcm_proj_flat, [2, 98])
axes[0, 1].imshow(dcm_proj_flat, cmap='gray', vmin=v0_d, vmax=v1_d, aspect='auto')
axes[0, 1].set_title(f"DICOM Projection (flat-resampled)")
axes[0, 1].axhline(mid_row, color='r', ls='--', alpha=0.5)

# Row 1: Normalized profiles
axes[1, 0].plot(fp_n, 'b-', alpha=0.7, label='FP (GT)')
axes[1, 0].plot(dcm_n, 'r-', alpha=0.7, label='DICOM (flat)')
axes[1, 0].axvline(fp_left, color='b', ls=':', alpha=0.5)
axes[1, 0].axvline(fp_right, color='b', ls=':', alpha=0.5)
axes[1, 0].axvline(dcm_left, color='r', ls=':', alpha=0.5)
axes[1, 0].axvline(dcm_right, color='r', ls=':', alpha=0.5)
axes[1, 0].legend()
axes[1, 0].set_title(f"Profiles (normalized): FP width={fp_width}, DICOM width={dcm_width}")
axes[1, 0].set_xlabel("Detector column")

# Overlay with optimal scale
from scipy.ndimage import zoom as zoom1d
fp_scaled = zoom1d(fp_n, optimal_profile_scale, order=1)
n = min(len(fp_scaled), len(dcm_n))
s_off = (len(fp_scaled) - n) // 2
d_off = (len(dcm_n) - n) // 2

axes[1, 1].plot(np.arange(n) + d_off, dcm_n[d_off:d_off+n], 'r-', alpha=0.7, label='DICOM')
axes[1, 1].plot(np.arange(n) + d_off, fp_scaled[s_off:s_off+n], 'b--', alpha=0.7,
                label=f'FP scaled {optimal_profile_scale:.3f}')
axes[1, 1].legend()
axes[1, 1].set_title(f"Optimal scale: {optimal_profile_scale:.4f}")

# Row 2: Raw values (not normalized)
axes[2, 0].plot(fp_profile, 'b-', alpha=0.7, label='FP (GT, mu*mm)')
axes[2, 0].set_xlabel("Detector column")
axes[2, 0].set_ylabel("Line integral (mu*mm)")
axes[2, 0].legend()
axes[2, 0].set_title("FP raw values")

axes[2, 1].plot(dcm_profile_flat, 'r-', alpha=0.7, label='DICOM')
axes[2, 1].set_xlabel("Detector column")
axes[2, 1].set_ylabel("Value")
axes[2, 1].legend()
axes[2, 1].set_title("DICOM raw values")

# Add summary text
info = (
    f"Geometry: SOD={sod}, SDD={sdd}\n"
    f"View angle: {np.degrees(view_angle):.1f}°, z={view_z:.1f}mm\n"
    f"FP width: {fp_width} px\n"
    f"DICOM width: {dcm_width} px\n"
    f"Width ratio: {dcm_width/max(fp_width,1):.4f}\n"
    f"Profile scale: {optimal_profile_scale:.4f}\n"
    f"Center shift: {dcm_center - fp_center:.1f} px\n"
)
fig.text(0.02, 0.02, info, fontsize=10, fontfamily='monospace',
         verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat'))

plt.suptitle("FP(GT) vs DICOM Projection: Geometry Verification", fontsize=14)
plt.tight_layout(rect=[0, 0.08, 1, 0.96])
out = os.path.join(OUT_DIR, "geometry_fp_comparison.png")
plt.savefig(out, dpi=150)
print(f"\nSaved -> {out}")
print("Done.", flush=True)
