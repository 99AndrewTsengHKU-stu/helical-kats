"""
Find optimal (dy, dx, angle) per slice to minimize MAE between recon and GT.
Phase correlation for coarse shift, then joint shift+rotation refinement.
Final audit: analyze patterns in the alignment parameters.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import tifffile
import pydicom
from pathlib import Path
from scipy.ndimage import shift as ndi_shift, rotate as ndi_rotate, affine_transform
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
TIFF_PATH = os.path.join(OUT_DIR, "full_recon_volume.tiff")
GT_DIR = Path(r"D:\AAPM-Data\L067\L067\full_1mm")
VOXEL_SIZE_XY = 0.6640625
VOXEL_SIZE_Z = 0.8

# ── Load data ────────────────────────────────────────────────────────────
print("Loading reconstructed volume...", flush=True)
vol = tifffile.imread(TIFF_PATH)  # (Z, Y, X) = (560, 512, 512)
print(f"Recon shape: {vol.shape}")

gt_files = sorted(GT_DIR.glob("*.IMA")) or sorted(GT_DIR.glob("*.dcm"))
print(f"GT files: {len(gt_files)}")

SLICES = min(vol.shape[0], len(gt_files))

def load_gt(idx):
    ds = pydicom.dcmread(str(gt_files[idx]))
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    return ds.pixel_array.astype(np.float32) * slope + intercept

def norm_slice(img):
    v0, v1 = np.percentile(img, [2, 98])
    if v1 - v0 < 1e-10:
        return np.zeros_like(img)
    return np.clip((img - v0) / (v1 - v0), 0, 1)

# ── Registration functions ───────────────────────────────────────────────
def find_shift_phase_corr(img1, img2):
    """Find (dy, dx) shift using phase correlation. img1=target, img2=moving."""
    f1 = np.fft.fft2(img1)
    f2 = np.fft.fft2(img2)
    cross = f1 * np.conj(f2)
    cross /= np.abs(cross) + 1e-10
    corr = np.real(np.fft.ifft2(cross))
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    dy, dx = peak
    if dy > corr.shape[0] // 2:
        dy -= corr.shape[0]
    if dx > corr.shape[1] // 2:
        dx -= corr.shape[1]
    return dy, dx

def apply_transform(img, dy, dx, angle_deg):
    """Apply shift + rotation (rotate around image center, then shift)."""
    if abs(angle_deg) < 1e-6:
        return ndi_shift(img, (dy, dx), order=1, mode='constant', cval=0)
    # Rotate first, then shift
    rotated = ndi_rotate(img, angle_deg, reshape=False, order=1, mode='constant', cval=0)
    return ndi_shift(rotated, (dy, dx), order=1, mode='constant', cval=0)

def refine_shift_rot(rec_n, gt_n, dy0, dx0,
                     shift_range=2, shift_step=0.5,
                     angle_range=1.0, angle_step=0.1):
    """Refine shift + rotation to minimize MAE."""
    best_mae = np.inf
    best_dy, best_dx, best_angle = float(dy0), float(dx0), 0.0

    shift_offsets = np.arange(-shift_range, shift_range + shift_step, shift_step)
    angles = np.arange(-angle_range, angle_range + angle_step, angle_step)

    # Stage 1: coarse rotation search at integer shift
    for angle in angles:
        transformed = apply_transform(rec_n, dy0, dx0, angle)
        mae = np.mean(np.abs(transformed - gt_n))
        if mae < best_mae:
            best_mae = mae
            best_angle = angle

    # Stage 2: joint shift refinement at best angle
    for ddy in shift_offsets:
        for ddx in shift_offsets:
            dy = dy0 + ddy
            dx = dx0 + ddx
            transformed = apply_transform(rec_n, dy, dx, best_angle)
            mae = np.mean(np.abs(transformed - gt_n))
            if mae < best_mae:
                best_mae = mae
                best_dy, best_dx = dy, dx

    # Stage 3: fine rotation at best shift
    fine_angles = np.arange(best_angle - angle_step, best_angle + angle_step + 0.02, 0.02)
    for angle in fine_angles:
        transformed = apply_transform(rec_n, best_dy, best_dx, angle)
        mae = np.mean(np.abs(transformed - gt_n))
        if mae < best_mae:
            best_mae = mae
            best_angle = angle

    return best_dy, best_dx, best_angle, best_mae

# ── Process all slices ───────────────────────────────────────────────────
print(f"\nFinding optimal shift+rotation for {SLICES} slices...", flush=True)

shifts_dy = np.zeros(SLICES)
shifts_dx = np.zeros(SLICES)
rotations = np.zeros(SLICES)
maes_before = np.zeros(SLICES)
maes_after = np.zeros(SLICES)

for sl in range(SLICES):
    rec_n = norm_slice(vol[sl])
    gt_n = norm_slice(load_gt(sl))

    maes_before[sl] = np.mean(np.abs(rec_n - gt_n))

    # Phase correlation for coarse shift
    dy0, dx0 = find_shift_phase_corr(gt_n, rec_n)

    # Joint shift + rotation refinement
    dy, dx, angle, mae = refine_shift_rot(rec_n, gt_n, dy0, dx0)

    shifts_dy[sl] = dy
    shifts_dx[sl] = dx
    rotations[sl] = angle
    maes_after[sl] = mae

    if sl % 50 == 0 or sl == SLICES - 1:
        print(f"  Slice {sl:3d}: shift=({dy:+.1f}, {dx:+.1f}), rot={angle:+.2f}deg, "
              f"MAE {maes_before[sl]:.4f} -> {mae:.4f}", flush=True)

# ── Audit summary ────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  ALIGNMENT AUDIT REPORT")
print(f"{'='*60}")
print(f"\n--- Shift statistics (pixels) ---")
print(f"  dy: mean={np.mean(shifts_dy):+.3f}, std={np.std(shifts_dy):.3f}, "
      f"range=[{np.min(shifts_dy):+.1f}, {np.max(shifts_dy):+.1f}]")
print(f"  dx: mean={np.mean(shifts_dx):+.3f}, std={np.std(shifts_dx):.3f}, "
      f"range=[{np.min(shifts_dx):+.1f}, {np.max(shifts_dx):+.1f}]")
print(f"\n--- Shift in mm ---")
print(f"  dy: mean={np.mean(shifts_dy)*VOXEL_SIZE_XY:+.3f}mm, "
      f"std={np.std(shifts_dy)*VOXEL_SIZE_XY:.3f}mm")
print(f"  dx: mean={np.mean(shifts_dx)*VOXEL_SIZE_XY:+.3f}mm, "
      f"std={np.std(shifts_dx)*VOXEL_SIZE_XY:.3f}mm")
print(f"\n--- Rotation statistics (degrees) ---")
print(f"  angle: mean={np.mean(rotations):+.4f}, std={np.std(rotations):.4f}, "
      f"range=[{np.min(rotations):+.2f}, {np.max(rotations):+.2f}]")
print(f"\n--- MAE ---")
print(f"  Before: mean={np.mean(maes_before):.4f}, std={np.std(maes_before):.4f}")
print(f"  After:  mean={np.mean(maes_after):.4f}, std={np.std(maes_after):.4f}")
print(f"  Reduction: {(1 - np.mean(maes_after)/np.mean(maes_before))*100:.1f}%")

# ── Pattern analysis ─────────────────────────────────────────────────────
print(f"\n--- Pattern analysis ---")

# Check if shift varies linearly with z (would indicate systematic tilt)
z_idx = np.arange(SLICES)
dy_fit = np.polyfit(z_idx, shifts_dy, 1)
dx_fit = np.polyfit(z_idx, shifts_dx, 1)
rot_fit = np.polyfit(z_idx, rotations, 1)

print(f"  dy vs z: slope={dy_fit[0]:.5f} px/slice ({dy_fit[0]*VOXEL_SIZE_XY/VOXEL_SIZE_Z:.5f} px/mm), "
      f"intercept={dy_fit[1]:+.2f}")
print(f"  dx vs z: slope={dx_fit[0]:.5f} px/slice ({dx_fit[0]*VOXEL_SIZE_XY/VOXEL_SIZE_Z:.5f} px/mm), "
      f"intercept={dx_fit[1]:+.2f}")
print(f"  rot vs z: slope={rot_fit[0]:.6f} deg/slice, intercept={rot_fit[1]:+.4f}")

# Correlation between dy and dx
corr_dydx = np.corrcoef(shifts_dy, shifts_dx)[0, 1]
print(f"  Correlation dy-dx: {corr_dydx:.3f}")

# Check for periodicity (related to gantry rotation)
from numpy.fft import fft
dy_fft = np.abs(fft(shifts_dy - np.mean(shifts_dy)))[:SLICES//2]
dx_fft = np.abs(fft(shifts_dx - np.mean(shifts_dx)))[:SLICES//2]
top_dy_freq = np.argsort(dy_fft[1:])[-3:] + 1  # skip DC
top_dx_freq = np.argsort(dx_fft[1:])[-3:] + 1
print(f"  dy dominant frequencies (in slice periods): {top_dy_freq}")
print(f"  dx dominant frequencies (in slice periods): {top_dx_freq}")

# Is the shift mostly constant? (systematic offset vs varying)
dy_const_frac = 1 - np.std(shifts_dy) / (np.abs(np.mean(shifts_dy)) + 1e-10)
dx_const_frac = 1 - np.std(shifts_dx) / (np.abs(np.mean(shifts_dx)) + 1e-10)
print(f"  dy constancy: {dy_const_frac*100:.0f}% (100%=perfectly constant shift)")
print(f"  dx constancy: {dx_const_frac*100:.0f}%")

# ── Plots ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(14, 14))

# Shifts
axes[0].plot(shifts_dy, 'b-', alpha=0.7, label='dy (rows)')
axes[0].plot(shifts_dx, 'r-', alpha=0.7, label='dx (cols)')
axes[0].plot(z_idx, np.polyval(dy_fit, z_idx), 'b--', alpha=0.4, label=f'dy trend')
axes[0].plot(z_idx, np.polyval(dx_fit, z_idx), 'r--', alpha=0.4, label=f'dx trend')
axes[0].axhline(0, color='gray', ls='-', alpha=0.3)
axes[0].set_ylabel('Shift (pixels)')
axes[0].set_title(f'Optimal shift per slice (mean: dy={np.mean(shifts_dy):+.2f}, dx={np.mean(shifts_dx):+.2f})')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Rotation
axes[1].plot(rotations, 'g-', alpha=0.7)
axes[1].plot(z_idx, np.polyval(rot_fit, z_idx), 'g--', alpha=0.4, label='trend')
axes[1].axhline(0, color='gray', ls='-', alpha=0.3)
axes[1].set_ylabel('Rotation (deg)')
axes[1].set_title(f'Optimal rotation per slice (mean: {np.mean(rotations):+.3f} deg)')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

# MAE
axes[2].plot(maes_before, 'k-', alpha=0.5, label='Before alignment')
axes[2].plot(maes_after, 'g-', alpha=0.7, label='After alignment')
axes[2].set_ylabel('MAE')
axes[2].set_title(f'MAE: {np.mean(maes_before):.4f} -> {np.mean(maes_after):.4f} '
                  f'({(1-np.mean(maes_after)/np.mean(maes_before))*100:.1f}% reduction)')
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

# Shift vector field (dx vs dy colored by slice)
sc = axes[3].scatter(shifts_dx, shifts_dy, c=z_idx, cmap='viridis', s=5, alpha=0.7)
plt.colorbar(sc, ax=axes[3], label='Slice index')
axes[3].set_xlabel('dx (pixels)')
axes[3].set_ylabel('dy (pixels)')
axes[3].set_title('Shift distribution (colored by z-position)')
axes[3].axhline(0, color='gray', ls='-', alpha=0.3)
axes[3].axvline(0, color='gray', ls='-', alpha=0.3)
axes[3].set_aspect('equal')
axes[3].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "optimal_shifts.png"), dpi=150)

# Example comparison
example_slices = [50, 200, 280, 400]
fig2, axes2 = plt.subplots(len(example_slices), 4, figsize=(16, 4*len(example_slices)))
for row, sl in enumerate(example_slices):
    rec_n = norm_slice(vol[sl])
    gt_n = norm_slice(load_gt(sl))
    aligned = apply_transform(rec_n, shifts_dy[sl], shifts_dx[sl], rotations[sl])

    axes2[row, 0].imshow(rec_n, cmap='gray')
    axes2[row, 0].set_title(f'Recon sl={sl}')
    axes2[row, 1].imshow(gt_n, cmap='gray')
    axes2[row, 1].set_title(f'GT sl={sl}')
    axes2[row, 2].imshow(rec_n - gt_n, cmap='RdBu', vmin=-0.3, vmax=0.3)
    axes2[row, 2].set_title(f'Before')
    axes2[row, 3].imshow(aligned - gt_n, cmap='RdBu', vmin=-0.3, vmax=0.3)
    axes2[row, 3].set_title(f'After ({shifts_dy[sl]:+.1f},{shifts_dx[sl]:+.1f},{rotations[sl]:+.2f}deg)')
    for ax in axes2[row]:
        ax.axis('off')
fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, "shift_comparison.png"), dpi=150)

# Save data
np.savez(os.path.join(OUT_DIR, "optimal_shifts.npz"),
         dy=shifts_dy, dx=shifts_dx, rotation=rotations,
         mae_before=maes_before, mae_after=maes_after)

# ── Apply and save corrected TIFF ────────────────────────────────────────
print("\nApplying alignment to full volume...", flush=True)
corrected = np.zeros_like(vol)
for sl in range(SLICES):
    corrected[sl] = apply_transform(vol[sl], shifts_dy[sl], shifts_dx[sl], rotations[sl])
    if sl % 100 == 0:
        print(f"  Slice {sl}/{SLICES}", flush=True)

corrected_path = os.path.join(OUT_DIR, "full_recon_volume_aligned.tiff")
tifffile.imwrite(corrected_path, corrected.astype(np.float32), imagej=True,
                 metadata={'spacing': VOXEL_SIZE_Z, 'unit': 'mm'},
                 resolution=(1.0/VOXEL_SIZE_XY, 1.0/VOXEL_SIZE_XY, 'MILLIMETER'))
print(f"Saved aligned TIFF -> {corrected_path}")

print(f"\nAll outputs saved to {OUT_DIR}")
print("Done.", flush=True)
