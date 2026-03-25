"""
Analyze per-slice optimal pixel shift and rotation vs GT,
then check for periodicity via FFT.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from pathlib import Path
import pydicom, tifffile, matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from scipy.ndimage import shift as ndi_shift, rotate as ndi_rotate
from scipy.signal import find_peaks

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
GT_DIR = Path(r"D:\AAPM-Data\L067\L067\full_1mm")

# Load reconstructed volume
print("Loading reconstructed volume...", flush=True)
vol = tifffile.imread(os.path.join(OUT_DIR, "full_recon_volume.tiff"))  # (Z, Y, X)
n_slices = vol.shape[0]
print(f"Recon shape: {vol.shape}")

gt_files = sorted(GT_DIR.glob("*.IMA")) or sorted(GT_DIR.glob("*.dcm"))
print(f"GT files: {len(gt_files)}")

def norm01(img):
    v0, v1 = np.percentile(img, [2, 98])
    return np.clip((img - v0) / max(v1 - v0, 1e-10), 0, 1)

def mae(a, b):
    return float(np.mean(np.abs(a - b)))

# Every slice
step = 1
slice_indices = list(range(0, n_slices))
print(f"Analyzing {len(slice_indices)} slices (every slice)...", flush=True)

shifts_dy = []
shifts_dx = []
rotations = []
mae_before = []
mae_after = []

# Search grid
shift_range = np.arange(-4, 4.5, 0.5)
rot_range = np.arange(-1.0, 1.05, 0.1)

for idx, sl in enumerate(slice_indices):
    rec = vol[sl].astype(np.float32)
    ds = pydicom.dcmread(str(gt_files[sl]))
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    gt = ds.pixel_array.astype(np.float32) * slope + intercept

    rec_n = norm01(rec)
    gt_n = norm01(gt)

    mae_orig = mae(rec_n, gt_n)
    mae_before.append(mae_orig)

    # Coarse shift search
    best_mae = mae_orig
    best_dy, best_dx, best_rot = 0.0, 0.0, 0.0

    for dy in shift_range:
        for dx in shift_range:
            shifted = ndi_shift(rec_n, [dy, dx], order=1, mode='constant')
            m = mae(shifted, gt_n)
            if m < best_mae:
                best_mae = m
                best_dy, best_dx = dy, dx

    # Refine rotation at best shift
    shifted_best = ndi_shift(rec_n, [best_dy, best_dx], order=1, mode='constant')
    for rot in rot_range:
        rotated = ndi_rotate(shifted_best, rot, reshape=False, order=1, mode='constant')
        m = mae(rotated, gt_n)
        if m < best_mae:
            best_mae = m
            best_rot = rot

    shifts_dy.append(best_dy)
    shifts_dx.append(best_dx)
    rotations.append(best_rot)
    mae_after.append(best_mae)

    if idx % 20 == 0:
        print(f"  Slice {sl:3d}: dy={best_dy:+.1f}, dx={best_dx:+.1f}, rot={best_rot:+.2f}deg, "
              f"MAE {mae_orig:.4f} -> {best_mae:.4f}", flush=True)

shifts_dy = np.array(shifts_dy)
shifts_dx = np.array(shifts_dx)
rotations = np.array(rotations)
mae_before = np.array(mae_before)
mae_after = np.array(mae_after)
z_positions = np.array(slice_indices)

print(f"\nDone. Analyzed {len(slice_indices)} slices.", flush=True)

# ── Plot ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 2, figsize=(16, 16))

# Row 0: shifts vs z
axes[0, 0].plot(z_positions, shifts_dy, 'r.-', markersize=3, label='dy')
axes[0, 0].plot(z_positions, shifts_dx, 'b.-', markersize=3, label='dx')
axes[0, 0].axhline(0, color='k', ls='--', alpha=0.3)
axes[0, 0].set_xlabel("Slice index")
axes[0, 0].set_ylabel("Optimal shift (pixels)")
axes[0, 0].set_title("Optimal shift vs slice")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(z_positions, rotations, 'g.-', markersize=3)
axes[0, 1].axhline(0, color='k', ls='--', alpha=0.3)
axes[0, 1].set_xlabel("Slice index")
axes[0, 1].set_ylabel("Optimal rotation (deg)")
axes[0, 1].set_title("Optimal rotation vs slice")
axes[0, 1].grid(True, alpha=0.3)

# Row 1: FFT of shifts
for i, (data, label, color) in enumerate([(shifts_dy, 'dy', 'r'), (shifts_dx, 'dx', 'b'), (rotations, 'rot', 'g')]):
    # Remove mean, compute FFT
    data_centered = data - np.mean(data)
    fft_mag = np.abs(np.fft.rfft(data_centered))
    freqs = np.fft.rfftfreq(len(data), d=step)  # cycles per slice

    # Skip DC
    fft_mag = fft_mag[1:]
    freqs = freqs[1:]

    # Convert to period in slices
    periods = 1.0 / (freqs + 1e-12)

    if i < 2:
        axes[1, i].stem(periods, fft_mag, linefmt=color+'-', markerfmt=color+'o', basefmt='k-')
        axes[1, i].set_xlabel("Period (slices)")
        axes[1, i].set_ylabel("FFT magnitude")
        axes[1, i].set_title(f"FFT of {label} shift")
        axes[1, i].set_xlim(0, 200)
        axes[1, i].grid(True, alpha=0.3)

        # Find peaks
        peaks, props = find_peaks(fft_mag, height=np.max(fft_mag)*0.3)
        for p in peaks:
            axes[1, i].annotate(f"{periods[p]:.0f}sl", (periods[p], fft_mag[p]),
                              fontsize=8, ha='center', va='bottom')

# Row 2: FFT of rotation + MAE
data_centered = rotations - np.mean(rotations)
fft_mag = np.abs(np.fft.rfft(data_centered))[1:]
freqs = np.fft.rfftfreq(len(rotations), d=step)[1:]
periods = 1.0 / (freqs + 1e-12)

axes[2, 0].stem(periods, fft_mag, linefmt='g-', markerfmt='go', basefmt='k-')
axes[2, 0].set_xlabel("Period (slices)")
axes[2, 0].set_ylabel("FFT magnitude")
axes[2, 0].set_title("FFT of rotation")
axes[2, 0].set_xlim(0, 200)
axes[2, 0].grid(True, alpha=0.3)
peaks, _ = find_peaks(fft_mag, height=np.max(fft_mag)*0.3)
for p in peaks:
    axes[2, 0].annotate(f"{periods[p]:.0f}sl", (periods[p], fft_mag[p]),
                      fontsize=8, ha='center', va='bottom')

axes[2, 1].plot(z_positions, mae_before, 'k.-', markersize=3, label='Before alignment')
axes[2, 1].plot(z_positions, mae_after, 'm.-', markersize=3, label='After alignment')
axes[2, 1].set_xlabel("Slice index")
axes[2, 1].set_ylabel("MAE")
axes[2, 1].set_title("MAE vs slice")
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

# Row 3: Autocorrelation of dy and dx
for i, (data, label, color) in enumerate([(shifts_dy, 'dy', 'r'), (shifts_dx, 'dx', 'b')]):
    data_c = data - np.mean(data)
    acf = np.correlate(data_c, data_c, mode='full')
    acf = acf[len(acf)//2:]  # positive lags only
    acf /= acf[0] + 1e-12  # normalize
    lags = np.arange(len(acf)) * step  # in slice units

    axes[3, i].plot(lags, acf, color+'-')
    axes[3, i].axhline(0, color='k', ls='--', alpha=0.3)
    axes[3, i].set_xlabel("Lag (slices)")
    axes[3, i].set_ylabel("Autocorrelation")
    axes[3, i].set_title(f"Autocorrelation of {label} shift")
    axes[3, i].set_xlim(0, 300)
    axes[3, i].grid(True, alpha=0.3)

plt.suptitle("Per-slice optimal shift/rotation periodicity analysis", fontsize=14)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "shift_periodicity_analysis.png")
plt.savefig(out_path, dpi=150)
print(f"Saved -> {out_path}")

# Print summary stats
print(f"\n--- Summary ---")
print(f"dy: mean={np.mean(shifts_dy):+.2f}, std={np.std(shifts_dy):.2f}, range=[{np.min(shifts_dy):+.1f}, {np.max(shifts_dy):+.1f}]")
print(f"dx: mean={np.mean(shifts_dx):+.2f}, std={np.std(shifts_dx):.2f}, range=[{np.min(shifts_dx):+.1f}, {np.max(shifts_dx):+.1f}]")
print(f"rot: mean={np.mean(rotations):+.3f}, std={np.std(rotations):.3f}, range=[{np.min(rotations):+.2f}, {np.max(rotations):+.2f}]")
print(f"MAE before: {np.mean(mae_before):.4f}, after: {np.mean(mae_after):.4f}")

# Check dominant periods
for label, data in [("dy", shifts_dy), ("dx", shifts_dx), ("rot", rotations)]:
    data_c = data - np.mean(data)
    fft_mag = np.abs(np.fft.rfft(data_c))[1:]
    freqs = np.fft.rfftfreq(len(data), d=step)[1:]
    periods = 1.0 / (freqs + 1e-12)
    top3 = np.argsort(fft_mag)[-3:][::-1]
    print(f"{label} dominant periods: {[f'{periods[t]:.0f}' for t in top3]} slices")

print("Done.", flush=True)
