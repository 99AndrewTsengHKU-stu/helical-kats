"""
Analyze table_positions_mm residual vs linear fit.
Check if there's a per-turn periodic deviation.
"""
import sys, os, struct
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from pathlib import Path
import pydicom, matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

DICOM_DIR = Path(r"D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Read all metadata (angles + table z + FFS z)
print("Reading metadata from all DICOM files...", flush=True)
paths = sorted(DICOM_DIR.glob("*.dcm"))
records = []
for p in paths:
    ds = pydicom.dcmread(p, stop_before_pixels=True)
    instance = int(getattr(ds, "InstanceNumber", 0))
    raw_ang = bytes(ds[(0x7031, 0x1001)].value)
    angle = struct.unpack("<f", raw_ang[:4])[0]
    # z position
    if (0x7031, 0x1002) in ds:
        z = struct.unpack("<f", bytes(ds[(0x7031, 0x1002)].value)[:4])[0]
    else:
        z = 0.0
    # FFS
    if (0x7033, 0x100B) in ds:
        ffs = struct.unpack("<f", bytes(ds[(0x7033, 0x100B)].value)[:4])[0]
    else:
        ffs = 0.0
    records.append((instance, angle, z, ffs))

records.sort(key=lambda x: x[0])
angles_raw = np.array([r[1] for r in records])
angles = np.unwrap(angles_raw).astype(np.float64)
table_z = np.array([r[2] for r in records], dtype=np.float64)
ffs_z = np.array([r[3] for r in records], dtype=np.float64)
n = len(angles)
idx = np.arange(n)

print(f"Total views: {n}")
print(f"Angle range: {angles[0]:.3f} to {angles[-1]:.3f} rad ({(angles[-1]-angles[0])/(2*np.pi):.1f} turns)")
print(f"Table z range: {table_z[0]:.2f} to {table_z[-1]:.2f} mm")

# Linear fit to table_z vs index
coeff_z = np.polyfit(idx, table_z, 1)
table_z_linear = np.polyval(coeff_z, idx)
residual_z = table_z - table_z_linear

# Linear fit to angles vs index
coeff_a = np.polyfit(idx, angles, 1)
angles_linear = np.polyval(coeff_a, idx)
residual_a = angles - angles_linear

# Compute projs_per_turn
angle_step = coeff_a[0]
projs_per_turn = 2 * np.pi / abs(angle_step)
z_per_turn = projs_per_turn * abs(coeff_z[0])

print(f"\nLinear fit:")
print(f"  angle_step = {angle_step:.8f} rad/proj")
print(f"  z_step = {coeff_z[0]:.8f} mm/proj")
print(f"  projs_per_turn = {projs_per_turn:.1f}")
print(f"  z_per_turn = {z_per_turn:.3f} mm")

print(f"\nTable z residual: max={np.max(np.abs(residual_z)):.6f} mm, std={np.std(residual_z):.6f} mm")
print(f"Angle residual: max={np.max(np.abs(residual_a)):.8f} rad, std={np.std(residual_a):.8f} rad")
print(f"Angle residual in degrees: max={np.degrees(np.max(np.abs(residual_a))):.4f}°")

# FFT of residuals
fig, axes = plt.subplots(4, 2, figsize=(16, 16))

# Row 0: Raw residuals
axes[0, 0].plot(idx, residual_z, 'b-', linewidth=0.3)
axes[0, 0].set_xlabel("View index")
axes[0, 0].set_ylabel("Residual (mm)")
axes[0, 0].set_title(f"Table z residual vs linear fit (std={np.std(residual_z):.4f}mm)")
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(idx, np.degrees(residual_a), 'r-', linewidth=0.3)
axes[0, 1].set_xlabel("View index")
axes[0, 1].set_ylabel("Residual (deg)")
axes[0, 1].set_title(f"Angle residual vs linear fit (std={np.degrees(np.std(residual_a)):.4f}°)")
axes[0, 1].grid(True, alpha=0.3)

# Row 1: FFT of residuals (in proj units)
for i, (data, label, unit) in enumerate([(residual_z, "table z", "mm"), (residual_a, "angle", "rad")]):
    fft_mag = np.abs(np.fft.rfft(data))[1:]
    freqs = np.fft.rfftfreq(len(data), d=1.0)[1:]  # cycles per projection
    periods_proj = 1.0 / freqs  # period in projections

    axes[1, i].plot(periods_proj, fft_mag, 'k-', linewidth=0.5)
    axes[1, i].axvline(projs_per_turn, color='r', ls='--', alpha=0.7, label=f'1 turn = {projs_per_turn:.0f} projs')
    axes[1, i].axvline(projs_per_turn/2, color='orange', ls='--', alpha=0.5, label=f'½ turn = {projs_per_turn/2:.0f}')
    axes[1, i].set_xlabel("Period (projections)")
    axes[1, i].set_ylabel("FFT magnitude")
    axes[1, i].set_title(f"FFT of {label} residual")
    axes[1, i].set_xlim(0, projs_per_turn * 3)
    axes[1, i].legend(fontsize=8)
    axes[1, i].grid(True, alpha=0.3)

# Row 2: Zoom on first few turns of residuals
n_show = int(projs_per_turn * 3)
axes[2, 0].plot(idx[:n_show], residual_z[:n_show], 'b.-', markersize=1, linewidth=0.5)
axes[2, 0].axvline(projs_per_turn, color='r', ls='--', alpha=0.5)
axes[2, 0].axvline(projs_per_turn*2, color='r', ls='--', alpha=0.5)
axes[2, 0].set_xlabel("View index")
axes[2, 0].set_ylabel("Residual z (mm)")
axes[2, 0].set_title("Table z residual (first 3 turns)")
axes[2, 0].grid(True, alpha=0.3)

axes[2, 1].plot(idx[:n_show], np.degrees(residual_a[:n_show]), 'r.-', markersize=1, linewidth=0.5)
axes[2, 1].axvline(projs_per_turn, color='b', ls='--', alpha=0.5)
axes[2, 1].axvline(projs_per_turn*2, color='b', ls='--', alpha=0.5)
axes[2, 1].set_xlabel("View index")
axes[2, 1].set_ylabel("Residual angle (deg)")
axes[2, 1].set_title("Angle residual (first 3 turns)")
axes[2, 1].grid(True, alpha=0.3)

# Row 3: FFS pattern + combined z (table + FFS) residual
axes[3, 0].plot(idx[:200], ffs_z[:200], 'g.-', markersize=2)
axes[3, 0].set_xlabel("View index")
axes[3, 0].set_ylabel("FFS z offset (mm)")
axes[3, 0].set_title("Flying focal spot z-offset (first 200 views)")
axes[3, 0].grid(True, alpha=0.3)

# Combined residual
combined_z = table_z + ffs_z
coeff_cz = np.polyfit(idx, combined_z, 1)
residual_cz = combined_z - np.polyval(coeff_cz, idx)
axes[3, 1].plot(idx[:n_show], residual_cz[:n_show], 'm.-', markersize=1, linewidth=0.5)
axes[3, 1].axvline(projs_per_turn, color='r', ls='--', alpha=0.5)
axes[3, 1].set_xlabel("View index")
axes[3, 1].set_ylabel("Residual (mm)")
axes[3, 1].set_title(f"Combined (table+FFS) z residual (std={np.std(residual_cz):.4f}mm)")
axes[3, 1].grid(True, alpha=0.3)

plt.suptitle("Table position & angle residual analysis", fontsize=14)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "table_residual_analysis.png")
plt.savefig(out_path, dpi=150)
print(f"\nSaved -> {out_path}")
print("Done.", flush=True)
