"""
H4 Test: T-D Weight Partition of Unity Check
=============================================
For a set of voxels, accumulate the T-D weight of every contributing projection.
The sum should equal 1.0 for every voxel (partition of unity).
If it oscillates with z at period = z_per_turn, the T-D boundary formula is the cause of flicker.

No GPU needed — pure NumPy.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import copy
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from pykatsevich import load_dicom_projections
from pykatsevich.initialize import create_configuration
import astra

DICOM_DIR = r"D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

ROWS = COLS = 512
SLICES = 560
VOXEL_SIZE_XY = 0.6640625
VOXEL_SIZE_Z = 0.800
RECON_OFFSET_X = +11.0

print("Loading DICOM...", flush=True)
_, meta = load_dicom_projections(DICOM_DIR)
sg = meta['scan_geometry']
angles_full = -meta['angles_rad'].copy() - np.pi / 2
ffs_z = meta.get('ffs_z_offsets_mm', np.zeros(len(meta['angles_rad']), dtype=np.float32))
src_z = meta['table_positions_mm'] + ffs_z  # physical source z for each projection

SOD = sg['SOD']
SDD = sg['SDD']
psize_cols = sg['detector'].get('detector psize cols', sg['detector']['detector psize'])
psize_rows = sg['detector'].get('detector psize rows', sg['detector']['detector psize'])
det_cols = sg['detector']['detector cols']
det_rows = sg['detector']['detector rows']
col_offset = sg['detector'].get('detector_col_offset', 0.0)
row_offset = sg['detector'].get('detector_row_offset', 0.0)
pitch_abs = float(abs(meta['pitch_mm_per_rad_signed']))

# Build a minimal conf (single-chunk covering full scan) just to get proj_row_mins/maxs
scan_geom = copy.deepcopy(sg)
scan_geom['helix']['pitch_mm_rad'] = pitch_abs
scan_geom['helix']['angles_count'] = len(angles_full)
scan_geom['helix']['angles_range'] = float(abs(angles_full[-1] - angles_full[0]))

half_xy = COLS * VOXEL_SIZE_XY * 0.5
total_half_z = SLICES * VOXEL_SIZE_Z * 0.5
vol_geom = astra.create_vol_geom(
    ROWS, COLS, SLICES,
    -half_xy + RECON_OFFSET_X, half_xy + RECON_OFFSET_X,
    -half_xy, half_xy,
    -total_half_z, total_half_z,
)
conf = create_configuration(scan_geom, vol_geom)

# T-D boundaries (shape: det_cols+1) — the "extended" col coord array
proj_row_mins = conf['proj_row_mins']   # shape (det_cols+1,)
proj_row_maxs = conf['proj_row_maxs']   # shape (det_cols+1,)
alpha = conf['T-D smoothing']

# col_coords and row_coords (extended by 1)
col_coords = conf['col_coords']  # (det_cols+1,)
row_coords = conf['row_coords']  # (det_rows+1,)

print(f"SOD={SOD}, SDD={SDD}, psize_col={psize_cols:.4f}, psize_row={psize_rows:.4f}")
print(f"col_offset={col_offset:.4f}, row_offset={row_offset:.4f}")
print(f"pitch_mm_rad={pitch_abs:.4f}, projs_per_turn={conf['projs_per_turn']:.2f}")
print(f"T-D smoothing alpha={alpha}")

def smoothstep(x, alpha):
    """Same smoothstep as in filter.py"""
    if alpha == 0.0:
        return np.where(x >= 0, 1.0, 0.0).astype(np.float32)
    t = np.clip(x / alpha, 0.0, 1.0)
    return (3 * t**2 - 2 * t**3).astype(np.float32)

# ── Test 1: Central voxel (x=0, y=0) across all z slices ────────────────────
print("\n=== Test 1: Partition of unity, central voxel (x=0,y=0) ===")

# For a voxel at (x=0, y=0, z_k), the source-to-voxel distance is just SOD
# (since x=0, y=0 is the isocenter). The projection hits the detector at:
#   col = u_center  (exactly the center column, since x=y=0)
#   row = (z_k - src_z[s]) / psize_row * SDD/SOD + v_center
# Note: col_coords are centered, so the T-D boundary at col=0 (center) is:
#   proj_row_mins[center_col_idx]  and  proj_row_maxs[center_col_idx]
# where center_col_idx corresponds to col_coord ≈ 0 + col_offset adjustment

# Find the index in col_coords closest to 0 (center of fan)
center_col_idx = int(round((det_cols - 1) * 0.5 + col_offset))
center_col_coord = col_coords[center_col_idx]
td_min_center = proj_row_mins[center_col_idx]
td_max_center = proj_row_maxs[center_col_idx]
print(f"Center col idx={center_col_idx}, coord={center_col_coord:.4f}mm")
print(f"T-D row range at center col: [{td_min_center:.4f}, {td_max_center:.4f}] mm")

z_values = -total_half_z + (np.arange(SLICES) + 0.5) * VOXEL_SIZE_Z
partition_sums = np.zeros(SLICES, dtype=np.float64)

v_center_mm = (det_rows - 1) * 0.5 * psize_rows + row_offset * psize_rows  # physical center row in mm
# Actually: row_coords are centered such that row_coord = (row_idx - v_center) * psize_rows
# For voxel at z_k, source at (angle s, src_z[s]):
#   physical row coord on detector = (z_k - src_z[s]) * SDD / SOD

for k in range(SLICES):
    z_k = z_values[k]
    # Row coordinate in mm for each projection (at center col, x=y=0, L=SOD)
    row_hit_mm = (z_k - src_z) * SDD / SOD   # shape (N_projs,)
    # T-D weight at center col for each projection
    w = smoothstep(row_hit_mm - td_min_center, alpha) * smoothstep(td_max_center - row_hit_mm, alpha)
    partition_sums[k] = float(np.sum(w))

print(f"\nPartition sum stats (center voxel):")
print(f"  mean  = {partition_sums.mean():.6f}  (should be ≈ 1.0)")
print(f"  std   = {partition_sums.std():.6f}  (should be ≈ 0.0)")
print(f"  min   = {partition_sums.min():.6f}")
print(f"  max   = {partition_sums.max():.6f}")

# FFT to check periodicity
z_per_turn = pitch_abs * 2 * np.pi
slices_per_turn = z_per_turn / VOXEL_SIZE_Z
fft_mag = np.abs(np.fft.rfft(partition_sums - partition_sums.mean()))
freqs = np.fft.rfftfreq(SLICES)
dominant_freq_idx = np.argmax(fft_mag[1:]) + 1
dominant_period_slices = 1.0 / freqs[dominant_freq_idx] if freqs[dominant_freq_idx] > 0 else np.inf
print(f"\nFFT analysis:")
print(f"  Dominant period = {dominant_period_slices:.1f} slices")
print(f"  Expected per-turn period = {slices_per_turn:.1f} slices")
print(f"  Amplitude at dominant freq = {fft_mag[dominant_freq_idx]:.6f}")
print(f"  → {'PER-TURN OSCILLATION DETECTED — H4 CONFIRMED' if abs(dominant_period_slices - slices_per_turn) < 2 else 'No per-turn periodicity — H4 likely not the cause'}")

# ── Test 2: Off-center voxels ────────────────────────────────────────────────
print("\n=== Test 2: Partition of unity, off-center voxels ===")

test_voxels = [
    (0.0,    0.0,    "center"),
    (80.0,   0.0,    "x=+80mm"),
    (-80.0,  0.0,    "x=-80mm"),
    (0.0,    80.0,   "y=+80mm"),
    (50.0,   50.0,   "xy=+50mm"),
]

z_sample = z_values[SLICES//4 : 3*SLICES//4 : 5]   # 56 z values in middle half

for (vx, vy, label) in test_voxels:
    sums = []
    for z_k in z_sample:
        # Source-voxel geometry for each projection
        # Source is at (SOD*cos(θ+π/2+π), SOD*sin(θ+π/2+π), src_z)
        # But we stored angles_full = -DICOM_angles - π/2
        # Voxel at (vx, vy, z_k)
        # L = SOD - vx*cos(θ_rad) - vy*sin(θ_rad)  where θ_rad = angles used in BP
        theta = angles_full
        cs = np.cos(theta)
        sn = np.sin(theta)
        L = SOD - vx * cs - vy * sn
        # Fan coord (column) at detector
        col_mm = SDD * (vy * cs - vx * sn) / L   # mm from center
        # Row coord at detector
        row_mm = (z_k - src_z) * SDD / L          # mm from center

        # Interpolate T-D boundaries at col_mm
        # col_coords are at indices 0..det_cols, so map col_mm to index
        col_idx_f = col_mm / psize_cols + (det_cols - 1) * 0.5 + col_offset
        col_idx_i = np.clip(col_idx_f.astype(int), 0, det_cols - 1)
        td_min = proj_row_mins[col_idx_i]
        td_max = proj_row_maxs[col_idx_i]
        # Mask out projections where L <= 0 (source behind voxel)
        valid = L > 0
        w = np.where(valid,
                     smoothstep(row_mm - td_min, alpha) * smoothstep(td_max - row_mm, alpha),
                     0.0)
        sums.append(float(np.sum(w)))
    sums = np.array(sums)
    print(f"  {label:12s}: mean={sums.mean():.4f}, std={sums.std():.4f}, "
          f"range=[{sums.min():.4f},{sums.max():.4f}]")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Plot 1: partition sum vs z
axes[0].plot(z_values, partition_sums, 'b-', lw=0.8)
axes[0].axhline(1.0, color='r', ls='--', lw=1, label='ideal = 1.0')
axes[0].set_xlabel('z (mm)')
axes[0].set_ylabel('Σ w_TD')
axes[0].set_title(f'T-D Partition of Unity — Central Voxel (x=0, y=0)\nmean={partition_sums.mean():.5f}, std={partition_sums.std():.5f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: deviation from 1.0
deviation = partition_sums - 1.0
axes[1].plot(z_values, deviation, 'r-', lw=0.8)
axes[1].axhline(0.0, color='k', ls='--', lw=0.8)
axes[1].set_xlabel('z (mm)')
axes[1].set_ylabel('Σ w_TD − 1')
axes[1].set_title(f'Deviation from Partition of Unity (per-turn period = {slices_per_turn:.1f} slices = {z_per_turn:.2f}mm)')
axes[1].grid(True, alpha=0.3)
# Mark expected per-turn boundaries
for n in range(int(2*total_half_z / z_per_turn) + 2):
    xv = -total_half_z + n * z_per_turn
    axes[1].axvline(xv, color='g', alpha=0.3, lw=0.5)

# Plot 3: FFT magnitude
axes[2].plot(1.0/freqs[1:] * VOXEL_SIZE_Z, fft_mag[1:], 'b-', lw=0.8)
axes[2].axvline(z_per_turn, color='r', ls='--', lw=1.5, label=f'z_per_turn={z_per_turn:.2f}mm')
axes[2].set_xlabel('Period (mm)')
axes[2].set_ylabel('FFT magnitude')
axes[2].set_title('FFT of partition sum deviation — looking for per-turn spike')
axes[2].set_xlim(0, z_per_turn * 4)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "td_partition_check.png")
plt.savefig(out_path, dpi=150)
print(f"\nSaved plot → {out_path}")
print("Done.")
