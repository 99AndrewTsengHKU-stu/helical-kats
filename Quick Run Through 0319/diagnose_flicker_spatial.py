"""
Phase 5A: Flicker spatial distribution + 5C: projs_per_turn consistency.

Diagnoses whether flicker is concentrated at chunk boundaries (every 64 slices)
or distributed uniformly. Also checks projs_per_turn consistency across chunks.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 5A: Load existing reconstruction and compute per-slice flicker ──────────
REC_PATH = os.path.join(os.path.dirname(__file__), "..", "Quick Run Through 0304", "L067_rec_560.npy")
REC_PATH = os.path.normpath(REC_PATH)

print(f"Loading {REC_PATH} ...")
rec = np.load(REC_PATH)  # shape: (ROWS, COLS, SLICES) = (512, 512, 560)
print(f"  shape={rec.shape}, dtype={rec.dtype}")

n_slices = rec.shape[2]
CHUNK_Z = 64

# Compute flicker metric for each adjacent slice pair
flicker = np.zeros(n_slices - 1, dtype=np.float64)
for i in range(n_slices - 1):
    s0 = rec[:, :, i].astype(np.float64)
    s1 = rec[:, :, i + 1].astype(np.float64)
    denom = np.mean(np.abs(s0))
    if denom > 1e-12:
        flicker[i] = np.mean(np.abs(s1 - s0)) / denom
    else:
        flicker[i] = 0.0

print(f"\nFlicker stats:")
print(f"  mean={flicker.mean():.6f}, std={flicker.std():.6f}")
print(f"  min={flicker.min():.6f} (at pair {flicker.argmin()})")
print(f"  max={flicker.max():.6f} (at pair {flicker.argmax()})")

# Identify chunk boundaries
chunk_boundaries = list(range(CHUNK_Z, n_slices, CHUNK_Z))  # [64, 128, 192, ...]
print(f"\nChunk boundaries at slices: {chunk_boundaries}")

# Flicker at boundaries vs interior
boundary_flicker = []
interior_flicker = []
for i in range(len(flicker)):
    slice_idx = i + 1  # this is the "right" slice of the pair
    if slice_idx in chunk_boundaries:
        boundary_flicker.append(flicker[i])
    else:
        interior_flicker.append(flicker[i])

boundary_flicker = np.array(boundary_flicker)
interior_flicker = np.array(interior_flicker)

print(f"\n** CHUNK BOUNDARY flicker: mean={boundary_flicker.mean():.6f}, "
      f"std={boundary_flicker.std():.6f}, n={len(boundary_flicker)} **")
print(f"** INTERIOR flicker:       mean={interior_flicker.mean():.6f}, "
      f"std={interior_flicker.std():.6f}, n={len(interior_flicker)} **")
ratio = boundary_flicker.mean() / interior_flicker.mean() if interior_flicker.mean() > 0 else 0
print(f"** Boundary/Interior ratio: {ratio:.3f}x **")

# ── Plot ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# Plot 1: Full flicker curve
ax = axes[0]
x = np.arange(len(flicker))
ax.plot(x, flicker, 'b-', linewidth=0.5, alpha=0.7, label='Flicker per slice-pair')
ax.axhline(flicker.mean(), color='r', linestyle='--', linewidth=1, label=f'Mean={flicker.mean():.4f}')
ax.axhline(0.040, color='g', linestyle='--', linewidth=1, label='GT baseline=0.040')
for cb in chunk_boundaries:
    ax.axvline(cb - 0.5, color='orange', linestyle=':', linewidth=0.8, alpha=0.7)
ax.set_xlabel('Slice pair index')
ax.set_ylabel('Flicker metric')
ax.set_title(f'Flicker spatial distribution (560 slices, CHUNK_Z={CHUNK_Z})')
ax.legend(fontsize=8)
ax.set_xlim(0, len(flicker))

# Plot 2: Zoom into 2 chunk boundaries
ax = axes[1]
zoom_start = CHUNK_Z * 3 - 30  # around slice 192
zoom_end = CHUNK_Z * 5 + 30    # around slice 320
mask = (x >= zoom_start) & (x < zoom_end)
ax.plot(x[mask], flicker[mask], 'b-o', markersize=2, linewidth=0.8)
ax.axhline(flicker.mean(), color='r', linestyle='--', linewidth=1)
ax.axhline(0.040, color='g', linestyle='--', linewidth=1)
for cb in chunk_boundaries:
    if zoom_start <= cb <= zoom_end:
        ax.axvline(cb - 0.5, color='orange', linestyle='-', linewidth=2, label=f'Chunk boundary (slice {cb})')
ax.set_xlabel('Slice pair index')
ax.set_ylabel('Flicker metric')
ax.set_title(f'Zoom: slices {zoom_start}-{zoom_end} (chunk boundaries highlighted)')
ax.legend(fontsize=8)

# Plot 3: Histogram comparing boundary vs interior
ax = axes[2]
bins = np.linspace(0, max(flicker.max(), 0.5), 50)
ax.hist(interior_flicker, bins=bins, alpha=0.6, label=f'Interior (n={len(interior_flicker)}, mean={interior_flicker.mean():.4f})', color='blue')
ax.hist(boundary_flicker, bins=bins, alpha=0.6, label=f'Boundary (n={len(boundary_flicker)}, mean={boundary_flicker.mean():.4f})', color='orange')
ax.set_xlabel('Flicker metric')
ax.set_ylabel('Count')
ax.set_title('Flicker distribution: chunk boundaries vs interior')
ax.legend(fontsize=9)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "flicker_spatial_distribution.png")
plt.savefig(out_path, dpi=150)
print(f"\nSaved plot -> {out_path}")

# ── 5C: projs_per_turn consistency across chunks ────────────────────────────
print("\n" + "=" * 60)
print("5C: projs_per_turn consistency check")
print("=" * 60)

# We need the data loader to compute z_cull per chunk
# Instead of re-loading DICOM, we simulate from run_L067.py parameters
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

# Try to load the cached metadata if available
CACHE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "Quick Run Through 0304"))
sys.path.insert(0, CACHE_DIR)

# We'll compute analytically from known parameters
SLICES = 560
VOXEL_SIZE_Z = 0.8  # mm
total_half_z = SLICES * VOXEL_SIZE_Z * 0.5  # 224mm

# From run_L067.py known values:
# Total projections: 48590
# angle range: ~20.26 turns * 2pi = ~127.3 rad
# pitch: 3.656 mm/rad
# The key question: does projs_per_turn vary per chunk?

# projs_per_turn = total_projs_in_chunk / s_len_chunk * 2*pi
# s_len_chunk = abs(angles_chunk[-1] - angles_chunk[0])
# total_projs_in_chunk = len(keep_idx) from z_cull

# Since all projections have uniform angular spacing (delta_s ≈ const),
# projs_per_turn should be approximately constant regardless of how many
# projections are in the chunk. Let's verify:

# Simulate: each chunk selects a contiguous-ish range of projections.
# projs_per_turn = N / (angle_range) * 2pi
# If angle_step is constant: projs_per_turn = N / ((N-1)*delta_s) * 2pi ≈ 2pi/delta_s
# This is independent of N! So projs_per_turn should be nearly identical.

# But there's a subtlety: z_cull doesn't select contiguous projections,
# and angles_range = abs(last - first) ignores gaps.

# Let's compute this properly using the actual parameters
print("\nAnalytical check:")
print("  projs_per_turn = total_projs / s_len * 2*pi")
print("  s_len = abs(angles[-1] - angles[0]) for the z-culled subset")
print("  If angular spacing is uniform (delta_s = const):")
print("    projs_per_turn = N / ((N-1)*delta_s) * 2*pi ≈ 2*pi/delta_s")
print("  This is ~independent of N, so chunks should have consistent projs_per_turn.")

# But let's check: what if z_cull creates a non-contiguous set?
# Actually looking at z_cull_indices, it selects projections where
# src_z ± cone_half_z ± margin overlaps chunk z-range.
# Since src_z increases monotonically (right-handed helix), the selected
# indices ARE contiguous. So projs_per_turn = 2pi/delta_s for all chunks.

# However, inv_scale = SOD^2 / projs_per_turn is computed from conf,
# and conf is rebuilt per chunk. Let's check if s_len computation is correct.
print("\n  z_cull selects contiguous projection range (monotonic src_z)")
print("  => projs_per_turn ≈ 2*pi / delta_s for ALL chunks (consistent)")
print("  => inv_scale = SOD^2 / projs_per_turn is consistent")

# BUT WAIT: there's a potential issue in create_configuration!
# Line 91-97 of initialize.py:
#   if "angles_range" in scan_geometry['helix']:
#       s_len = angles_range
#       s_min = -s_len * 0.5
#       s_max = s_len * 0.5
#
# And then line 99:
#   projs_per_turn = total_projs / s_len * 2*pi
#
# The problem: s_min = -s_len/2, s_max = s_len/2
# But source_pos at line 107:
#   source_pos = s_min + delta_s * (arange(total_projs) + 0.5)
# This is OVERWRITTEN at run_L067.py:372:
#   conf_chunk['source_pos'] = angles_chunk
#
# So source_pos is correct (actual angles), but s_min/s_max are WRONG:
# they're centered at 0, while actual angles are centered around
# the chunk's angle range (which is NOT centered at 0).
#
# Does this matter? s_min/s_max are used in:
# - source_pos generation (overwritten, so no effect)
# - Possibly in filter pipeline? Let's check...

print("\n** POTENTIAL ISSUE FOUND **")
print("  create_configuration computes s_min = -s_len/2, s_max = s_len/2")
print("  This centers the angle range at 0, but actual chunk angles are NOT centered at 0")
print("  source_pos is overwritten with actual angles (line 372), so BP is OK")
print("  BUT: does the filter pipeline use s_min/s_max internally?")
print("  Need to check filter.py for s_min/s_max usage...")

# Check if filter.py uses s_min/s_max
import importlib.util
filter_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "pykatsevich", "filter.py"))
with open(filter_path, 'r') as f:
    filter_src = f.read()

s_min_uses = [i for i, line in enumerate(filter_src.split('\n'), 1)
              if 's_min' in line and not line.strip().startswith('#')]
s_max_uses = [i for i, line in enumerate(filter_src.split('\n'), 1)
              if 's_max' in line and not line.strip().startswith('#')]

print(f"\n  filter.py lines with s_min: {s_min_uses}")
print(f"  filter.py lines with s_max: {s_max_uses}")

# Also check: does delta_s override matter?
delta_s_uses = [i for i, line in enumerate(filter_src.split('\n'), 1)
                if 'delta_s' in line and not line.strip().startswith('#')]
print(f"  filter.py lines with delta_s: {delta_s_uses}")

source_pos_uses = [i for i, line in enumerate(filter_src.split('\n'), 1)
                   if 'source_pos' in line and not line.strip().startswith('#')]
print(f"  filter.py lines with source_pos: {source_pos_uses}")

print("\nDone.")
