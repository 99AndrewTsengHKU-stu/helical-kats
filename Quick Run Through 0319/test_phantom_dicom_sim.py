"""
Phase 6B: Simulate DICOM convention on phantom data.

Tests whether flip_rows + negate_angles + angle_offset introduce geometric
shift by applying these transforms to a synthetic phantom's sinogram, then
reconstructing with our L067 pipeline logic.

Comparisons:
  A) Clean pipeline: pure pykatsevich (no transforms)
  B) Simulated DICOM: apply reverse transforms to sinogram, then reconstruct
     with flip/negate/offset (mimicking our L067 pipeline)
  C) Measure sphere center positions in both → detect geometric shift
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Quick Run Through 0304"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from time import time
import yaml
import copy
import astra

from common import phantom_objects_3d, project
from pykatsevich.geometry import astra_helical_views
from pykatsevich.initialize import create_configuration
from pykatsevich.filter import (
    differentiate, fw_height_rebinning, compute_hilbert_kernel,
    hilbert_conv, rev_rebin_vec, sino_weight_td, filter_katsevich
)
from backproject_cupy import backproject_cupy

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "tests"))


def flicker_metric(vol):
    n = vol.shape[2]
    m = []
    for i in range(n - 1):
        s0 = vol[:, :, i].astype(np.float64)
        s1 = vol[:, :, i + 1].astype(np.float64)
        denom = 0.5 * (np.mean(np.abs(s0)) + np.mean(np.abs(s1)))
        m.append(np.mean(np.abs(s1 - s0)) / max(denom, 1e-12))
    return np.array(m)


def find_sphere_center_z(vol, voxel_size_z):
    """Find z-center of the brightest object by intensity-weighted centroid."""
    # Sum over x,y for each z-slice
    z_profile = np.array([vol[:, :, i].sum() for i in range(vol.shape[2])])
    z_coords = (np.arange(vol.shape[2]) - vol.shape[2] / 2.0 + 0.5) * voxel_size_z
    # Weighted centroid
    total = z_profile.sum()
    if abs(total) > 1e-12:
        center = np.sum(z_profile * z_coords) / total
    else:
        center = 0.0
    return center, z_profile, z_coords


def measure_couch_drift(vol, n_sample=20):
    """Measure horizontal shift of a feature across z-slices using cross-correlation."""
    mid = vol.shape[2] // 2
    ref_slice = vol[:, :, mid]
    shifts = []
    slice_indices = np.linspace(max(mid - n_sample, 0),
                                min(mid + n_sample, vol.shape[2] - 1),
                                2 * n_sample + 1, dtype=int)
    for iz in slice_indices:
        test_slice = vol[:, :, iz]
        # Cross-correlate rows (y-direction) to find shift
        from scipy.ndimage import shift as ndshift
        from scipy.signal import correlate2d
        cc = correlate2d(ref_slice, test_slice, mode='same')
        peak = np.unravel_index(cc.argmax(), cc.shape)
        dy = peak[0] - ref_slice.shape[0] // 2
        dx = peak[1] - ref_slice.shape[1] // 2
        shifts.append((iz, dx, dy))
    return shifts


# ══════════════════════════════════════════════════════════════════════
# USE test01 (256x256x256, best quality)
# ══════════════════════════════════════════════════════════════════════
yaml_path = os.path.join(TEST_DIR, "test01.yaml")
with open(yaml_path, "r") as f:
    settings = yaml.safe_load(f)

ps = settings['phantom']
geom = settings['geometry']

print("Generating phantom...", flush=True)
phantom = phantom_objects_3d(
    ps['rows'], ps['columns'], ps['slices'],
    voxel_size=ps['voxel_size'],
    objects_list=ps['objects'],
)
voxel_size = ps['voxel_size']
print(f"  Phantom: {phantom.shape}, voxel={voxel_size}mm")

# ══════════════════════════════════════════════════════════════════════
# A) CLEAN PIPELINE: Pure pykatsevich, no transforms
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("TEST A: Clean pipeline (no transforms)")
print(f"{'='*60}", flush=True)

sinogram, vol_geom, proj_geom = project(phantom, voxel_size, geom)
sinogram_swapped = np.asarray(np.swapaxes(sinogram, 0, 1), order='C')

conf_clean = create_configuration(geom, vol_geom, geom.get('options', {}))

# Get the angles and views that project() used internally
# (reproduce from common.py::project logic)
s_len = geom['helix']['angles_range']
s_min = -s_len * 0.5
projs_per_turn = geom['helix']['angles_count'] / s_len * 2 * np.pi
delta_s = 2 * np.pi / projs_per_turn
angles_clean = s_min + delta_s * (np.arange(geom['helix']['angles_count'], dtype=np.float32) + 0.5)
stride_mm = geom['helix']['pitch_mm_rad'] * delta_s

views_clean = astra_helical_views(
    geom["SOD"], geom["SDD"],
    geom['detector']["detector psize"],
    angles_clean, stride_mm,
)

t0 = time()
filtered_clean = filter_katsevich(
    sinogram_swapped, conf_clean,
    {"Diff": {"Print time": False}, "FwdRebin": {"Print time": False},
     "BackRebin": {"Print time": False}},
)
sino_td_clean = sino_weight_td(filtered_clean, conf_clean, False)
rec_clean = backproject_cupy(sino_td_clean, conf_clean, vol_geom, proj_geom, tqdm_bar=False)
print(f"  Done in {time()-t0:.1f}s, range=[{rec_clean.min():.4f}, {rec_clean.max():.4f}]")
fm_clean = flicker_metric(rec_clean)
print(f"  Flicker: mean={fm_clean.mean():.6f}")

# ══════════════════════════════════════════════════════════════════════
# B) SIMULATED DICOM: Reverse-engineer what DICOM would look like,
#    then apply our L067 transforms and reconstruct.
#
# Our L067 pipeline does:
#   1) negate_angles: angles = -angles
#   2) angle_offset:  angles = angles - pi/2
#   3) flip_rows:     sino = sino[:, ::-1, :]
#
# To simulate "DICOM" data from the clean sinogram:
#   - The clean sinogram was projected with angles_clean
#   - DICOM would have angles_dicom such that after negate+offset
#     we get angles_clean back:
#       angles_clean = -(angles_dicom) - pi/2
#       => angles_dicom = -(angles_clean + pi/2) = -angles_clean - pi/2
#   - DICOM sinogram rows are un-flipped:
#       sino_dicom = sino_clean[:, ::-1, :]  (undo the flip)
#
# Then our pipeline applies:
#   angles = -angles_dicom - pi/2 = -(-angles_clean - pi/2) - pi/2 = angles_clean
#   sino = sino_dicom[:, ::-1, :] = sino_clean
#
# This should give IDENTICAL results to Test A (round-trip).
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("TEST B: Simulated DICOM (round-trip: should match A)")
print(f"{'='*60}", flush=True)

# Simulate DICOM angles and sinogram
angles_dicom = -angles_clean - np.pi / 2
sino_dicom = sinogram_swapped[:, ::-1, :].copy()  # un-flip rows

# Now apply our L067 transforms
angles_b = -angles_dicom - np.pi / 2  # negate + offset
sino_b = sino_dicom[:, ::-1, :].copy()  # flip_rows

print(f"  angles_clean[0:3]: {angles_clean[:3]}")
print(f"  angles_b[0:3]:     {angles_b[:3]}")
print(f"  Max angle diff: {np.abs(angles_b - angles_clean).max():.2e}")
print(f"  Sino match: {np.allclose(sino_b, sinogram_swapped)}")

# Build views with the "DICOM" z-shifts (same as clean, since perfect helix)
views_b = astra_helical_views(
    geom["SOD"], geom["SDD"],
    geom['detector']["detector psize"],
    angles_b, stride_mm,
)

proj_geom_b = astra.create_proj_geom(
    "cone_vec",
    geom['detector']["detector rows"],
    geom['detector']["detector cols"],
    views_b,
)

conf_b = create_configuration(geom, vol_geom, geom.get('options', {}))
conf_b['source_pos'] = angles_b.astype(np.float32)
conf_b['delta_s'] = float(np.mean(np.diff(angles_b)))

t0 = time()
filtered_b = filter_katsevich(
    sino_b.astype(np.float32), conf_b,
    {"Diff": {"Print time": False}, "FwdRebin": {"Print time": False},
     "BackRebin": {"Print time": False}},
)
sino_td_b = sino_weight_td(filtered_b, conf_b, False)
rec_b = backproject_cupy(sino_td_b, conf_b, vol_geom, proj_geom_b, tqdm_bar=False)
print(f"  Done in {time()-t0:.1f}s, range=[{rec_b.min():.4f}, {rec_b.max():.4f}]")
fm_b = flicker_metric(rec_b)
print(f"  Flicker: mean={fm_b.mean():.6f}")

# ══════════════════════════════════════════════════════════════════════
# C) BROKEN DICOM: Apply transforms but DON'T adjust geometry properly.
#    This simulates a real mistake: e.g., flip sinogram rows but keep
#    the same views (v-vector not flipped).
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("TEST C: Flip rows only (no geometry compensation)")
print(f"{'='*60}", flush=True)

# Just flip rows of clean sinogram, keep everything else the same
sino_c = sinogram_swapped[:, ::-1, :].copy()

t0 = time()
# Use clean conf (row_coords not adjusted)
filtered_c = filter_katsevich(
    sino_c.astype(np.float32), conf_clean,
    {"Diff": {"Print time": False}, "FwdRebin": {"Print time": False},
     "BackRebin": {"Print time": False}},
)
sino_td_c = sino_weight_td(filtered_c, conf_clean, False)
# Use clean proj_geom (v-vector not flipped)
rec_c = backproject_cupy(sino_td_c, conf_clean, vol_geom, proj_geom, tqdm_bar=False)
print(f"  Done in {time()-t0:.1f}s, range=[{rec_c.min():.4f}, {rec_c.max():.4f}]")
fm_c = flicker_metric(rec_c)
print(f"  Flicker: mean={fm_c.mean():.6f}")

# ══════════════════════════════════════════════════════════════════════
# D) Negate angles only (no geometry compensation)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("TEST D: Negate angles only (no other compensation)")
print(f"{'='*60}", flush=True)

angles_d = -angles_clean
views_d = astra_helical_views(
    geom["SOD"], geom["SDD"],
    geom['detector']["detector psize"],
    angles_d, stride_mm,
)
proj_geom_d = astra.create_proj_geom(
    "cone_vec", geom['detector']["detector rows"],
    geom['detector']["detector cols"], views_d,
)
conf_d = create_configuration(geom, vol_geom, geom.get('options', {}))
conf_d['source_pos'] = angles_d.astype(np.float32)
conf_d['delta_s'] = float(np.mean(np.diff(angles_d)))

t0 = time()
filtered_d = filter_katsevich(
    sinogram_swapped.astype(np.float32), conf_d,
    {"Diff": {"Print time": False}, "FwdRebin": {"Print time": False},
     "BackRebin": {"Print time": False}},
)
sino_td_d = sino_weight_td(filtered_d, conf_d, False)
rec_d = backproject_cupy(sino_td_d, conf_d, vol_geom, proj_geom_d, tqdm_bar=False)
print(f"  Done in {time()-t0:.1f}s, range=[{rec_d.min():.4f}, {rec_d.max():.4f}]")
fm_d = flicker_metric(rec_d)
print(f"  Flicker: mean={fm_d.mean():.6f}")

# ══════════════════════════════════════════════════════════════════════
# E) Flip rows + negate angles (mimics L067, no angle_offset)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("TEST E: Flip rows + negate angles (L067-like, no offset)")
print(f"{'='*60}", flush=True)

sino_e = sinogram_swapped[:, ::-1, :].copy()
angles_e = -angles_clean
views_e = astra_helical_views(
    geom["SOD"], geom["SDD"],
    geom['detector']["detector psize"],
    angles_e, stride_mm,
)
proj_geom_e = astra.create_proj_geom(
    "cone_vec", geom['detector']["detector rows"],
    geom['detector']["detector cols"], views_e,
)
conf_e = create_configuration(geom, vol_geom, geom.get('options', {}))
conf_e['source_pos'] = angles_e.astype(np.float32)
conf_e['delta_s'] = float(np.mean(np.diff(angles_e)))

t0 = time()
filtered_e = filter_katsevich(
    sino_e.astype(np.float32), conf_e,
    {"Diff": {"Print time": False}, "FwdRebin": {"Print time": False},
     "BackRebin": {"Print time": False}},
)
sino_td_e = sino_weight_td(filtered_e, conf_e, False)
rec_e = backproject_cupy(sino_td_e, conf_e, vol_geom, proj_geom_e, tqdm_bar=False)
print(f"  Done in {time()-t0:.1f}s, range=[{rec_e.min():.4f}, {rec_e.max():.4f}]")
fm_e = flicker_metric(rec_e)
print(f"  Flicker: mean={fm_e.mean():.6f}")

# ══════════════════════════════════════════════════════════════════════
# GEOMETRIC SHIFT ANALYSIS
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("GEOMETRIC SHIFT ANALYSIS")
print(f"{'='*60}", flush=True)

for label, rec in [("A: Clean", rec_clean), ("B: Round-trip", rec_b),
                    ("C: Flip only", rec_c), ("D: Negate only", rec_d),
                    ("E: Flip+Negate", rec_e)]:
    z_center, z_prof, z_coords = find_sphere_center_z(rec, voxel_size)
    print(f"  {label:<25} z-centroid = {z_center:+.4f} mm  "
          f"(range [{rec.min():.3f}, {rec.max():.3f}])")

# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"  {'Test':<40} {'Flicker':>10} {'z-shift':>10}")
print(f"  {'-'*60}")
for label, rec, fm in [
    ("A: Clean (no transforms)", rec_clean, fm_clean),
    ("B: Sim DICOM round-trip", rec_b, fm_b),
    ("C: Flip rows only", rec_c, fm_c),
    ("D: Negate angles only", rec_d, fm_d),
    ("E: Flip + Negate", rec_e, fm_e),
]:
    z_c, _, _ = find_sphere_center_z(rec, voxel_size)
    print(f"  {label:<40} {fm.mean():>10.6f} {z_c:>+10.4f} mm")
print(f"  {'Our L067':<40} {'0.229':>10}")
print(f"  {'GT L067':<40} {'~0.040':>10}")

# ══════════════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 5, figsize=(25, 15))
mid_z = phantom.shape[2] // 2

all_recs = [
    ("A: Clean", rec_clean, fm_clean),
    ("B: Round-trip", rec_b, fm_b),
    ("C: Flip only", rec_c, fm_c),
    ("D: Negate only", rec_d, fm_d),
    ("E: Flip+Negate", rec_e, fm_e),
]

# Row 0: center slice
vmin = min(r.min() for _, r, _ in all_recs)
vmax = max(r.max() for _, r, _ in all_recs)
for col, (label, rec, fm) in enumerate(all_recs):
    ax = axes[0, col]
    ax.imshow(rec[:, :, mid_z], cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(f"{label}\nfl={fm.mean():.4f}", fontsize=9)
    ax.axis('off')

# Row 1: difference from Clean (A)
for col, (label, rec, fm) in enumerate(all_recs):
    ax = axes[1, col]
    diff = rec[:, :, mid_z].astype(np.float64) - rec_clean[:, :, mid_z].astype(np.float64)
    vlim = max(abs(diff).max(), 1e-6)
    ax.imshow(diff, cmap='RdBu', vmin=-vlim, vmax=vlim)
    ax.set_title(f"vs Clean: max|diff|={abs(diff).max():.4f}", fontsize=9)
    ax.axis('off')

# Row 2: z-profile (sum over x,y per slice)
for col, (label, rec, fm) in enumerate(all_recs):
    ax = axes[2, col]
    z_prof = np.array([rec[:, :, i].sum() for i in range(rec.shape[2])])
    z_prof_phantom = np.array([phantom[:, :, i].sum() for i in range(phantom.shape[2])])
    ax.plot(z_prof / max(z_prof.max(), 1e-12), 'b-', label='Recon')
    ax.plot(z_prof_phantom / max(z_prof_phantom.max(), 1e-12), 'g--', label='Phantom')
    z_c, _, _ = find_sphere_center_z(rec, voxel_size)
    ax.axvline(mid_z, color='r', linestyle=':', linewidth=0.5)
    ax.set_title(f"z-profile (centroid={z_c:+.2f}mm)", fontsize=9)
    ax.legend(fontsize=7)

axes[0, 0].set_ylabel("Center slice", fontsize=11)
axes[1, 0].set_ylabel("Diff from Clean", fontsize=11)
axes[2, 0].set_ylabel("Z-profile", fontsize=11)

plt.suptitle("Phantom DICOM Simulation: Geometric Shift Analysis (test01)", fontsize=14)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "phantom_dicom_sim.png")
plt.savefig(out_path, dpi=150)
print(f"\nSaved -> {out_path}")

print("\nDone.", flush=True)
