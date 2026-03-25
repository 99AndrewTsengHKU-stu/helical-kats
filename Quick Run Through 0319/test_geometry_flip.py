"""
Phase 6C: Geometry-level flip vs data-level flip.

Instead of flipping sinogram rows (sino[:, ::-1, :]), handle the detector
z-flip by negating psize_row in the geometry. This way the filter pipeline
sees unflipped data with correct row_coords alignment.

Tests on phantom:
  A) Clean (baseline)
  B) Data-level flip: sino[:, ::-1, :] + original psize_row (current L067 approach)
  C) Geometry-level flip: no sino flip + negative psize_row

Then tests on L067 DICOM data:
  D) Current L067 pipeline (data-level flip)
  E) Geometry-level flip on L067
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
from pykatsevich.filter import filter_katsevich, sino_weight_td
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
    z_profile = np.array([vol[:, :, i].sum() for i in range(vol.shape[2])])
    z_coords = (np.arange(vol.shape[2]) - vol.shape[2] / 2.0 + 0.5) * voxel_size_z
    total = z_profile.sum()
    center = np.sum(z_profile * z_coords) / total if abs(total) > 1e-12 else 0.0
    return center


def z_cull_indices(views, SOD, SDD, det_rows, pixel_size, vol_z_min, vol_z_max,
                   margin_turns=1.0, pitch_mm_per_angle=None):
    source_z = views[:, 2]
    cone_half_z = 0.5 * det_rows * pixel_size * (SOD / SDD)
    if pitch_mm_per_angle is not None and len(views) > 1:
        angle_step = abs(np.mean(np.diff(np.arctan2(views[:10, 1], views[:10, 0]))))
        projs_per_turn = 2 * np.pi / max(angle_step, 1e-12)
        margin_z = margin_turns * projs_per_turn * abs(pitch_mm_per_angle)
    else:
        margin_z = cone_half_z
    z_lo = source_z - cone_half_z - margin_z
    z_hi = source_z + cone_half_z + margin_z
    mask = (z_hi >= vol_z_min) & (z_lo <= vol_z_max)
    return np.where(mask)[0]


# ══════════════════════════════════════════════════════════════════════
# PART 1: PHANTOM TESTS
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("PART 1: PHANTOM TESTS (test01, 256x256x256)")
print("=" * 60, flush=True)

yaml_path = os.path.join(TEST_DIR, "test01.yaml")
with open(yaml_path, "r") as f:
    settings = yaml.safe_load(f)

ps = settings['phantom']
geom = settings['geometry']

phantom = phantom_objects_3d(
    ps['rows'], ps['columns'], ps['slices'],
    voxel_size=ps['voxel_size'], objects_list=ps['objects'],
)
voxel_size = ps['voxel_size']

# Forward project (clean)
sinogram, vol_geom, proj_geom = project(phantom, voxel_size, geom)
sinogram_swapped = np.asarray(np.swapaxes(sinogram, 0, 1), order='C')

# Reconstruct angles (reproduce from project())
s_len = geom['helix']['angles_range']
s_min = -s_len * 0.5
projs_per_turn = geom['helix']['angles_count'] / s_len * 2 * np.pi
delta_s = 2 * np.pi / projs_per_turn
angles_clean = s_min + delta_s * (np.arange(geom['helix']['angles_count'], dtype=np.float32) + 0.5)
stride_mm = geom['helix']['pitch_mm_rad'] * delta_s
psize_row = geom['detector']['detector psize']

# ── A: Clean baseline ──
print(f"\nTEST A: Clean (no flip)", flush=True)
conf_a = create_configuration(geom, vol_geom, geom.get('options', {}))
t0 = time()
filt_a = filter_katsevich(sinogram_swapped, conf_a,
    {"Diff": {"Print time": False}, "FwdRebin": {"Print time": False}, "BackRebin": {"Print time": False}})
td_a = sino_weight_td(filt_a, conf_a, False)
rec_a = backproject_cupy(td_a, conf_a, vol_geom, proj_geom, tqdm_bar=False)
fm_a = flicker_metric(rec_a)
print(f"  Done in {time()-t0:.1f}s, flicker={fm_a.mean():.6f}, range=[{rec_a.min():.4f}, {rec_a.max():.4f}]")

# ── B: Data-level flip (current L067 approach) ──
print(f"\nTEST B: Data-level flip (sino[:,::-1,:])", flush=True)
sino_flipped = sinogram_swapped[:, ::-1, :].copy()
conf_b = create_configuration(geom, vol_geom, geom.get('options', {}))
t0 = time()
filt_b = filter_katsevich(sino_flipped.astype(np.float32), conf_b,
    {"Diff": {"Print time": False}, "FwdRebin": {"Print time": False}, "BackRebin": {"Print time": False}})
td_b = sino_weight_td(filt_b, conf_b, False)
rec_b = backproject_cupy(td_b, conf_b, vol_geom, proj_geom, tqdm_bar=False)
fm_b = flicker_metric(rec_b)
print(f"  Done in {time()-t0:.1f}s, flicker={fm_b.mean():.6f}, range=[{rec_b.min():.4f}, {rec_b.max():.4f}]")

# ── C: Geometry-level flip (negate psize_row in views + BP kernel) ──
print(f"\nTEST C: Geometry-level flip (negate psize_row)", flush=True)

# Build views with negative psize_row
views_neg = astra_helical_views(
    geom["SOD"], geom["SDD"],
    geom['detector']["detector psize"],
    angles_clean, stride_mm,
    pixel_size_row=-psize_row,  # NEGATIVE
)
proj_geom_neg = astra.create_proj_geom(
    "cone_vec", geom['detector']["detector rows"],
    geom['detector']["detector cols"], views_neg,
)

# Use FLIPPED sinogram (simulating DICOM raw data where rows are "wrong" direction)
# but filter with UNFLIPPED data (geometry handles the flip)
# Wait - the point is: if DICOM data has reversed detector rows, instead of flipping
# the data, we tell the geometry that psize_row is negative.
# So we pass the FLIPPED sinogram (raw DICOM) WITHOUT un-flipping, and negate psize_row.
# Actually no - let's think again:
#   - Clean sinogram: row 0 = bottom of detector (negative z)
#   - "DICOM" sinogram: row 0 = top of detector (positive z) → flip of clean
#   - Data-level fix: flip DICOM back → clean order, but filter sees correct order
#   - Geometry-level fix: keep DICOM as-is, but negate psize_row so BP maps correctly
#
# For phantom: sino_flipped simulates "DICOM" (reversed rows)
# Geometry-level: feed sino_flipped to filter (unflipped), but use neg psize_row in BP

conf_c = create_configuration(geom, vol_geom, geom.get('options', {}))
t0 = time()
# Filter sees the flipped sinogram (DICOM order) — row_coords still [-h/2, +h/2]
# This means filter maps physical coordinates wrong... unless we also fix row_coords.
# Actually let's try two sub-tests:
#   C1: geometry flip only in BP (filter still gets wrong row order)
#   C2: geometry flip in BP + flip row_coords in conf (filter gets corrected coords)

# C1: just negate psize_row in BP
filt_c1 = filter_katsevich(sino_flipped.astype(np.float32), conf_c,
    {"Diff": {"Print time": False}, "FwdRebin": {"Print time": False}, "BackRebin": {"Print time": False}})
td_c1 = sino_weight_td(filt_c1, conf_c, False)
rec_c1 = backproject_cupy(td_c1, conf_c, vol_geom, proj_geom_neg, tqdm_bar=False)
fm_c1 = flicker_metric(rec_c1)
print(f"  C1 (neg psize_row BP only): flicker={fm_c1.mean():.6f}, range=[{rec_c1.min():.4f}, {rec_c1.max():.4f}]")

# C2: negate psize_row in BP + flip row_coords & TD bounds in conf
conf_c2 = create_configuration(geom, vol_geom, geom.get('options', {}))
conf_c2['row_coords'] = conf_c2['row_coords'][::-1].copy()
old_mins = conf_c2['proj_row_mins'].copy()
old_maxs = conf_c2['proj_row_maxs'].copy()
conf_c2['proj_row_mins'] = -old_maxs[::-1]
conf_c2['proj_row_maxs'] = -old_mins[::-1]

filt_c2 = filter_katsevich(sino_flipped.astype(np.float32), conf_c2,
    {"Diff": {"Print time": False}, "FwdRebin": {"Print time": False}, "BackRebin": {"Print time": False}})
td_c2 = sino_weight_td(filt_c2, conf_c2, False)
rec_c2 = backproject_cupy(td_c2, conf_c2, vol_geom, proj_geom_neg, tqdm_bar=False)
fm_c2 = flicker_metric(rec_c2)
elapsed = time() - t0
print(f"  C2 (neg psize + fix row_coords): flicker={fm_c2.mean():.6f}, range=[{rec_c2.min():.4f}, {rec_c2.max():.4f}]")
print(f"  Total C time: {elapsed:.1f}s")

# ── D: No flip at all, just use clean sinogram with negative psize_row ──
# This tests: what if we DON'T need flip at all, and just negate psize_row?
print(f"\nTEST D: Clean sino + negative psize_row (double-flip = identity?)", flush=True)
conf_d = create_configuration(geom, vol_geom, geom.get('options', {}))
t0 = time()
filt_d = filter_katsevich(sinogram_swapped.astype(np.float32), conf_d,
    {"Diff": {"Print time": False}, "FwdRebin": {"Print time": False}, "BackRebin": {"Print time": False}})
td_d = sino_weight_td(filt_d, conf_d, False)
rec_d = backproject_cupy(td_d, conf_d, vol_geom, proj_geom_neg, tqdm_bar=False)
fm_d = flicker_metric(rec_d)
print(f"  Done in {time()-t0:.1f}s, flicker={fm_d.mean():.6f}, range=[{rec_d.min():.4f}, {rec_d.max():.4f}]")

# ══════════════════════════════════════════════════════════════════════
# PHANTOM SUMMARY
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PHANTOM SUMMARY")
print(f"{'='*60}")
print(f"  {'Test':<50} {'Flicker':>10} {'z-cent':>10} {'Range':>20}")
print(f"  {'-'*90}")
for label, rec, fm in [
    ("A: Clean (baseline)", rec_a, fm_a),
    ("B: Data-flip sino[:,::-1,:]", rec_b, fm_b),
    ("C1: Geom-flip (neg psize_row BP only)", rec_c1, fm_c1),
    ("C2: Geom-flip (neg psize + fix row_coords)", rec_c2, fm_c2),
    ("D: Clean + neg psize_row", rec_d, fm_d),
]:
    zc = find_sphere_center_z(rec, voxel_size)
    print(f"  {label:<50} {fm.mean():>10.6f} {zc:>+10.4f} [{rec.min():.4f}, {rec.max():.4f}]")

# ══════════════════════════════════════════════════════════════════════
# PART 2: L067 DICOM TEST
# ══════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*60}")
print("PART 2: L067 DICOM TEST")
print("=" * 60, flush=True)

from pykatsevich import load_dicom_projections

DICOM_DIR = r"D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD"
ROWS_L = 512
COLS_L = 512
SLICES_L = 560
VOXEL_SIZE_XY = 0.664
VOXEL_SIZE_Z = 0.8
TEST_START = 276
TEST_END = 286
TEST_SLICES = TEST_END - TEST_START

print("Loading DICOM...", flush=True)
t0 = time()
sino_raw, meta = load_dicom_projections(DICOM_DIR)
print(f"Loaded {sino_raw.shape[0]} projections in {time()-t0:.1f}s", flush=True)

angles_orig = meta["angles_rad"].copy()
z_shift = meta["table_positions_mm"] - meta["table_positions_mm"].mean()
scan_geom = copy.deepcopy(meta["scan_geometry"])
pitch_abs = float(abs(meta["pitch_mm_per_rad_signed"]))
scan_geom["helix"]["pitch_mm_rad"] = pitch_abs

psize_cols_l = scan_geom["detector"].get("detector psize cols", scan_geom["detector"]["detector psize"])
psize_rows_l = scan_geom["detector"].get("detector psize rows", scan_geom["detector"]["detector psize"])
det_rows_l = scan_geom["detector"]["detector rows"]
psize_l = scan_geom["detector"]["detector psize"]

total_half_z = SLICES_L * VOXEL_SIZE_Z * 0.5
chunk_z_min = -total_half_z + TEST_START * VOXEL_SIZE_Z
chunk_z_max = -total_half_z + TEST_END * VOXEL_SIZE_Z

# ── D: Current L067 pipeline (data-level flip) ──
print(f"\nTEST D-L067: Current pipeline (flip_rows + negate + offset)", flush=True)

angles_d = -angles_orig - np.pi / 2
sino_d = sino_raw[:, ::-1, :].copy()

scan_geom_d = copy.deepcopy(scan_geom)
scan_geom_d["helix"]["angles_range"] = float(abs(angles_d[-1] - angles_d[0]))

views_d = astra_helical_views(
    scan_geom["SOD"], scan_geom["SDD"], psize_l,
    angles_d, meta["pitch_mm_per_angle"],
    vertical_shifts=z_shift,
    pixel_size_col=psize_cols_l, pixel_size_row=psize_rows_l,
)

keep_d = z_cull_indices(
    views_d, scan_geom["SOD"], scan_geom["SDD"],
    det_rows_l, psize_l, chunk_z_min, chunk_z_max,
    margin_turns=1.0, pitch_mm_per_angle=meta["pitch_mm_per_angle"],
)

sino_chunk_d = sino_d[keep_d].copy()
angles_chunk_d = angles_d[keep_d]
views_chunk_d = views_d[keep_d]

sg_d = copy.deepcopy(scan_geom_d)
sg_d["helix"]["angles_range"] = float(abs(angles_chunk_d[-1] - angles_chunk_d[0]))
sg_d["helix"]["angles_count"] = len(angles_chunk_d)

half_x = COLS_L * VOXEL_SIZE_XY * 0.5
half_y = ROWS_L * VOXEL_SIZE_XY * 0.5
vol_geom_l = astra.create_vol_geom(
    ROWS_L, COLS_L, TEST_SLICES,
    -half_x, half_x, -half_y, half_y, chunk_z_min, chunk_z_max,
)
proj_geom_d = astra.create_proj_geom("cone_vec", det_rows_l,
    scan_geom["detector"]["detector cols"], views_chunk_d)

conf_d = create_configuration(sg_d, vol_geom_l)
conf_d['source_pos'] = angles_chunk_d.astype(np.float32)
conf_d['delta_s'] = float(np.mean(np.diff(angles_chunk_d)))

t0 = time()
filt_d = filter_katsevich(sino_chunk_d.astype(np.float32), conf_d,
    {"Diff": {"Print time": False}, "FwdRebin": {"Print time": False}, "BackRebin": {"Print time": False}})
td_d = sino_weight_td(filt_d, conf_d, False)
rec_d_l067 = backproject_cupy(td_d, conf_d, vol_geom_l, proj_geom_d, tqdm_bar=False)
fm_d_l067 = flicker_metric(rec_d_l067)
print(f"  Done in {time()-t0:.1f}s, flicker={fm_d_l067.mean():.6f}, range=[{rec_d_l067.min():.4f}, {rec_d_l067.max():.4f}]")

# ── E: Geometry-level flip on L067 ──
# NO sino flip. Negate psize_row in views. Filter sees original row order.
print(f"\nTEST E-L067: Geometry-level flip (no sino flip, neg psize_row)", flush=True)

angles_e = -angles_orig - np.pi / 2
sino_e = sino_raw.copy()  # NO FLIP

views_e = astra_helical_views(
    scan_geom["SOD"], scan_geom["SDD"], psize_l,
    angles_e, meta["pitch_mm_per_angle"],
    vertical_shifts=z_shift,
    pixel_size_col=psize_cols_l, pixel_size_row=-psize_rows_l,  # NEGATIVE
)

scan_geom_e = copy.deepcopy(scan_geom)
scan_geom_e["helix"]["angles_range"] = float(abs(angles_e[-1] - angles_e[0]))

keep_e = z_cull_indices(
    views_e, scan_geom["SOD"], scan_geom["SDD"],
    det_rows_l, psize_l, chunk_z_min, chunk_z_max,
    margin_turns=1.0, pitch_mm_per_angle=meta["pitch_mm_per_angle"],
)

sino_chunk_e = sino_e[keep_e].copy()
angles_chunk_e = angles_e[keep_e]
views_chunk_e = views_e[keep_e]

sg_e = copy.deepcopy(scan_geom_e)
sg_e["helix"]["angles_range"] = float(abs(angles_chunk_e[-1] - angles_chunk_e[0]))
sg_e["helix"]["angles_count"] = len(angles_chunk_e)

proj_geom_e = astra.create_proj_geom("cone_vec", det_rows_l,
    scan_geom["detector"]["detector cols"], views_chunk_e)

conf_e = create_configuration(sg_e, vol_geom_l)
conf_e['source_pos'] = angles_chunk_e.astype(np.float32)
conf_e['delta_s'] = float(np.mean(np.diff(angles_chunk_e)))

t0 = time()
filt_e = filter_katsevich(sino_chunk_e.astype(np.float32), conf_e,
    {"Diff": {"Print time": False}, "FwdRebin": {"Print time": False}, "BackRebin": {"Print time": False}})
td_e = sino_weight_td(filt_e, conf_e, False)
rec_e_l067 = backproject_cupy(td_e, conf_e, vol_geom_l, proj_geom_e, tqdm_bar=False)
fm_e_l067 = flicker_metric(rec_e_l067)
print(f"  Done in {time()-t0:.1f}s, flicker={fm_e_l067.mean():.6f}, range=[{rec_e_l067.min():.4f}, {rec_e_l067.max():.4f}]")

# ── F: Geometry-level flip + fix row_coords/TD bounds ──
print(f"\nTEST F-L067: Geom-flip + fix row_coords & TD bounds", flush=True)

conf_f = create_configuration(sg_e, vol_geom_l)
conf_f['source_pos'] = angles_chunk_e.astype(np.float32)
conf_f['delta_s'] = float(np.mean(np.diff(angles_chunk_e)))
# Flip row_coords and TD bounds to match unflipped sinogram
conf_f['row_coords'] = conf_f['row_coords'][::-1].copy()
old_mins_f = conf_f['proj_row_mins'].copy()
old_maxs_f = conf_f['proj_row_maxs'].copy()
conf_f['proj_row_mins'] = -old_maxs_f[::-1]
conf_f['proj_row_maxs'] = -old_mins_f[::-1]

t0 = time()
filt_f = filter_katsevich(sino_chunk_e.astype(np.float32), conf_f,
    {"Diff": {"Print time": False}, "FwdRebin": {"Print time": False}, "BackRebin": {"Print time": False}})
td_f = sino_weight_td(filt_f, conf_f, False)
rec_f_l067 = backproject_cupy(td_f, conf_f, vol_geom_l, proj_geom_e, tqdm_bar=False)
fm_f_l067 = flicker_metric(rec_f_l067)
print(f"  Done in {time()-t0:.1f}s, flicker={fm_f_l067.mean():.6f}, range=[{rec_f_l067.min():.4f}, {rec_f_l067.max():.4f}]")

# ══════════════════════════════════════════════════════════════════════
# L067 SUMMARY
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("L067 SUMMARY")
print(f"{'='*60}")
print(f"  {'Test':<55} {'Flicker':>10} {'Range':>25}")
print(f"  {'-'*90}")
for label, rec, fm in [
    ("D: Current pipeline (data-flip)", rec_d_l067, fm_d_l067),
    ("E: Geom-flip (neg psize_row, no sino flip)", rec_e_l067, fm_e_l067),
    ("F: Geom-flip + fix row_coords/TD", rec_f_l067, fm_f_l067),
]:
    print(f"  {label:<55} {fm.mean():>10.6f} [{rec.min():.4f}, {rec.max():.4f}]")
print(f"  {'GT L067':<55} {'~0.040':>10}")

# ══════════════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

mid = TEST_SLICES // 2
for col, (label, rec, fm) in enumerate([
    ("D: Current (data-flip)", rec_d_l067, fm_d_l067),
    ("E: Geom-flip only", rec_e_l067, fm_e_l067),
    ("F: Geom-flip + fix coords", rec_f_l067, fm_f_l067),
]):
    ax = axes[0, col]
    ax.imshow(rec[:, :, mid], cmap='gray')
    ax.set_title(f"{label}\nfl={fm.mean():.4f}", fontsize=10)
    ax.axis('off')

    ax = axes[1, col]
    if mid > 0:
        d = rec[:, :, mid].astype(np.float64) - rec[:, :, mid-1].astype(np.float64)
        vlim = max(abs(d).max(), 1e-6)
        ax.imshow(d, cmap='RdBu', vmin=-vlim, vmax=vlim)
        ax.set_title(f"Slice diff (fl={fm[mid-1]:.4f})", fontsize=9)
    ax.axis('off')

plt.suptitle("L067: Data-level flip vs Geometry-level flip", fontsize=14)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "geometry_flip_comparison.png")
plt.savefig(out_path, dpi=150)
print(f"\nSaved -> {out_path}")

print("\nDone.", flush=True)
