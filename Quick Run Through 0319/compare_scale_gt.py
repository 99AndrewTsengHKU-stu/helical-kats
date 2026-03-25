"""
Scale comparison: Katsevich reconstruction vs Ground Truth.

Reconstructs the same chunk as test_perfect_helix.py (slices 270-290),
loads the matching GT slices from full_1mm, converts recon to HU,
and juxtaposes them with intensity statistics.
"""
import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Quick Run Through 0304"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from backproject_safe import ensure_astra_cuda_init
ensure_astra_cuda_init()

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from pathlib import Path
from time import time
import astra
import pydicom

from pykatsevich import load_dicom_projections
from pykatsevich.geometry import astra_helical_views
from pykatsevich.initialize import create_configuration
from pykatsevich.filter import filter_katsevich, sino_weight_td
from backproject_cupy import backproject_cupy

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Settings ──────────────────────────────────────────────────────────
DICOM_DIR = r"D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD"
GT_DIR = Path(r"D:\AAPM-Data\L067\L067\full_1mm")
ROWS = 512
COLS = 512
SLICES = 560
VOXEL_SIZE_XY = 0.664
VOXEL_SIZE_Z = 0.8

TEST_START = 270
TEST_END = 290
TEST_SLICES = TEST_END - TEST_START


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


def load_gt_slices(gt_dir: Path):
    """Load all GT slices, return (volume_HU, z_positions_mm, pixel_spacing)."""
    files = sorted(gt_dir.glob("*.IMA"))
    if not files:
        files = sorted(gt_dir.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"No IMA/dcm files in {gt_dir}")

    ds0 = pydicom.dcmread(str(files[0]))
    slope = float(getattr(ds0, "RescaleSlope", 1.0))
    intercept = float(getattr(ds0, "RescaleIntercept", 0.0))
    pix_spacing = [float(x) for x in ds0.PixelSpacing]

    n = len(files)
    rows, cols = ds0.Rows, ds0.Columns
    vol = np.empty((rows, cols, n), dtype=np.float32)
    z_pos = np.empty(n, dtype=np.float64)

    for i, f in enumerate(files):
        ds = pydicom.dcmread(str(f))
        vol[:, :, i] = ds.pixel_array.astype(np.float32) * slope + intercept
        z_pos[i] = float(getattr(ds, "SliceLocation", i))

    return vol, z_pos, pix_spacing


def mu_to_hu(vol_mu, mu_water=0.019):
    return (vol_mu - mu_water) / mu_water * 1000.0


# ══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════
print("Loading DICOM projections...", flush=True)
t0 = time()
sino, meta = load_dicom_projections(DICOM_DIR)
print(f"Loaded {sino.shape[0]} projections in {time()-t0:.1f}s", flush=True)

angles_full = -meta["angles_rad"].copy() - np.pi / 2
z_shift_dicom = meta["table_positions_mm"] - meta["table_positions_mm"].mean()
sino_work = sino[:, ::-1, :].copy()

scan_geom_full = copy.deepcopy(meta["scan_geometry"])
pitch_abs = float(abs(meta["pitch_mm_per_rad_signed"]))
scan_geom_full["helix"]["pitch_mm_rad"] = pitch_abs
scan_geom_full["helix"]["angles_range"] = float(abs(angles_full[-1] - angles_full[0]))

psize_cols = scan_geom_full["detector"].get("detector psize cols", scan_geom_full["detector"]["detector psize"])
psize_rows = scan_geom_full["detector"].get("detector psize rows", scan_geom_full["detector"]["detector psize"])
det_rows_n = scan_geom_full["detector"]["detector rows"]
psize = scan_geom_full["detector"]["detector psize"]

# ══════════════════════════════════════════════════════════════════════
# RECONSTRUCT (using DICOM table positions, baseline)
# ══════════════════════════════════════════════════════════════════════
views_dicom = astra_helical_views(
    scan_geom_full["SOD"], scan_geom_full["SDD"],
    scan_geom_full["detector"]["detector psize"],
    angles_full, meta["pitch_mm_per_angle"],
    vertical_shifts=z_shift_dicom,
    pixel_size_col=psize_cols, pixel_size_row=psize_rows,
)

total_half_z = SLICES * VOXEL_SIZE_Z * 0.5
chunk_z_min = -total_half_z + TEST_START * VOXEL_SIZE_Z
chunk_z_max = -total_half_z + TEST_END * VOXEL_SIZE_Z

keep_idx = z_cull_indices(
    views_dicom, scan_geom_full["SOD"], scan_geom_full["SDD"],
    det_rows_n, psize, chunk_z_min, chunk_z_max,
    margin_turns=1.0, pitch_mm_per_angle=meta["pitch_mm_per_angle"],
)

sino_chunk = sino_work[keep_idx].copy()
angles_chunk = angles_full[keep_idx].copy()
views_chunk = views_dicom[keep_idx]

scan_geom_chunk = copy.deepcopy(scan_geom_full)
scan_geom_chunk["helix"]["angles_range"] = float(abs(angles_chunk[-1] - angles_chunk[0]))
scan_geom_chunk["helix"]["angles_count"] = len(angles_chunk)

half_x = COLS * VOXEL_SIZE_XY * 0.5
half_y = ROWS * VOXEL_SIZE_XY * 0.5
vol_geom = astra.create_vol_geom(
    ROWS, COLS, TEST_SLICES,
    -half_x, half_x, -half_y, half_y, chunk_z_min, chunk_z_max,
)
proj_geom = astra.create_proj_geom("cone_vec", det_rows_n,
    scan_geom_chunk["detector"]["detector cols"], views_chunk)

conf = create_configuration(scan_geom_chunk, vol_geom)
conf['source_pos'] = angles_chunk.astype(np.float32)
conf['delta_s'] = float(np.mean(np.diff(angles_chunk)))

print(f"Reconstructing slices [{TEST_START}:{TEST_END}]...", flush=True)
t0 = time()
sino_f32 = np.asarray(sino_chunk, dtype=np.float32, order="C")
filtered = filter_katsevich(
    sino_f32, conf,
    {"Diff": {"Print time": False}, "FwdRebin": {"Print time": False},
     "BackRebin": {"Print time": False}},
)
sino_td = sino_weight_td(filtered, conf, False)
rec = backproject_cupy(sino_td, conf, vol_geom, proj_geom, tqdm_bar=False)
print(f"Done in {time()-t0:.1f}s, shape={rec.shape}", flush=True)

# ══════════════════════════════════════════════════════════════════════
# LOAD GT & MATCH SLICES
# ══════════════════════════════════════════════════════════════════════
print(f"\nLoading GT from {GT_DIR}...", flush=True)
gt_vol, gt_z, gt_pix_spacing = load_gt_slices(GT_DIR)
print(f"GT volume: shape={gt_vol.shape}, z=[{gt_z[0]:.1f}, {gt_z[-1]:.1f}] mm")
print(f"GT pixel spacing: {gt_pix_spacing} mm")
print(f"GT HU range: [{gt_vol.min():.0f}, {gt_vol.max():.0f}]")

# Recon z-positions for each slice
rec_z = np.linspace(chunk_z_min, chunk_z_max, TEST_SLICES, endpoint=False)
rec_z += (chunk_z_max - chunk_z_min) / TEST_SLICES * 0.5  # center of each voxel

# GT z is in patient coordinates; recon z is centered at 0.
# We need to figure out the offset. Use the DICOM table_positions to map.
# The recon volume center is at z=0 which corresponds to the mean table position.
dicom_table_mean = meta["table_positions_mm"].mean()
# recon z=0 => patient z = dicom_table_mean (approximately)
# So recon_z in patient coords:
rec_z_patient = rec_z + dicom_table_mean

print(f"\nRecon z range (centered): [{rec_z[0]:.1f}, {rec_z[-1]:.1f}] mm")
print(f"Recon z range (patient):  [{rec_z_patient[0]:.1f}, {rec_z_patient[-1]:.1f}] mm")
print(f"GT z range:               [{gt_z[0]:.1f}, {gt_z[-1]:.1f}] mm")

# Find closest GT slice for each recon slice
gt_match_idx = []
for rz in rec_z_patient:
    idx = np.argmin(np.abs(gt_z - rz))
    gt_match_idx.append(idx)
gt_match_idx = np.array(gt_match_idx)

print(f"Matched GT slice indices: [{gt_match_idx[0]}..{gt_match_idx[-1]}]")
print(f"Matched GT z: [{gt_z[gt_match_idx[0]]:.1f}, {gt_z[gt_match_idx[-1]]:.1f}] mm")
z_errors = np.array([abs(rec_z_patient[i] - gt_z[gt_match_idx[i]]) for i in range(TEST_SLICES)])
print(f"Z matching error: max={z_errors.max():.2f}mm, mean={z_errors.mean():.2f}mm")

# ══════════════════════════════════════════════════════════════════════
# CONVERT RECON TO HU
# ══════════════════════════════════════════════════════════════════════
print(f"\nRecon raw value range: [{rec.min():.6f}, {rec.max():.6f}]")

# Auto-calibrate: find mu_water from central soft tissue region
mid = rec[:, :, TEST_SLICES // 2]
r0, r1 = int(0.3 * mid.shape[0]), int(0.7 * mid.shape[0])
c0, c1 = int(0.3 * mid.shape[1]), int(0.7 * mid.shape[1])
center_vals = mid[r0:r1, c0:c1]
positive_vals = center_vals[center_vals > 0]
if len(positive_vals) > 0:
    mu_water_auto = float(np.median(positive_vals))
else:
    mu_water_auto = 0.019
print(f"Auto-calibrated mu_water = {mu_water_auto:.6f} mm^-1")

# Also try theoretical
mu_water_theory = 0.019
rec_hu_auto = mu_to_hu(rec, mu_water_auto)
rec_hu_theory = mu_to_hu(rec, mu_water_theory)

print(f"Recon HU (auto mu_water={mu_water_auto:.5f}): [{rec_hu_auto.min():.0f}, {rec_hu_auto.max():.0f}]")
print(f"Recon HU (theory mu_water=0.019):             [{rec_hu_theory.min():.0f}, {rec_hu_theory.max():.0f}]")

# ══════════════════════════════════════════════════════════════════════
# PLOT: JUXTAPOSITION
# ══════════════════════════════════════════════════════════════════════
# Pick 5 evenly spaced slices from the chunk
n_show = 5
show_idx = np.linspace(0, TEST_SLICES - 1, n_show, dtype=int)

fig, axes = plt.subplots(4, n_show, figsize=(4 * n_show, 16))

# Row 0: Recon (auto HU), soft tissue window
# Row 1: GT, soft tissue window
# Row 2: Recon (auto HU), auto window
# Row 3: GT, auto window

win_lo, win_hi = -200, 300  # soft tissue

for col, si in enumerate(show_idx):
    gi = gt_match_idx[si]
    r_img = rec_hu_auto[:, :, si]
    g_img = gt_vol[:, :, gi]

    # Row 0: Recon soft tissue
    axes[0, col].imshow(r_img, cmap='gray', vmin=win_lo, vmax=win_hi)
    axes[0, col].set_title(f"Recon sl{TEST_START+si}\n[{r_img.mean():.0f} HU]", fontsize=9)
    axes[0, col].axis('off')

    # Row 1: GT soft tissue
    axes[1, col].imshow(g_img, cmap='gray', vmin=win_lo, vmax=win_hi)
    axes[1, col].set_title(f"GT sl{gi}\n[{g_img.mean():.0f} HU]", fontsize=9)
    axes[1, col].axis('off')

    # Row 2: Recon auto window
    rv0, rv1 = np.percentile(r_img, [1, 99])
    axes[2, col].imshow(r_img, cmap='gray', vmin=rv0, vmax=rv1)
    axes[2, col].set_title(f"Recon auto [{rv0:.0f},{rv1:.0f}]", fontsize=9)
    axes[2, col].axis('off')

    # Row 3: GT auto window
    gv0, gv1 = np.percentile(g_img, [1, 99])
    axes[3, col].imshow(g_img, cmap='gray', vmin=gv0, vmax=gv1)
    axes[3, col].set_title(f"GT auto [{gv0:.0f},{gv1:.0f}]", fontsize=9)
    axes[3, col].axis('off')

# Row labels
for ax, label in zip(axes[:, 0], [
    "Recon (soft tissue)", "GT (soft tissue)", "Recon (auto)", "GT (auto)"
]):
    ax.set_ylabel(label, fontsize=11, rotation=90, labelpad=10)

plt.suptitle(
    f"Scale Comparison: Katsevich vs GT\n"
    f"mu_water(auto)={mu_water_auto:.5f} | "
    f"Recon HU: [{rec_hu_auto.min():.0f}, {rec_hu_auto.max():.0f}] | "
    f"GT HU: [{gt_vol.min():.0f}, {gt_vol.max():.0f}]",
    fontsize=12,
)
plt.tight_layout()
out1 = os.path.join(OUT_DIR, "scale_comparison_gt.png")
plt.savefig(out1, dpi=150)
plt.close(fig)
print(f"\nSaved -> {out1}")

# ══════════════════════════════════════════════════════════════════════
# PLOT: HISTOGRAMS
# ══════════════════════════════════════════════════════════════════════
center_si = TEST_SLICES // 2
center_gi = gt_match_idx[center_si]
r_center = rec_hu_auto[:, :, center_si].ravel()
g_center = gt_vol[:, :, center_gi].ravel()

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Histogram comparison
bins = np.linspace(-1200, 2000, 200)
ax1.hist(r_center, bins=bins, alpha=0.6, label=f"Recon (mu_w={mu_water_auto:.5f})", density=True)
ax1.hist(g_center, bins=bins, alpha=0.6, label="GT", density=True)
ax1.set_xlabel("HU")
ax1.set_ylabel("Density")
ax1.set_title("HU Histogram: Recon vs GT (center slice)")
ax1.legend()
ax1.set_xlim(-1200, 2000)

# Profile comparison: horizontal line through center
mid_row = ROWS // 2
r_profile = rec_hu_auto[mid_row, :, center_si]
g_profile = gt_vol[mid_row, :, center_gi]
ax2.plot(r_profile, label="Recon", alpha=0.8)
ax2.plot(g_profile, label="GT", alpha=0.8)
ax2.set_xlabel("Column pixel")
ax2.set_ylabel("HU")
ax2.set_title(f"Horizontal profile at row {mid_row}")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out2 = os.path.join(OUT_DIR, "scale_histogram_gt.png")
plt.savefig(out2, dpi=150)
plt.close(fig2)
print(f"Saved -> {out2}")

# ══════════════════════════════════════════════════════════════════════
# NUMERICAL SUMMARY
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("SCALE SUMMARY (center slice)")
print(f"{'='*60}")
print(f"  {'Metric':<35} {'Recon':>12} {'GT':>12}")
print(f"  {'-'*60}")
print(f"  {'Mean HU':<35} {r_center.mean():>12.1f} {g_center.mean():>12.1f}")
print(f"  {'Std HU':<35} {r_center.std():>12.1f} {g_center.std():>12.1f}")
print(f"  {'Median HU':<35} {np.median(r_center):>12.1f} {np.median(g_center):>12.1f}")
print(f"  {'1st percentile':<35} {np.percentile(r_center,1):>12.1f} {np.percentile(g_center,1):>12.1f}")
print(f"  {'99th percentile':<35} {np.percentile(r_center,99):>12.1f} {np.percentile(g_center,99):>12.1f}")
print(f"  {'Min':<35} {r_center.min():>12.1f} {g_center.min():>12.1f}")
print(f"  {'Max':<35} {r_center.max():>12.1f} {g_center.max():>12.1f}")

# Body-only stats (exclude air background)
r_body = r_center[r_center > -500]
g_body = g_center[g_center > -500]
if len(r_body) > 100 and len(g_body) > 100:
    print(f"\n  Body region (HU > -500):")
    print(f"  {'Mean HU (body)':<35} {r_body.mean():>12.1f} {g_body.mean():>12.1f}")
    print(f"  {'Std HU (body)':<35} {r_body.std():>12.1f} {g_body.std():>12.1f}")
    scale_ratio = r_body.mean() / g_body.mean() if abs(g_body.mean()) > 1 else float('inf')
    print(f"  {'Scale ratio (recon/GT)':<35} {scale_ratio:>12.4f}")

print(f"\n  mu_water used: {mu_water_auto:.6f} mm^-1")
print(f"  If scale is off, try adjusting mu_water.")
print("\nDone.", flush=True)
