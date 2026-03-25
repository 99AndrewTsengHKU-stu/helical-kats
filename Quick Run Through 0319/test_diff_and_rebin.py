"""
Test D: Negate d_row term in Diff step
Test E: Skip rebin (Diff -> Hilbert -> TD -> Backproject, no FwdRebin/BackRebin)
Test F: Negate row_coords to test if flip_rows created a sign mismatch

Goal: Isolate whether flicker comes from Diff d_row sign, rebin interpolation,
      or row_coords direction.
"""
import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Quick Run Through 0304"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from backproject_safe import ensure_astra_cuda_init
ensure_astra_cuda_init()

import numpy as np
from matplotlib import pyplot as plt
from time import time
import astra

from pykatsevich import load_dicom_projections
from pykatsevich.geometry import astra_helical_views
from pykatsevich.initialize import create_configuration
from pykatsevich.filter import (
    differentiate, fw_height_rebinning, compute_hilbert_kernel,
    hilbert_conv, rev_rebin_vec, sino_weight_td
)
from backproject_cupy import backproject_cupy

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Settings ──────────────────────────────────────────────────────────
DICOM_DIR = r"D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD"
ROWS = 512
COLS = 512
SLICES = 560
VOXEL_SIZE_XY = 0.664
VOXEL_SIZE_Z = 0.8

TEST_START = 276
TEST_END = 286
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


def flicker_metric(vol):
    n_slices = vol.shape[2]
    metrics = []
    for i in range(n_slices - 1):
        s0 = vol[:, :, i]
        s1 = vol[:, :, i + 1]
        diff = np.mean(np.abs(s1 - s0))
        base = 0.5 * (np.mean(np.abs(s0)) + np.mean(np.abs(s1)))
        metrics.append(diff / max(base, 1e-12))
    return np.array(metrics)


def diff_negate_drow(sinogram, conf, tqdm_bar=False):
    """Same as differentiate() but with NEGATED d_row term."""
    output_array = np.zeros_like(sinogram)
    delta_s = conf['delta_s']
    pixel_height = conf['pixel_height']
    pixel_span = conf['pixel_span']
    dia = conf['scan_diameter']
    dia_sqr = dia ** 2

    col_coords = conf['col_coords'][:-2]
    row_coords = conf['row_coords'][:-2]

    row_col_prod = np.zeros_like(sinogram[0, :-1, :-1])
    row_col_prod += col_coords
    row_transposed = np.zeros_like(row_coords)
    row_transposed += row_coords
    row_transposed.shape = (len(row_coords), 1)
    row_col_prod *= row_transposed
    col_sqr = np.zeros_like(sinogram[0, :-1, :-1])
    col_sqr += col_coords ** 2
    row_sqr = np.zeros_like(sinogram[0, :-1, :-1])
    row_sqr += row_transposed ** 2

    for proj in range(sinogram.shape[0] - 1):
        d_proj = (sinogram[proj + 1, :-1, :-1] - sinogram[proj, :-1, :-1]
                  + sinogram[proj + 1, 1:, :-1] - sinogram[proj, 1:, :-1]
                  + sinogram[proj + 1, :-1, 1:] - sinogram[proj, :-1, 1:]
                  + sinogram[proj + 1, 1:, 1:] - sinogram[proj, 1:, 1:]) / (4 * delta_s)
        d_row = (sinogram[proj, 1:, :-1] - sinogram[proj, :-1, :-1]
                 + sinogram[proj, 1:, 1:] - sinogram[proj, :-1, 1:]
                 + sinogram[proj + 1, 1:, :-1] - sinogram[proj + 1, :-1, :-1]
                 + sinogram[proj + 1, 1:, 1:] - sinogram[proj + 1, :-1, 1:]) / (4 * pixel_height)
        d_col = (sinogram[proj, :-1, 1:] - sinogram[proj, :-1, :-1]
                 + sinogram[proj, 1:, 1:] - sinogram[proj, 1:, :-1]
                 + sinogram[proj + 1, :-1, 1:] - sinogram[proj + 1, :-1, :-1]
                 + sinogram[proj + 1, 1:, 1:] - sinogram[proj + 1, 1:, :-1]) / (4 * pixel_span)

        # NEGATED d_row term (the key change)
        output_array[proj, :-1, :-1] = d_proj + d_col * (col_sqr + dia_sqr) / dia + (-d_row) * row_col_prod / dia
        output_array[proj, :-1, :-1] *= dia / np.sqrt(col_sqr + dia_sqr + row_sqr)

    return output_array


def reconstruct_custom(sino_chunk, angles_chunk, views_chunk,
                       scan_geom_chunk, meta, chunk_z_min, chunk_z_max,
                       chunk_slices, conf_override=None,
                       negate_drow=False, skip_rebin=False,
                       td_smoothing=0.025):
    """Reconstruct with optional modifications to filter pipeline."""
    det_rows_n = scan_geom_chunk["detector"]["detector rows"]
    det_cols_n = scan_geom_chunk["detector"]["detector cols"]

    half_x = COLS * VOXEL_SIZE_XY * 0.5
    half_y = ROWS * VOXEL_SIZE_XY * 0.5
    vol_geom = astra.create_vol_geom(
        ROWS, COLS, chunk_slices,
        -half_x, half_x, -half_y, half_y, chunk_z_min, chunk_z_max,
    )
    proj_geom = astra.create_proj_geom("cone_vec", det_rows_n, det_cols_n, views_chunk)

    conf = create_configuration(scan_geom_chunk, vol_geom)
    conf['source_pos'] = angles_chunk.astype(np.float32)
    conf['delta_s'] = float(np.mean(np.diff(angles_chunk)))
    conf['T-D smoothing'] = td_smoothing

    if conf_override:
        conf.update(conf_override)

    sino_f32 = np.asarray(sino_chunk, dtype=np.float32, order="C")

    if negate_drow:
        # Test D: use modified Diff with negated d_row
        print("    [Diff with negated d_row]")
        diff_proj = diff_negate_drow(sino_f32, conf)
    else:
        diff_proj = differentiate(sino_f32, conf)

    if skip_rebin:
        # Test E: skip rebin, go Diff -> Hilbert -> TD -> BP
        print("    [Skipping FwdRebin/BackRebin]")
        hilbert_array = compute_hilbert_kernel(conf)
        filtered = hilbert_conv(diff_proj, hilbert_array, conf)
    else:
        fwd = fw_height_rebinning(diff_proj, conf)
        hilbert_array = compute_hilbert_kernel(conf)
        hilbert_out = hilbert_conv(fwd, hilbert_array, conf)
        filtered = rev_rebin_vec(hilbert_out, conf)

    sino_td = sino_weight_td(filtered, conf, False)
    rec = backproject_cupy(sino_td, conf, vol_geom, proj_geom, tqdm_bar=False)
    return rec


# ══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════
print("Loading DICOM...", flush=True)
t0 = time()
sino, meta = load_dicom_projections(DICOM_DIR)
print(f"Loaded {sino.shape[0]} projections in {time()-t0:.1f}s", flush=True)

# Transforms
angles_full = -meta["angles_rad"].copy() - np.pi / 2
z_shift_full = meta["table_positions_mm"] - meta["table_positions_mm"].mean()
sino_work = sino[:, ::-1, :].copy()  # flip_rows

scan_geom_full = copy.deepcopy(meta["scan_geometry"])
scan_geom_full["helix"]["pitch_mm_rad"] = float(abs(meta["pitch_mm_per_rad_signed"]))
scan_geom_full["helix"]["angles_range"] = float(abs(angles_full[-1] - angles_full[0]))

psize_cols = scan_geom_full["detector"].get("detector psize cols", scan_geom_full["detector"]["detector psize"])
psize_rows = scan_geom_full["detector"].get("detector psize rows", scan_geom_full["detector"]["detector psize"])

views_full = astra_helical_views(
    scan_geom_full["SOD"], scan_geom_full["SDD"],
    scan_geom_full["detector"]["detector psize"],
    angles_full, meta["pitch_mm_per_angle"],
    vertical_shifts=z_shift_full,
    pixel_size_col=psize_cols, pixel_size_row=psize_rows,
)

# Chunk setup
total_half_z = SLICES * VOXEL_SIZE_Z * 0.5
chunk_z_min = -total_half_z + TEST_START * VOXEL_SIZE_Z
chunk_z_max = -total_half_z + TEST_END * VOXEL_SIZE_Z
det_rows_n = scan_geom_full["detector"]["detector rows"]
det_cols_n = scan_geom_full["detector"]["detector cols"]
psize = scan_geom_full["detector"]["detector psize"]

keep_idx = z_cull_indices(
    views_full, scan_geom_full["SOD"], scan_geom_full["SDD"],
    det_rows_n, psize, chunk_z_min, chunk_z_max,
    margin_turns=1.0, pitch_mm_per_angle=meta["pitch_mm_per_angle"],
)

sino_chunk = sino_work[keep_idx].copy()
angles_chunk = angles_full[keep_idx].copy()
views_chunk = views_full[keep_idx]

scan_geom_chunk = copy.deepcopy(scan_geom_full)
scan_geom_chunk["helix"]["angles_range"] = float(abs(angles_chunk[-1] - angles_chunk[0]))
scan_geom_chunk["helix"]["angles_count"] = len(angles_chunk)

print(f"Chunk: {len(keep_idx)} projections, z=[{chunk_z_min:.1f}, {chunk_z_max:.1f}] mm", flush=True)


# ══════════════════════════════════════════════════════════════════════
# TEST D: NEGATE d_row IN DIFF STEP
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}", flush=True)
print("TEST D: Negate d_row term in Diff step", flush=True)
print(f"{'='*60}", flush=True)

t0 = time()
rec_D = reconstruct_custom(
    sino_chunk, angles_chunk, views_chunk,
    scan_geom_chunk, meta, chunk_z_min, chunk_z_max,
    TEST_SLICES, negate_drow=True,
)
elapsed = time() - t0
fm_D = flicker_metric(rec_D)
print(f"  Done in {elapsed:.1f}s, range=[{rec_D.min():.4f}, {rec_D.max():.4f}]", flush=True)
print(f"  Flicker mean: {fm_D.mean():.6f}, max: {fm_D.max():.6f}", flush=True)


# ══════════════════════════════════════════════════════════════════════
# TEST E: SKIP REBIN (Diff -> Hilbert -> TD -> BP)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}", flush=True)
print("TEST E: Skip FwdRebin/BackRebin", flush=True)
print(f"{'='*60}", flush=True)

t0 = time()
rec_E = reconstruct_custom(
    sino_chunk, angles_chunk, views_chunk,
    scan_geom_chunk, meta, chunk_z_min, chunk_z_max,
    TEST_SLICES, skip_rebin=True,
)
elapsed = time() - t0
fm_E = flicker_metric(rec_E)
print(f"  Done in {elapsed:.1f}s, range=[{rec_E.min():.4f}, {rec_E.max():.4f}]", flush=True)
print(f"  Flicker mean: {fm_E.mean():.6f}, max: {fm_E.max():.6f}", flush=True)


# ══════════════════════════════════════════════════════════════════════
# TEST F: NEGATE row_coords (flip_rows sign mismatch test)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}", flush=True)
print("TEST F: Negate row_coords (test flip_rows sign consistency)", flush=True)
print(f"{'='*60}", flush=True)

t0 = time()
# We override row_coords after create_configuration
rec_F = reconstruct_custom(
    sino_chunk, angles_chunk, views_chunk,
    scan_geom_chunk, meta, chunk_z_min, chunk_z_max,
    TEST_SLICES,
    conf_override=None,  # will be handled inline
)
# Actually need to do this properly - reconstruct with flipped row_coords
# Let's do it manually
half_x = COLS * VOXEL_SIZE_XY * 0.5
half_y = ROWS * VOXEL_SIZE_XY * 0.5
vol_geom_F = astra.create_vol_geom(
    ROWS, COLS, TEST_SLICES,
    -half_x, half_x, -half_y, half_y, chunk_z_min, chunk_z_max,
)
proj_geom_F = astra.create_proj_geom("cone_vec", det_rows_n, det_cols_n, views_chunk)
conf_F = create_configuration(scan_geom_chunk, vol_geom_F)
conf_F['source_pos'] = angles_chunk.astype(np.float32)
conf_F['delta_s'] = float(np.mean(np.diff(angles_chunk)))
conf_F['T-D smoothing'] = 0.025

# Negate row_coords
print(f"  Original row_coords: [{conf_F['row_coords'][0]:.4f}, ..., {conf_F['row_coords'][-2]:.4f}]", flush=True)
conf_F['row_coords'] = -conf_F['row_coords'][::-1]
# Also swap proj_row_mins and proj_row_maxs
old_mins = conf_F['proj_row_mins'].copy()
old_maxs = conf_F['proj_row_maxs'].copy()
conf_F['proj_row_mins'] = -old_maxs[::-1]
conf_F['proj_row_maxs'] = -old_mins[::-1]
print(f"  Negated row_coords: [{conf_F['row_coords'][0]:.4f}, ..., {conf_F['row_coords'][-2]:.4f}]", flush=True)

from pykatsevich.filter import filter_katsevich
sino_f32_F = np.asarray(sino_chunk, dtype=np.float32, order="C")
filtered_F = filter_katsevich(
    sino_f32_F, conf_F,
    {"Diff": {"Print time": False}, "FwdRebin": {"Print time": False},
     "BackRebin": {"Print time": False}},
)
sino_td_F = sino_weight_td(filtered_F, conf_F, False)
rec_F = backproject_cupy(sino_td_F, conf_F, vol_geom_F, proj_geom_F, tqdm_bar=False)
elapsed_F = time() - t0
fm_F = flicker_metric(rec_F)
print(f"  Done in {elapsed_F:.1f}s, range=[{rec_F.min():.4f}, {rec_F.max():.4f}]", flush=True)
print(f"  Flicker mean: {fm_F.mean():.6f}, max: {fm_F.max():.6f}", flush=True)


# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
# Previous baseline from Test 3
BASELINE_FLICKER = 0.229097

print(f"\n{'='*60}", flush=True)
print("SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
print(f"  {'Test':<40} {'Flicker':>10}", flush=True)
print(f"  {'-'*50}", flush=True)
print(f"  {'Baseline (from prev test)':<40} {BASELINE_FLICKER:10.6f}", flush=True)
print(f"  {'D: Negated d_row in Diff':<40} {fm_D.mean():10.6f}", flush=True)
print(f"  {'E: Skip rebin (Diff->Hilbert->TD->BP)':<40} {fm_E.mean():10.6f}", flush=True)
print(f"  {'F: Negated row_coords':<40} {fm_F.mean():10.6f}", flush=True)
print(f"  {'GT baseline':<40} {'~0.040':>10}", flush=True)

# Save comparison figure
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
titles = ["Baseline (prev)", "D: Neg d_row", "E: No rebin", "F: Neg row_coords"]
# For baseline, reuse rec_F's first reconstruction (the one without override that we accidentally ran)
# Actually we don't have baseline rec here. Just show D, E, F and a placeholder.
recs = [rec_D, rec_D, rec_E, rec_F]  # first is placeholder
fms_list = [fm_D, fm_D, fm_E, fm_F]

# Re-run baseline quickly for fair comparison
print("\n  Running baseline for comparison plot...", flush=True)
t0 = time()
rec_base = reconstruct_custom(
    sino_chunk, angles_chunk, views_chunk,
    scan_geom_chunk, meta, chunk_z_min, chunk_z_max,
    TEST_SLICES,
)
fm_base = flicker_metric(rec_base)
print(f"  Baseline flicker: {fm_base.mean():.6f} (verification)", flush=True)

recs = [rec_base, rec_D, rec_E, rec_F]
fms_list = [fm_base, fm_D, fm_E, fm_F]
titles = [f"Baseline ({fm_base.mean():.4f})",
          f"D: Neg d_row ({fm_D.mean():.4f})",
          f"E: No rebin ({fm_E.mean():.4f})",
          f"F: Neg row_coords ({fm_F.mean():.4f})"]

for col, (title, rec, fm) in enumerate(zip(titles, recs, fms_list)):
    mid = rec.shape[2] // 2
    vmin, vmax = np.percentile(rec[:, :, mid], [1, 99])
    axes[0, col].imshow(rec[:, :, mid], cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, col].set_title(title)
    axes[0, col].axis('off')

    if rec.shape[2] > mid + 1:
        diff_img = rec[:, :, mid + 1] - rec[:, :, mid]
        vd = max(np.percentile(np.abs(diff_img), 99), 1e-10)
        axes[1, col].imshow(diff_img, cmap='RdBu', vmin=-vd, vmax=vd)
        axes[1, col].set_title(f"Slice diff, flicker={fm.mean():.5f}")
        axes[1, col].axis('off')

plt.suptitle("Flicker Diagnostic: Diff/Rebin/RowCoords Tests", fontsize=14)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "diff_rebin_comparison.png")
plt.savefig(out_path, dpi=150)
print(f"\n  Saved -> {out_path}", flush=True)
print("\nDone.", flush=True)
