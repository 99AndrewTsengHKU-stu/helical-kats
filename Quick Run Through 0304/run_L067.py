"""
Reconstruct L067 quarter-dose DICOM data using backproject_safe.
Includes Auto-Focus to detect correct rotation/flip parameters.

Logic:
  1. Load DICOM projections (equiangular → flat detector resampling built-in)
  2. Auto-Focus: test 9 geometry modes on 1-slice recon, pick sharpest
  3. Full reconstruction with best mode
  4. Compare with ground truth DICOM
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

# CRITICAL: Initialize ASTRA CUDA context BEFORE CuPy is imported
from backproject_safe import ensure_astra_cuda_init
ensure_astra_cuda_init()

import copy
import numpy as np
from matplotlib import pyplot as plt
from time import time
import astra

from pykatsevich import load_dicom_projections
from pykatsevich.geometry import astra_helical_views
from pykatsevich.initialize import create_configuration
from pykatsevich.filter import filter_katsevich, sino_weight_td
from backproject_safe import backproject_safe
from backproject_cupy import backproject_cupy

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Settings ─────────────────────────────────────────────────────────────
DICOM_DIR = r"D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD"
GT_DIR    = r"D:\AAPM-Data\L067\L067\full_1mm"   # ground truth FBP slices

ROWS      = 512
COLS      = 512
SLICES    = 560    # match GT: 560 slices @ 1mm
VOXEL_SIZE_XY = 0.664  # mm — match GT pixel spacing (FOV=340mm / 512)
VOXEL_SIZE_Z  = None   # computed from DICOM table positions after loading
CHUNK_Z   = 64     # slices per z-chunk for chunked reconstruction

DECIMATE  = 1      # use every N-th projection (1 = all 2000)
MAX_VIEWS = None    # None = use all; set to e.g. 1000 for ~1 turn


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════

def reverse_views_per_turn(angles, sino, z_shift):
    """Reverse projection order within each 2*pi rotation."""
    unwrapped = np.unwrap(angles)
    turn_ids = np.floor((unwrapped - unwrapped[0]) / (2 * np.pi)).astype(int)
    unique_turns = np.unique(turn_ids)
    reorder_idx = np.concatenate([np.flatnonzero(turn_ids == turn)[::-1] for turn in unique_turns])
    return angles[reorder_idx], sino[reorder_idx], z_shift[reorder_idx] if z_shift is not None else None


def measure_sharpness(img):
    """Estimate image sharpness using gradient magnitude mean."""
    gy, gx = np.gradient(img)
    return np.mean(np.sqrt(gx**2 + gy**2))


def z_cull_indices(views, SOD, SDD, det_rows, pixel_size, vol_z_min, vol_z_max, margin_turns=1.0, pitch_mm_per_angle=None):
    """
    Return indices of projections whose cone beam overlaps the volume z-range.

    For each projection, the source z is views[i,2]. The cone beam covers
    z_src ± (det_rows/2 * pixel_size * SOD/SDD) at the isocenter.
    We add an extra margin (in turns worth of z-travel) for safety.
    """
    source_z = views[:, 2]
    # Half-height of cone beam at isocenter
    cone_half_z = 0.5 * det_rows * pixel_size * (SOD / SDD)
    # Extra margin: one full turn of z-travel
    if pitch_mm_per_angle is not None and len(views) > 1:
        angle_step = abs(np.mean(np.diff(np.arctan2(views[:10, 1], views[:10, 0]))))
        projs_per_turn = 2 * np.pi / max(angle_step, 1e-12)
        margin_z = margin_turns * projs_per_turn * abs(pitch_mm_per_angle)
    else:
        margin_z = cone_half_z  # fallback: add one cone-height as margin

    z_lo = source_z - cone_half_z - margin_z
    z_hi = source_z + cone_half_z + margin_z
    mask = (z_hi >= vol_z_min) & (z_lo <= vol_z_max)
    return np.where(mask)[0]


def build_and_reconstruct(sino, meta, opts, rows, cols, slices, voxel_size, verbose=True, voxel_size_z=None):
    """
    Build geometry, filter, and reconstruct with given mode options.

    opts keys: negate_angles, reverse_angle_order, reverse_per_turn,
               flip_cols, flip_rows, pitch_signed, angle_offset_rad
    """
    angles = meta["angles_rad"].copy()
    scan_geom = copy.deepcopy(meta["scan_geometry"])

    if meta["table_positions_mm"] is not None:
        z_shift = meta["table_positions_mm"] - meta["table_positions_mm"].mean()
    else:
        z_shift = np.zeros(len(angles), dtype=np.float32)

    # Apply transforms
    sino_work = sino.copy()

    if opts.get("reverse_per_turn", False):
        angles, sino_work, z_shift = reverse_views_per_turn(angles, sino_work, z_shift)

    if opts.get("reverse_angle_order", False):
        angles = angles[::-1].copy()
        sino_work = sino_work[::-1].copy()
        z_shift = z_shift[::-1].copy()

    if opts.get("negate_angles", False):
        angles = -angles

    angle_offset = opts.get("angle_offset_rad", 0)
    if angle_offset != 0:
        angles = angles + angle_offset

    if opts.get("flip_cols", False):
        sino_work = sino_work[:, :, ::-1].copy()

    if opts.get("flip_rows", False):
        sino_work = sino_work[:, ::-1, :].copy()

    pitch_signed = meta["pitch_mm_per_rad_signed"]
    pitch_used = pitch_signed if opts.get("pitch_signed", False) else abs(pitch_signed)

    scan_geom["helix"]["pitch_mm_rad"] = float(pitch_used)
    scan_geom["helix"]["angles_range"] = float(abs(angles[-1] - angles[0]))

    # Build ASTRA geometries (full set first for z-culling)
    views = astra_helical_views(
        scan_geom["SOD"],
        scan_geom["SDD"],
        scan_geom["detector"]["detector psize"],
        angles,
        meta["pitch_mm_per_angle"],
        vertical_shifts=z_shift,
        pixel_size_col=scan_geom["detector"].get("detector psize cols"),
        pixel_size_row=scan_geom["detector"].get("detector psize rows"),
    )

    vz = voxel_size_z if voxel_size_z is not None else voxel_size
    half_x = cols * voxel_size * 0.5
    half_y = rows * voxel_size * 0.5
    half_z = slices * vz * 0.5

    # ── Z-range culling: skip projections that can't illuminate the volume ──
    det_rows_n = scan_geom["detector"]["detector rows"]
    det_cols_n = scan_geom["detector"]["detector cols"]
    psize = scan_geom["detector"]["detector psize"]
    n_before = len(angles)

    keep_idx = z_cull_indices(
        views, scan_geom["SOD"], scan_geom["SDD"],
        det_rows_n, psize, -half_z, half_z,
        margin_turns=1.0, pitch_mm_per_angle=meta["pitch_mm_per_angle"],
    )

    if len(keep_idx) < n_before:
        if verbose:
            print(f"  [Z-cull] {n_before} -> {len(keep_idx)} projections "
                  f"({100*len(keep_idx)/n_before:.1f}% kept)")
        sino_work = sino_work[keep_idx].copy()
        angles = angles[keep_idx].copy()
        z_shift = z_shift[keep_idx].copy()
        views = views[keep_idx]
        scan_geom["helix"]["angles_range"] = float(abs(angles[-1] - angles[0]))
        scan_geom["helix"]["angles_count"] = len(angles)

    proj_geom = astra.create_proj_geom("cone_vec", det_rows_n, det_cols_n, views)
    vol_geom = astra.create_vol_geom(
        rows, cols, slices,
        -half_x, half_x, -half_y, half_y, -half_z, half_z,
    )

    # Helical configuration
    conf = create_configuration(scan_geom, vol_geom)
    conf['source_pos'] = angles.astype(np.float32)
    conf['delta_s'] = float(np.mean(np.diff(angles)))

    if verbose:
        print(f"  [Angle] step={conf['delta_s']:.5f} rad  "
              f"direction={'CW' if conf['delta_s'] < 0 else 'CCW'}")
        print(f"  [Pitch] progress_per_radian={conf['progress_per_radian']:.4f}  "
              f"per_turn={conf['progress_per_turn']:.2f}")

    # Filter
    sino_f32 = np.asarray(sino_work, dtype=np.float32, order="C")
    filtered = filter_katsevich(
        sino_f32, conf,
        {"Diff": {"Print time": verbose}, "FwdRebin": {"Print time": verbose}, "BackRebin": {"Print time": verbose}},
    )
    sino_td = sino_weight_td(filtered, conf, False)

    # Backproject
    rec = backproject_safe(sino_td, conf, vol_geom, proj_geom, tqdm_bar=verbose)
    return rec


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

# ── Load DICOM ───────────────────────────────────────────────────────────
print(f"Loading DICOM projections from {DICOM_DIR} ...")
t0 = time()
sino, meta = load_dicom_projections(DICOM_DIR)
print(f"Loaded {sino.shape[0]} projections, detector {sino.shape[1]}x{sino.shape[2]}, in {time()-t0:.1f}s")
print(f"  SOD={meta['scan_geometry']['SOD']:.1f}  SDD={meta['scan_geometry']['SDD']:.1f}")
print(f"  angle range: {np.degrees(meta['angles_rad'][-1]-meta['angles_rad'][0]):.1f} deg "
      f"({(meta['angles_rad'][-1]-meta['angles_rad'][0])/(2*np.pi):.2f} turns)")
print(f"  pitch_mm_per_rad (signed): {meta['pitch_mm_per_rad_signed']:.4f}")

# Z voxel size: match GT DICOM (verified: 560 slices at 0.8mm spacing, z-span=447.2mm)
# The GT volume is smaller than the table span (484.5mm) because edge slices
# lack sufficient helical data for reconstruction.
VOXEL_SIZE_Z = 0.8  # mm — from GT DICOM SliceLocation analysis
if meta["table_positions_mm"] is not None:
    table_z_span = meta["table_positions_mm"].max() - meta["table_positions_mm"].min()
    vol_z_span = SLICES * VOXEL_SIZE_Z
    print(f"  table z-span: {table_z_span:.2f} mm, volume z-span: {vol_z_span:.1f} mm "
          f"(margin: {(table_z_span - vol_z_span)/2:.1f} mm each side)")
print(f"  VOXEL_SIZE_XY = {VOXEL_SIZE_XY:.4f} mm, VOXEL_SIZE_Z = {VOXEL_SIZE_Z:.4f} mm")

# ── Trim / decimate ─────────────────────────────────────────────────────
if MAX_VIEWS is not None and MAX_VIEWS < len(sino):
    print(f"Trimming to first {MAX_VIEWS} views ...")
    sino = sino[:MAX_VIEWS].copy()
    meta["angles_rad"] = meta["angles_rad"][:MAX_VIEWS]
    if meta["table_positions_mm"] is not None:
        meta["table_positions_mm"] = meta["table_positions_mm"][:MAX_VIEWS]
    meta["scan_geometry"]["helix"]["angles_count"] = MAX_VIEWS
    meta["pitch_mm_per_angle"] = float(np.mean(np.abs(np.diff(meta["angles_rad"]))))

if DECIMATE > 1:
    print(f"Decimating by {DECIMATE}x ...")
    sino = sino[::DECIMATE].copy()
    meta["angles_rad"] = meta["angles_rad"][::DECIMATE]
    if meta["table_positions_mm"] is not None:
        meta["table_positions_mm"] = meta["table_positions_mm"][::DECIMATE]
    meta["scan_geometry"]["helix"]["angles_count"] = len(sino)
    meta["pitch_mm_per_angle"] *= DECIMATE
    print(f"  Now {len(sino)} projections")

# ═══════════════════════════════════════════════════════════════════════════
# Mode: Flip Rows + Negate + Angle Offset -π/2
#   - negate_angles: DICOM stores CW angles, code expects CCW
#   - flip_rows: DICOM detector z-axis is flipped vs code convention
#   - angle_offset: DICOM 0° is at Y-axis, code 0° is at X-axis (-90° shift)
#     This replaces the old post-recon rot90(k=1).
#     Verified 2026-03-13: max pixel diff < 0.00013 vs rot90 approach.
# ═══════════════════════════════════════════════════════════════════════════
best_mode_name = "Flip Rows + Negate + Offset(-pi/2)"
best_opts = {"flip_rows": True, "negate_angles": True, "angle_offset_rad": -np.pi/2}

# ═══════════════════════════════════════════════════════════════════════════
# Chunked z-reconstruction: split 560 slices into CHUNK_Z-sized blocks,
# z-cull projections per chunk so each chunk only processes ~few thousand
# projections instead of all 48590.
# ═══════════════════════════════════════════════════════════════════════════
total_half_z = SLICES * VOXEL_SIZE_Z * 0.5  # total volume half-height in mm
n_chunks = (SLICES + CHUNK_Z - 1) // CHUNK_Z

print(f"\n{'='*60}")
print(f"CHUNKED RECONSTRUCTION: {best_mode_name}")
print(f"  {ROWS}x{COLS}x{SLICES}, voxel_xy={VOXEL_SIZE_XY}mm, voxel_z={VOXEL_SIZE_Z:.4f}mm")
print(f"  {n_chunks} chunks of {CHUNK_Z} slices")
print(f"{'='*60}")

# Pre-compute transforms and full views ONCE
angles_full = meta["angles_rad"].copy()
scan_geom_full = copy.deepcopy(meta["scan_geometry"])

if meta["table_positions_mm"] is not None:
    z_shift_full = meta["table_positions_mm"] - meta["table_positions_mm"].mean()
else:
    z_shift_full = np.zeros(len(angles_full), dtype=np.float32)

sino_work = sino.copy()

if best_opts.get("negate_angles", False):
    angles_full = -angles_full
angle_offset = best_opts.get("angle_offset_rad", 0)
if angle_offset != 0:
    angles_full = angles_full + angle_offset
if best_opts.get("flip_rows", False):
    sino_work = sino_work[:, ::-1, :].copy()

pitch_signed = meta["pitch_mm_per_rad_signed"]
pitch_used = abs(pitch_signed)
scan_geom_full["helix"]["pitch_mm_rad"] = float(pitch_used)
scan_geom_full["helix"]["angles_range"] = float(abs(angles_full[-1] - angles_full[0]))

# Build full views array ONCE
psize_cols = scan_geom_full["detector"]["detector psize cols"]
psize_rows = scan_geom_full["detector"]["detector psize rows"]
views_full = astra_helical_views(
    scan_geom_full["SOD"],
    scan_geom_full["SDD"],
    scan_geom_full["detector"]["detector psize"],  # fallback (unused if col/row given)
    angles_full,
    meta["pitch_mm_per_angle"],
    vertical_shifts=z_shift_full,
    pixel_size_col=psize_cols,
    pixel_size_row=psize_rows,
)

det_rows_n = scan_geom_full["detector"]["detector rows"]
det_cols_n = scan_geom_full["detector"]["detector cols"]
psize = scan_geom_full["detector"]["detector psize"]

# Allocate full output volume
rec_full = np.zeros((ROWS, COLS, SLICES), dtype=np.float32)

t_total = time()

for chunk_i in range(n_chunks):
    z_start_slice = chunk_i * CHUNK_Z
    z_end_slice = min(z_start_slice + CHUNK_Z, SLICES)
    chunk_slices = z_end_slice - z_start_slice

    # Chunk z-range in mm (volume centered at z=0)
    chunk_z_min = -total_half_z + z_start_slice * VOXEL_SIZE_Z
    chunk_z_max = -total_half_z + z_end_slice * VOXEL_SIZE_Z

    print(f"\n--- Chunk {chunk_i+1}/{n_chunks}: slices [{z_start_slice}:{z_end_slice}], "
          f"z=[{chunk_z_min:.1f}, {chunk_z_max:.1f}]mm ---")

    # Z-cull: find projections relevant to this chunk
    keep_idx = z_cull_indices(
        views_full, scan_geom_full["SOD"], scan_geom_full["SDD"],
        det_rows_n, psize, chunk_z_min, chunk_z_max,
        margin_turns=1.0, pitch_mm_per_angle=meta["pitch_mm_per_angle"],
    )
    print(f"  [Z-cull] {len(angles_full)} -> {len(keep_idx)} projections "
          f"({100*len(keep_idx)/len(angles_full):.1f}%)")

    if len(keep_idx) == 0:
        print(f"  [SKIP] No projections illuminate this chunk")
        continue

    # Extract subset
    sino_chunk = sino_work[keep_idx].copy()
    angles_chunk = angles_full[keep_idx].copy()
    views_chunk = views_full[keep_idx]

    # Build chunk geometry
    scan_geom_chunk = copy.deepcopy(scan_geom_full)
    scan_geom_chunk["helix"]["angles_range"] = float(abs(angles_chunk[-1] - angles_chunk[0]))
    scan_geom_chunk["helix"]["angles_count"] = len(angles_chunk)

    proj_geom_chunk = astra.create_proj_geom("cone_vec", det_rows_n, det_cols_n, views_chunk)

    half_x = COLS * VOXEL_SIZE_XY * 0.5
    half_y = ROWS * VOXEL_SIZE_XY * 0.5
    vol_geom_chunk = astra.create_vol_geom(
        ROWS, COLS, chunk_slices,
        -half_x, half_x, -half_y, half_y, chunk_z_min, chunk_z_max,
    )

    # Configuration for this chunk
    conf_chunk = create_configuration(scan_geom_chunk, vol_geom_chunk)
    conf_chunk['source_pos'] = angles_chunk.astype(np.float32)
    conf_chunk['delta_s'] = float(np.mean(np.diff(angles_chunk)))

    # Filter
    t0 = time()
    sino_f32 = np.asarray(sino_chunk, dtype=np.float32, order="C")
    filtered = filter_katsevich(
        sino_f32, conf_chunk,
        {"Diff": {"Print time": False}, "FwdRebin": {"Print time": False}, "BackRebin": {"Print time": False}},
    )
    sino_td = sino_weight_td(filtered, conf_chunk, False)
    t_filt = time() - t0

    # Backproject (fused CuPy kernel — no per-projection ASTRA loop)
    t0 = time()
    rec_chunk = backproject_cupy(sino_td, conf_chunk, vol_geom_chunk, proj_geom_chunk, tqdm_bar=True)
    t_bp = time() - t0

    # No post-recon rotation needed — angle_offset=-pi/2 handles the
    # DICOM-vs-code axis convention directly in the geometry.

    # Store in full volume
    rec_full[:, :, z_start_slice:z_end_slice] = rec_chunk

    print(f"  Filter: {t_filt:.1f}s  BP: {t_bp:.1f}s  "
          f"chunk range: [{rec_chunk.min():.4f}, {rec_chunk.max():.4f}]")

print(f"\n{'='*60}")
print(f"ALL CHUNKS DONE in {time()-t_total:.1f}s")
print(f"Full volume: {rec_full.shape}  min={rec_full.min():.4f}  max={rec_full.max():.4f}")
print(f"{'='*60}")

rec = rec_full

# ── Save volume as .npy and .tiff ─────────────────────────────────────
npy_path = os.path.join(OUT_DIR, "L067_rec_560.npy")
np.save(npy_path, rec)
print(f"Saved volume -> {npy_path}")

try:
    import tifffile
    tiff_path = os.path.join(OUT_DIR, "L067_rec_560.tiff")
    # (slices, rows, cols) ordering for TIFF stack
    tiff_data = np.moveaxis(rec, 2, 0).astype(np.float32, copy=False)
    tifffile.imwrite(tiff_path, tiff_data)
    print(f"Saved TIFF -> {tiff_path}")
except ImportError:
    print("tifffile not installed, skipping TIFF save")

# ── Load ground truth for comparison ────────────────────────────────────
gt_slice = None
if os.path.isdir(GT_DIR):
    try:
        import pydicom
        gt_files = sorted([f for f in os.listdir(GT_DIR) if f.lower().endswith(('.dcm', '.ima'))])
        if gt_files:
            mid_idx = len(gt_files) // 2
            ds = pydicom.dcmread(os.path.join(GT_DIR, gt_files[mid_idx]))
            gt_slice = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            gt_slice = gt_slice * slope + intercept
            print(f"Loaded GT slice {mid_idx}/{len(gt_files)}: shape={gt_slice.shape}")
    except Exception as e:
        print(f"Could not load GT: {e}")

# ── Visualize final result ──────────────────────────────────────────────
n_panels = 4 if gt_slice is not None else 3
fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

sl_indices = [rec.shape[2] // 4, rec.shape[2] // 2, rec.shape[2] * 3 // 4]
for i, si in enumerate(sl_indices):
    vmin, vmax = np.percentile(rec[:, :, si], [1, 99])
    cs = axes[i].imshow(rec[:, :, si], cmap="gray", vmin=vmin, vmax=vmax)
    axes[i].set_title(f"Recon z={si}")
    axes[i].axis("off")
    fig.colorbar(cs, ax=axes[i], fraction=0.046)

if gt_slice is not None:
    vmin_gt, vmax_gt = np.percentile(gt_slice, [1, 99])
    cs = axes[3].imshow(gt_slice, cmap="gray", vmin=vmin_gt, vmax=vmax_gt)
    axes[3].set_title("GT (full-dose FBP)")
    axes[3].axis("off")
    fig.colorbar(cs, ax=axes[3], fraction=0.046)

fig.suptitle(f"L067 Katsevich [{best_mode_name}] ({ROWS}x{COLS}x{SLICES}, vxy={VOXEL_SIZE_XY}, vz={VOXEL_SIZE_Z:.4f}mm)")
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "L067_result.png")
plt.savefig(out_path, dpi=150)
print(f"\nSaved -> {out_path}")
plt.close()
