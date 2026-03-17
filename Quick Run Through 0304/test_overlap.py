"""
Quick test: compare no-overlap vs overlap chunk blending on a small region.
Reconstruct slices 240-320 (80 slices) in two ways:
  A) Single chunk (no boundary) — ground truth
  B) Two chunks of 48 with 16 overlap, feather-blended at boundary (slice 280)
Compare flickering at the boundary region.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

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
from backproject_cupy import backproject_cupy

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

DICOM_DIR = r"D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD"
ROWS, COLS = 512, 512
VOXEL_SIZE = 0.664
TOTAL_SLICES = 560

# Region to test
SLAB_START = 240
SLAB_END   = 320
SLAB_SIZE  = SLAB_END - SLAB_START  # 80

def z_cull_indices(views, SOD, SDD, det_rows, pixel_size, vol_z_min, vol_z_max, margin_turns=1.0, pitch_mm_per_angle=None):
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

def reconstruct_slab(sino_work, angles_full, views_full, scan_geom, meta,
                     slab_start, slab_end):
    """Reconstruct a z-slab as a single chunk."""
    total_half_z = TOTAL_SLICES * VOXEL_SIZE * 0.5
    z_min = -total_half_z + slab_start * VOXEL_SIZE
    z_max = -total_half_z + slab_end * VOXEL_SIZE
    n_slices = slab_end - slab_start

    det_rows_n = scan_geom["detector"]["detector rows"]
    det_cols_n = scan_geom["detector"]["detector cols"]
    psize = scan_geom["detector"]["detector psize"]

    keep = z_cull_indices(views_full, scan_geom["SOD"], scan_geom["SDD"],
                          det_rows_n, psize, z_min, z_max,
                          margin_turns=1.0, pitch_mm_per_angle=meta["pitch_mm_per_angle"])
    print(f"  Z-cull: {len(angles_full)} -> {len(keep)} projs")

    sc = sino_work[keep].copy()
    ac = angles_full[keep].copy()
    vc = views_full[keep]

    sg = copy.deepcopy(scan_geom)
    sg["helix"]["angles_range"] = float(abs(ac[-1] - ac[0]))
    sg["helix"]["angles_count"] = len(ac)

    pg = astra.create_proj_geom("cone_vec", det_rows_n, det_cols_n, vc)
    hx = COLS * VOXEL_SIZE * 0.5
    hy = ROWS * VOXEL_SIZE * 0.5
    vg = astra.create_vol_geom(ROWS, COLS, n_slices, -hx, hx, -hy, hy, z_min, z_max)

    conf = create_configuration(sg, vg)
    conf['source_pos'] = ac.astype(np.float32)
    conf['delta_s'] = float(np.mean(np.diff(ac)))

    sf = np.asarray(sc, dtype=np.float32, order="C")
    filtered = filter_katsevich(sf, conf,
        {"Diff": {"Print time": False}, "FwdRebin": {"Print time": False}, "BackRebin": {"Print time": False}})
    std = sino_weight_td(filtered, conf, False)

    rec = backproject_cupy(std, conf, vg, pg, tqdm_bar=False)
    rec = np.rot90(rec, k=1, axes=(0, 1))
    return rec

# ── Load DICOM ──────────────────────────────────────────────────────────
print(f"Loading DICOM ...")
t0 = time()
sino, meta = load_dicom_projections(DICOM_DIR)
print(f"Loaded {sino.shape[0]} projections in {time()-t0:.1f}s")

angles_full = -meta["angles_rad"].copy()
scan_geom = copy.deepcopy(meta["scan_geometry"])
z_shift_full = meta["table_positions_mm"] - meta["table_positions_mm"].mean() if meta["table_positions_mm"] is not None else np.zeros(len(angles_full), dtype=np.float32)
sino_work = sino[:, ::-1, :].copy()
scan_geom["helix"]["pitch_mm_rad"] = float(abs(meta["pitch_mm_per_rad_signed"]))
scan_geom["helix"]["angles_range"] = float(abs(angles_full[-1] - angles_full[0]))

views_full = astra_helical_views(
    scan_geom["SOD"], scan_geom["SDD"],
    scan_geom["detector"]["detector psize"],
    angles_full, meta["pitch_mm_per_angle"],
    vertical_shifts=z_shift_full)

# ═══════════════════════════════════════════════════════════════════════════
# Method A: Single chunk (80 slices, no boundary)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"METHOD A: Single chunk [{SLAB_START}:{SLAB_END}] = {SLAB_SIZE} slices")
print(f"{'='*60}")
t0 = time()
rec_A = reconstruct_slab(sino_work, angles_full, views_full, scan_geom, meta,
                         SLAB_START, SLAB_END)
print(f"  Done in {time()-t0:.1f}s, shape={rec_A.shape}")

# ═══════════════════════════════════════════════════════════════════════════
# Method B: Two chunks with overlap, feather-blended
# ═══════════════════════════════════════════════════════════════════════════
OVERLAP = 16
SPLIT = 40  # split the 80-slice slab in half
# Chunk 1: own [240:280], padded [240:296]  (extends 16 past owned end)
# Chunk 2: own [280:320], padded [264:320]  (extends 16 before owned start)

print(f"\n{'='*60}")
print(f"METHOD B: Two chunks, overlap={OVERLAP}")
print(f"{'='*60}")

# Chunk 1: [240:296] = 56 slices
print(f"\nChunk 1: [{SLAB_START}:{SLAB_START+SPLIT+OVERLAP}]")
t0 = time()
rec_c1 = reconstruct_slab(sino_work, angles_full, views_full, scan_geom, meta,
                           SLAB_START, SLAB_START + SPLIT + OVERLAP)
print(f"  Done in {time()-t0:.1f}s")

# Chunk 2: [264:320] = 56 slices
print(f"\nChunk 2: [{SLAB_START+SPLIT-OVERLAP}:{SLAB_END}]")
t0 = time()
rec_c2 = reconstruct_slab(sino_work, angles_full, views_full, scan_geom, meta,
                           SLAB_START + SPLIT - OVERLAP, SLAB_END)
print(f"  Done in {time()-t0:.1f}s")

# Blend
rec_B = np.zeros_like(rec_A)
# Non-overlap from chunk 1: slices 0..39 (local) -> slab 240..279
rec_B[:, :, :SPLIT] = rec_c1[:, :, :SPLIT]
# Non-overlap from chunk 2: slices 16+.. (local) -> slab 280+..
rec_B[:, :, SPLIT:] = rec_c2[:, :, OVERLAP:]

# Overlap region: slab slices [280-16 : 280] = [264:280] -> local B indices [24:40]
# In rec_c1: local [40:56], in rec_c2: local [0:16]
# Actually let me redo this more carefully:
# rec_c1 covers slab [240:296], local indices 0..55
# rec_c2 covers slab [264:320], local indices 0..55
# Overlap region in slab: [264:296] but we only blend where both chunks have data
# Actually the boundary is at slab 280 (SPLIT from start)
# Let me just do feather in the overlap region [264:296]

rec_B = np.zeros_like(rec_A)
wgt = np.zeros(SLAB_SIZE, dtype=np.float32)

# Chunk 1: covers slab [240:296] -> local 0..55
c1_slab_start = 0
c1_slab_end = SPLIT + OVERLAP  # 56
w1 = np.ones(c1_slab_end - c1_slab_start, dtype=np.float32)
# Ramp down at end
w1[-OVERLAP:] = np.linspace(1, 0, OVERLAP + 1, dtype=np.float32)[1:]
for i in range(c1_slab_end - c1_slab_start):
    rec_B[:, :, c1_slab_start + i] += rec_c1[:, :, i] * w1[i]
    wgt[c1_slab_start + i] += w1[i]

# Chunk 2: covers slab [264:320] -> maps to B indices [24:80]
c2_slab_start = SPLIT - OVERLAP  # 24
c2_slab_end = SLAB_SIZE  # 80
w2 = np.ones(c2_slab_end - c2_slab_start, dtype=np.float32)
# Ramp up at start
w2[:OVERLAP] = np.linspace(0, 1, OVERLAP + 1, dtype=np.float32)[:-1]
for i in range(c2_slab_end - c2_slab_start):
    rec_B[:, :, c2_slab_start + i] += rec_c2[:, :, i] * w2[i]
    wgt[c2_slab_start + i] += w2[i]

# Normalize
for s in range(SLAB_SIZE):
    if wgt[s] > 0:
        rec_B[:, :, s] /= wgt[s]

# ═══════════════════════════════════════════════════════════════════════════
# Method C: Two chunks, hard boundary (no overlap) — like original code
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"METHOD C: Two chunks, no overlap (hard boundary at slice 280)")
print(f"{'='*60}")

rec_C = np.zeros_like(rec_A)
# Re-use rec_c1[:, :, :SPLIT] and reconstruct second half without overlap
print(f"\nChunk 2 (no overlap): [{SLAB_START+SPLIT}:{SLAB_END}]")
t0 = time()
rec_c2_hard = reconstruct_slab(sino_work, angles_full, views_full, scan_geom, meta,
                                SLAB_START + SPLIT, SLAB_END)
print(f"  Done in {time()-t0:.1f}s")

rec_C[:, :, :SPLIT] = rec_c1[:, :, :SPLIT]
rec_C[:, :, SPLIT:] = rec_c2_hard

# ═══════════════════════════════════════════════════════════════════════════
# Compare
# ═══════════════════════════════════════════════════════════════════════════
cx, cy = ROWS // 2, COLS // 2
roi_r = 80

def slice_stats(rec, name):
    means = [rec[cx-roi_r:cx+roi_r, cy-roi_r:cy+roi_r, s].mean() for s in range(rec.shape[2])]
    means = np.array(means)
    diffs = np.diff(means)
    flicker = np.std(diffs)
    # Focus on boundary region (around slice 40 local = slab 280)
    boundary_diffs = np.abs(diffs[SPLIT-3:SPLIT+3])
    return means, diffs, flicker, boundary_diffs

mA, dA, fA, bA = slice_stats(rec_A, "A")
mB, dB, fB, bB = slice_stats(rec_B, "B")
mC, dC, fC, bC = slice_stats(rec_C, "C")

print(f"\n{'='*60}")
print(f"RESULTS")
print(f"{'='*60}")
print(f"  A (single chunk):     flicker={fA:.7f}  boundary_max={bA.max():.7f}")
print(f"  B (overlap+feather):  flicker={fB:.7f}  boundary_max={bB.max():.7f}")
print(f"  C (hard boundary):    flicker={fC:.7f}  boundary_max={bC.max():.7f}")

# ── Visualize ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(16, 14))
slices_x = np.arange(SLAB_START, SLAB_END)

for col, (rec, means, diffs, name, flk) in enumerate([
    (rec_A, mA, dA, "A: Single chunk", fA),
    (rec_B, mB, dB, "B: Overlap+feather", fB),
    (rec_C, mC, dC, "C: Hard boundary", fC),
]):
    # Row 0: mid slice image
    mid = SPLIT  # at the boundary
    vmin, vmax = np.percentile(rec[:, :, mid], [1, 99])
    axes[0, col].imshow(rec[:, :, mid], cmap="gray", vmin=vmin, vmax=vmax)
    axes[0, col].set_title(f"{name}\nSlice {SLAB_START+mid}")
    axes[0, col].axis("off")

    # Row 1: diff at boundary
    diff_img = rec[:, :, mid] - rec[:, :, mid-1]
    dlim = max(abs(diff_img).max(), 1e-6)
    axes[1, col].imshow(diff_img, cmap="RdBu_r", vmin=-dlim, vmax=dlim)
    axes[1, col].set_title(f"Diff(s{mid}-s{mid-1})\nmax={dlim:.6f}")
    axes[1, col].axis("off")

    # Row 2: per-slice means
    axes[2, col].plot(slices_x, means, "o-", markersize=2)
    axes[2, col].axvline(SLAB_START + SPLIT, color='r', alpha=0.5, linestyle='--')
    axes[2, col].set_title(f"ROI mean, flicker={flk:.7f}")
    axes[2, col].set_xlabel("Slice")

fig.suptitle(f"Overlap Blending Test (slices {SLAB_START}-{SLAB_END}, boundary at {SLAB_START+SPLIT})", fontsize=14)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "overlap_compare.png")
plt.savefig(out_path, dpi=150)
print(f"\nSaved -> {out_path}")
plt.close()
