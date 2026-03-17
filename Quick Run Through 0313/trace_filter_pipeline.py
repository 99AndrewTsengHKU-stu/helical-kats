"""
Trace the Katsevich filtering pipeline to find the source of flickering artifacts.

Checks:
  A. Detector pixel size asymmetry (rows vs cols)
  B. Volume z-range vs table position z-range alignment
  C. Katsevich filter intermediate outputs for periodic patterns
  D. T-D window coverage and boundary behavior
  E. source_pos precision in sincosf (large angle values)

Usage:
  ~/anaconda3/envs/MNIST/python.exe -u "Quick Run Through 0313/trace_filter_pipeline.py"
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
from pykatsevich.filter import differentiate, fw_height_rebinning, compute_hilbert_kernel, hilbert_conv, rev_rebin_vec, sino_weight_td

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DICOM_DIR = r"D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD"
ROWS = 512
COLS = 512
SLICES = 560
VOXEL_SIZE = 0.664
CHUNK_Z = 64

# ====================================================================
# CHECK A: Detector pixel size
# ====================================================================
print("=" * 70)
print("CHECK A: Detector pixel size asymmetry")
print("=" * 70)

print("\nLoading DICOM...")
t0 = time()
sino, meta = load_dicom_projections(DICOM_DIR)
print(f"Loaded {sino.shape} in {time()-t0:.1f}s")

psize_rows = meta["detector_pixel_size_rows_mm"]
psize_cols = meta["detector_pixel_size_cols_mm"]
psize_avg = meta["scan_geometry"]["detector"]["detector psize"]

print(f"  Pixel size rows (z):  {psize_rows:.6f} mm")
print(f"  Pixel size cols (fan): {psize_cols:.6f} mm")
print(f"  Average (used):       {psize_avg:.6f} mm")
print(f"  Ratio cols/rows:      {psize_cols/psize_rows:.6f}")
print(f"  Relative diff:        {abs(psize_cols - psize_rows) / psize_avg * 100:.3f}%")

if abs(psize_cols - psize_rows) / psize_avg > 0.01:
    print("  [!] Pixel sizes differ! Using average may cause geometric distortion.")
    print(f"      Error in row projection: {abs(psize_rows - psize_avg)/psize_avg*100:.3f}%")
    print(f"      Error in col projection: {abs(psize_cols - psize_avg)/psize_avg*100:.3f}%")
else:
    print("  [OK] Pixel sizes are nearly equal.")

# ====================================================================
# CHECK B: Volume z vs table z alignment
# ====================================================================
print("\n" + "=" * 70)
print("CHECK B: Volume z-range vs source z-range alignment")
print("=" * 70)

total_half_z = SLICES * VOXEL_SIZE * 0.5
vol_z_min = -total_half_z
vol_z_max = total_half_z
vol_z_span = vol_z_max - vol_z_min

table_pos = meta["table_positions_mm"]
table_centered = table_pos - table_pos.mean()
table_z_span = table_centered.max() - table_centered.min()

print(f"  Volume z: [{vol_z_min:.2f}, {vol_z_max:.2f}] mm (span={vol_z_span:.2f})")
print(f"  Table z (centered): [{table_centered.min():.2f}, {table_centered.max():.2f}] mm (span={table_z_span:.2f})")
print(f"  Span ratio (table/volume): {table_z_span/vol_z_span:.4f}")
print(f"  Table spans {table_z_span - vol_z_span:.2f} mm MORE than volume")

# The volume z and table z should be roughly aligned
# Volume uses SLICES * VOXEL_SIZE, table uses actual table positions
# If VOXEL_SIZE doesn't match the actual pitch-based spacing, there's a mismatch
expected_voxel_z = table_z_span / SLICES
print(f"\n  Expected voxel z-size from table: {expected_voxel_z:.6f} mm")
print(f"  Actual VOXEL_SIZE used:           {VOXEL_SIZE:.6f} mm")
print(f"  Ratio: {VOXEL_SIZE / expected_voxel_z:.6f}")
if abs(VOXEL_SIZE / expected_voxel_z - 1.0) > 0.05:
    print("  [!] VOXEL_SIZE doesn't match table-based z-spacing!")
    print("      This means volume z-coordinates don't align with source z-coordinates.")
else:
    print("  [OK] VOXEL_SIZE roughly matches table spacing.")

# ====================================================================
# CHECK C: Filter pipeline intermediate outputs
# ====================================================================
print("\n" + "=" * 70)
print("CHECK C: Filter pipeline step-by-step (single chunk)")
print("=" * 70)

# Prepare geometry
angles_full = meta["angles_rad"].copy()
scan_geom_full = copy.deepcopy(meta["scan_geometry"])

z_shift_full = table_pos - table_pos.mean()

angles_full = -angles_full  # negate
angles_full = angles_full + (-np.pi/2)  # offset

sino_work = sino[:, ::-1, :].copy()  # flip rows

pitch_used = abs(meta["pitch_mm_per_rad_signed"])
scan_geom_full["helix"]["pitch_mm_rad"] = float(pitch_used)
scan_geom_full["helix"]["angles_range"] = float(abs(angles_full[-1] - angles_full[0]))

views_full = astra_helical_views(
    scan_geom_full["SOD"], scan_geom_full["SDD"],
    scan_geom_full["detector"]["detector psize"],
    angles_full, meta["pitch_mm_per_angle"],
    vertical_shifts=z_shift_full,
)

# Pick middle chunk
SOD = scan_geom_full["SOD"]
SDD = scan_geom_full["SDD"]
det_rows = scan_geom_full["detector"]["detector rows"]
psize = scan_geom_full["detector"]["detector psize"]

mid_chunk = SLICES // 2 // CHUNK_Z
z_start = mid_chunk * CHUNK_Z
z_end = min(z_start + CHUNK_Z, SLICES)
chunk_slices = z_end - z_start
chunk_z_min = -total_half_z + z_start * VOXEL_SIZE
chunk_z_max = -total_half_z + z_end * VOXEL_SIZE

# Z-cull
src_z_astra = views_full[:, 2]
cone_half_z = 0.5 * det_rows * psize * (SOD / SDD)
margin_z = cone_half_z
z_lo = src_z_astra - cone_half_z - margin_z
z_hi = src_z_astra + cone_half_z + margin_z
mask = (z_hi >= chunk_z_min) & (z_lo <= chunk_z_max)
keep_idx = np.where(mask)[0]

sino_chunk = sino_work[keep_idx].copy()
angles_chunk = angles_full[keep_idx].copy()
views_chunk = views_full[keep_idx]

print(f"  Chunk [{z_start}, {z_end}), z=[{chunk_z_min:.2f}, {chunk_z_max:.2f}] mm")
print(f"  {len(keep_idx)} projections after z-cull")

# Build conf
scan_geom_chunk = copy.deepcopy(scan_geom_full)
scan_geom_chunk["helix"]["angles_range"] = float(abs(angles_chunk[-1] - angles_chunk[0]))
scan_geom_chunk["helix"]["angles_count"] = len(angles_chunk)

half_x = COLS * VOXEL_SIZE * 0.5
half_y = ROWS * VOXEL_SIZE * 0.5
vol_geom_chunk = astra.create_vol_geom(
    ROWS, COLS, chunk_slices,
    -half_x, half_x, -half_y, half_y, chunk_z_min, chunk_z_max,
)
proj_geom_chunk = astra.create_proj_geom("cone_vec", det_rows,
    scan_geom_full["detector"]["detector cols"], views_chunk)

conf = create_configuration(scan_geom_chunk, vol_geom_chunk)
conf['source_pos'] = angles_chunk.astype(np.float32)
conf['delta_s'] = float(np.mean(np.diff(angles_chunk)))

print(f"  conf delta_s: {conf['delta_s']:.8f}")
print(f"  conf projs_per_turn: {conf['projs_per_turn']:.2f}")

# Step-by-step filtering
sino_f32 = np.asarray(sino_chunk, dtype=np.float32, order="C")
print(f"\n  Input sinogram: shape={sino_f32.shape}, range=[{sino_f32.min():.4f}, {sino_f32.max():.4f}]")

# Step C1: Differentiate
print("\n  [C1] Differentiation...")
t0 = time()
diff_proj = differentiate(sino_f32, conf)
print(f"    Done in {time()-t0:.1f}s")
print(f"    Output range: [{diff_proj.min():.6f}, {diff_proj.max():.6f}]")
print(f"    Has NaN: {np.any(np.isnan(diff_proj))}")
print(f"    Has Inf: {np.any(np.isinf(diff_proj))}")

# Check for periodic patterns in differentiated data
# Look at a single detector pixel across all projections
mid_row = diff_proj.shape[1] // 2
mid_col = diff_proj.shape[2] // 2
trace_diff = diff_proj[:, mid_row, mid_col]
# Compute FFT to find periodic components
fft_diff = np.abs(np.fft.rfft(trace_diff))
freqs = np.fft.rfftfreq(len(trace_diff))
# Find dominant frequency (skip DC)
peak_idx = np.argmax(fft_diff[1:]) + 1
peak_freq = freqs[peak_idx]
peak_period = 1.0 / peak_freq if peak_freq > 0 else float('inf')
projs_per_turn = conf['projs_per_turn']
print(f"    Dominant periodic component at pixel [{mid_row},{mid_col}]:")
print(f"      Frequency: {peak_freq:.6f} (period={peak_period:.1f} projections)")
print(f"      projs_per_turn: {projs_per_turn:.1f}")
print(f"      Period/projs_per_turn ratio: {peak_period/projs_per_turn:.4f}")

# Step C2: Forward height rebinning
print("\n  [C2] Forward height rebinning...")
t0 = time()
fwd_rebin = fw_height_rebinning(diff_proj, conf)
print(f"    Done in {time()-t0:.1f}s")
print(f"    Output shape: {fwd_rebin.shape}, range: [{fwd_rebin.min():.6f}, {fwd_rebin.max():.6f}]")
print(f"    Has NaN: {np.any(np.isnan(fwd_rebin))}")

# Step C3: Hilbert transform
print("\n  [C3] Hilbert convolution...")
t0 = time()
hilbert_kernel = compute_hilbert_kernel(conf)
hilbert_out = hilbert_conv(fwd_rebin, hilbert_kernel, conf)
print(f"    Done in {time()-t0:.1f}s")
print(f"    Output range: [{hilbert_out.min():.6f}, {hilbert_out.max():.6f}]")
print(f"    Has NaN: {np.any(np.isnan(hilbert_out))}")

# Step C4: Reverse rebinning
print("\n  [C4] Reverse height rebinning...")
t0 = time()
rev_rebin = rev_rebin_vec(hilbert_out, conf)
print(f"    Done in {time()-t0:.1f}s")
print(f"    Output shape: {rev_rebin.shape}, range: [{rev_rebin.min():.6f}, {rev_rebin.max():.6f}]")
print(f"    Has NaN: {np.any(np.isnan(rev_rebin))}")

# Step C5: T-D weighting
print("\n  [C5] Tam-Danielsson weighting...")
t0 = time()
sino_td = sino_weight_td(rev_rebin, conf, False)
print(f"    Done in {time()-t0:.1f}s")
print(f"    Output range: [{sino_td.min():.6f}, {sino_td.max():.6f}]")

# Check how many projections have significant T-D masked data
td_energy = np.sum(sino_td**2, axis=(1,2))
total_energy = np.sum(rev_rebin**2, axis=(1,2))
td_ratio = td_energy / (total_energy + 1e-30)
print(f"    T-D energy retention per projection: mean={td_ratio.mean():.4f}, min={td_ratio.min():.4f}, max={td_ratio.max():.4f}")
zero_projs = np.sum(td_energy < 1e-20)
print(f"    Projections with zero T-D output: {zero_projs}/{len(td_energy)}")

# ====================================================================
# CHECK D: T-D window shape
# ====================================================================
print("\n" + "=" * 70)
print("CHECK D: T-D window analysis")
print("=" * 70)

w_bottom = conf['proj_row_mins'][:-1]
w_top = conf['proj_row_maxs'][:-1]
row_coords = conf['row_coords'][:-1]
col_coords = conf['col_coords'][:-1]

print(f"  T-D bottom range: [{w_bottom.min():.4f}, {w_bottom.max():.4f}] mm")
print(f"  T-D top range:    [{w_top.min():.4f}, {w_top.max():.4f}] mm")
print(f"  Row coords range: [{row_coords.min():.4f}, {row_coords.max():.4f}] mm")
print(f"  Col coords range: [{col_coords.min():.4f}, {col_coords.max():.4f}] mm")

# Check if T-D window covers the detector
mid_col_idx = len(col_coords) // 2
td_height_center = w_top[mid_col_idx] - w_bottom[mid_col_idx]
det_height = row_coords[-1] - row_coords[0]
print(f"\n  T-D window height at center col: {td_height_center:.4f} mm")
print(f"  Detector height: {det_height:.4f} mm")
print(f"  T-D covers {td_height_center/det_height*100:.1f}% of detector height at center")

# Edge cols
td_height_edge0 = w_top[0] - w_bottom[0]
td_height_edge1 = w_top[-1] - w_bottom[-1]
print(f"  T-D window height at edge cols: {td_height_edge0:.4f}, {td_height_edge1:.4f} mm")

# Check if smoothing parameter is reasonable
smoothing = conf['T-D smoothing']
dw = det_rows * conf['pixel_height']
print(f"\n  T-D smoothing: {smoothing}")
print(f"  Smoothing zone width: {smoothing * dw:.4f} mm ({smoothing*100:.1f}% of detector height)")

# ====================================================================
# CHECK E: sincosf precision with large angles
# ====================================================================
print("\n" + "=" * 70)
print("CHECK E: Angle precision for cos/sin")
print("=" * 70)

# Compare cos/sin of large DICOM angles vs reduced angles
angles_reduced = np.mod(angles_chunk, 2 * np.pi)  # reduce to [0, 2pi)
cos_full = np.cos(angles_chunk.astype(np.float32))
sin_full = np.sin(angles_chunk.astype(np.float32))
cos_reduced = np.cos(angles_reduced.astype(np.float32))
sin_reduced = np.sin(angles_reduced.astype(np.float32))

cos_diff = np.abs(cos_full - cos_reduced)
sin_diff = np.abs(sin_full - sin_reduced)

print(f"  Angle range: [{angles_chunk.min():.2f}, {angles_chunk.max():.2f}] rad")
print(f"  max|cos(full) - cos(reduced)|: {cos_diff.max():.2e}")
print(f"  max|sin(full) - sin(reduced)|: {sin_diff.max():.2e}")
print(f"  mean|cos diff|: {cos_diff.mean():.2e}")
print(f"  mean|sin diff|: {sin_diff.mean():.2e}")

# Also check float64 vs float32
cos_f64 = np.cos(angles_chunk.astype(np.float64))
cos_f32 = np.cos(angles_chunk.astype(np.float32))
f32_error = np.abs(cos_f64 - cos_f32.astype(np.float64))
print(f"\n  float32 vs float64 cos error:")
print(f"    max: {f32_error.max():.2e}")
print(f"    mean: {f32_error.mean():.2e}")

if cos_diff.max() > 1e-5 or sin_diff.max() > 1e-5:
    print("  [!] Large angle values cause precision loss in cos/sin!")
    print("      Consider reducing angles modulo 2*pi before passing to kernel.")
else:
    print("  [OK] cos/sin precision is adequate for these angle values.")

# ====================================================================
# CHECK F: Backproject with CuPy - compare reduced vs full angles
# ====================================================================
print("\n" + "=" * 70)
print("CHECK F: Backproject comparison (full angles vs reduced angles)")
print("=" * 70)

from backproject_cupy import backproject_cupy

# Run with original (large) angles
print("  Running BP with original angles...")
t0 = time()
rec_orig = backproject_cupy(sino_td, conf, vol_geom_chunk, proj_geom_chunk, tqdm_bar=False)
print(f"    Done in {time()-t0:.1f}s, range=[{rec_orig.min():.6f}, {rec_orig.max():.6f}]")

# Run with reduced angles (mod 2pi)
conf_reduced = copy.deepcopy(conf)
conf_reduced['source_pos'] = np.mod(angles_chunk, 2 * np.pi).astype(np.float32)
print("  Running BP with reduced angles (mod 2pi)...")
t0 = time()
rec_reduced = backproject_cupy(sino_td, conf_reduced, vol_geom_chunk, proj_geom_chunk, tqdm_bar=False)
print(f"    Done in {time()-t0:.1f}s, range=[{rec_reduced.min():.6f}, {rec_reduced.max():.6f}]")

diff = rec_orig - rec_reduced
print(f"\n  Difference (orig - reduced):")
print(f"    max|diff|: {np.abs(diff).max():.2e}")
print(f"    mean|diff|: {np.abs(diff).mean():.2e}")
print(f"    relative max|diff|: {np.abs(diff).max() / (np.abs(rec_orig).max() + 1e-30) * 100:.6f}%")

# ====================================================================
# Plot diagnostics
# ====================================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Row 1: Filter pipeline stages (single projection, mid-chunk)
mid_proj = len(sino_chunk) // 2
axes[0,0].imshow(sino_f32[mid_proj], cmap='gray', aspect='auto')
axes[0,0].set_title(f"Input sinogram\nproj {mid_proj}")

axes[0,1].imshow(diff_proj[mid_proj], cmap='gray', aspect='auto')
axes[0,1].set_title("After differentiate")

axes[0,2].imshow(rev_rebin[mid_proj], cmap='gray', aspect='auto')
axes[0,2].set_title("After rebin+Hilbert+rebin")

axes[0,3].imshow(sino_td[mid_proj], cmap='gray', aspect='auto')
axes[0,3].set_title("After T-D weighting")

# Row 2: Sinogram traces and T-D window
# Sinogram slice across projections at detector center
axes[1,0].plot(td_ratio, linewidth=0.3)
axes[1,0].set_xlabel("Projection index")
axes[1,0].set_ylabel("T-D energy ratio")
axes[1,0].set_title("T-D energy retention")

# T-D window shape
td_mask_img = np.zeros((det_rows, scan_geom_full["detector"]["detector cols"]))
W, U = np.meshgrid(row_coords, col_coords, indexing='ij')
w_bot_2d = np.broadcast_to(w_bottom, (det_rows, len(col_coords)))
w_top_2d = np.broadcast_to(w_top, (det_rows, len(col_coords)))
td_mask_img[(W >= w_bot_2d) & (W <= w_top_2d)] = 1.0
axes[1,1].imshow(td_mask_img, cmap='gray', aspect='auto')
axes[1,1].set_title("T-D window (sharp)")

# Reconstruction comparison
mid_z = chunk_slices // 2
vmin, vmax = np.percentile(rec_orig[:,:,mid_z], [1, 99])
axes[1,2].imshow(rec_orig[:,:,mid_z], cmap='gray', vmin=vmin, vmax=vmax)
axes[1,2].set_title(f"Recon (full angles)\nz={mid_z}")

if np.abs(diff).max() > 0:
    vd = max(np.abs(np.percentile(diff[:,:,mid_z], [1, 99])))
    if vd > 0:
        axes[1,3].imshow(diff[:,:,mid_z], cmap='RdBu_r', vmin=-vd, vmax=vd)
    else:
        axes[1,3].imshow(diff[:,:,mid_z], cmap='RdBu_r')
else:
    axes[1,3].imshow(diff[:,:,mid_z], cmap='RdBu_r')
axes[1,3].set_title(f"Diff (full-reduced)\nmax={np.abs(diff).max():.2e}")

for ax in axes.flat:
    ax.axis("off") if ax.images else None

plt.suptitle("Katsevich Filter Pipeline Diagnostics", fontsize=14)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "filter_pipeline_trace.png")
plt.savefig(out_path, dpi=150)
plt.close()
print(f"\nSaved -> {out_path}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
A. Pixel size: rows={psize_rows:.6f}, cols={psize_cols:.6f}, diff={abs(psize_cols-psize_rows)/psize_avg*100:.3f}%
B. Volume vs table z: volume span={vol_z_span:.2f}, table span={table_z_span:.2f}, ratio={table_z_span/vol_z_span:.4f}
C. Filter pipeline: check plot for visual artifacts
D. T-D window: covers {td_height_center/det_height*100:.1f}% of detector at center
E. cos/sin precision: max error = {max(cos_diff.max(), sin_diff.max()):.2e}
F. BP full vs reduced angles: max|diff| = {np.abs(diff).max():.2e}
""")

print("Done.")
