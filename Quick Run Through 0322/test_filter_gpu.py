"""
Test GPU Katsevich filter against CPU version.
Validates correctness then benchmarks speed.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Quick Run Through 0304"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from backproject_safe import ensure_astra_cuda_init
ensure_astra_cuda_init()

import numpy as np
from time import time
from pathlib import Path
import copy, astra

from pykatsevich import load_dicom_projections
from pykatsevich.geometry import astra_helical_views
from pykatsevich.initialize import create_configuration
from pykatsevich.filter import filter_katsevich, sino_weight_td
from pykatsevich.filter_gpu import filter_katsevich_gpu, sino_weight_td_gpu_np

DICOM_DIR = r"D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD"

# ── Load a small subset of projections ──────────────────────────────────
print("Loading DICOM projections...", flush=True)
t0 = time()
sino, meta = load_dicom_projections(DICOM_DIR)
print(f"Loaded in {time()-t0:.0f}s: {sino.shape}", flush=True)

sg = meta['scan_geometry']
angles_full = -meta['angles_rad'].copy() - np.pi / 2
z_shift = meta['table_positions_mm'] - meta['table_positions_mm'].mean()
# DON'T copy full sinogram — select subset first to save RAM

scan_geom = copy.deepcopy(sg)
pitch_abs = float(abs(meta['pitch_mm_per_rad_signed']))
scan_geom['helix']['pitch_mm_rad'] = pitch_abs

psize_cols = sg['detector'].get('detector psize cols', sg['detector']['detector psize'])
psize_rows = sg['detector'].get('detector psize rows', sg['detector']['detector psize'])
psize = sg['detector']['detector psize']
det_rows_n = sg['detector']['detector rows']

views = astra_helical_views(
    sg['SOD'], sg['SDD'], psize, angles_full, meta['pitch_mm_per_angle'],
    vertical_shifts=z_shift, pixel_size_col=psize_cols, pixel_size_row=psize_rows,
)

# ── Select a subset for testing ──
VOXEL_SIZE_XY = 0.6640625
VOXEL_SIZE_Z = 0.8
ROWS = COLS = 512
target_z = 0.0
half_xy = COLS * VOXEL_SIZE_XY * 0.5

source_z = views[:, 2]
cone_half_z = 0.5 * det_rows_n * psize * (sg['SOD'] / sg['SDD'])
angle_step = abs(np.mean(np.diff(np.arctan2(views[:10, 1], views[:10, 0]))))
projs_per_turn = 2 * np.pi / max(angle_step, 1e-12)
margin_z = projs_per_turn * abs(meta['pitch_mm_per_angle'])

keep = np.where(
    (source_z + cone_half_z + margin_z >= target_z - VOXEL_SIZE_Z) &
    (source_z - cone_half_z - margin_z <= target_z + VOXEL_SIZE_Z)
)[0]

# Limit projections to fit in RAM (CPU code uses float64 internally)
MAX_PROJS = 2000
if len(keep) > MAX_PROJS:
    mid = len(keep) // 2
    keep = keep[mid - MAX_PROJS // 2 : mid + MAX_PROJS // 2]

print(f"Using {len(keep)} projections for test", flush=True)

# Select subset and flip rows, then free full sinogram immediately
import gc
sino_c = sino[keep][:, ::-1, :].copy()
angles_c = angles_full[keep].copy()
views_c = views[keep]
del sino, angles_full, views, z_shift, source_z
gc.collect()
print(f"Freed full sinogram, subset shape: {sino_c.shape}", flush=True)

sg_c = copy.deepcopy(scan_geom)
sg_c['helix']['angles_range'] = float(abs(angles_c[-1] - angles_c[0]))
sg_c['helix']['angles_count'] = len(angles_c)

vol_geom = astra.create_vol_geom(
    ROWS, COLS, 1,
    -half_xy, half_xy, -half_xy, half_xy,
    target_z - VOXEL_SIZE_Z * 0.5, target_z + VOXEL_SIZE_Z * 0.5,
)

conf = create_configuration(sg_c, vol_geom)
conf['source_pos'] = angles_c.astype(np.float32)
conf['delta_s'] = float(np.mean(np.diff(angles_c)))

sino_input = np.asarray(sino_c, dtype=np.float32, order='C')

# ── CPU reference ───────────────────────────────────────────────────────
print("\n=== CPU filter ===", flush=True)
t1 = time()
filtered_cpu = filter_katsevich(sino_input, conf,
    {'Diff': {'Print time': True}, 'FwdRebin': {'Print time': True},
     'BackRebin': {'Print time': True}})
t_cpu_filter = time() - t1
print(f"CPU filter total: {t_cpu_filter:.2f}s")

t1 = time()
td_cpu = sino_weight_td(filtered_cpu, conf, False)
t_cpu_td = time() - t1
print(f"CPU T-D weight:   {t_cpu_td:.2f}s")
print(f"CPU total:        {t_cpu_filter + t_cpu_td:.2f}s")

# ── GPU version ─────────────────────────────────────────────────────────
print("\n=== GPU filter ===", flush=True)

# Warmup (first CuPy call compiles kernels)
import cupy as cp
_ = cp.zeros(10)

t2 = time()
filtered_gpu = filter_katsevich_gpu(sino_input, conf,
    {'Diff': {'Print time': True}, 'FwdRebin': {'Print time': True},
     'BackRebin': {'Print time': True}})
t_gpu_filter = time() - t2
print(f"GPU filter total: {t_gpu_filter:.2f}s")

t2 = time()
td_gpu = sino_weight_td_gpu_np(filtered_gpu, conf, False)
t_gpu_td = time() - t2
print(f"GPU T-D weight:   {t_gpu_td:.2f}s")
print(f"GPU total:        {t_gpu_filter + t_gpu_td:.2f}s")

# ── Compare results ─────────────────────────────────────────────────────
print("\n=== Accuracy comparison ===")

# Filter output
diff_filter = np.abs(filtered_cpu - filtered_gpu)
max_abs = np.max(np.abs(filtered_cpu))
rel_err = np.max(diff_filter) / max(max_abs, 1e-10)
print(f"Filter max abs error: {np.max(diff_filter):.6e}")
print(f"Filter max rel error: {rel_err:.6e}")
print(f"Filter mean abs error: {np.mean(diff_filter):.6e}")

# T-D weighted output
diff_td = np.abs(td_cpu - td_gpu)
max_abs_td = np.max(np.abs(td_cpu))
rel_err_td = np.max(diff_td) / max(max_abs_td, 1e-10)
print(f"T-D max abs error:    {np.max(diff_td):.6e}")
print(f"T-D max rel error:    {rel_err_td:.6e}")

# ── Speedup ─────────────────────────────────────────────────────────────
print(f"\n=== Speedup ===")
print(f"Filter: {t_cpu_filter:.1f}s (CPU) → {t_gpu_filter:.1f}s (GPU) = {t_cpu_filter/t_gpu_filter:.1f}x")
print(f"T-D:    {t_cpu_td:.1f}s (CPU) → {t_gpu_td:.1f}s (GPU) = {t_cpu_td/max(t_gpu_td,0.001):.1f}x")
total_cpu = t_cpu_filter + t_cpu_td
total_gpu = t_gpu_filter + t_gpu_td
print(f"Total:  {total_cpu:.1f}s (CPU) → {total_gpu:.1f}s (GPU) = {total_cpu/total_gpu:.1f}x")

# Pass/fail
TOLERANCE = 1e-3
if rel_err < TOLERANCE and rel_err_td < TOLERANCE:
    print(f"\n✓ PASS: GPU results match CPU within {TOLERANCE} relative error")
else:
    print(f"\n✗ FAIL: GPU results differ from CPU beyond tolerance {TOLERANCE}")
    print(f"  Filter rel err: {rel_err:.6e}")
    print(f"  T-D rel err:    {rel_err_td:.6e}")

print("\nDone.", flush=True)
