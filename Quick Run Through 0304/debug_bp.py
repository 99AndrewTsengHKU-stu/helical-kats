"""Debug: compare ASTRA BP vs fused kernel for a single projection."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from backproject_safe import ensure_astra_cuda_init
ensure_astra_cuda_init()

import numpy as np
import cupy as cp
import yaml
import astra

from common import phantom_objects_3d, project
from pykatsevich.initialize import create_configuration
from pykatsevich.filter import filter_katsevich, sino_weight_td
from backproject_cupy import _KERNEL_SRC

YAML_PATH = os.path.join(os.path.dirname(__file__), "..", "tests", "test03.yaml")
with open(YAML_PATH, "r") as f:
    settings = yaml.safe_load(f)
ps = settings["phantom"]
geom = settings["geometry"]

phantom = phantom_objects_3d(ps["rows"], ps["columns"], ps["slices"],
                              voxel_size=ps["voxel_size"], objects_list=ps["objects"])
sinogram, vol_geom, proj_geom = project(phantom, ps["voxel_size"], geom)
conf = create_configuration(geom, vol_geom, geom.get("options", {}))
sinogram_swapped = np.asarray(np.swapaxes(sinogram, 0, 1), order="C")
filtered = filter_katsevich(sinogram_swapped, conf, {"Diff": {}, "FwdRebin": {}, "BackRebin": {}})
sino_td = sino_weight_td(filtered, conf, False)

print(f"sino_td shape: {sino_td.shape}")  # (n_projs, det_rows, det_cols)
print(f"vol_shape: {astra.geom_size(vol_geom)}")  # (Z, Y, X)
n_projs = sino_td.shape[0]

# ── Test single projection with ASTRA ──
proj_idx = n_projs // 2
sino_for_astra = np.asarray(np.swapaxes(sino_td, 1, 0), dtype=np.float32, order='C')
single_pg = astra.create_proj_geom("cone_vec",
    proj_geom["DetectorRowCount"], proj_geom["DetectorColCount"],
    proj_geom["Vectors"][proj_idx:proj_idx+1])
sino_slice = np.ascontiguousarray(sino_for_astra[:, proj_idx:proj_idx+1, :])
sino_id = astra.data3d.create("-sino", single_pg, sino_slice)
bp_id = astra.data3d.create("-vol", vol_geom, 0)
cfg = astra.astra_dict("BP3D_CUDA")
cfg["ReconstructionDataId"] = bp_id
cfg["ProjectionDataId"] = sino_id
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id)
astra_bp = astra.data3d.get(bp_id)  # (Z, Y, X)
astra.algorithm.delete([alg_id])
astra.data3d.delete([sino_id, bp_id])

print(f"\nASTRA BP for proj {proj_idx}: shape={astra_bp.shape}, "
      f"range=[{astra_bp.min():.6e}, {astra_bp.max():.6e}], "
      f"nonzero={np.count_nonzero(astra_bp)}/{astra_bp.size}")

# ── Test single projection with fused kernel (just the interpolation part, no 1/L) ──
kernel_raw = cp.RawKernel(r'''
extern "C" __global__
void debug_bp(
    float* __restrict__ rec,
    const float* __restrict__ sino,
    float angle, float src_z_val,
    int det_rows, int det_cols,
    float SOD, float SDD, float psize,
    float x_min, float y_min, float z_min,
    float dx, float dy, float dz,
    int nx, int ny, int nz)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int iz = blockDim.z * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz) return;

    float X = x_min + ((float)ix + 0.5f) * dx;
    float Y = y_min + ((float)iy + 0.5f) * dy;
    float Z = z_min + ((float)iz + 0.5f) * dz;

    float cs, sn;
    sincosf(angle, &sn, &cs);

    float L = SOD - X * cs - Y * sn;
    if (L <= 0.0f) return;

    float SDD_over_psize = SDD / psize;
    float u = SDD_over_psize * (Y * cs - X * sn) / L + (float)(det_cols - 1) * 0.5f;
    float v = SDD_over_psize * (Z - src_z_val) / L + (float)(det_rows - 1) * 0.5f;

    if (u < 0.0f || u > (float)(det_cols - 1) ||
        v < 0.0f || v > (float)(det_rows - 1)) return;

    int u0 = (int)u;
    int v0 = (int)v;
    if (u0 >= det_cols - 1) u0 = det_cols - 2;
    if (v0 >= det_rows - 1) v0 = det_rows - 2;
    if (u0 < 0) u0 = 0;
    if (v0 < 0) v0 = 0;

    float fu = u - (float)u0;
    float fv = v - (float)v0;

    long long base = 0;  // single projection, p=0
    float s00 = sino[base + v0 * det_cols + u0];
    float s01 = sino[base + v0 * det_cols + u0 + 1];
    float s10 = sino[base + (v0 + 1) * det_cols + u0];
    float s11 = sino[base + (v0 + 1) * det_cols + u0 + 1];

    float interp = (1.0f - fv) * ((1.0f - fu) * s00 + fu * s01)
                  + fv * ((1.0f - fu) * s10 + fu * s11);

    long long idx = (long long)iz * ny * nx + (long long)iy * nx + ix;
    rec[idx] = interp;  // RAW interpolation, NO 1/L weighting
}
''', 'debug_bp', options=('--use_fast_math',))

nz, ny, nx = astra.geom_size(vol_geom)
rec_cupy_raw = cp.zeros((nz, ny, nx), dtype=cp.float32)

# Single projection sinogram
sino_single = cp.asarray(sino_td[proj_idx:proj_idx+1].astype(np.float32))
angle_val = float(conf['source_pos'][proj_idx])
src_z_val = float(proj_geom['Vectors'][proj_idx, 2])

block = (8, 8, 4)
grid = ((nx+7)//8, (ny+7)//8, (nz+3)//4)

kernel_raw(grid, block, (
    rec_cupy_raw, sino_single,
    np.float32(angle_val), np.float32(src_z_val),
    np.int32(proj_geom['DetectorRowCount']), np.int32(proj_geom['DetectorColCount']),
    np.float32(conf['scan_radius']), np.float32(conf['scan_diameter']),
    np.float32(conf['pixel_span']),
    np.float32(conf['x_min']), np.float32(conf['y_min']), np.float32(conf['z_min']),
    np.float32(conf['delta_x']), np.float32(conf['delta_y']), np.float32(conf['delta_z']),
    np.int32(nx), np.int32(ny), np.int32(nz),
))
cp.cuda.Device().synchronize()
cupy_raw = rec_cupy_raw.get()

print(f"CuPy raw BP (no 1/L): shape={cupy_raw.shape}, "
      f"range=[{cupy_raw.min():.6e}, {cupy_raw.max():.6e}], "
      f"nonzero={np.count_nonzero(cupy_raw)}/{cupy_raw.size}")

# ── Compare at several locations ──
print(f"\n{'='*60}")
print(f"Angle={angle_val:.4f} rad, src_z={src_z_val:.4f}")
print(f"SOD={conf['scan_radius']}, SDD={conf['scan_diameter']}, psize={conf['pixel_span']}")
print(f"Vol: nx={nx}, ny={ny}, nz={nz}")
print(f"x_min={conf['x_min']:.4f}, y_min={conf['y_min']:.4f}, z_min={conf['z_min']:.4f}")
print(f"dx={conf['delta_x']:.4f}, dy={conf['delta_y']:.4f}, dz={conf['delta_z']:.4f}")
print(f"det_rows={proj_geom['DetectorRowCount']}, det_cols={proj_geom['DetectorColCount']}")

# Center voxel
iz, iy, ix = nz//2, ny//2, nx//2
print(f"\nCenter voxel (iz={iz}, iy={iy}, ix={ix}):")
print(f"  ASTRA:    {astra_bp[iz, iy, ix]:.6e}")
print(f"  CuPy raw: {cupy_raw[iz, iy, ix]:.6e}")
if cupy_raw[iz, iy, ix] != 0:
    print(f"  Ratio:    {astra_bp[iz, iy, ix] / cupy_raw[iz, iy, ix]:.6f}")

# Check a grid of points
print(f"\nSampled comparison (ASTRA vs CuPy_raw):")
for iz_s in [nz//4, nz//2, 3*nz//4]:
    for iy_s in [ny//4, ny//2, 3*ny//4]:
        for ix_s in [nx//4, nx//2, 3*nx//4]:
            a = astra_bp[iz_s, iy_s, ix_s]
            c = cupy_raw[iz_s, iy_s, ix_s]
            ratio = a / c if abs(c) > 1e-12 else float('inf')
            if abs(a) > 1e-10 or abs(c) > 1e-10:
                print(f"  [{iz_s:2d},{iy_s:2d},{ix_s:2d}] astra={a:+.6e} cupy={c:+.6e} ratio={ratio:.4f}")

# Also check: compute u, v for center voxel manually
X = conf['x_min'] + (nx//2 + 0.5) * conf['delta_x']
Y = conf['y_min'] + (ny//2 + 0.5) * conf['delta_y']
Z = conf['z_min'] + (nz//2 + 0.5) * conf['delta_z']
cos_a = np.cos(angle_val)
sin_a = np.sin(angle_val)
SOD = conf['scan_radius']
SDD = conf['scan_diameter']
psize = conf['pixel_span']
L = SOD - X * cos_a - Y * sin_a
u_det = SDD * (Y * cos_a - X * sin_a) / (L * psize) + (proj_geom['DetectorColCount'] - 1) / 2
v_det = SDD * (Z - src_z_val) / (L * psize) + (proj_geom['DetectorRowCount'] - 1) / 2
print(f"\nManual projection for center voxel:")
print(f"  X={X:.4f}, Y={Y:.4f}, Z={Z:.4f}")
print(f"  L={L:.4f}, u={u_det:.4f}, v={v_det:.4f}")
print(f"  det bounds: u=[0,{proj_geom['DetectorColCount']-1}], v=[0,{proj_geom['DetectorRowCount']-1}]")
print(f"  In bounds: {0 <= u_det <= proj_geom['DetectorColCount']-1 and 0 <= v_det <= proj_geom['DetectorRowCount']-1}")
