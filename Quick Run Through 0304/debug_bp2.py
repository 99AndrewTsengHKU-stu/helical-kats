"""Debug: test ASTRA BP with constant sinogram to isolate implicit weighting."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from backproject_safe import ensure_astra_cuda_init
ensure_astra_cuda_init()

import numpy as np
import yaml
import astra

from common import phantom_objects_3d, project
from pykatsevich.initialize import create_configuration

YAML_PATH = os.path.join(os.path.dirname(__file__), "..", "tests", "test03.yaml")
with open(YAML_PATH, "r") as f:
    settings = yaml.safe_load(f)
ps = settings["phantom"]
geom = settings["geometry"]

# Create the geometry (we need vol_geom and proj_geom)
phantom = phantom_objects_3d(ps["rows"], ps["columns"], ps["slices"],
                              voxel_size=ps["voxel_size"], objects_list=ps["objects"])
_, vol_geom, proj_geom = project(phantom, ps["voxel_size"], geom)
conf = create_configuration(geom, vol_geom, geom.get("options", {}))

nz, ny, nx = astra.geom_size(vol_geom)
det_rows = proj_geom['DetectorRowCount']
det_cols = proj_geom['DetectorColCount']

# Use middle projection only, with CONSTANT sinogram = 1.0
proj_idx = proj_geom['Vectors'].shape[0] // 2
single_pg = astra.create_proj_geom("cone_vec", det_rows, det_cols,
    proj_geom["Vectors"][proj_idx:proj_idx+1])

sino_ones = np.ones((det_rows, 1, det_cols), dtype=np.float32)
sino_id = astra.data3d.create("-sino", single_pg, sino_ones)
bp_id = astra.data3d.create("-vol", vol_geom, 0)
cfg = astra.astra_dict("BP3D_CUDA")
cfg["ReconstructionDataId"] = bp_id
cfg["ProjectionDataId"] = sino_id
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id)
astra_bp = astra.data3d.get(bp_id)
astra.algorithm.delete([alg_id])
astra.data3d.delete([sino_id, bp_id])

# For constant sinogram, my kernel would give bilinear_interp = 1.0 everywhere
# (except out-of-bounds voxels which get 0)
# So cupy_raw = 1.0 for in-bounds voxels
# And the ratio ASTRA/cupy_raw = ASTRA weighting function

angle = float(conf['source_pos'][proj_idx])
src_z = float(proj_geom['Vectors'][proj_idx, 2])
SOD = conf['scan_radius']
SDD = conf['scan_diameter']
psize = conf['pixel_span']

print(f"Angle={angle:.6f}, src_z={src_z:.4f}")
print(f"SOD={SOD}, SDD={SDD}, psize={psize}")
print(f"det_rows={det_rows}, det_cols={det_cols}")
print(f"vol: {nz}x{ny}x{nx}")
print(f"\nASTRA BP of constant-1 sinogram (single proj):")
print(f"  range: [{astra_bp.min():.6e}, {astra_bp.max():.6e}]")
print(f"  nonzero: {np.count_nonzero(astra_bp)}/{astra_bp.size}")

# Check if ASTRA output = 1/L^2 * C
print(f"\nSampled voxels (ASTRA BP value vs 1/L and 1/L^2):")
for iz_s in [nz//4, nz//2, 3*nz//4]:
    for iy_s in [ny//4, ny//2, 3*ny//4]:
        for ix_s in [nx//4, nx//2, 3*nx//4]:
            val = astra_bp[iz_s, iy_s, ix_s]
            if abs(val) < 1e-15:
                continue
            X = conf['x_min'] + (ix_s + 0.5) * conf['delta_x']
            Y = conf['y_min'] + (iy_s + 0.5) * conf['delta_y']
            Z = conf['z_min'] + (iz_s + 0.5) * conf['delta_z']
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            L = SOD - X * cos_a - Y * sin_a

            # Check various hypotheses
            val_times_L = val * L
            val_times_L2 = val * L**2
            val_times_SDD2 = val * SDD**2
            print(f"  [{iz_s:2d},{iy_s:2d},{ix_s:2d}] X={X:+.2f} Y={Y:+.2f} Z={Z:+.2f}"
                  f"  ASTRA={val:+.4e}  L={L:.2f}"
                  f"  val*L={val_times_L:.4f}  val*L^2={val_times_L2:.2f}"
                  f"  val*SDD^2={val_times_SDD2:.2f}")
