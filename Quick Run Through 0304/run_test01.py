"""
Reproduce test01.yaml: 256^3 phantom, two spheres, SOD=50, SDD=80.
Katsevich filtering + backprojection with distance weighting (no GPULink).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

# CRITICAL: Initialize ASTRA CUDA context BEFORE CuPy is imported (via filter.py)
from backproject_safe import ensure_astra_cuda_init
ensure_astra_cuda_init()

import numpy as np
import yaml
from matplotlib import pyplot as plt
from time import time

from common import phantom_objects_3d, project
from pykatsevich.initialize import create_configuration
from pykatsevich.filter import filter_katsevich, sino_weight_td
from backproject_safe import backproject_safe

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
YAML_PATH = os.path.join(os.path.dirname(__file__), "..", "tests", "test01.yaml")

# --- Load settings ---
with open(YAML_PATH, "r") as f:
    settings = yaml.safe_load(f)

ps = settings["phantom"]
geom = settings["geometry"]

# --- Generate phantom ---
print(f"[test01] Generating phantom {ps['rows']}x{ps['columns']}x{ps['slices']} ...")
phantom = phantom_objects_3d(
    ps["rows"], ps["columns"], ps["slices"],
    voxel_size=ps["voxel_size"],
    objects_list=ps["objects"],
)

# --- Forward projection ---
print("[test01] Forward projection with ASTRA ...", end=" ", flush=True)
t0 = time()
sinogram, vol_geom, proj_geom = project(phantom, ps["voxel_size"], geom)
print(f"done in {time()-t0:.2f}s")

# --- Create configuration ---
conf = create_configuration(geom, vol_geom, geom.get("options", {}))

# --- Reorder sinogram for pykatsevich (angles, rows, cols) ---
sinogram_swapped = np.asarray(np.swapaxes(sinogram, 0, 1), order="C")

# --- Katsevich filtering ---
print("[test01] Katsevich filtering ...")
filtered = filter_katsevich(
    sinogram_swapped, conf,
    {"Diff": {"Print time": True}, "FwdRebin": {"Print time": True}, "BackRebin": {"Print time": True}},
)

# --- Tam-Danielsson weighting ---
sino_td = sino_weight_td(filtered, conf, False)

# --- Katsevich backprojection with distance weighting (no GPULink) ---
print("[test01] Backprojection (Katsevich weighted) ...")
t0 = time()
rec = backproject_safe(sino_td, conf, vol_geom, proj_geom, tqdm_bar=True)
print(f"done in {time()-t0:.2f}s")

# --- Save result slices ---
sl_indices = [rec.shape[2] // 4, rec.shape[2] // 2, rec.shape[2] * 3 // 4]
fig, axes = plt.subplots(2, len(sl_indices), figsize=(12, 8))
fig.suptitle("test01: Phantom vs Katsevich Reconstruction (256^3)")
for i, si in enumerate(sl_indices):
    axes[0, i].imshow(phantom[:, :, si], cmap="gray")
    axes[0, i].set_title(f"Phantom z={si}")
    axes[0, i].axis("off")
    cs = axes[1, i].imshow(rec[:, :, si], cmap="gray")
    axes[1, i].set_title(f"Recon z={si}")
    axes[1, i].axis("off")
    fig.colorbar(cs, ax=axes[1, i], fraction=0.046)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "test01_result.png")
plt.savefig(out_path, dpi=150)
print(f"[test01] Saved -> {out_path}")
plt.close()
