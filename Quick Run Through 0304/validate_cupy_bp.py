"""
Validate backproject_cupy against backproject_safe using the test03 phantom.
Runs both implementations on the same filtered sinogram and compares results.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

# CRITICAL: Initialize ASTRA CUDA context BEFORE CuPy is imported
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
from backproject_cupy import backproject_cupy

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
YAML_PATH = os.path.join(os.path.dirname(__file__), "..", "tests", "test03.yaml")

# --- Load settings ---
with open(YAML_PATH, "r") as f:
    settings = yaml.safe_load(f)

ps = settings["phantom"]
geom = settings["geometry"]

# --- Generate phantom ---
print(f"Generating phantom {ps['rows']}x{ps['columns']}x{ps['slices']} ...")
phantom = phantom_objects_3d(
    ps["rows"], ps["columns"], ps["slices"],
    voxel_size=ps["voxel_size"],
    objects_list=ps["objects"],
)

# --- Forward projection ---
print("Forward projection ...", end=" ", flush=True)
t0 = time()
sinogram, vol_geom, proj_geom = project(phantom, ps["voxel_size"], geom)
print(f"done in {time()-t0:.2f}s")

# --- Create configuration ---
conf = create_configuration(geom, vol_geom, geom.get("options", {}))

# --- Reorder sinogram for pykatsevich (angles, rows, cols) ---
sinogram_swapped = np.asarray(np.swapaxes(sinogram, 0, 1), order="C")

# --- Katsevich filtering ---
print("Katsevich filtering ...")
filtered = filter_katsevich(
    sinogram_swapped, conf,
    {"Diff": {"Print time": True}, "FwdRebin": {"Print time": True}, "BackRebin": {"Print time": True}},
)
sino_td = sino_weight_td(filtered, conf, False)

# --- BP with backproject_safe (reference) ---
print("\n=== backproject_safe (ASTRA per-proj loop) ===")
t0 = time()
rec_safe = backproject_safe(sino_td, conf, vol_geom, proj_geom, tqdm_bar=True)
t_safe = time() - t0
print(f"  Time: {t_safe:.2f}s")

# --- BP with backproject_cupy (fused kernel) ---
print("\n=== backproject_cupy (fused CuPy kernel) ===")
t0 = time()
rec_cupy = backproject_cupy(sino_td, conf, vol_geom, proj_geom, tqdm_bar=True)
t_cupy = time() - t0
print(f"  Time: {t_cupy:.2f}s")

# --- Compare ---
diff = rec_safe - rec_cupy
max_abs_diff = np.max(np.abs(diff))
rel_diff = max_abs_diff / max(np.max(np.abs(rec_safe)), 1e-12)
rmse = np.sqrt(np.mean(diff**2))
safe_range = rec_safe.max() - rec_safe.min()

print(f"\n{'='*50}")
print(f"COMPARISON:")
print(f"  rec_safe  range: [{rec_safe.min():.6f}, {rec_safe.max():.6f}]")
print(f"  rec_cupy  range: [{rec_cupy.min():.6f}, {rec_cupy.max():.6f}]")
print(f"  Max |diff|:      {max_abs_diff:.6e}")
print(f"  Relative diff:   {rel_diff:.6e}")
print(f"  RMSE:            {rmse:.6e}")
print(f"  Speedup:         {t_safe/t_cupy:.1f}x")
print(f"{'='*50}")

if rel_diff < 0.01:
    print("PASS: Results match within 1% relative error")
else:
    print(f"WARNING: Relative difference is {rel_diff*100:.2f}%")

# --- Visualize comparison ---
mid_z = rec_safe.shape[2] // 2
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(f"BP Validation: safe vs cupy (speedup {t_safe/t_cupy:.1f}x)")

vmin, vmax = np.percentile(rec_safe[:, :, mid_z], [1, 99])

axes[0, 0].imshow(rec_safe[:, :, mid_z], cmap="gray", vmin=vmin, vmax=vmax)
axes[0, 0].set_title(f"backproject_safe ({t_safe:.1f}s)")
axes[0, 0].axis("off")

axes[0, 1].imshow(rec_cupy[:, :, mid_z], cmap="gray", vmin=vmin, vmax=vmax)
axes[0, 1].set_title(f"backproject_cupy ({t_cupy:.1f}s)")
axes[0, 1].axis("off")

im = axes[0, 2].imshow(diff[:, :, mid_z], cmap="RdBu_r")
axes[0, 2].set_title(f"|Diff| max={max_abs_diff:.2e}")
axes[0, 2].axis("off")
fig.colorbar(im, ax=axes[0, 2], fraction=0.046)

axes[1, 0].imshow(phantom[:, :, mid_z], cmap="gray")
axes[1, 0].set_title("Phantom (ground truth)")
axes[1, 0].axis("off")

axes[1, 1].imshow(rec_cupy[:, :, mid_z], cmap="gray", vmin=vmin, vmax=vmax)
axes[1, 1].set_title("CuPy recon (zoomed)")
axes[1, 1].axis("off")

# Horizontal profile comparison
mid_row = rec_safe.shape[0] // 2
axes[1, 2].plot(rec_safe[mid_row, :, mid_z], label="safe", alpha=0.8)
axes[1, 2].plot(rec_cupy[mid_row, :, mid_z], label="cupy", alpha=0.8, linestyle="--")
axes[1, 2].set_title("Profile comparison (mid row)")
axes[1, 2].legend()

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "validate_cupy_bp.png")
plt.savefig(out_path, dpi=150)
print(f"Saved -> {out_path}")
plt.close()
