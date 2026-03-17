"""
Step 1: 仿真 Phantom 测试
运行原版 t03 pipeline，把双球体重建结果保存为 PNG。
输出到当前目录（Quick Run Through 0228）。
"""

import sys
import os
from pathlib import Path

# Make sure we use the local pykatsevich package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Change working directory so test/common.py imports correctly.
TESTS_DIR = ROOT / "tests"
sys.path.insert(0, str(TESTS_DIR))

OUT_DIR = Path(__file__).resolve().parent

import numpy as np
import astra
import yaml
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from common import phantom_objects_3d, project
from pykatsevich.initialize import create_configuration
from pykatsevich.reconstruct import reconstruct

settings_file = str(TESTS_DIR / "test03.yaml")

with open(settings_file, "r") as f:
    cfg = yaml.safe_load(f)

phantom_s = cfg["phantom"]
voxel_size = phantom_s["voxel_size"]
print(f"[Step 1] 生成仿真 phantom: rows={phantom_s['rows']}, cols={phantom_s['columns']}, slices={phantom_s['slices']}")
phantom = phantom_objects_3d(
    phantom_s["rows"], phantom_s["columns"], phantom_s["slices"],
    voxel_size=voxel_size, objects_list=phantom_s["objects"]
)

geom = cfg["geometry"]
print("[Step 1] 正投影 phantom ...")
sinogram, vol_geom, proj_geom = project(phantom, voxel_size, geom)

conf = create_configuration(geom, vol_geom, geom.get("options", {}))

sinogram_swapped = np.asarray(np.swapaxes(sinogram, 0, 1), order="C")
print(f"[Step 1] sinogram shape: {sinogram_swapped.shape}, min={sinogram_swapped.min():.4f}, max={sinogram_swapped.max():.4f}")

# Save central projection
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.imshow(sinogram_swapped[sinogram_swapped.shape[0] // 2], cmap="gray")
ax.set_title("Central projection (simulated)")
ax.axis("off")
fig.tight_layout()
fig.savefig(str(OUT_DIR / "t03_central_projection.png"), dpi=150)
plt.close(fig)
print(f"[Step 1] 保存投影图: {OUT_DIR / 't03_central_projection.png'}")

print("[Step 1] 开始 Katsevich 重建 ...")
rec = reconstruct(
    sinogram_swapped,
    conf,
    vol_geom,
    proj_geom,
    {
        "Diff":     {"Print time": True},
        "FwdRebin": {"Print time": True},
        "BackRebin":{"Print time": True},
        "BackProj": {"Print time": True},
    }
)
print(f"[Step 1] 重建完成，volume shape: {rec.shape}, min={rec.min():.4f}, max={rec.max():.4f}")

# Save 3 slices
im_idx = [rec.shape[2] // 4, rec.shape[2] // 2, rec.shape[2] * 3 // 4]
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle("Katsevich Reconstruction — Simulated Phantom (t03)")
for i, ax in enumerate(axes):
    img = rec[:, :, im_idx[i]]
    im = ax.imshow(img, cmap="gray")
    ax.set_title(f"Slice z={im_idx[i]}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
out_path = str(OUT_DIR / "t03_reconstruction.png")
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"[Step 1] 保存重建图: {out_path}")
print("[Step 1] DONE — 检查 t03_reconstruction.png 看双球是否可见。")
