"""
Step 3: 单层重建测试
用 run_dicom_recon.py 重建第 280 层（L067 GT 共 560 层，取中间层）。
输出保存到 Quick Run Through 0228 文件夹。
"""

import sys
import subprocess
from pathlib import Path

python_exe = r"C:\Users\Andrew\anaconda3\envs\MNIST\python.exe"
script     = r"D:\Github\helical-kats\tests\run_dicom_recon.py"
out_dir    = Path(__file__).resolve().parent

dicom_dir = r"D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD_2000"
out_npy   = str(out_dir / "rec_1slice.npy")

print("=" * 60)
print("Step 3: 单层重建（中间层）")
print("=" * 60)

cmd = [
    python_exe, script,
    "--dicom-dir", dicom_dir,
    "--rows",  "512",
    "--cols",  "512",
    "--slices", "1",
    "--voxel-size", "1.0",
    "--save-npy", out_npy,
]

print("命令:", " ".join(cmd))
print()
result = subprocess.run(cmd, capture_output=False)

if result.returncode != 0:
    print("\n[ERROR] 重建失败，返回码:", result.returncode)
    sys.exit(1)

print("\n[Step 3] 重建成功，开始可视化 ...")

import numpy as np
from matplotlib import pyplot as plt

rec = np.load(out_npy)
print(f"[Step 3] 重建 shape: {rec.shape}, min={rec.min():.4f}, max={rec.max():.4f}")

# rec shape is (rows, cols, slices) = (512, 512, 1)
img = rec[:, :, 0]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Step 3: Single-Slice Reconstruction (L067, 2000 views, 1 slice)")

# Full range
im0 = axes[0].imshow(img, cmap="gray")
axes[0].set_title("Full range")
axes[0].axis("off")
fig.colorbar(im0, ax=axes[0])

# Percentile windowed
vmin, vmax = np.percentile(img, [1, 99])
im1 = axes[1].imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
axes[1].set_title(f"1-99% windowed [{vmin:.3f}, {vmax:.3f}]")
axes[1].axis("off")
fig.colorbar(im1, ax=axes[1])

out_png = str(out_dir / "rec_1slice.png")
fig.tight_layout()
fig.savefig(out_png, dpi=150)
plt.close(fig)
print(f"[Step 3] 图像已保存: {out_png}")
print("[Step 3] DONE — 检查 rec_1slice.png，确认是否有旋涡伪影。")
