"""
TD smoothing 对比测试：0.025 / 0.05 / 0.10 三个值
以及螺距验证（进给量是否合理）
"""
import subprocess, sys, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

PYTHON = r"C:\Users\Andrew\anaconda3\envs\MNIST\python.exe"
SCRIPT = r"D:\Github\helical-kats\tests\run_dicom_recon.py"
DICOM  = r"D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD_2000"
OUT    = Path(r"D:\Github\helical-kats\Quick Run Through 0228")

td_values = [0.025, 0.05, 0.10]
npy_paths = [OUT / f"rec_td{str(v).replace('.','')}.npy" for v in td_values]

for td, npy in zip(td_values, npy_paths):
    print(f"\n{'='*60}")
    print(f"Running td_smoothing={td} ...")
    cmd = [
        PYTHON, SCRIPT,
        "--dicom-dir", DICOM,
        "--rows", "512", "--cols", "512", "--slices", "1",
        "--voxel-size", "1.0",
        "--td-smoothing", str(td),
        "--save-npy", str(npy),
    ]
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"ERROR: td={td} failed")
    else:
        print(f"td={td} DONE -> {npy.name}")

# 可视化三张对比图
print("\nVisualizing ...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("T-D Smoothing Comparison — Quarter-2000 views, 1 slice", fontsize=13)

for ax, td, npy in zip(axes, td_values, npy_paths):
    rec = np.load(str(npy))[:, :, 0]
    vmin, vmax = np.percentile(rec, [1, 99])
    ax.imshow(rec, cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(f"T-D smoothing = {td}\n1-99% [{vmin:.3f}, {vmax:.3f}]")
    ax.axis('off')

fig.tight_layout()
out_png = OUT / "td_smoothing_compare.png"
fig.savefig(str(out_png), dpi=150)
plt.close(fig)
print(f"Saved: {out_png.name}")
