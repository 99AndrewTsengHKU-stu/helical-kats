"""
可视化 10 层重建结果，并尝试与 GT 对比。
"""
import sys, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pydicom

rec = np.load(str(OUT_DIR / "rec_10slices.npy"))
print(f"重建 shape: {rec.shape}  min={rec.min():.4f}  max={rec.max():.4f}")
# rec shape: (rows=512, cols=512, slices=10)

# ---- 1. 所有 10 层 ----
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle("Katsevich Reconstruction — L067 quarter-dose 2000 views  (512×512×10, 1mm voxel)", fontsize=13)
vmin, vmax = np.percentile(rec[rec != 0], [1, 99])
for i, ax in enumerate(axes.flat):
    img = rec[:, :, i]
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(f"z-slice {i}", fontsize=9)
    ax.axis("off")
fig.tight_layout()
fig.savefig(str(OUT_DIR / "rec_10slices_all.png"), dpi=120)
plt.close(fig)
print(f"保存: rec_10slices_all.png")

# ---- 2. 中间层 vs GT ----
gt_dir = Path(r"D:\AAPM-Data\L067\L067\full_1mm")
mid_slice_z = 5  # 中间层

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle("Katsevich vs GT — Middle Slice Comparison", fontsize=12)

rec_mid = rec[:, :, mid_slice_z]
vmin2, vmax2 = np.percentile(rec_mid[rec_mid != 0], [1, 99])
axes2[0].imshow(rec_mid, cmap="gray", vmin=vmin2, vmax=vmax2)
axes2[0].set_title(f"Katsevich (z-slice {mid_slice_z})\n1-99% windowed [{vmin2:.3f},{vmax2:.3f}]")
axes2[0].axis("off")

# Percentile stretch (full range visible)
vm0, vm1 = rec_mid.min(), rec_mid.max()
axes2[1].imshow(rec_mid, cmap="gray", vmin=vm0, vmax=vm1)
axes2[1].set_title(f"Katsevich (full range)\n[{vm0:.4f},{vm1:.4f}]")
axes2[1].axis("off")

# GT middle slice ~ slice 280 (of 560)
try:
    gt_paths = sorted(p for p in gt_dir.iterdir() if p.suffix in ('.IMA', '.dcm', '.ima', ''))
    if gt_paths:
        gt_mid_path = gt_paths[len(gt_paths) // 2]
        ds_gt = pydicom.dcmread(str(gt_mid_path))
        gt_img = ds_gt.pixel_array.astype(np.float32)
        slope = float(getattr(ds_gt, 'RescaleSlope', 1.0))
        intercept = float(getattr(ds_gt, 'RescaleIntercept', 0.0))
        gt_img = gt_img * slope + intercept  # HU
        axes2[2].imshow(gt_img, cmap="gray", vmin=-1000, vmax=400)
        axes2[2].set_title(f"GT FBP (slice {len(gt_paths)//2} of {len(gt_paths)})\nHU window [-1000, 400]")
        axes2[2].axis("off")
        print(f"GT 图像: {gt_mid_path.name}  min={gt_img.min():.0f}  max={gt_img.max():.0f} HU")
    else:
        axes2[2].text(0.5, 0.5, "No GT files found", ha='center', va='center')
except Exception as e:
    axes2[2].text(0.5, 0.5, f"GT error:\n{e}", ha='center', va='center', fontsize=8)
    print(f"GT error: {e}")

fig2.tight_layout()
fig2.savefig(str(OUT_DIR / "rec_vs_gt.png"), dpi=150)
plt.close(fig2)
print(f"保存: rec_vs_gt.png")
print("DONE")
