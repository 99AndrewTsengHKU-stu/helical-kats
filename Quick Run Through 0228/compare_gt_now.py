import numpy as np
import matplotlib.pyplot as plt
import pydicom
from pathlib import Path

OUT    = Path(r'D:\Github\helical-kats\Quick Run Through 0228')
GT_DIR = Path(r'D:\AAPM-Data\L067\L067\full_1mm')

rec = np.load(str(OUT / 'rec_signed_delta.npy'))[:, :, 0]
print(f'rec shape={rec.shape}  range=[{rec.min():.4f}, {rec.max():.4f}]')

gt_files = sorted(GT_DIR.glob('*.IMA'))
print(f'GT slices: {len(gt_files)}')

ds0 = pydicom.dcmread(str(gt_files[0]))
slope     = float(getattr(ds0, 'RescaleSlope', 1.0))
intercept = float(getattr(ds0, 'RescaleIntercept', 0.0))
print(f'GT RescaleSlope={slope}  Intercept={intercept}')
print(f'GT SliceThickness={getattr(ds0, "SliceThickness", "?")}  PixelSpacing={getattr(ds0, "PixelSpacing", "?")}')

# GT middle slice
mid = len(gt_files) // 2
ds_mid = pydicom.dcmread(str(gt_files[mid]))
gt_img = ds_mid.pixel_array.astype(np.float32) * slope + intercept
print(f'GT HU range: [{gt_img.min():.0f}, {gt_img.max():.0f}]')
print(f'GT file: {gt_files[mid].name}')

# -- Plot --
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Katsevich Recon vs GT (middle slice)', fontsize=13)

vmin_rec, vmax_rec = np.percentile(rec, [1, 99])
axes[0].imshow(rec, cmap='gray', vmin=vmin_rec, vmax=vmax_rec)
axes[0].set_title(f'Katsevich (signed delta_s)\n1-99%=[{vmin_rec:.3f}, {vmax_rec:.3f}]')
axes[0].axis('off')

# Standard CT lung window
axes[1].imshow(gt_img, cmap='gray', vmin=-1000, vmax=200)
axes[1].set_title(f'GT FBP (lung window)\n[-1000, 200] HU  —  slice {mid}')
axes[1].axis('off')

vmin_gt, vmax_gt = np.percentile(gt_img, [1, 99])
axes[2].imshow(gt_img, cmap='gray', vmin=vmin_gt, vmax=vmax_gt)
axes[2].set_title(f'GT FBP (auto window)\n1-99%=[{vmin_gt:.0f}, {vmax_gt:.0f}] HU')
axes[2].axis('off')

fig.tight_layout()
out_path = str(OUT / 'rec_vs_gt.png')
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f'Saved: {out_path}')
