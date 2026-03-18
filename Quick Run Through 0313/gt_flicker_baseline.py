"""
Compute flicker metric on GT DICOM slices for comparison baseline.
If GT also has ~23% adjacent-slice difference, then our metric is normal anatomy variation,
not artifact flickering.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pydicom
from pathlib import Path

GT_DIR = Path(r"D:\AAPM-Data\L067\L067\full_1mm")

# Load GT slices 276-285 (same range as our test)
paths = sorted(list(GT_DIR.glob("*.dcm")) + list(GT_DIR.glob("*.IMA")))
print(f"GT: {len(paths)} files")

# Read all GT slices and compute flicker
print("\nReading all GT slices...")
gt_slices = []
z_positions = []
for p in paths:
    ds = pydicom.dcmread(p)
    img = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    img = img * slope + intercept
    gt_slices.append(img)
    ipp = getattr(ds, 'ImagePositionPatient', None)
    z = float(ipp[2]) if ipp else float(getattr(ds, 'SliceLocation', 0))
    z_positions.append(z)

gt_vol = np.stack(gt_slices, axis=0)  # (560, 512, 512)
z_positions = np.array(z_positions)
print(f"GT volume: {gt_vol.shape}, dtype={gt_vol.dtype}")
print(f"GT value range: [{gt_vol.min():.1f}, {gt_vol.max():.1f}] (Hounsfield units)")
print(f"GT z-spacing: {np.mean(np.abs(np.diff(z_positions))):.4f} mm")

# Compute flicker metric for all adjacent pairs
print("\nComputing GT flicker metrics...")
fm_all = []
for i in range(len(gt_slices) - 1):
    s0 = gt_vol[i]
    s1 = gt_vol[i + 1]
    diff = np.mean(np.abs(s1 - s0))
    base = 0.5 * (np.mean(np.abs(s0)) + np.mean(np.abs(s1)))
    fm_all.append(diff / max(base, 1e-12))

fm_all = np.array(fm_all)

# Focus on slices 276-285 (same as our test)
fm_test = fm_all[276:285]
print(f"\nGT flicker (slices 276-285): {fm_test}")
print(f"GT flicker mean (276-285): {fm_test.mean():.6f}")
print(f"GT flicker max  (276-285): {fm_test.max():.6f}")

print(f"\nGT flicker overall:")
print(f"  mean: {fm_all.mean():.6f}")
print(f"  std:  {fm_all.std():.6f}")
print(f"  min:  {fm_all.min():.6f}")
print(f"  max:  {fm_all.max():.6f}")

# Check for periodic patterns
print(f"\nGT flicker at chunk boundaries (every 64 slices):")
for boundary in [63, 127, 191, 255, 319, 383, 447, 511]:
    if boundary < len(fm_all):
        # Show flicker at boundary and neighbors
        start = max(0, boundary - 2)
        end = min(len(fm_all), boundary + 3)
        print(f"  Around slice {boundary}: {fm_all[start:end]}")

# Our reconstruction vs GT comparison
print(f"\n{'='*60}")
print(f"COMPARISON")
print(f"{'='*60}")
print(f"Our reconstruction (VOXEL_SIZE_Z=0.8, TD=0.025):")
print(f"  Mean flicker (276-285): 0.229097")
print(f"GT baseline:")
print(f"  Mean flicker (276-285): {fm_test.mean():.6f}")
ratio = 0.229097 / max(fm_test.mean(), 1e-12)
print(f"  Ratio (ours/GT): {ratio:.2f}x")
if ratio > 2.0:
    print(f"  [!] Our flicker is {ratio:.1f}x higher than GT -- artifact present")
elif ratio > 1.5:
    print(f"  [!] Our flicker is {ratio:.1f}x higher than GT -- mild artifact")
else:
    print(f"  [OK] Our flicker is within normal range of GT variation")

print("\nDone.")
