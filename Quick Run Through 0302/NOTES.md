# Run Through 0302 - Katsevich Reconstruction Debugging

## Key Finding: Detector Pixel Size Misinterpretation

The DICOM tag `(0x7031, 0x1033)` stores detector extents **at isocenter**, not at
the physical detector plane. The Katsevich pipeline (differentiate, rebinning,
ASTRA backprojection) operates in physical detector coordinates.

### The Problem
- Tag returns: (369.625, 32.5) mm
- Old pixel_size: 369.625 / 736 = **0.502 mm** (isocenter)
- FOV at isocenter: 736 * 0.502 * SOD/SDD = **202.6 mm** (too small!)
- Body appeared at ~60% of image width

### The Fix (dicom.py)
Multiply tag-derived pixel sizes by magnification `SDD/SOD = 1.824`:
- New pixel_size: 0.502 * 1.824 = **0.916 mm** (physical detector)
- FOV at isocenter: 736 * 0.916 * SOD/SDD = **369.6 mm** (correct!)
- Body now appears at ~75-80% of image width, matching GT

### Evidence
1. **Test case (test03.yaml)** uses physical detector pixel_size directly
2. **Backprojection scaling** (filter.py:860) explicitly divides pixel_size by
   magnification: `(pixel_size / (sdd/sod))^2` - expects physical input
3. **Differentiate formula** (filter.py:108-109) uses `col_coords / dia` where
   `dia = SDD` - col_coords must be in physical detector units

## Reconstruction Results

### Before fix (rec_L067_raw.npy)
- Value range: [-0.105, 0.204] mm^-1
- Detector height: 32.32 mm, T-D window: 73.9%
- Body ~60% of image, anatomy barely visible

### After fix (rec_L067_fixed.npy)
- Value range: [-0.079, 0.125] mm^-1
- Detector height: 58.97 mm, T-D window: 46.5%
- Body ~75-80% of image, anatomy clearly visible

### Comparison with GT (auto-calibrated HU)
- Katsevich: mean=-712 HU, std=537, 1-99%: [-1589, 913]
- GT:        mean=-475 HU, std=515, 1-99%: [-1020, 674]
- Auto-calibrated mu_water: 0.0264 mm^-1 (expected ~0.019)

## Remaining Issues
1. **Ring/streak artifacts** - concentric circles visible, likely from 10x decimation
2. **HU calibration offset** - mu_water estimated too high (0.026 vs 0.019)
3. **Body slightly smaller than GT** - might need further geometry tuning
4. **Value range still wider than GT** - extreme values from decimation undersampling

## Files in this folder
- `compare_soft_tissue.png` - Final comparison, soft tissue window [-200, 300] HU
- `compare_lung.png` - Final comparison, lung window [-1000, 200] HU
- `compare_auto.png` - Final comparison, auto percentile window
- `compare_2x_soft_tissue.png` - Earlier 2x scaling test (failed)
- `compare_fixed_*.png` - Intermediate comparison results
- `compare_katsevich_gt.py` - Comparison script (snapshot)
- `debug_pipeline.py` - Pipeline debugging script

## Command used
```bash
~/anaconda3/envs/MNIST/python.exe tests/run_dicom_recon.py \
  --dicom-dir "D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD" \
  --rows 512 --cols 512 --slices 560 --voxel-size 0.664 \
  --decimate 10 --save-npy rec_L067_fixed.npy
```
