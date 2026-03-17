
import pydicom
import numpy as np
from PIL import Image
import os
import glob

def compare_slices():
    # Reconstruction path
    rec_path = r"d:\Github\helical-kats\rec_L067_FULL_FIXED.npy"
    # GT directory
    gt_dir = r"D:\AAPM-Data\L067\L067\full_1mm"

    # 1. Load reconstruction
    rec = np.load(rec_path)
    # rec is (rows, cols, slices)
    mid_slice_idx = rec.shape[2] // 2
    rec_slice = rec[:, :, mid_slice_idx]
    
    # 2. Extract GT middle slice
    # Need to find the file with index ~280
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.IMA")))
    if not gt_files:
        print("No GT files found!")
        return
        
    gt_file = gt_files[len(gt_files)//2]
    print(f"Using GT file: {os.path.basename(gt_file)}")
    ds = pydicom.dcmread(gt_file)
    gt_slice = ds.pixel_array.astype(np.float32)
    
    # Apply RescaleSlope/Intercept to GT if needed (HU)
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    gt_slice = gt_slice * slope + intercept

    # Normalize for comparison
    def normalize(img):
        vmin, vmax = np.percentile(img, [1, 99])
        if vmax > vmin:
            img = np.clip(img, vmin, vmax)
            img = (img - vmin) / (vmax - vmin)
        return img

    rec_norm = normalize(rec_slice)
    gt_norm = normalize(gt_slice)

    # Save PNGs
    Image.fromarray((rec_norm * 255).astype(np.uint8)).save("slice_rec_final.png")
    Image.fromarray((gt_norm * 255).astype(np.uint8)).save("slice_gt_ref.png")
    
    # Combined side-by-side
    combined = np.hstack([gt_norm, rec_norm])
    Image.fromarray((combined * 255).astype(np.uint8)).save("compare_aapm.png")
    print("Saved comparison image: compare_aapm.png")

if __name__ == "__main__":
    compare_slices()
