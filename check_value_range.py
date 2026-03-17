
import pydicom
import numpy as np
import os
from pathlib import Path

def check_projection_values():
    proj_dir = r"D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD"
    files = list(Path(proj_dir).glob("*.dcm"))
    if not files:
        print("No files found.")
        return

    # Check first few files
    for i in range(5):
        ds = pydicom.dcmread(files[i])
        pixels = ds.pixel_array.astype(np.float32)
        slope = getattr(ds, "RescaleSlope", 1.0)
        intercept = getattr(ds, "RescaleIntercept", 0.0)
        actual_values = pixels * slope + intercept
        
        print(f"File: {files[i].name}")
        print(f"  Shape: {actual_values.shape}")
        print(f"  Range: [{actual_values.min():.2f}, {actual_values.max():.2f}]")
        print(f"  Mean: {actual_values.mean():.2f}")
        
    # If values are large (>1000), it's likely Intensity (I) and needs -log(I/I0)
    # If values are small (0-10), it might already be attenuation.

if __name__ == "__main__":
    check_projection_values()
