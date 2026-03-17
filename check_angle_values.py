
import pydicom
import struct
import numpy as np
import os

def check_angles():
    dicom_dir = r"D:\1212_High_Pitch_argparse\C001\C001_bundle_nview_div4_pitch0.6\dcm_proj"
    files = sorted([f for f in os.listdir(dicom_dir) if f.endswith('.dcm')])
    if not files: return
    
    first = os.path.join(dicom_dir, files[0])
    ds = pydicom.dcmread(first)
    
    tag = (0x7031, 0x1001) # Private Angle
    
    # Read first few angles
    angles = []
    for f in files[:10]:
        ds = pydicom.dcmread(os.path.join(dicom_dir, f), stop_before_pixels=True)
        if tag in ds:
            raw = bytes(ds[tag].value)
            val = struct.unpack("<f", raw[:4])[0]
            angles.append(val)
            
    print("First 10 Angles:", angles)
    angles_arr = np.array(angles)
    diffs = np.diff(angles_arr)
    print("Diffs:", diffs)
    print("Mean step:", np.mean(diffs))
    
    # Check if range implies degrees (e.g. > 2pi)
    # Read last file
    last = os.path.join(dicom_dir, files[-1])
    ds_last = pydicom.dcmread(last, stop_before_pixels=True)
    raw = bytes(ds_last[tag].value)
    last_angle = struct.unpack("<f", raw[:4])[0]
    
    print(f"First Angle: {angles[0]}, Last Angle: {last_angle}")
    print(f"Total Range: {last_angle - angles[0]}")

if __name__ == "__main__":
    check_angles()
