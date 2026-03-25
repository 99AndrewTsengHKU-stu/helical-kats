"""Quick calc: how many slices = 1 helical turn?"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from pykatsevich import load_dicom_projections

DICOM_DIR = r"D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD"

# Only need metadata, but load_dicom_projections reads all pixels.
# Instead, just read a few DICOM headers manually.
from pathlib import Path
import pydicom, struct

paths = sorted(Path(DICOM_DIR).glob("*.dcm"))[:100]
angles = []
table_z = []
for p in paths:
    ds = pydicom.dcmread(p, stop_before_pixels=True)
    raw = bytes(ds[(0x7031, 0x1001)].value)
    ang = struct.unpack("<f", raw[:4])[0]
    angles.append(ang)
    if (0x7031, 0x1002) in ds:
        z = struct.unpack("<f", bytes(ds[(0x7031, 0x1002)].value)[:4])[0]
        table_z.append(z)
    elif (0x0018, 0x9327) in ds:
        table_z.append(float(ds[(0x0018, 0x9327)].value))

angles = np.unwrap(np.array(angles))
table_z = np.array(table_z[:len(angles)])

angle_step = np.mean(np.diff(angles))  # rad/projection
z_step = np.mean(np.diff(table_z))     # mm/projection
projs_per_turn = 2 * np.pi / abs(angle_step)
z_per_turn = projs_per_turn * abs(z_step)  # mm per full turn
slices_per_turn = z_per_turn / 0.8  # at 0.8mm/slice

print(f"angle_step = {angle_step:.6f} rad/proj")
print(f"z_step = {z_step:.6f} mm/proj")
print(f"projs_per_turn = {projs_per_turn:.1f}")
print(f"z_per_turn = {z_per_turn:.2f} mm")
print(f"slices_per_turn = {slices_per_turn:.1f} (at 0.8mm/slice)")
print(f"\nObserved period = 29 slices = {29*0.8:.1f} mm")
print(f"Match: {'YES' if abs(slices_per_turn - 29) < 2 else 'NO'}")
