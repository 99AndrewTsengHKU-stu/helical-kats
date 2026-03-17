"""
Diagnose two remaining issues:
A. Does the .T transpose in dicom.py cause pixel size swap?
B. Does Katsevich filter z-mapping diverge from actual table positions?
"""
import sys, os, struct
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pydicom
from pathlib import Path

DICOM_DIR = Path(r"D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD")

print("=" * 60)
print("CHECK A: .T transpose and pixel size assignment")
print("=" * 60)

# Read first DICOM file
paths = sorted(p for p in DICOM_DIR.iterdir() if p.suffix.lower() == ".dcm")
ds = pydicom.dcmread(paths[0])

# Raw DICOM dimensions
print(f"\n  DICOM Rows attribute:    {ds.Rows}")
print(f"  DICOM Columns attribute: {ds.Columns}")
print(f"  pixel_array.shape:       {ds.pixel_array.shape}")
print(f"  pixel_array.shape[0] = DICOM Rows = {ds.pixel_array.shape[0]}")
print(f"  pixel_array.shape[1] = DICOM Columns = {ds.pixel_array.shape[1]}")

# After .T
transposed = ds.pixel_array.T
print(f"\n  After .T: shape = {transposed.shape}")
print(f"  transposed axis 0 (code 'rows') = DICOM Columns = {transposed.shape[0]}")
print(f"  transposed axis 1 (code 'cols') = DICOM Rows    = {transposed.shape[1]}")

# Code's naming:
det_rows = int(ds.Columns)  # code line 188
det_cols = int(ds.Rows)     # code line 189
print(f"\n  Code det_rows = Columns = {det_rows}  (z/height direction)")
print(f"  Code det_cols = Rows    = {det_cols}  (fan/horizontal direction)")

# Detector extents tag
DETECTOR_EXTENTS_TAG = (0x7029, 0x1002)
def _decode_float32_tag(dataset, tag, count=1):
    if tag not in dataset:
        return None
    raw = dataset[tag].value
    if isinstance(raw, (int, float)):
        return float(raw) if count == 1 else None
    raw_bytes = bytes(raw)
    vals = struct.unpack(f"<{count}f", raw_bytes[:count * 4])
    return vals[0] if count == 1 else list(vals)

det_extents = _decode_float32_tag(ds, DETECTOR_EXTENTS_TAG, count=2)
print(f"\n  DETECTOR_EXTENTS_TAG raw: {det_extents}")
if det_extents:
    ext0, ext1 = det_extents
    print(f"  det_extents[0] = {ext0:.2f} mm")
    print(f"  det_extents[1] = {ext1:.2f} mm")
    print(f"\n  If extents[0]=fan, extents[1]=z:")
    print(f"    fan psize = {ext0}/{det_cols} = {ext0/det_cols:.6f} mm/pixel (for {det_cols} cols)")
    print(f"    z psize   = {ext1}/{det_rows} = {ext1/det_rows:.6f} mm/pixel (for {det_rows} rows)")
    print(f"\n  If extents[0]=z, extents[1]=fan (SWAPPED!):")
    print(f"    z psize   = {ext0}/{det_rows} = {ext0/det_rows:.6f} mm/pixel (for {det_rows} rows)")
    print(f"    fan psize = {ext1}/{det_cols} = {ext1/det_cols:.6f} mm/pixel (for {det_cols} cols)")

    # Code's actual assignment (line 194):
    # det_length_cols_mm, det_length_rows_mm = det_extents
    det_length_cols_mm = ext0
    det_length_rows_mm = ext1
    SOD_TAG = (0x0018, 0x1110)
    SDD_TAG = (0x0018, 0x1111)
    sod = float(ds[SOD_TAG].value) if SOD_TAG in ds else _decode_float32_tag(ds, (0x7031, 0x1003))
    sdd = float(ds[SDD_TAG].value) if SDD_TAG in ds else _decode_float32_tag(ds, (0x7031, 0x1031))
    print(f"\n  SOD = {sod:.2f}, SDD = {sdd:.2f}")

    arc_psize_cols = det_length_cols_mm / det_cols
    arc_psize_rows = det_length_rows_mm / det_rows
    print(f"\n  arc_psize_cols (code) = extents[0]/{det_cols} = {arc_psize_cols:.6f}")
    print(f"  arc_psize_rows (code) = extents[1]/{det_rows} = {arc_psize_rows:.6f}")

    # After equiangular conversion for cols
    delta_gamma = arc_psize_cols / sod
    j_center = (det_cols - 1) / 2.0
    j_indices = np.arange(det_cols, dtype=np.float64)
    gamma = (j_indices - j_center) * delta_gamma
    u_flat = sdd * np.tan(gamma)
    flat_psize_cols = float((u_flat[-1] - u_flat[0]) / (det_cols - 1))

    # For rows: magnification scaling
    det_pixel_size_cols = flat_psize_cols  # fan direction
    det_pixel_size_rows = float(arc_psize_rows * sdd / sod)  # z direction

    print(f"\n  FINAL pixel sizes (as code computes):")
    print(f"    det_pixel_size_cols (fan) = {det_pixel_size_cols:.6f} mm")
    print(f"    det_pixel_size_rows (z)   = {det_pixel_size_rows:.6f} mm")

    # Sanity: total detector extent at isocenter
    print(f"\n  Sanity check:")
    print(f"    Total fan extent at detector = {det_cols} * {det_pixel_size_cols:.4f} = {det_cols * det_pixel_size_cols:.1f} mm")
    print(f"    Total z extent at detector   = {det_rows} * {det_pixel_size_rows:.4f} = {det_rows * det_pixel_size_rows:.1f} mm")
    print(f"    Fan angle span = 2*atan({det_cols*det_pixel_size_cols/2}/{sdd}) = {2*np.degrees(np.arctan(det_cols*det_pixel_size_cols/2/sdd)):.1f} deg")
    print(f"    Z cone angle   = 2*atan({det_rows*det_pixel_size_rows/2}/{sdd}) = {2*np.degrees(np.arctan(det_rows*det_pixel_size_rows/2/sdd)):.1f} deg")

print()
print("=" * 60)
print("CHECK B: Katsevich filter z-mapping vs actual table positions")
print("=" * 60)

from pykatsevich import load_dicom_projections
print("\nLoading DICOM metadata (quick, no pixels)...")
# We only need angles and table positions — read from meta we already have
meta_list = []
for path in paths[:100]:  # just first 100 for speed
    d = pydicom.dcmread(path, stop_before_pixels=True)
    instance = int(getattr(d, "InstanceNumber", 0))
    angle_raw = d[(0x7031, 0x1001)].value
    angle = struct.unpack('<f', bytes(angle_raw)[:4])[0]
    table_tag = (0x0018, 0x9327)
    table_pos = float(d[table_tag].value) if table_tag in d else None
    meta_list.append((instance, angle, table_pos))

meta_list.sort(key=lambda x: x[0])
angles_raw = np.array([m[1] for m in meta_list])
table_raw = np.array([m[2] for m in meta_list if m[2] is not None])
angles_unwrapped = np.unwrap(angles_raw).astype(np.float32)

if len(table_raw) == len(angles_raw):
    print(f"  First 100 projections analyzed")
    print(f"  Angle range: [{angles_unwrapped[0]:.4f}, {angles_unwrapped[-1]:.4f}] rad")
    print(f"  Table range: [{table_raw[0]:.2f}, {table_raw[-1]:.2f}] mm")

    # Linear fit: table = a * angle + b
    coeffs = np.polyfit(angles_unwrapped, table_raw, 1)
    table_linear = np.polyval(coeffs, angles_unwrapped)
    residuals = table_raw - table_linear
    print(f"\n  Linear fit: table = {coeffs[0]:.6f} * angle + {coeffs[1]:.2f}")
    print(f"  pitch_mm_per_rad from fit = {coeffs[0]:.6f}")

    # Katsevich uses: z = source_pos * progress_per_radian
    # After negate: source_pos = -angle - pi/2
    source_pos = -angles_unwrapped - np.pi/2
    # progress_per_radian from DICOM
    pitch_mm_per_angle = float(np.mean(np.abs(np.diff(table_raw))))
    pitch_mm_per_rad = float(np.mean(np.abs(np.diff(table_raw) / np.diff(angles_unwrapped))))
    print(f"  pitch_mm_per_rad from mean(diff) = {pitch_mm_per_rad:.6f}")

    # Katsevich z-mapping (what the filter sees)
    filter_z = source_pos * pitch_mm_per_rad
    filter_z_centered = filter_z - filter_z.mean()

    # Actual table z-mapping (what CuPy uses)
    table_z_centered = table_raw - table_raw.mean()

    # Compare
    diff = filter_z_centered - table_z_centered
    print(f"\n  Filter z (centered) range: [{filter_z_centered[0]:.4f}, {filter_z_centered[-1]:.4f}]")
    print(f"  Table z (centered) range:  [{table_z_centered[0]:.4f}, {table_z_centered[-1]:.4f}]")
    print(f"\n  Difference (filter_z - table_z):")
    print(f"    max|diff|: {np.max(np.abs(diff)):.6f} mm")
    print(f"    mean|diff|: {np.mean(np.abs(diff)):.6f} mm")
    print(f"    std(diff): {np.std(diff):.6f} mm")

    # Check for systematic drift (linear trend in residuals)
    drift_coeffs = np.polyfit(np.arange(len(diff)), diff, 1)
    print(f"    Linear drift: {drift_coeffs[0]:.8f} mm/projection")
    print(f"    Over 100 projections: {drift_coeffs[0]*100:.4f} mm")
    print(f"    Over all 48590 projections: {drift_coeffs[0]*48590:.2f} mm")

    # Also check: linearity of table positions
    print(f"\n  Table position linearity:")
    print(f"    max|residual from linear fit|: {np.max(np.abs(residuals)):.6f} mm")
    print(f"    std(residual): {np.std(residuals):.6f} mm")
    if np.max(np.abs(residuals)) > 0.1:
        print(f"    [!] Table positions are NOT perfectly linear!")
    else:
        print(f"    [OK] Table positions are very linear")

    # Quadratic fit to see if there's acceleration
    coeffs2 = np.polyfit(angles_unwrapped, table_raw, 2)
    print(f"\n  Quadratic fit: a2={coeffs2[0]:.8f}, a1={coeffs2[1]:.6f}, a0={coeffs2[2]:.2f}")
    if abs(coeffs2[0]) > 1e-6:
        print(f"    [!] Non-zero quadratic term suggests non-constant pitch")
        print(f"    Pitch drift over full scan: ~{coeffs2[0] * (angles_unwrapped[-1]-angles_unwrapped[0]):.4f} mm/rad")
    else:
        print(f"    [OK] Pitch is constant (quadratic term negligible)")

else:
    print(f"  [WARN] Could not match all table positions")

print("\nDone.")
