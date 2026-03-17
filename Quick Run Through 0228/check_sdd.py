"""
检查 L067 DICOM 里 SOD/SDD 实际值，和 Helix2Fan 的读取结果对比
"""
import pydicom, struct, numpy as np, sys
sys.path.insert(0, r'D:\Github\helical-kats')
from pathlib import Path

d = Path(r'D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD_2000')
ds = pydicom.dcmread(str(sorted(d.iterdir())[0]), stop_before_pixels=True)

def f32(ds, tag):
    if tag not in ds: return None
    raw = bytes(ds[tag].value)
    return struct.unpack('<f', raw[:4])[0] if len(raw) >= 4 else None

def fstd(ds, tag):
    if tag not in ds: return None
    try: return float(ds[tag].value)
    except: return None

print("="*60)
print("SOD/SDD 全量对比")
print("="*60)
v_7031_1003 = f32(ds, (0x7031,0x1003))
v_7031_1031 = f32(ds, (0x7031,0x1031))
v_0018_9402 = fstd(ds, (0x0018,0x9402))   # SOD_TAG
v_0018_1110 = fstd(ds, (0x0018,0x1110))   # SDD_TAG

print(f"(7031,1003) DetectorFocalCenterRadialDistance = {v_7031_1003}")
print(f"(7031,1031) ConstantRadialDistance            = {v_7031_1031}")
print(f"(0018,9402) DistanceSrcToPatient (SOD_TAG)    = {v_0018_9402}")
print(f"(0018,1110) DistanceSrcToDetector (SDD_TAG)   = {v_0018_1110}")

print()
print("--- Helix2Fan 读取逻辑 ---")
print(f"  dso (SOD) = (7031,1003) = {v_7031_1003} mm")
print(f"  dsd (SDD) = (7031,1031) = {v_7031_1031} mm")

print()
print("--- Katsevich dicom.py 读取逻辑 ---")
sod_kats = v_0018_9402 if v_0018_9402 is not None else v_7031_1003
sdd_kats = v_0018_1110 if v_0018_1110 is not None else v_7031_1031
print(f"  SOD_TAG (0018,9402) = {v_0018_9402}  ->  fallback (7031,1003) = {v_7031_1003}")
print(f"  SDD_TAG (0018,1110) = {v_0018_1110}  ->  fallback (7031,1031) = {v_7031_1031}")
print(f"  => sod = {sod_kats}  sdd = {sdd_kats}")

print()
print("="*60)
print("du/dv 对比 (像素间距)")
print("="*60)
du_h2f = f32(ds, (0x7029,0x1002))   # Helix2Fan 用
dv_h2f = f32(ds, (0x7029,0x1006))   # Helix2Fan 用
extents = None
if (0x7031,0x1033) in ds:
    raw = bytes(ds[(0x7031,0x1033)].value)
    if len(raw) >= 8:
        extents = struct.unpack('<2f', raw[:8])
print(f"(7029,1002) DetectorElementTransverseSpacing (du) = {du_h2f} mm")
print(f"(7029,1006) DetectorElementAxialSpacing      (dv) = {dv_h2f} mm")
print(f"(7031,1033) DetectorExtents (cols_mm, rows_mm)    = {extents}")

det_rows = int(ds.Columns)
det_cols = int(ds.Rows)
if extents is not None:
    det_pixel_cols = extents[0] / det_cols
    det_pixel_rows = extents[1] / det_rows
    print(f"  => pixel size from extents: cols={det_pixel_cols:.4f} rows={det_pixel_rows:.4f}")
    print(f"  (7029) du={du_h2f}  vs extents/cols={det_pixel_cols:.4f}  MATCH={abs(du_h2f - det_pixel_cols) < 0.01}")

print()
print("="*60)
print("角度 Helix2Fan transform 检查")
print("="*60)
paths = sorted(d.iterdir())[:5]
raw_angles = []
for p in paths:
    ds2 = pydicom.dcmread(str(p), stop_before_pixels=True)
    raw_angles.append(f32(ds2, (0x7031,0x1001)))

ra = np.array(raw_angles)
h2f = -(np.unwrap(ra + np.pi/2)) - np.pi
print(f"Raw angles[:5]:           {ra}")
print(f"Helix2Fan transformed:    {h2f}")
print(f"Katsevich (unwrapped):    {np.unwrap(ra)}")
print(f"Direction (Katsevich): {'decreasing (CW)' if np.mean(np.diff(ra)) < 0 else 'increasing (CCW)'}")
