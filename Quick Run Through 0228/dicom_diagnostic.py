"""
Step 2: DICOM 读取诊断
打印 L067 quarter_DICOM-CT-PD_2000 数据集的关键标签，
判断像素值类型（是否需要 log 校正），并输出摘要到 NOTES.md。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = Path(__file__).resolve().parent

import struct
import numpy as np
import pydicom

DICOM_DIR = Path(r"D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD_2000")
if not DICOM_DIR.exists():
    print(f"[ERROR] 找不到 DICOM 目录: {DICOM_DIR}")
    print("请确认数据路径，或修改此脚本中的 DICOM_DIR 变量。")
    sys.exit(1)

paths = sorted(p for p in DICOM_DIR.iterdir() if p.suffix.lower() == ".dcm")
print(f"[Step 2] 找到 {len(paths)} 个 .dcm 文件")

# Read first 3 and last file
probe_files = paths[:3] + [paths[-1]]

# Private tags we care about
ANGLE_TAG    = (0x7031, 0x1001)
TABLE_POS    = (0x0018, 0x9327)
SOD_TAG      = (0x0018, 0x9402)
SDD_TAG      = (0x0018, 0x1110)
DET_EXT_TAG  = (0x7031, 0x1033)

def decode_f32(ds, tag, count=1):
    if tag not in ds: return None
    raw = bytes(ds[tag].value)
    if len(raw) < 4 * count: return None
    vals = struct.unpack("<" + "f" * count, raw[:4*count])
    return vals if count > 1 else vals[0]

lines = []
def pr(s=""):
    print(s)
    lines.append(s)

pr("=" * 60)
pr("Py-Kat DICOM Diagnostic — 0228")
pr(f"Data dir : {DICOM_DIR}")
pr(f"Total files: {len(paths)}")
pr("=" * 60)

# ---- HEADER INFO (first file) ----
first_path = paths[0]
ds0 = pydicom.dcmread(str(first_path))

pr(f"\n--- First file: {first_path.name} ---")
pr(f"  InstanceNumber    : {getattr(ds0, 'InstanceNumber', 'N/A')}")
pr(f"  Rows × Cols       : {getattr(ds0, 'Rows', '?')} × {getattr(ds0, 'Columns', '?')}")
pr(f"  BitsAllocated     : {getattr(ds0, 'BitsAllocated', '?')}")
pr(f"  PixelRepresentation: {getattr(ds0, 'PixelRepresentation', '?')}")
pr(f"  RescaleSlope      : {getattr(ds0, 'RescaleSlope', 1.0)}")
pr(f"  RescaleIntercept  : {getattr(ds0, 'RescaleIntercept', 0.0)}")
pr(f"  PixelSpacing      : {getattr(ds0, 'PixelSpacing', 'N/A')}")

sod = None
if SOD_TAG in ds0:
    sod = float(ds0[SOD_TAG].value)
elif (0x7031, 0x1003) in ds0:
    sod = decode_f32(ds0, (0x7031, 0x1003))
pr(f"  SOD               : {sod} mm")

sdd = None
if SDD_TAG in ds0:
    sdd = float(ds0[SDD_TAG].value)
elif (0x7031, 0x1031) in ds0:
    sdd = decode_f32(ds0, (0x7031, 0x1031))
pr(f"  SDD               : {sdd} mm")

angle0 = decode_f32(ds0, ANGLE_TAG)
pr(f"  Angle (private)   : {angle0} rad")
table0 = float(ds0[TABLE_POS].value) if TABLE_POS in ds0 else "N/A"
pr(f"  TablePosition     : {table0} mm")

det_ext = decode_f32(ds0, DET_EXT_TAG, count=2)
pr(f"  DetectorExtents   : {det_ext}  (col_mm, row_mm)")

# ---- PIXEL VALUE ANALYSIS ----
pr("\n--- Pixel value analysis (5 files sampled) ---")
sample_paths = paths[::max(1, len(paths)//5)][:5]
all_mins, all_maxes, all_means = [], [], []
for p in sample_paths:
    ds = pydicom.dcmread(str(p))
    pix = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    pix_cal = pix * slope + intercept
    all_mins.append(pix_cal.min())
    all_maxes.append(pix_cal.max())
    all_means.append(pix_cal.mean())
    pr(f"  {p.name}: min={pix_cal.min():.2f}  max={pix_cal.max():.2f}  mean={pix_cal.mean():.2f}")

pr(f"\n  Overall calibrated range: [{min(all_mins):.2f}, {max(all_maxes):.2f}]")

# ---- ANGLE / PITCH ANALYSIS ----
pr("\n--- Angle / Pitch analysis (first vs last 3 views) ---")
meta_head = []
meta_tail = []
for p in paths[:3]:
    ds = pydicom.dcmread(str(p), stop_before_pixels=True)
    ang = decode_f32(ds, ANGLE_TAG)
    tbl = float(ds[TABLE_POS].value) if TABLE_POS in ds else None
    meta_head.append((int(getattr(ds, 'InstanceNumber', 0)), ang, tbl))
for p in paths[-3:]:
    ds = pydicom.dcmread(str(p), stop_before_pixels=True)
    ang = decode_f32(ds, ANGLE_TAG)
    tbl = float(ds[TABLE_POS].value) if TABLE_POS in ds else None
    meta_tail.append((int(getattr(ds, 'InstanceNumber', 0)), ang, tbl))
for inst, ang, tbl in meta_head:
    pr(f"  inst={inst:5d}  angle={ang:.5f} rad  table={tbl} mm")
pr("  ...")
for inst, ang, tbl in meta_tail:
    pr(f"  inst={inst:5d}  angle={ang:.5f} rad  table={tbl} mm")

# Estimate pitch
all_meta = []
for p in paths:
    ds = pydicom.dcmread(str(p), stop_before_pixels=True)
    inst = int(getattr(ds, 'InstanceNumber', 0))
    ang = decode_f32(ds, ANGLE_TAG)
    tbl = float(ds[TABLE_POS].value) if TABLE_POS in ds else None
    all_meta.append((inst, ang, tbl))
all_meta.sort(key=lambda x: x[0])
angles = np.unwrap(np.array([m[1] for m in all_meta]))
tables = np.array([m[2] for m in all_meta]) if all_meta[0][2] is not None else None

if tables is not None:
    dangle = np.diff(angles)
    dtable = np.diff(tables)
    pitch_per_rad = float(np.mean(dtable / dangle))
    pr(f"\n  Mean angle step   : {np.mean(dangle):.6f} rad ({np.degrees(np.mean(dangle)):.4f} deg)")
    pr(f"  Mean pitch        : {pitch_per_rad:.4f} mm/rad  ({pitch_per_rad * 2 * np.pi:.2f} mm/turn)")

# ---- INTERPRETATION ----
pix_max = max(all_maxes)
pix_min = min(all_mins)
pr("\n--- 诊断结论 ---")
if pix_min >= 0 and pix_max > 1000:
    pr("  [判断] 像素值范围大且非负 → 可能是原始计数或已校正的 I 值（非 -ln 形式）")
    pr("  [建议] 需要 log 校正：sino = -log(sino / sino.max()) 或类似处理")
elif -5 < pix_min < 0 and pix_max < 50:
    pr("  [判断] 值域在 (-5, 50) → 大概率已经是线积分（-log 形式），可直接输入 Katsevich")
elif pix_min > -1100 and pix_max < 3000:
    pr("  [判断] 值域类似 HU → 这是重建图像，不是投影数据！检查 DICOM 文件路径是否正确")
else:
    pr(f"  [判断] 值域 [{pix_min:.1f}, {pix_max:.1f}]，需要人工确认数据类型")

# Save notes
with open(str(OUT_DIR / "NOTES.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
pr(f"\n[Step 2] 诊断报告保存至: {OUT_DIR / 'NOTES.md'}")
