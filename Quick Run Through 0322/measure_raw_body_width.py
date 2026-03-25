"""
Measure body width on RAW equi-angular DICOM (no resampling) and compare
with predicted width from geometry tags. This avoids circular dependency
on delta_gamma that contaminated the flat-resampled comparison.
"""
import sys, os, struct
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pydicom
from pathlib import Path

DICOM_DIR = Path(r"D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD")
GT_DIR = Path(r"D:\AAPM-Data\L067\L067\full_1mm")

def decode_f32(ds, tag, count=1):
    if tag not in ds:
        return None
    raw = bytes(ds[tag].value)
    vals = struct.unpack("<" + "f" * count, raw[:4*count])
    return vals if count > 1 else vals[0]

# ── Load geometry from DICOM ──
dcm_files = sorted(DICOM_DIR.glob("*.dcm"))
ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
sod = float(ds[(0x0018, 0x9402)].value) if (0x0018, 0x9402) in ds else decode_f32(ds, (0x7031, 0x1003))
sdd = float(ds[(0x0018, 0x1110)].value) if (0x0018, 0x1110) in ds else decode_f32(ds, (0x7031, 0x1031))
det_extents = decode_f32(ds, (0x7031, 0x1033), count=2)
det_rows = int(ds.Columns)
det_cols = int(ds.Rows)

det_extent_fan = det_extents[0]  # 369.625 mm
arc_psize_iso = det_extent_fan / det_cols
delta_gamma = arc_psize_iso / sod

print(f"SOD={sod}, SDD={sdd}")
print(f"Detector: {det_rows} rows x {det_cols} cols")
print(f"Det extent fan: {det_extent_fan} mm (tag says 'at isocenter')")
print(f"arc_psize_iso = {arc_psize_iso:.6f} mm")
print(f"delta_gamma = {delta_gamma:.6e} rad")
print(f"Total fan angle = {delta_gamma * det_cols:.4f} rad = {np.degrees(delta_gamma * det_cols):.2f} deg")

# ── Load a projection near z-center ──
mid_idx = len(dcm_files) // 2
ds_proj = pydicom.dcmread(str(dcm_files[mid_idx]))
proj_raw = ds_proj.pixel_array.astype(np.float32)
slope = float(getattr(ds_proj, 'RescaleSlope', 1.0))
intercept = float(getattr(ds_proj, 'RescaleIntercept', 0.0))
proj_raw = proj_raw * slope + intercept
proj_raw = proj_raw.T  # → (det_rows, det_cols)

print(f"\nLoaded projection {mid_idx}, shape={proj_raw.shape}")
print(f"  Value range: [{proj_raw.min():.4f}, {proj_raw.max():.4f}]")

# ── Measure body width on RAW equi-angular data ──
mid_row = det_rows // 2
profile_raw = proj_raw[mid_row, :]

# Normalize
p_min, p_max = np.percentile(profile_raw[profile_raw > 0], [5, 95])
profile_norm = np.clip((profile_raw - p_min) / max(p_max - p_min, 1e-10), 0, 1)

# Find body edges at multiple thresholds
print(f"\n{'='*60}")
print(f"RAW EQUI-ANGULAR BODY WIDTH (center row {mid_row})")
print(f"{'='*60}")
for thresh in [0.05, 0.10, 0.15, 0.20]:
    above = profile_norm > thresh
    if not above.any():
        continue
    left = np.argmax(above)
    right = len(above) - 1 - np.argmax(above[::-1])
    width = right - left
    center = (left + right) / 2.0

    # Angular extent
    angular_extent = width * delta_gamma
    # Predicted body diameter from this angular extent
    body_diam_predicted = 2 * sod * np.sin(angular_extent / 2)

    print(f"  thresh={thresh:.2f}: left={left}, right={right}, width={width}px, center={center:.1f}")
    print(f"    angular extent = {angular_extent:.4f} rad = {np.degrees(angular_extent):.2f} deg")
    print(f"    implied body diam = {body_diam_predicted:.1f} mm")

# ── GT body diameter for reference ──
gt_files = sorted(GT_DIR.glob("*.IMA")) or sorted(GT_DIR.glob("*.dcm"))
ds_gt = pydicom.dcmread(str(gt_files[len(gt_files)//2]))
gt_pix = [float(x) for x in ds_gt.PixelSpacing]
gt_recon_diam = float(getattr(ds_gt, 'ReconstructionDiameter', 340))
gt_dcd = float(getattr(ds_gt, 'DataCollectionDiameter', 500))
gt_img = ds_gt.pixel_array.astype(np.float32) * float(getattr(ds_gt, 'RescaleSlope', 1)) + float(getattr(ds_gt, 'RescaleIntercept', 0))

# Measure GT body width
gt_mid = gt_img[gt_img.shape[0]//2, :]
gt_min, gt_max = np.percentile(gt_img, [2, 98])
gt_norm = np.clip((gt_mid - gt_min) / max(gt_max - gt_min, 1e-10), 0, 1)
gt_above = gt_norm > 0.10
gt_left = np.argmax(gt_above)
gt_right = len(gt_above) - 1 - np.argmax(gt_above[::-1])
gt_body_px = gt_right - gt_left
gt_body_mm = gt_body_px * gt_pix[0]

print(f"\n{'='*60}")
print(f"GROUND TRUTH BODY SIZE")
print(f"{'='*60}")
print(f"  ReconstructionDiameter: {gt_recon_diam} mm")
print(f"  DataCollectionDiameter: {gt_dcd} mm")
print(f"  PixelSpacing: {gt_pix[0]:.6f} mm")
print(f"  Body width (thresh=0.10): {gt_body_px} px = {gt_body_mm:.1f} mm")

# ── What delta_gamma SHOULD be for the body to match? ──
print(f"\n{'='*60}")
print(f"GEOMETRY CONSISTENCY CHECK")
print(f"{'='*60}")

# Use thresh=0.10 measurement
above = profile_norm > 0.10
left = np.argmax(above)
right = len(above) - 1 - np.argmax(above[::-1])
raw_body_width_px = right - left

# The body half-angle from source
body_half_angle = np.arcsin(gt_body_mm / 2 / sod)
body_full_angle = 2 * body_half_angle
print(f"  GT body diameter: {gt_body_mm:.1f} mm")
print(f"  Body half-angle from source: {np.degrees(body_half_angle):.2f} deg")
print(f"  Body full angle: {body_full_angle:.4f} rad = {np.degrees(body_full_angle):.2f} deg")

# What delta_gamma gives the correct body width?
delta_gamma_correct = body_full_angle / raw_body_width_px
arc_psize_correct = delta_gamma_correct * sod
det_extent_correct = arc_psize_correct * det_cols

print(f"\n  Raw DICOM body width: {raw_body_width_px} px")
print(f"  delta_gamma needed: {delta_gamma_correct:.6e} rad")
print(f"  arc_psize needed: {arc_psize_correct:.6f} mm")
print(f"  det_extent needed: {det_extent_correct:.1f} mm")
print(f"\n  Current delta_gamma: {delta_gamma:.6e} rad")
print(f"  Current det_extent: {det_extent_fan:.1f} mm")
print(f"  Ratio (correct/current): {delta_gamma_correct/delta_gamma:.4f}")
print(f"  → This should ≈ {gt_body_mm / (raw_body_width_px * delta_gamma * sod / np.sin(body_half_angle * 2 / 2)):.4f}")

# ── Also check: what if det_extent is at DETECTOR (SDD) not isocenter? ──
delta_gamma_at_det = arc_psize_iso / sdd
body_width_predicted_at_det = body_full_angle / delta_gamma_at_det
print(f"\n  If det_extent is at DETECTOR (SDD):")
print(f"    delta_gamma = {delta_gamma_at_det:.6e} rad")
print(f"    predicted body width = {body_width_predicted_at_det:.0f} px (actual: {raw_body_width_px} px)")

# ── Check DataCollectionDiameter consistency ──
dcd_half_angle = np.arcsin(gt_dcd / 2 / sod)
dcd_full_angle = 2 * dcd_half_angle
dcd_pixels = dcd_full_angle / delta_gamma
print(f"\n  DataCollectionDiameter = {gt_dcd} mm:")
print(f"    full angle = {dcd_full_angle:.4f} rad = {np.degrees(dcd_full_angle):.2f} deg")
print(f"    pixels (current delta_gamma) = {dcd_pixels:.0f} (detector has {det_cols})")
print(f"    pixels (correct delta_gamma) = {dcd_full_angle/delta_gamma_correct:.0f}")

# Total fan angle
total_fan_current = delta_gamma * det_cols
total_fan_correct = delta_gamma_correct * det_cols
print(f"\n  Total fan angle (current): {np.degrees(total_fan_current):.2f} deg")
print(f"  Total fan angle (correct): {np.degrees(total_fan_correct):.2f} deg")
print(f"  DCD from current fan: {2*sod*np.sin(total_fan_current/2):.1f} mm")
print(f"  DCD from correct fan: {2*sod*np.sin(total_fan_correct/2):.1f} mm")
print(f"  GT DCD tag: {gt_dcd} mm")
