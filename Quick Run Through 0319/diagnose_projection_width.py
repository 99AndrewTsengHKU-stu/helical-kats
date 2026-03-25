"""
Fast diagnostic: measure the body width in a SINGLE DICOM projection and
compare with what the geometry predicts.

If the measured body width on the detector matches the predicted width from
SOD/SDD/psize geometry → geometry is correct, bug is elsewhere.
If they disagree → geometry parameters don't describe the data correctly.
"""
import sys, os, struct
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pydicom
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DICOM_DIR = Path(r"D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD")
GT_DIR = Path(r"D:\AAPM-Data\L067\L067\full_1mm")

# ── Load ONE DICOM projection ────────────────────────────────────────
dcm_files = sorted(DICOM_DIR.glob("*.dcm"))
# Pick a file roughly in the middle
mid_idx = len(dcm_files) // 2
ds = pydicom.dcmread(str(dcm_files[mid_idx]))

print(f"File: {dcm_files[mid_idx].name}")
print(f"  Rows={ds.Rows}, Columns={ds.Columns}")
print(f"  InstanceNumber={getattr(ds, 'InstanceNumber', '?')}")

# Extract geometry
def decode_f32(ds, tag, count=1):
    if tag not in ds:
        return None
    raw = bytes(ds[tag].value)
    vals = struct.unpack("<" + "f" * count, raw[:4*count])
    return vals if count > 1 else vals[0]

sod = float(ds[(0x0018, 0x9402)].value) if (0x0018, 0x9402) in ds else decode_f32(ds, (0x7031, 0x1003))
sdd = float(ds[(0x0018, 0x1110)].value) if (0x0018, 0x1110) in ds else decode_f32(ds, (0x7031, 0x1031))
det_extents = decode_f32(ds, (0x7031, 0x1033), count=2)
angle_rad = decode_f32(ds, (0x7031, 0x1001))

print(f"\n  SOD = {sod}")
print(f"  SDD = {sdd}")
print(f"  det_extents = {det_extents}")
print(f"  angle = {angle_rad:.4f} rad = {np.degrees(angle_rad):.1f}°")

det_rows_tag = int(ds.Columns)  # z direction (as in dicom.py)
det_cols_tag = int(ds.Rows)     # fan direction

print(f"  det_rows (z) = {det_rows_tag}")
print(f"  det_cols (fan) = {det_cols_tag}")

# Get pixel data
slope = float(getattr(ds, "RescaleSlope", 1.0))
intercept = float(getattr(ds, "RescaleIntercept", 0.0))
pixels = ds.pixel_array.astype(np.float32) * slope + intercept
print(f"  pixel_array shape = {pixels.shape}, range=[{pixels.min():.1f}, {pixels.max():.1f}]")

# Transpose to (rows_z, cols_fan) convention
proj = pixels.T  # now (det_rows_tag, det_cols_tag) = (64, 736)
print(f"  After .T: proj shape = {proj.shape}")

# ── Measure body width in projection ─────────────────────────────────
# Take the central row (middle of detector in z direction)
mid_row = proj.shape[0] // 2
profile = proj[mid_row, :]

print(f"\n  Central row {mid_row} profile:")
print(f"    range = [{profile.min():.2f}, {profile.max():.2f}]")
print(f"    mean = {profile.mean():.2f}")

# Find body edges: where signal drops below some threshold
# Body has high attenuation → high projection value
# Air has low attenuation → low projection value
threshold = profile.max() * 0.1  # 10% of peak
above = profile > threshold
left_edge = np.argmax(above)
right_edge = len(above) - 1 - np.argmax(above[::-1])
body_width_pixels = right_edge - left_edge
body_center_pixel = (left_edge + right_edge) / 2.0

print(f"\n  Body edges: left={left_edge}, right={right_edge}")
print(f"  Body width = {body_width_pixels} pixels")
print(f"  Body center pixel = {body_center_pixel:.1f} (detector center = {(det_cols_tag-1)/2:.1f})")

# ── Compute geometry predictions ─────────────────────────────────────
# The DICOM data is equi-angular. The detector extent at isocenter is
# det_extents[0] (= 369.625mm for fan direction).
# Each channel subtends: delta_gamma = extent / (n_channels * SOD) rad
# (extent / n_channels is the arc length per channel at isocenter)
det_extent_fan = det_extents[0]  # mm at isocenter
arc_psize_iso = det_extent_fan / det_cols_tag  # arc spacing at isocenter
delta_gamma = arc_psize_iso / sod  # angular step per channel

print(f"\n  arc_psize_iso = {arc_psize_iso:.4f} mm")
print(f"  delta_gamma = {delta_gamma:.6f} rad = {np.degrees(delta_gamma):.4f}°")
print(f"  Total fan angle = {delta_gamma * det_cols_tag:.4f} rad = {np.degrees(delta_gamma * det_cols_tag):.1f}°")
print(f"  Half fan angle = {delta_gamma * det_cols_tag / 2:.4f} rad = {np.degrees(delta_gamma * det_cols_tag / 2):.1f}°")

# From the GT, we know the body diameter.
# Load one GT slice to measure.
gt_files = sorted(GT_DIR.glob("*.IMA")) or sorted(GT_DIR.glob("*.dcm"))
gt_ds = pydicom.dcmread(str(gt_files[len(gt_files)//2]))
gt_slope = float(getattr(gt_ds, "RescaleSlope", 1.0))
gt_intercept = float(getattr(gt_ds, "RescaleIntercept", 0.0))
gt_pix_spacing = [float(x) for x in gt_ds.PixelSpacing]
gt_img = gt_ds.pixel_array.astype(np.float32) * gt_slope + gt_intercept

print(f"\nGT image shape = {gt_img.shape}")
print(f"GT pixel spacing = {gt_pix_spacing} mm")
print(f"GT ReconstructionDiameter = {getattr(gt_ds, 'ReconstructionDiameter', '?')}")
print(f"GT DataCollectionDiameter = {getattr(gt_ds, 'DataCollectionDiameter', '?')}")
print(f"GT range = [{gt_img.min():.0f}, {gt_img.max():.0f}] HU")

# Find body edges in GT
gt_body_mask = gt_img > -500  # body > -500 HU
gt_cols_present = np.any(gt_body_mask, axis=0)
gt_rows_present = np.any(gt_body_mask, axis=1)
gt_left = np.argmax(gt_cols_present)
gt_right = len(gt_cols_present) - 1 - np.argmax(gt_cols_present[::-1])
gt_top = np.argmax(gt_rows_present)
gt_bottom = len(gt_rows_present) - 1 - np.argmax(gt_rows_present[::-1])

gt_width_px = gt_right - gt_left
gt_height_px = gt_bottom - gt_top
gt_width_mm = gt_width_px * gt_pix_spacing[1]
gt_height_mm = gt_height_px * gt_pix_spacing[0]
gt_center_x_px = (gt_left + gt_right) / 2.0
gt_center_y_px = (gt_top + gt_bottom) / 2.0
gt_center_x_mm = (gt_center_x_px - gt_img.shape[1]/2) * gt_pix_spacing[1]
gt_center_y_mm = (gt_center_y_px - gt_img.shape[0]/2) * gt_pix_spacing[0]

print(f"\nGT body bounding box:")
print(f"  width  = {gt_width_px} px = {gt_width_mm:.1f} mm")
print(f"  height = {gt_height_px} px = {gt_height_mm:.1f} mm")
print(f"  center offset from image center = ({gt_center_x_mm:.1f}, {gt_center_y_mm:.1f}) mm")

# ── Predict body width on detector ───────────────────────────────────
# If the body is at isocenter with half-width hw (in the direction perpendicular
# to the X-ray beam at this angle), it projects onto the equi-angular detector.
#
# For a body edge at perpendicular distance R from rotation center:
# In equi-angular: fan_angle = arctan(R / SOD) ≈ R / SOD for small angles
# Channel position = center + fan_angle / delta_gamma

# Use average body radius as proxy
body_radius_mm = (gt_width_mm + gt_height_mm) / 4.0  # average half-width
print(f"\n  Average body radius ≈ {body_radius_mm:.1f} mm")

# Predicted body width on equi-angular detector:
fan_angle_edge = np.arctan(body_radius_mm / sod)
predicted_body_half_width_channels = fan_angle_edge / delta_gamma
predicted_body_width_pixels = 2 * predicted_body_half_width_channels

print(f"\n  Predicted body half-fan-angle = {fan_angle_edge:.4f} rad = {np.degrees(fan_angle_edge):.2f}°")
print(f"  Predicted body width on detector = {predicted_body_width_pixels:.1f} pixels")

print(f"\n  ACTUAL body width on detector = {body_width_pixels} pixels")
print(f"  Ratio (actual/predicted) = {body_width_pixels / predicted_body_width_pixels:.4f}")

# ── Also try with flat-detector pixel size ────────────────────────────
# If the data were already on a flat detector with flat_psize:
j_center = (det_cols_tag - 1) / 2.0
j_indices = np.arange(det_cols_tag, dtype=np.float64)
gamma = (j_indices - j_center) * delta_gamma
u_flat = sdd * np.tan(gamma)
flat_psize = float((u_flat[-1] - u_flat[0]) / (det_cols_tag - 1))

# On flat detector at SDD:
# body edge projects to u_mm = SDD * body_radius / SOD
flat_edge_mm = sdd * body_radius_mm / sod
flat_edge_pixels = flat_edge_mm / flat_psize
flat_body_width = 2 * flat_edge_pixels

print(f"\n  flat_psize = {flat_psize:.4f} mm")
print(f"  Predicted body width (flat detector) = {flat_body_width:.1f} pixels")

# ── What if the data is already "magnification-corrected" at isocenter? ──
# i.e., the projection data represents a detector at isocenter (SOD) not at SDD?
iso_edge_pixels = body_radius_mm / arc_psize_iso
iso_body_width = 2 * iso_edge_pixels

print(f"  Predicted body width (iso detector, no magnification) = {iso_body_width:.1f} pixels")

# ── What if pixel spacing is stored in the DICOM PixelSpacing tag? ────
proj_pix_spacing = getattr(ds, "PixelSpacing", None)
if proj_pix_spacing is not None:
    pps = [float(x) for x in proj_pix_spacing]
    print(f"\n  Projection PixelSpacing = {pps}")
    ps_edge_0 = body_radius_mm / pps[0]
    ps_edge_1 = body_radius_mm / pps[1]
    print(f"  Predicted body width using PixelSpacing[0]={pps[0]}: {2*ps_edge_0:.1f} px")
    print(f"  Predicted body width using PixelSpacing[1]={pps[1]}: {2*ps_edge_1:.1f} px")

# ── Summary ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"  GT body radius    = {body_radius_mm:.1f} mm")
print(f"  SOD = {sod}, SDD = {sdd}")
print(f"  Magnification (SDD/SOD) = {sdd/sod:.4f}")
print(f"")
print(f"  Measured body width on detector: {body_width_pixels} px")
print(f"  Predicted (equi-angular):        {predicted_body_width_pixels:.1f} px")
print(f"  Predicted (flat at SDD):         {flat_body_width:.1f} px")
print(f"  Predicted (iso, no magnify):     {iso_body_width:.1f} px")
print(f"")
print(f"  Ratios (actual/predicted):")
print(f"    equi-angular: {body_width_pixels / predicted_body_width_pixels:.4f}")
print(f"    flat at SDD:  {body_width_pixels / flat_body_width:.4f}")
print(f"    iso (no mag): {body_width_pixels / iso_body_width:.4f}")

# ── Plot ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Projection image
axes[0, 0].imshow(proj, cmap='gray', aspect='auto')
axes[0, 0].axhline(mid_row, color='r', ls='--', alpha=0.5)
axes[0, 0].axvline(left_edge, color='lime', ls='--', alpha=0.7)
axes[0, 0].axvline(right_edge, color='lime', ls='--', alpha=0.7)
axes[0, 0].set_title(f"DICOM projection (view {mid_idx})")
axes[0, 0].set_xlabel("Fan channel")
axes[0, 0].set_ylabel("Z row")

# Profile with edges
axes[0, 1].plot(profile, 'b-', linewidth=0.5)
axes[0, 1].axhline(threshold, color='r', ls=':', label=f'threshold={threshold:.0f}')
axes[0, 1].axvline(left_edge, color='lime', ls='--', label=f'left={left_edge}')
axes[0, 1].axvline(right_edge, color='lime', ls='--', label=f'right={right_edge}')
det_center = (det_cols_tag - 1) / 2
predicted_left = det_center - predicted_body_half_width_channels
predicted_right = det_center + predicted_body_half_width_channels
axes[0, 1].axvline(predicted_left, color='orange', ls='-.', label=f'predicted L={predicted_left:.0f}')
axes[0, 1].axvline(predicted_right, color='orange', ls='-.', label=f'predicted R={predicted_right:.0f}')
axes[0, 1].set_title(f"Profile row {mid_row}")
axes[0, 1].legend(fontsize=7)
axes[0, 1].set_xlabel("Fan channel")

# GT image with body edges
axes[1, 0].imshow(gt_img, cmap='gray', vmin=-200, vmax=300)
axes[1, 0].axvline(gt_left, color='lime', ls='--')
axes[1, 0].axvline(gt_right, color='lime', ls='--')
axes[1, 0].axhline(gt_top, color='lime', ls='--')
axes[1, 0].axhline(gt_bottom, color='lime', ls='--')
axes[1, 0].set_title(f"GT slice (body: {gt_width_mm:.0f}×{gt_height_mm:.0f} mm)")

# Summary text
info = (
    f"Geometry:\n"
    f"  SOD = {sod}\n"
    f"  SDD = {sdd}\n"
    f"  magnification = {sdd/sod:.4f}\n"
    f"  det_cols = {det_cols_tag}\n"
    f"  det_extent_fan = {det_extent_fan:.3f} mm\n"
    f"  arc_psize_iso = {arc_psize_iso:.4f} mm\n"
    f"  delta_gamma = {delta_gamma:.6f} rad\n"
    f"  flat_psize = {flat_psize:.4f} mm\n"
    f"\nBody:\n"
    f"  GT radius = {body_radius_mm:.1f} mm\n"
    f"  Measured width = {body_width_pixels} px\n"
    f"  Predicted (equi-ang) = {predicted_body_width_pixels:.1f} px\n"
    f"  Predicted (flat) = {flat_body_width:.1f} px\n"
    f"  Predicted (iso) = {iso_body_width:.1f} px\n"
    f"\nRatios (actual/predicted):\n"
    f"  equi-ang: {body_width_pixels / predicted_body_width_pixels:.4f}\n"
    f"  flat: {body_width_pixels / flat_body_width:.4f}\n"
    f"  iso: {body_width_pixels / iso_body_width:.4f}\n"
)
axes[1, 1].axis('off')
axes[1, 1].text(0.05, 0.95, info, fontsize=10, transform=axes[1, 1].transAxes,
                verticalalignment='top', fontfamily='monospace')

plt.suptitle("Projection Body Width: Measured vs Predicted from Geometry", fontsize=13)
plt.tight_layout()
out = os.path.join(OUT_DIR, "projection_width_diagnostic.png")
plt.savefig(out, dpi=150)
print(f"\nSaved -> {out}")
print("Done.")
