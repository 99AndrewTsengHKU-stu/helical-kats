"""
Diagnose the remaining 2.3% scale offset after the geometry fix.
Check: center offset, SOD/SDD accuracy, col_coords alignment, z-position match.
"""
import sys, os, struct
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pydicom
from pathlib import Path
from pykatsevich.dicom import (
    _decode_float32_tag, _get_float,
    DETECTOR_COL_SPACING_TAG, DETECTOR_ROW_SPACING_TAG,
    DETECTOR_CENTRAL_ELEMENT_TAG, DETECTOR_SHAPE_TAG,
    SOD_TAG, SDD_TAG,
)

DICOM_DIR = Path(r"D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD")
GT_DIR = Path(r"D:\AAPM-Data\L067\L067\full_1mm")

# ── 1. Read geometry from first DICOM ────────────────────────────────
dcm_files = sorted(DICOM_DIR.glob("*.dcm"))
ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)

sod = _get_float(ds, SOD_TAG, fallback=_decode_float32_tag(ds, (0x7031, 0x1003)))
sdd = _get_float(ds, SDD_TAG, fallback=_decode_float32_tag(ds, (0x7031, 0x1031)))
psize_col = float(_decode_float32_tag(ds, DETECTOR_COL_SPACING_TAG))
psize_row = float(_decode_float32_tag(ds, DETECTOR_ROW_SPACING_TAG))
central = _decode_float32_tag(ds, DETECTOR_CENTRAL_ELEMENT_TAG, count=2)
j_center = float(central[0])
i_center = float(central[1])

det_rows = int(ds.Columns)  # transposed in loader
det_cols = int(ds.Rows)

print("=" * 60)
print("GEOMETRY FROM DICOM TAGS")
print("=" * 60)
print(f"SOD = {sod} mm")
print(f"SDD = {sdd} mm")
print(f"Magnification = {sdd/sod:.4f}")
print(f"Detector: {det_rows} rows x {det_cols} cols")
print(f"psize_col (at det) = {psize_col} mm")
print(f"psize_row (at det) = {psize_row} mm")
print(f"Central element: col={j_center}, row={i_center}")
print(f"Geometric center: col={(det_cols-1)/2}, row={(det_rows-1)/2}")
print(f"Center offset: {j_center - (det_cols-1)/2:.3f} pixels = {(j_center - (det_cols-1)/2)*psize_col:.3f} mm")

# ── 2. Compute derived geometry ──────────────────────────────────────
delta_gamma = psize_col / sdd
print(f"\ndelta_gamma = {delta_gamma:.6e} rad")

# Flat resampling
j_indices = np.arange(det_cols, dtype=np.float64)
gamma = (j_indices - j_center) * delta_gamma
u_flat = sdd * np.tan(gamma)
flat_psize = (u_flat[-1] - u_flat[0]) / (det_cols - 1)
print(f"flat_psize_cols = {flat_psize:.6f} mm")

# DCD
half_fan = j_center * delta_gamma  # to the short side
half_fan_max = (det_cols - 1 - j_center) * delta_gamma  # to the long side
dcd_short = 2 * sod * np.sin(half_fan)
dcd_long = 2 * sod * np.sin(half_fan_max)
dcd_geom_center = 2 * sod * np.sin((det_cols - 1) / 2 * delta_gamma)
print(f"\nDCD (using j_center={j_center}): short side = {dcd_short:.1f} mm")
print(f"DCD (using j_center={j_center}): long side  = {dcd_long:.1f} mm")
print(f"DCD (using geom center {(det_cols-1)/2}): {dcd_geom_center:.1f} mm")

# ── 3. Check col_coords center mismatch ─────────────────────────────
print("\n" + "=" * 60)
print("COL_COORDS CENTER ANALYSIS")
print("=" * 60)

# create_configuration uses geom center for col_coords
col_coords_current = flat_psize * (np.arange(det_cols + 1) - 0.5 * (det_cols - 1))
print(f"col_coords center (current): {col_coords_current[(det_cols)//2]:.4f} mm")
print(f"col_coords range: [{col_coords_current[0]:.2f}, {col_coords_current[-1]:.2f}] mm")

# After resampling, u=0 (isocenter) is at pixel:
u_target_0 = u_flat[0]
j_isocenter = -u_target_0 / flat_psize
print(f"\nIsocenter pixel in resampled grid: {j_isocenter:.2f}")
print(f"Geometric center pixel: {(det_cols-1)/2:.1f}")
print(f"Offset: {j_isocenter - (det_cols-1)/2:.2f} pixels = {(j_isocenter - (det_cols-1)/2) * flat_psize:.2f} mm")

# The col_coords should be centered at j_isocenter, not (det_cols-1)/2
# This means the Katsevich filter sees shifted column coordinates
# Physical offset at detector:
center_offset_mm = (j_isocenter - (det_cols - 1) / 2) * flat_psize
print(f"\nKatsevich filter center offset: {center_offset_mm:.3f} mm")
print(f"As fraction of half-detector: {center_offset_mm / (flat_psize * det_cols / 2) * 100:.2f}%")

# ── 4. What SOD/SDD gives scale=1.0 ─────────────────────────────────
print("\n" + "=" * 60)
print("PARAMETER SENSITIVITY")
print("=" * 60)
observed_scale = 1.0244
print(f"Observed optimal scale: {observed_scale}")
print(f"Recon is {(observed_scale-1)*100:.1f}% too small")
print()

# If scale = 1.024 means recon needs 2.4% upscale,
# the backprojection magnification is effectively too high by 2.4%
# Physical size at isocenter = pixel_on_det * SOD / SDD
# Larger SOD → larger physical size → larger recon
sod_needed = sod * observed_scale
print(f"SOD needed for scale=1.0: {sod_needed:.1f} mm (current {sod})")
print(f"  → SOD increase: {sod_needed - sod:.1f} mm ({(observed_scale-1)*100:.1f}%)")

sdd_needed = sdd / observed_scale
print(f"SDD needed for scale=1.0: {sdd_needed:.1f} mm (current {sdd})")
print(f"  → SDD decrease: {sdd - sdd_needed:.1f} mm ({(1-1/observed_scale)*100:.1f}%)")

psize_needed = psize_col * observed_scale
print(f"psize needed for scale=1.0: {psize_needed:.4f} mm (current {psize_col})")

# ── 5. Check SOD from all available tags ─────────────────────────────
print("\n" + "=" * 60)
print("SOD/SDD FROM ALL AVAILABLE TAGS")
print("=" * 60)

sod_tags = [
    ((0x0018, 0x9402), "PositionOfIsocenterProjection"),
    ((0x7031, 0x1003), "DistanceSourceToIsocenter (private)"),
    ((0x0018, 0x1111), "DistanceSourceToPatient"),
]
for tag, name in sod_tags:
    val = _get_float(ds, tag)
    if val is None:
        val_dec = _decode_float32_tag(ds, tag)
        if val_dec is not None:
            print(f"  {tag} {name}: {val_dec} (decoded float32)")
        else:
            print(f"  {tag} {name}: NOT PRESENT")
    else:
        print(f"  {tag} {name}: {val}")

sdd_tags = [
    ((0x0018, 0x1110), "DistanceSourceToDetector"),
    ((0x7031, 0x1031), "DistanceSourceToDetector (private)"),
]
for tag, name in sdd_tags:
    val = _get_float(ds, tag)
    if val is None:
        val_dec = _decode_float32_tag(ds, tag)
        if val_dec is not None:
            print(f"  {tag} {name}: {val_dec} (decoded float32)")
        else:
            print(f"  {tag} {name}: NOT PRESENT")
    else:
        print(f"  {tag} {name}: {val}")

# Check other potentially relevant tags
extra_tags = [
    ((0x0018, 0x9306), "SingleCollimationWidth"),
    ((0x0018, 0x9307), "TotalCollimationWidth"),
    ((0x0018, 0x9311), "SpiralPitchFactor"),
    ((0x0018, 0x1100), "ReconstructionDiameter"),
    ((0x0018, 0x0090), "DataCollectionDiameter"),
]
print("\nOther tags:")
for tag, name in extra_tags:
    val = _get_float(ds, tag)
    if val is not None:
        print(f"  {tag} {name}: {val}")

# ── 6. Check GT z-position vs reconstruction z-position ──────────────
print("\n" + "=" * 60)
print("Z-POSITION MATCH CHECK")
print("=" * 60)
gt_files = sorted(GT_DIR.glob("*.IMA")) or sorted(GT_DIR.glob("*.dcm"))
print(f"GT files: {len(gt_files)}")

if len(gt_files) > 280:
    gt_ds = pydicom.dcmread(str(gt_files[280]), stop_before_pixels=True)
    gt_pos = getattr(gt_ds, 'ImagePositionPatient', None)
    gt_spacing = getattr(gt_ds, 'PixelSpacing', None)
    gt_thickness = getattr(gt_ds, 'SliceThickness', None)
    gt_recon_diam = _get_float(gt_ds, (0x0018, 0x1100))
    gt_dcd = _get_float(gt_ds, (0x0018, 0x0090))
    print(f"GT slice 280:")
    print(f"  ImagePositionPatient: {[float(x) for x in gt_pos] if gt_pos else 'N/A'}")
    print(f"  PixelSpacing: {[float(x) for x in gt_spacing] if gt_spacing else 'N/A'}")
    print(f"  SliceThickness: {gt_thickness}")
    print(f"  ReconstructionDiameter: {gt_recon_diam}")
    print(f"  DataCollectionDiameter: {gt_dcd}")

    # Also check first and last GT slices for z-range
    gt_ds0 = pydicom.dcmread(str(gt_files[0]), stop_before_pixels=True)
    gt_dsN = pydicom.dcmread(str(gt_files[-1]), stop_before_pixels=True)
    z0 = float(getattr(gt_ds0, 'ImagePositionPatient', [0,0,0])[2])
    zN = float(getattr(gt_dsN, 'ImagePositionPatient', [0,0,0])[2])
    z280 = float(gt_pos[2]) if gt_pos else 0
    print(f"\n  GT z-range: [{z0:.1f}, {zN:.1f}] mm")
    print(f"  GT slice 280 z: {z280:.1f} mm")
    print(f"  GT z-step: {(zN - z0) / (len(gt_files) - 1):.3f} mm")

# ── 7. Check if (0018,9402) is really SOD ────────────────────────────
print("\n" + "=" * 60)
print("TAG (0018,9402) INVESTIGATION")
print("=" * 60)
tag_9402 = (0x0018, 0x9402)
if tag_9402 in ds:
    elem = ds[tag_9402]
    print(f"  VR: {elem.VR}")
    print(f"  Value: {elem.value}")
    print(f"  Name/keyword: {elem.keyword if hasattr(elem, 'keyword') else 'unknown'}")
    # Try reading as different types
    raw = bytes(elem.value) if not isinstance(elem.value, (int, float, str)) else None
    if raw:
        if len(raw) >= 4:
            as_float32 = struct.unpack("<f", raw[:4])[0]
            print(f"  As float32: {as_float32}")
        if len(raw) >= 8:
            as_float64 = struct.unpack("<d", raw[:8])[0]
            print(f"  As float64: {as_float64}")
else:
    print("  Tag (0018,9402) NOT PRESENT in this DICOM file")
    print(f"  SOD came from fallback tag (7031,1003)")
    val = _decode_float32_tag(ds, (0x7031, 0x1003))
    print(f"  (7031,1003) = {val}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"For a 2.4% scale correction:")
print(f"  Option A: SOD = {sod_needed:.1f} (currently {sod})")
print(f"  Option B: SDD = {sdd_needed:.1f} (currently {sdd})")
print(f"  Option C: psize_col = {psize_needed:.4f} (currently {psize_col})")
print(f"  Option D: col_coords center offset = {center_offset_mm:.2f} mm")
print(f"\nMost likely cause: TBD based on above diagnostics")
