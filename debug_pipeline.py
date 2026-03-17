"""
Debug Katsevich reconstruction pipeline to identify scaling issues.
Check values at each stage: sinogram -> differentiate -> hilbert -> backproject.
"""
import sys
from pathlib import Path
import numpy as np
import pydicom
from pykatsevich import load_dicom_projections
from pykatsevich.geometry import astra_helical_views
from pykatsevich.initialize import create_configuration
from pykatsevich.filter import differentiate, hilbert_filter
import astra

# --- Load L067 quarter dataset ---
dicom_dir = Path(r"D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD")
print(f"Loading DICOM from {dicom_dir}...")

proj_array, geometry = load_dicom_projections(str(dicom_dir))
angle_array = geometry['angles']

print(f"Projections shape: {proj_array.shape}")
print(f"Angles shape: {angle_array.shape}")
print(f"Sinogram range: [{proj_array.min():.6f}, {proj_array.max():.6f}]")
print(f"Angle range: [{angle_array.min():.4f}, {angle_array.max():.4f}] rad")

# --- Decimate to reduce compute ---
DECIM = 10
proj_dec = proj_array[::DECIM, :, :]
angle_dec = angle_array[::DECIM]
print(f"\nAfter decimation ({DECIM}x):")
print(f"  Projections: {proj_dec.shape}")
print(f"  Angles: {angle_dec.shape}")

# --- Create configuration ---
print("\nCreating configuration...")
conf = create_configuration(
    geometry=geometry,
    angle_array=angle_dec,
    image_rows=512,
    image_cols=512,
    image_slices=560,
    voxel_size=0.664,
)

print(f"delta_s (angular sampling): {conf['delta_s']:.6f} rad")
print(f"pixel_height: {conf['pixel_height']:.6f} mm")
print(f"pixel_span: {conf['pixel_span']:.6f} mm")
print(f"source_distance: {conf['source_distance']:.2f} mm")
print(f"detector_distance: {conf['detector_distance']:.2f} mm")

# --- Stage 1: Raw sinogram ---
print("\n=== STAGE 1: Raw Sinogram ===")
print(f"Range: [{proj_dec.min():.6f}, {proj_dec.max():.6f}]")
print(f"Mean: {proj_dec.mean():.6f}")

# Take a single view and check it
view_0 = proj_dec[0, :, :]
print(f"View 0: min={view_0.min():.6f}, max={view_0.max():.6f}, mean={view_0.mean():.6f}")

# --- Stage 2: Differentiate ---
print("\n=== STAGE 2: Differentiate ===")
proj_deriv = differentiate(proj_dec, conf, tqdm_bar=False)
print(f"Range: [{proj_deriv.min():.6f}, {proj_deriv.max():.6f}]")
print(f"Mean: {proj_deriv.mean():.6f}")
deriv_0 = proj_deriv[0, :, :]
print(f"View 0: min={deriv_0.min():.6f}, max={deriv_0.max():.6f}, mean={deriv_0.mean():.6f}")

# --- Stage 3: Hilbert filter ---
print("\n=== STAGE 3: Hilbert Filter ===")
proj_hilbert = hilbert_filter(proj_deriv, conf)
print(f"Range: [{proj_hilbert.min():.6f}, {proj_hilbert.max():.6f}]")
print(f"Mean: {proj_hilbert.mean():.6f}")
hil_0 = proj_hilbert[0, :, :]
print(f"View 0: min={hil_0.min():.6f}, max={hil_0.max():.6f}, mean={hil_0.mean():.6f}")

# --- Stage 4: Apply cone-angle weighting ---
print("\n=== STAGE 4: Cone-angle weighting ===")
# Apply sine of angle weighting (standard cone-beam correction)
detector_rows = proj_hilbert.shape[1]
detector_cols = proj_hilbert.shape[2]
row_coords = conf['row_coords']
pixel_height = conf['pixel_height']

# Compute cone angles: tan(theta) = row / SDD
sdd = conf['detector_distance'] - conf['source_distance']
row_coords_mm = (row_coords - detector_rows / 2) * pixel_height
cone_angles = np.arctan(row_coords_mm / sdd)

# Apply weighting: multiply by cos^3(angle)
weights = np.cos(cone_angles) ** 3
print(f"Cone angle weights: min={weights.min():.6f}, max={weights.max():.6f}")

proj_weighted = proj_hilbert * weights[np.newaxis, :-2, np.newaxis]
print(f"After weighting:")
print(f"  Range: [{proj_weighted.min():.6f}, {proj_weighted.max():.6f}]")
print(f"  Mean: {proj_weighted.mean():.6f}")

print("\n=== SUMMARY ===")
print("If the issue is in scaling:")
print(f"  - Check delta_s value: {conf['delta_s']:.6f}")
print(f"  - Check row/col coordinate ranges")
print(f"  - Check if differentiate is dividing by correct factor")
print(f"\nIf all intermediate values look reasonable but final is wrong:")
print(f"  - Issue is likely in backprojection normalization")
