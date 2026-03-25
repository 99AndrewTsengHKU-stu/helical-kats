# -----------------------------------------------------------------------
# This file is part of Pykatsevich distribution (https://github.com/astra-toolbox/helical-kats).
# Copyright (c) 2024 imec-Vision Lab, University of Antwerp.
# 
# This program is free software: you can redistribute it and/or modify  
# it under the terms of the GNU General Public License as published by  
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License 
# along with this program. If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------

"""
Utilities for reading projection DICOM series into NumPy arrays together with
the scanner geometry needed by the Katsevich pipeline.

The loader assumes a private tag (0x7031, 0x1001) containing the source angle
in radians for every view and uses the standard `Table Position` tag for the
axial shift. This matches the projection set stored under
`D:\\1212_High_Pitch_argparse\\L067\\L067_bundle_nview_div4_pitch1.5\\dcm_proj`.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pydicom

# Private tag that stores the source angle in radians.
ANGLE_TAG = (0x7031, 0x1001)
# Standard Table Position tag (mm).
TABLE_POSITION_TAG = (0x0018, 0x9327)
# Private tag: DetectorFocalCenterAxialPosition = z_0, the z location of the
# detector's focal center (mm). Defined in DICOM-CT-PD User Manual Table 2.
# This is the per-view z position for helical scans when the standard tag is absent.
DETECTOR_AXIAL_POSITION_TAG = (0x7031, 0x1002)
# Standard distance source to isocenter and distance source to detector tags.
SOD_TAG = (0x0018, 0x9402)
SDD_TAG = (0x0018, 0x1110)
# Per DICOM-CT-PD User Manual v12, Table 1:
# (7031,1033) DetectorCentralElement: (col_idx, row_idx) of the detector
#   element aligning with the isocenter and the detector's focal center.
DETECTOR_CENTRAL_ELEMENT_TAG = (0x7031, 0x1033)
# (7029,1002) DetectorElementTransverseSpacing: width of each detector column,
#   measured at the detector (mm).
DETECTOR_COL_SPACING_TAG = (0x7029, 0x1002)
# (7029,1006) DetectorElementAxialSpacing: width of each detector row,
#   measured at the detector (mm).
DETECTOR_ROW_SPACING_TAG = (0x7029, 0x1006)
# (7029,100B) DetectorShape: "CYLINDRICAL", "SPHERICAL", or "FLAT".
DETECTOR_SHAPE_TAG = (0x7029, 0x100B)
# (7033,100B) Flying Focal Spot z-offset (mm) per view.
# (7033,100E) indicates FFS mode, e.g. "FFSZ" = Flying Focal Spot in Z.
FFS_Z_OFFSET_TAG = (0x7033, 0x100B)
FFS_MODE_TAG = (0x7033, 0x100E)


def _decode_float32_tag(ds: pydicom.Dataset, tag: Tuple[int, int], count: int = 1):
    """
    Decode a private tag that stores little-endian IEEE754 floats.
    """
    if tag not in ds:
        return None
    raw = bytes(ds[tag].value)
    needed = 4 * count
    if len(raw) < needed:
        return None
    values = struct.unpack("<" + "f" * count, raw[:needed])
    return values if count > 1 else values[0]


def _get_float(ds: pydicom.Dataset, tag: Tuple[int, int], fallback: float | None = None):
    """
    Try to read a numeric value from a DICOM tag, optionally returning a fallback.
    """
    if tag in ds:
        try:
            return float(ds[tag].value)
        except Exception:
            pass
    return fallback


def load_dicom_projections(
    dicom_dir: Union[str, Path]
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Load a directory with projection DICOM files into a sinogram and extract
    the associated scan geometry.

    The returned NumPy array has the shape (views, detector_rows, detector_cols)
    that the rest of the package expects. Pixel values are converted to float32
    and rescale slope/intercept are applied. The detector rows/cols follow the
    Katsevich code convention (rows along z, cols along the fan).

    Parameters
    ----------
    dicom_dir : str or Path
        Path to the directory that contains only the projection DICOM files.

    Returns
    -------
    projections : ndarray
        Array with shape (n_views, detector_rows, detector_cols).
    metadata : dict
        Dictionary with keys:
            - angles_rad: unwrapped source angles in radians.
            - table_positions_mm: table positions in mm, ordered by view.
            - angle_step_rad: mean angular step (signed).
            - pitch_mm_per_angle: axial shift per projection.
            - pitch_mm_per_rad: pitch magnitude in mm per radian.
            - pitch_mm_per_rad_signed: signed pitch (matches the raw tags).
            - detector_pixel_size_rows_mm / detector_pixel_size_cols_mm.
            - scan_geometry: dict ready for initialize.create_configuration().
    """
    dicom_dir = Path(dicom_dir)
    if not dicom_dir.exists():
        raise FileNotFoundError(f"{dicom_dir} does not exist")

    paths = sorted(p for p in dicom_dir.iterdir() if p.suffix.lower() == ".dcm")
    if not paths:
        raise FileNotFoundError(f"No .dcm files found in {dicom_dir}")

    # Read lightweight metadata first so we can sort by InstanceNumber.
    meta = []
    table_positions_available = True
    axial_positions_available = True  # fallback: (7031,1002) DetectorFocalCenterAxialPosition
    for path in paths:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        if ANGLE_TAG not in ds:
            raise ValueError(f"{path.name} is missing private angle tag {ANGLE_TAG}")
        if TABLE_POSITION_TAG not in ds:
            table_positions_available = False
        if DETECTOR_AXIAL_POSITION_TAG not in ds:
            axial_positions_available = False

        instance = int(getattr(ds, "InstanceNumber", 0))
        angle = _decode_float32_tag(ds, ANGLE_TAG)
        # Primary: standard table position tag
        if TABLE_POSITION_TAG in ds:
            table_pos = float(ds[TABLE_POSITION_TAG].value)
        # Secondary: DetectorFocalCenterAxialPosition per DICOM-CT-PD User Manual
        elif DETECTOR_AXIAL_POSITION_TAG in ds:
            table_pos = _decode_float32_tag(ds, DETECTOR_AXIAL_POSITION_TAG)
        else:
            table_pos = None
        # Flying focal spot z-offset (mm)
        ffs_z = _decode_float32_tag(ds, FFS_Z_OFFSET_TAG)
        if ffs_z is None:
            ffs_z = 0.0
        meta.append((instance, path, angle, table_pos, ffs_z))

    meta.sort(key=lambda item: item[0])
    ordered_paths = [item[1] for item in meta]
    angles = np.asarray([item[2] for item in meta], dtype=np.float64)
    angles_unwrapped = np.unwrap(angles).astype(np.float32)
    ffs_z_offsets = np.asarray([item[4] for item in meta], dtype=np.float32)

    # Use table positions if any source gave us valid data
    has_table = any(item[3] is not None for item in meta)
    if has_table:
        table_positions = np.asarray(
            [item[3] if item[3] is not None else 0.0 for item in meta],
            dtype=np.float64
        )
    else:
        table_positions = None

    if table_positions is not None:
        table_positions = table_positions.astype(np.float32)

    angle_step = float(np.mean(np.diff(angles_unwrapped)))
    first_ds = pydicom.dcmread(ordered_paths[0], stop_before_pixels=True)

    sod = _get_float(first_ds, SOD_TAG, fallback=_decode_float32_tag(first_ds, (0x7031, 0x1003)))
    sdd = _get_float(first_ds, SDD_TAG, fallback=_decode_float32_tag(first_ds, (0x7031, 0x1031)))
    if sod is None or sdd is None:
        raise ValueError("Missing SOD/SDD tags; cannot build scan geometry")

    det_rows = int(first_ds.Columns)
    det_cols = int(first_ds.Rows)

    if table_positions is not None:
        pitch_mm_per_angle = float(np.mean(np.diff(table_positions)))
        pitch_mm_per_rad_signed = float(
            np.mean(np.diff(table_positions) / np.diff(angles_unwrapped))
        )
    else:
        # Fallback: derive pitch from Spiral Pitch Factor and detector collimation (height).
        pitch_factor = _get_float(first_ds, (0x0018, 0x9311))
        row_spacing = _decode_float32_tag(first_ds, DETECTOR_ROW_SPACING_TAG)
        n_rows_val = int(struct.unpack("<H", bytes(first_ds[(0x7029, 0x1010)].value))[0]) \
            if (0x7029, 0x1010) in first_ds else det_rows
        if row_spacing is not None:
            collimation_mm = float(n_rows_val * row_spacing * sod / sdd)
        else:
            collimation_mm = None
        if pitch_factor is None or collimation_mm is None:
            pitch_mm_per_rad_signed = 0.0
        else:
            pitch_mm_per_rad_signed = float(pitch_factor * collimation_mm / (2 * np.pi))
            # Match sign of angular increment.
            pitch_mm_per_rad_signed *= np.sign(angle_step) if angle_step != 0 else 1.0
        table_positions = pitch_mm_per_rad_signed * (angles_unwrapped - angles_unwrapped[0])
        pitch_mm_per_angle = float(np.mean(np.diff(table_positions)))

    # Read detector element spacing (measured at detector) from DICOM-CT-PD tags.
    psize_col_at_det = _decode_float32_tag(first_ds, DETECTOR_COL_SPACING_TAG)
    psize_row_at_det = _decode_float32_tag(first_ds, DETECTOR_ROW_SPACING_TAG)
    central_element = _decode_float32_tag(first_ds, DETECTOR_CENTRAL_ELEMENT_TAG, count=2)
    det_shape_raw = first_ds[DETECTOR_SHAPE_TAG].value if DETECTOR_SHAPE_TAG in first_ds else None
    det_shape = bytes(det_shape_raw).decode("ascii", errors="ignore").strip() \
        if det_shape_raw is not None else None

    resample_equiangular = False
    if psize_col_at_det is not None:
        psize_col_at_det = float(psize_col_at_det)
        psize_row_at_det = float(psize_row_at_det) if psize_row_at_det is not None else psize_col_at_det

        # Centre pixel index from DetectorCentralElement, fallback to geometric centre.
        j_center = float(central_element[0]) if central_element is not None else (det_cols - 1) / 2.0
        i_center = float(central_element[1]) if central_element is not None else (det_rows - 1) / 2.0

        is_cylindrical = det_shape is not None and "CYLINDRICAL" in det_shape.upper()

        if is_cylindrical:
            # For a cylindrical detector the element spacing at the detector
            # subtends angle delta_gamma = psize_at_det / SDD.
            delta_gamma = psize_col_at_det / sdd

            # Build equi-angular -> flat mapping (same structure as before).
            j_indices = np.arange(det_cols, dtype=np.float64)
            gamma = (j_indices - j_center) * delta_gamma
            u_flat = sdd * np.tan(gamma)

            flat_psize_cols = float((u_flat[-1] - u_flat[0]) / (det_cols - 1))
            u_target = u_flat[0] + np.arange(det_cols, dtype=np.float64) * flat_psize_cols

            j_src = j_center + np.arctan(u_target / sdd) / delta_gamma
            resample_equiangular = True

            det_pixel_size_cols = flat_psize_cols
            det_pixel_size_rows = psize_row_at_det  # at detector plane

            # Column center offset after rebinning: central ray (u=0) pixel index
            # vs geometric center.
            j_central_rebinned = float(-u_flat[0] / flat_psize_cols)
            det_col_offset = j_central_rebinned - (det_cols - 1) / 2.0
        else:
            # FLAT detector — pixel spacing is directly the flat pitch.
            det_pixel_size_cols = psize_col_at_det
            det_pixel_size_rows = psize_row_at_det
            det_col_offset = j_center - (det_cols - 1) / 2.0

        # Row center offset: DICOM central element vs geometric center.
        det_row_offset = i_center - (det_rows - 1) / 2.0
    else:
        pixel_spacing = getattr(first_ds, "PixelSpacing", None)
        if pixel_spacing is not None and len(pixel_spacing) >= 2:
            det_pixel_size_rows = float(pixel_spacing[1])
            det_pixel_size_cols = float(pixel_spacing[0])
        else:
            det_pixel_size_cols = det_pixel_size_rows = 1.0
        det_col_offset = 0.0
        det_row_offset = 0.0

    projections = np.empty((len(ordered_paths), det_rows, det_cols), dtype=np.float32)
    for idx, path in enumerate(ordered_paths):
        ds = pydicom.dcmread(path)
        try:
            pixels = ds.pixel_array.astype(np.float32)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to decode pixel data for {path.name}. "
                "Some datasets use compressed or raw-data storage that requires an external decoder "
                "(e.g., install gdcm via `conda install -c conda-forge gdcm` or "
                "`pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg`)."
            ) from exc
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        pixels = pixels * slope + intercept
        proj = pixels.T  # transpose so shape matches (rows, cols)

        if resample_equiangular:
            # Resample each row from equi-angular to flat-detector grid
            from scipy.ndimage import map_coordinates
            # j_src contains the fractional column indices in the source grid
            # We need to interpolate proj[row, :] at these positions for every row
            col_coords = np.broadcast_to(j_src[np.newaxis, :], (det_rows, det_cols))
            row_coords = np.broadcast_to(
                np.arange(det_rows, dtype=np.float64)[:, np.newaxis],
                (det_rows, det_cols),
            )
            proj = map_coordinates(
                proj.astype(np.float64), [row_coords, col_coords],
                order=3, mode='constant', cval=0.0,
            ).astype(np.float32)

        projections[idx] = proj

    # Keep the native signed pitch; caller can take abs if desired.
    pitch_mm_rad = pitch_mm_per_rad_signed

    scan_geometry = {
        "SOD": float(sod),
        "SDD": float(sdd),
        "detector": {
            "detector psize": float(np.mean([det_pixel_size_cols, det_pixel_size_rows])),
            "detector psize cols": float(det_pixel_size_cols),
            "detector psize rows": float(det_pixel_size_rows),
            "detector rows": det_rows,
            "detector cols": det_cols,
            "detector_col_offset": float(det_col_offset),
            "detector_row_offset": float(det_row_offset),
        },
        "helix": {
            "angles_count": len(ordered_paths),
            "pitch_mm_rad": pitch_mm_rad,
        },
    }

    metadata = {
        "angles_rad": angles_unwrapped,
        "table_positions_mm": table_positions,
        "angle_step_rad": angle_step,
        "pitch_mm_per_angle": pitch_mm_per_angle,
        "pitch_mm_per_rad": abs(pitch_mm_per_rad_signed),
        "pitch_mm_per_rad_signed": pitch_mm_per_rad_signed,
        "detector_pixel_size_rows_mm": det_pixel_size_rows,
        "detector_pixel_size_cols_mm": det_pixel_size_cols,
        "scan_geometry": scan_geometry,
        "ffs_z_offsets_mm": ffs_z_offsets,
    }

    return projections, metadata
