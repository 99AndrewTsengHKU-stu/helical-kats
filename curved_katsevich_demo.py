"""
Curved-detector Katsevich reconstruction - minimal standalone demo.

Extracted and converted from CPH CT Toolbox (Python 2 -> Python 3).
Original: Copyright (C) 2011-2013 The CT-Toolbox Project lead by Brian Vinter
License: GPLv2+

This implements the full Katsevich pipeline for CURVED (equi-angular) detectors:
  1. Differentiate  (curved formula - no row derivative needed)
  2. Forward rebin   (same as flat, but col_coords are in radians)
  3. Hilbert convolve (same as flat)
  4. Reverse rebin    (flat + cos(gamma) weighting)
  5. Backproject      (arctan column projection instead of linear)

Usage:
  python curved_katsevich_demo.py --dicom-dir <path> [options]
"""

import numpy as np
from numpy import (pi, zeros, zeros_like, arange, sin, cos, tan, arctan,
                   sqrt, convolve, clip, floor, ceil, arcsin, int32)
from time import time
import argparse
import sys


# ============================================================================
# Core Katsevich kernels for CURVED detector (from CPH CT Toolbox base.py)
# ============================================================================

def curved_differentiate(sinogram, conf):
    """
    Differentiation step for curved detector.

    Key difference from flat:
    - No d_row component (curved detector removes row coupling)
    - d_proj uses 4 neighbors instead of 8
    - pixel_span is in RADIANS (angular step), not mm
    - Length correction: dia/sqrt(dia^2 + row^2) -- no col^2 term

    Output has same (rows, cols) shape as input but last row/col are zeros
    (one fewer projection due to forward diff).
    """
    n_projs, n_rows, n_cols = sinogram.shape
    # CPH convention: output has full detector size, last row/col = 0
    out = zeros((n_projs - 1, n_rows, n_cols), dtype=sinogram.dtype)

    dia = conf['scan_diameter']
    dia_sqr = dia ** 2
    delta_s = conf['delta_s']
    pixel_span = conf['detector_pixel_span']  # radians!

    row_coords = conf['row_coords'][:-2]  # skip extension + last for diff
    row_transposed = row_coords.reshape(-1, 1)
    row_sqr = zeros((n_rows - 1, n_cols - 1), dtype=sinogram.dtype)
    row_sqr += row_transposed ** 2

    print(f"  Differentiating {n_projs-1} projections...", end='', flush=True)
    t0 = time()
    for p in range(n_projs - 1):
        d_proj = (sinogram[p + 1, :-1, :-1] - sinogram[p, :-1, :-1]
                  + sinogram[p + 1, :-1, 1:] - sinogram[p, :-1, 1:]) \
                 / (2 * delta_s)
        d_col = (sinogram[p, :-1, 1:] - sinogram[p, :-1, :-1]
                 + sinogram[p + 1, :-1, 1:] - sinogram[p + 1, :-1, :-1]) \
                / (2 * pixel_span)
        out[p, :-1, :-1] = d_proj + d_col
        # Length correction (flat only in row direction)
        out[p, :-1, :-1] *= dia / sqrt(dia_sqr + row_sqr)
    print(f" {time()-t0:.1f}s")
    return out


def forward_rebin(diff_data, conf):
    """
    Forward height rebinning. Same algorithm for flat and curved,
    but fwd_rebin_row was computed with the curved formula.
    """
    n_projs, n_rows, n_cols = diff_data.shape
    n_rebin = conf['detector_rebin_rows']
    out = zeros((n_projs, n_rebin, n_cols), dtype=diff_data.dtype)

    pixel_height = conf['detector_pixel_height']
    fwd_rebin_row = conf['fwd_rebin_row']
    row_coords = conf['row_coords'][:-1]
    det_row_offset = conf.get('detector_row_offset', 0.0)

    print(f"  Forward rebinning to {n_rebin} rows...", end='', flush=True)
    t0 = time()
    for p in range(n_projs):
        for col in range(n_cols):
            rebin_scaled = fwd_rebin_row[:, col] / pixel_height
            out[p, :, col] = np.interp(rebin_scaled, row_coords, diff_data[p, :, col])
    print(f" {time()-t0:.1f}s")
    return out


def hilbert_convolve(rebin_data, conf):
    """
    1D Hilbert convolution along columns for each rebin row.
    Same for flat and curved detectors.
    """
    n_projs, n_rebin, n_cols = rebin_data.shape
    out = zeros_like(rebin_data)
    hilbert = conf['hilbert_filter']

    print(f"  Hilbert convolution...", end='', flush=True)
    t0 = time()
    for p in range(n_projs):
        for r in range(n_rebin):
            conv_full = convolve(hilbert, rebin_data[p, r, :])
            out[p, r, :] = conv_full[n_cols - 1: 2 * n_cols - 1]
    print(f" {time()-t0:.1f}s")
    return out


def reverse_rebin(conv_data, conf):
    """
    Reverse height rebinning.
    Curved version = flat reverse rebin + multiply by cos(col_coords).
    """
    n_projs, n_rebin, n_cols = conv_data.shape
    n_rows = conf['detector_rows']
    out = zeros((n_projs, n_rows, n_cols), dtype=conv_data.dtype)

    fwd_rebin_row = conf['fwd_rebin_row']
    row_coords = conf['row_coords'][:-1]
    col_coords = conf['col_coords'][:-1]  # radians for curved
    det_col_offset = conf.get('detector_column_offset', 0.0)
    pos_start = int(0.5 * n_cols - det_col_offset)

    print(f"  Reverse rebinning to {n_rows} rows...", end='', flush=True)
    t0 = time()
    for p in range(n_projs):
        for row in range(n_rows):
            # Positive column range
            for col in range(pos_start, n_cols):
                rebin_row = 0
                frac0, frac1 = 0.0, 1.0
                for rebin in range(n_rebin - 1):
                    if (row_coords[row] >= fwd_rebin_row[rebin, col] and
                            row_coords[row] <= fwd_rebin_row[rebin + 1, col]):
                        rebin_row = rebin
                        frac0 = ((row_coords[row] - fwd_rebin_row[rebin_row, col])
                                 / (fwd_rebin_row[rebin_row + 1, col]
                                    - fwd_rebin_row[rebin_row, col]))
                        frac1 = 1.0 - frac0
                        break
                out[p, row, col] = (frac1 * conv_data[p, rebin_row, col]
                                    + frac0 * conv_data[p, rebin_row + 1, col])

            # Negative column range
            for col in range(pos_start):
                rebin_row = 1
                frac0, frac1 = 0.0, 1.0
                for rebin in range(n_rebin - 1, 0, -1):
                    if (row_coords[row] >= fwd_rebin_row[rebin - 1, col] and
                            row_coords[row] <= fwd_rebin_row[rebin, col]):
                        rebin_row = rebin
                        frac0 = ((row_coords[row] - fwd_rebin_row[rebin_row - 1, col])
                                 / (fwd_rebin_row[rebin_row, col]
                                    - fwd_rebin_row[rebin_row - 1, col]))
                        frac1 = 1.0 - frac0
                        break
                out[p, row, col] = (frac1 * conv_data[p, rebin_row - 1, col]
                                    + frac0 * conv_data[p, rebin_row, col])

        # *** CURVED-SPECIFIC: multiply by cos(gamma) ***
        out[p] *= cos(col_coords)

    print(f" {time()-t0:.1f}s")
    return out


def curved_backproject(filtered, conf, progress_interval=50):
    """
    Backprojection for curved detector.

    Key differences from flat:
    - Column projection uses arctan (angular coord) instead of linear
    - Row projection includes cos(gamma) factor
    - pixel_span is in radians
    """
    n_projs, n_rows, n_cols = filtered.shape
    x_voxels = conf['x_voxels']
    y_voxels = conf['y_voxels']
    z_voxels = conf['z_voxels']

    recon = zeros((x_voxels, y_voxels, z_voxels), dtype=filtered.dtype)

    scan_radius = conf['scan_radius']
    scan_diameter = conf['scan_diameter']
    x_min, y_min, z_min = conf['x_min'], conf['y_min'], conf['z_min']
    delta_x, delta_y, delta_z = conf['delta_x'], conf['delta_y'], conf['delta_z']
    pixel_span = conf['detector_pixel_span']   # radians
    pixel_height = conf['detector_pixel_height']
    det_row_shift = conf['detector_row_shift']
    det_col_offset = conf.get('detector_column_offset', 0.0)
    progress_per_radian = conf['progress_per_radian']
    source_pos = conf['source_pos']

    fov_radius = conf['fov_radius']
    rad_sqr = fov_radius ** 2
    row_mins = conf['proj_row_mins']
    row_maxs = conf['proj_row_maxs']

    x_coords = arange(x_voxels, dtype=np.float32) * delta_x + x_min
    y_coords = arange(y_voxels, dtype=np.float32) * delta_y + y_min

    print(f"  Backprojecting {n_projs} projections -> ({x_voxels},{y_voxels},{z_voxels})...")
    t0 = time()
    for x in range(x_voxels):
        if x % progress_interval == 0:
            elapsed = time() - t0
            pct = x / x_voxels * 100
            print(f"    x={x}/{x_voxels} ({pct:.0f}%) {elapsed:.1f}s", flush=True)

        x_coord = x_coords[x]
        x_sqr = x_coord ** 2
        for y in range(y_voxels):
            y_coord = y_coords[y]
            y_sqr = y_coord ** 2
            if x_sqr + y_sqr > rad_sqr:
                continue

            # Scale helpers (denominator): R - x*cos(s) - y*sin(s)
            U = scan_radius - x_coord * cos(source_pos) - y_coord * sin(source_pos)

            # *** CURVED: column projection = arctan(t / U) -> angular coord ***
            t_coord = -x_coord * sin(source_pos) + y_coord * cos(source_pos)
            proj_col_gamma = arctan(t_coord / U)
            cos_gamma = cos(proj_col_gamma)

            # Column index (in angular pixel units)
            proj_col_reals = proj_col_gamma / pixel_span + 0.5 * n_cols - det_col_offset
            proj_col_ints = proj_col_reals.astype(int32)
            np.clip(proj_col_ints, 0, n_cols - 2, out=proj_col_ints)
            proj_col_fracs = proj_col_reals - proj_col_ints

            # *** CURVED: row projection includes cos(gamma) ***
            proj_row_diffs = scan_diameter * cos_gamma / U
            proj_row_z_min = (scan_diameter * cos_gamma
                              * (z_min - progress_per_radian * source_pos) / U)
            proj_row_ind_z_min = proj_row_z_min / pixel_height + det_row_shift

            proj_row_ind_diffs = proj_row_diffs * delta_z / pixel_height

            # T-D window z limits
            p_row_mins = ((1 - proj_col_fracs) * row_mins[proj_col_ints]
                          + proj_col_fracs * row_mins[proj_col_ints + 1])
            p_row_maxs = ((1 - proj_col_fracs) * row_maxs[proj_col_ints]
                          + proj_col_fracs * row_maxs[proj_col_ints + 1])

            z_mins = (source_pos * progress_per_radian
                      + p_row_mins * U / (scan_diameter * cos_gamma))
            z_maxs = (source_pos * progress_per_radian
                      + p_row_maxs * U / (scan_diameter * cos_gamma))
            z_firsts = np.ceil((z_mins - z_min) / delta_z).astype(int32)
            z_lasts = np.floor((z_maxs - z_min) / delta_z).astype(int32)

            for proj in range(n_projs):
                z_lo = max(z_firsts[proj], 0)
                z_hi = min(z_lasts[proj], z_voxels - 1)
                if z_lo > z_hi:
                    continue

                col_int = min(max(proj_col_ints[proj], 0), n_cols - 2)
                col_frac = proj_col_fracs[proj]
                row_ind_base = proj_row_ind_z_min[proj]
                row_step = proj_row_ind_diffs[proj]

                for z in range(z_lo, z_hi + 1):
                    row_real = row_ind_base + z * row_step
                    row_int = int(row_real)
                    row_int = min(max(row_int, 0), n_rows - 2)
                    row_frac = row_real - row_int

                    # Bilinear interpolation
                    val = ((1 - row_frac) * (1 - col_frac) * filtered[proj, row_int, col_int]
                           + row_frac * (1 - col_frac) * filtered[proj, row_int + 1, col_int]
                           + (1 - row_frac) * col_frac * filtered[proj, row_int, col_int + 1]
                           + row_frac * col_frac * filtered[proj, row_int + 1, col_int + 1])
                    recon[x, y, z] += val / U[proj]

    elapsed = time() - t0
    print(f"  Backprojection done in {elapsed:.1f}s")
    return recon


# ============================================================================
# Configuration setup for curved detector
# ============================================================================

def create_curved_conf(
    sod, sdd, det_rows, det_cols,
    pixel_span_rad, pixel_height_mm,
    progress_per_turn,
    x_voxels, y_voxels, z_voxels,
    voxel_size,
    detector_rebin_rows=128,
):
    """
    Build configuration dict for curved-detector Katsevich.

    Parameters
    ----------
    sod : float
        Source-to-object (isocenter) distance in mm.
    sdd : float
        Source-to-detector distance in mm.
    det_rows : int
        Number of detector rows.
    det_cols : int
        Number of detector columns (channels).
    pixel_span_rad : float
        Angular step per channel in RADIANS.
    pixel_height_mm : float
        Physical height per detector row in mm.
    progress_per_turn : float
        Helical pitch: table advance per full 360-degree turn in mm.
    x/y/z_voxels : int
        Reconstruction grid size.
    voxel_size : float
        Voxel edge length in mm.
    detector_rebin_rows : int
        Number of rows for height rebinning (default 128).
    """
    conf = {}
    fdt = np.float32

    conf['scan_radius'] = fdt(sod)
    conf['scan_diameter'] = fdt(sdd)
    conf['detector_rows'] = det_rows
    conf['detector_columns'] = det_cols
    conf['detector_shape'] = 'curved'

    conf['detector_pixel_height'] = fdt(pixel_height_mm)
    # For curved detector: pixel_span is in RADIANS
    conf['detector_pixel_span'] = fdt(pixel_span_rad)
    conf['detector_pixel_width'] = fdt(pixel_span_rad * sdd)  # arc length at SDD

    conf['x_voxels'] = x_voxels
    conf['y_voxels'] = y_voxels
    conf['z_voxels'] = z_voxels

    half_fov = voxel_size * max(x_voxels, y_voxels) / 2.0
    conf['x_min'] = fdt(-half_fov)
    conf['x_max'] = fdt(half_fov)
    conf['y_min'] = fdt(-half_fov)
    conf['y_max'] = fdt(half_fov)
    conf['delta_x'] = fdt(voxel_size)
    conf['delta_y'] = fdt(voxel_size)

    conf['fov_radius'] = fdt(half_fov)
    conf['fov_diameter'] = fdt(2 * half_fov)

    half_z = voxel_size * z_voxels / 2.0
    conf['z_min'] = fdt(-half_z)
    conf['z_max'] = fdt(half_z)
    conf['z_len'] = fdt(2 * half_z)
    conf['delta_z'] = fdt(voxel_size)

    conf['progress_per_turn'] = fdt(progress_per_turn)
    conf['progress_per_radian'] = fdt(progress_per_turn / (2 * pi))

    # Helix path
    conf['s_min'] = fdt(-pi + conf['z_min'] / conf['progress_per_radian'])
    conf['s_max'] = fdt(pi + conf['z_max'] / conf['progress_per_radian'])
    conf['s_len'] = fdt(conf['s_max'] - conf['s_min'])

    # Projections per turn
    conf['half_fan_angle'] = fdt(arcsin(conf['fov_radius'] / conf['scan_radius']))

    # Auto-calculate total turns and projections
    core_turns = int(ceil(conf['z_len'] / conf['progress_per_turn']))
    total_turns = core_turns + 1  # +1 for overscan
    conf['projs_per_turn'] = -1  # set later from data
    conf['total_turns'] = total_turns

    # Offsets
    conf['detector_row_offset'] = 0.0
    conf['detector_column_offset'] = 0.0
    conf['detector_row_shift'] = fdt(0.5 * (det_rows - 1))
    conf['detector_column_shift'] = fdt(0.5 * (det_cols - 1))

    conf['detector_rebin_rows'] = detector_rebin_rows

    return conf


def finalize_conf(conf, total_projs, actual_angles=None):
    """
    Complete the configuration after knowing the total number of projections.
    Sets up source positions, coordinate arrays, rebinning tables, Hilbert filter, etc.

    Parameters
    ----------
    conf : dict
        Configuration from create_curved_conf.
    total_projs : int
        Number of projections.
    actual_angles : ndarray, optional
        Actual DICOM gantry angles (radians, unwrapped). If provided, these are
        used directly as source_pos and delta_s is derived from them.
        If None, synthetic uniformly-spaced angles are generated from s_min/s_max.
    """
    fdt = np.float32

    conf['total_projs'] = total_projs

    if actual_angles is not None:
        # Use actual DICOM angles as source positions
        conf['source_pos'] = fdt(actual_angles)
        conf['delta_s'] = fdt(np.mean(np.diff(actual_angles)))
        conf['projs_per_turn'] = int(round(2 * pi / abs(conf['delta_s'])))
        # Update s_min/s_max from actual data range + overscan
        conf['s_min'] = fdt(actual_angles[0])
        conf['s_max'] = fdt(actual_angles[-1])
        conf['s_len'] = fdt(actual_angles[-1] - actual_angles[0])
    else:
        conf['projs_per_turn'] = int(total_projs / conf['s_len'] * 2 * pi)
        conf['delta_s'] = fdt(2 * pi / conf['projs_per_turn'])
        conf['source_pos'] = fdt(conf['s_min'] + conf['delta_s']
                                 * (arange(total_projs, dtype=fdt) + 0.5))

    # Detector coordinate arrays (EXTENDED by 1 for interpolation)
    # For curved detector: col_coords are in RADIANS
    conf['col_coords'] = conf['detector_pixel_span'] * (
        arange(conf['detector_columns'] + 1, dtype=fdt)
        - conf['detector_column_shift'])

    conf['row_coords'] = conf['detector_pixel_height'] * (
        arange(conf['detector_rows'] + 1, dtype=fdt)
        - conf['detector_row_shift'])

    # Rebinning coordinates
    conf['detector_rebin_rows_height'] = fdt(
        (pi + 2 * conf['half_fan_angle']) / (conf['detector_rebin_rows'] - 1))

    rebin_coords = fdt(-pi / 2 - conf['half_fan_angle']
                        + conf['detector_rebin_rows_height']
                        * arange(conf['detector_rebin_rows'], dtype=fdt))

    rebin_scale = fdt(2 * conf['progress_per_radian'])

    # Forward rebin row table -- CURVED version
    n_rebin = conf['detector_rebin_rows']
    n_cols = conf['detector_columns']
    col_coords = conf['col_coords'][:-1]  # skip extension

    fwd_rebin_row = zeros((n_rebin, n_cols), dtype=fdt)
    for col in range(n_cols):
        # *** CURVED formula: uses cos/sin of angular col_coords ***
        row = rebin_scale * (rebin_coords * cos(col_coords[col])
                             + rebin_coords / tan(rebin_coords)
                             * sin(col_coords[col]))
        fwd_rebin_row[:, col] = row
    conf['fwd_rebin_row'] = fwd_rebin_row

    # Tam-Danielsson boundaries -- CURVED version
    col_coords_ext = conf['col_coords']  # extended
    conf['proj_row_mins'] = fdt(
        -conf['progress_per_turn'] / pi
        * (pi / 2 + col_coords_ext) / cos(col_coords_ext))
    conf['proj_row_maxs'] = fdt(
        conf['progress_per_turn'] / pi
        * (pi / 2 - col_coords_ext) / cos(col_coords_ext))

    # Hilbert filter
    kernel_radius = n_cols - 1
    kernel_width = 1 + 2 * kernel_radius
    hilbert = zeros(kernel_width, dtype=fdt)
    for i in range(kernel_width):
        hilbert[i] = (1.0 - cos(pi * (i - kernel_radius - 0.5))) \
                     / (pi * (i - kernel_radius - 0.5))
    conf['hilbert_filter'] = hilbert
    conf['kernel_radius'] = kernel_radius
    conf['kernel_width'] = kernel_width

    return conf


# ============================================================================
# Full pipeline
# ============================================================================

def reconstruct_curved(sinogram, conf):
    """
    Run the full curved-detector Katsevich pipeline.

    Parameters
    ----------
    sinogram : ndarray, shape (n_projs, n_rows, n_cols)
        Projection data.
    conf : dict
        Configuration from create_curved_conf + finalize_conf.

    Returns
    -------
    recon : ndarray, shape (x_voxels, y_voxels, z_voxels)
    """
    print("=== Curved Katsevich Pipeline ===")
    print(f"  Input: {sinogram.shape}")
    print(f"  detector_pixel_span (rad): {conf['detector_pixel_span']:.6f}")
    print(f"  detector_pixel_height (mm): {conf['detector_pixel_height']:.4f}")
    print(f"  half_fan_angle: {np.degrees(conf['half_fan_angle']):.2f} deg")

    # Step 1: Differentiate
    diff = curved_differentiate(sinogram, conf)

    # Step 2: Forward rebin
    rebin = forward_rebin(diff, conf)

    # Step 3: Hilbert convolution
    conv = hilbert_convolve(rebin, conf)

    # Step 4: Reverse rebin
    filtered = reverse_rebin(conv, conf)

    # Step 5: Backprojection
    recon = curved_backproject(filtered, conf)

    # Final scaling: delta_s / (2*pi)
    recon *= conf['delta_s'] / (2 * pi)

    return recon


# ============================================================================
# DICOM loader (adapted from pykatsevich/dicom.py)
# ============================================================================

def load_dicom_projections(dicom_dir, decimate=1):
    """
    Load DICOM-CT-PD projections and extract geometry including proper pitch.

    Two-pass approach (following pykatsevich/dicom.py):
      Pass 1: Read metadata only (fast) - sort by InstanceNumber, unwrap angles,
              extract table positions for pitch computation.
      Pass 2: Read pixel data in sorted order.

    Pitch extraction priority:
      1. Table position tag (0x0018, 0x9327) or DetectorFocalCenterAxialPosition (0x7031, 0x1002)
         -> pitch_mm_per_rad = mean(diff(table_pos) / diff(angles_unwrapped))
      2. SpiralPitchFactor (0x0018, 0x9311) * collimation / (2*pi)
    """
    import pydicom
    from pathlib import Path
    import struct

    def decode_float32(ds, tag, count=1):
        if tag not in ds:
            return None
        raw = bytes(ds[tag].value)
        needed = 4 * count
        if len(raw) < needed:
            return None
        vals = struct.unpack("<" + "f" * count, raw[:needed])
        return vals if count > 1 else vals[0]

    def get_float(ds, tag, fallback=None):
        if tag in ds:
            try:
                return float(ds[tag].value)
            except Exception:
                pass
        return fallback

    # Tag definitions
    ANGLE_TAG = (0x7031, 0x1001)
    TABLE_POSITION_TAG = (0x0018, 0x9327)
    DETECTOR_AXIAL_POSITION_TAG = (0x7031, 0x1002)
    SOD_TAG_STD = (0x0018, 0x9402)
    SDD_TAG_STD = (0x0018, 0x1110)
    SOD_TAG_PVT = (0x7031, 0x1003)
    SDD_TAG_PVT = (0x7031, 0x1031)
    DET_EXTENTS_TAG = (0x7031, 0x1033)
    PITCH_FACTOR_TAG = (0x0018, 0x9311)

    dicom_dir = Path(dicom_dir)
    files = sorted(p for p in dicom_dir.iterdir() if p.suffix.lower() == ".dcm")
    if not files:
        raise FileNotFoundError(f"No .dcm files found in {dicom_dir}")

    print(f"Found {len(files)} DICOM files")

    # ---- Pass 1: metadata only (no pixel data) ----
    print("  Reading metadata...", end='', flush=True)
    meta = []
    for path in files:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        instance = int(getattr(ds, "InstanceNumber", 0))
        angle = decode_float32(ds, ANGLE_TAG)
        if angle is None:
            raise ValueError(f"{path.name} missing angle tag {ANGLE_TAG}")

        # Table position: try standard tag, then private axial position
        if TABLE_POSITION_TAG in ds:
            table_pos = float(ds[TABLE_POSITION_TAG].value)
        elif DETECTOR_AXIAL_POSITION_TAG in ds:
            table_pos = decode_float32(ds, DETECTOR_AXIAL_POSITION_TAG)
        else:
            table_pos = None
        meta.append((instance, path, angle, table_pos))
    print(" done")

    # Sort by InstanceNumber
    meta.sort(key=lambda item: item[0])

    # Extract and unwrap angles
    angles_raw = np.array([item[2] for item in meta], dtype=np.float64)
    angles_unwrapped = np.unwrap(angles_raw).astype(np.float32)

    # Table positions
    has_table = any(item[3] is not None for item in meta)
    if has_table:
        table_positions = np.array(
            [item[3] if item[3] is not None else 0.0 for item in meta],
            dtype=np.float32)
    else:
        table_positions = None

    angle_step = float(np.mean(np.diff(angles_unwrapped)))
    total_angle_range = float(angles_unwrapped[-1] - angles_unwrapped[0])
    n_turns = total_angle_range / (2 * pi)
    print(f"  Angle range: {np.degrees(total_angle_range):.1f} deg = {n_turns:.2f} turns")
    print(f"  Angle step: {np.degrees(angle_step):.4f} deg")

    # Read geometry from first file
    ds0 = pydicom.dcmread(meta[0][1], stop_before_pixels=True)

    sod = get_float(ds0, SOD_TAG_STD, fallback=decode_float32(ds0, SOD_TAG_PVT))
    sdd = get_float(ds0, SDD_TAG_STD, fallback=decode_float32(ds0, SDD_TAG_PVT))
    if sod is None or sdd is None:
        raise ValueError("Missing SOD/SDD tags")

    det_cols = int(ds0.Rows)     # fan direction
    det_rows = int(ds0.Columns)  # z direction
    det_extents = decode_float32(ds0, DET_EXTENTS_TAG, count=2)

    print(f"  SOD = {sod:.1f} mm, SDD = {sdd:.1f} mm, magnification = {sdd/sod:.3f}")
    print(f"  Detector: {det_cols} cols x {det_rows} rows")

    # Detector extents -> equi-angular geometry
    det_extent_cols, det_extent_rows = det_extents
    print(f"  Detector extents at isocenter: ({det_extent_cols:.2f}, {det_extent_rows:.2f}) mm")

    arc_psize_cols = det_extent_cols / det_cols
    arc_psize_rows = det_extent_rows / det_rows
    delta_gamma = arc_psize_cols / sod  # angular step per channel (rad)
    pixel_height = arc_psize_rows * sdd / sod  # row height at detector plane
    print(f"  delta_gamma = {delta_gamma:.6f} rad = {np.degrees(delta_gamma):.4f} deg")
    print(f"  pixel_height (at detector) = {pixel_height:.4f} mm")

    # ---- Compute pitch ----
    if table_positions is not None:
        pitch_mm_per_rad = float(
            np.mean(np.diff(table_positions) / np.diff(angles_unwrapped)))
        pitch_mm_per_turn = abs(pitch_mm_per_rad) * 2 * pi
        print(f"  Pitch from table positions: {pitch_mm_per_rad:.4f} mm/rad"
              f" = {pitch_mm_per_turn:.2f} mm/turn")
    else:
        # Fallback: SpiralPitchFactor * collimation / (2*pi)
        pitch_factor = get_float(ds0, PITCH_FACTOR_TAG)
        collimation_mm = det_extent_rows  # z-extent at isocenter
        if pitch_factor is not None and collimation_mm is not None:
            pitch_mm_per_rad = float(pitch_factor * collimation_mm / (2 * pi))
            pitch_mm_per_rad *= np.sign(angle_step) if angle_step != 0 else 1.0
            pitch_mm_per_turn = abs(pitch_mm_per_rad) * 2 * pi
            print(f"  Pitch from SpiralPitchFactor ({pitch_factor:.3f}):"
                  f" {pitch_mm_per_rad:.4f} mm/rad = {pitch_mm_per_turn:.2f} mm/turn")
            # Synthesize table positions for downstream use
            table_positions = (pitch_mm_per_rad
                               * (angles_unwrapped - angles_unwrapped[0])).astype(np.float32)
        else:
            raise ValueError("No table position or pitch factor available in DICOM tags")

    # ---- Pass 2: read pixel data in sorted order ----
    ordered_paths = [item[1] for item in meta]
    if decimate > 1:
        indices = list(range(0, len(ordered_paths), decimate))
        ordered_paths = [ordered_paths[i] for i in indices]
        angles_unwrapped = angles_unwrapped[indices]
        if table_positions is not None:
            table_positions = table_positions[indices]
        print(f"  After decimation ({decimate}x): {len(ordered_paths)} projections")

    print(f"  Loading pixel data...", end='', flush=True)
    n_views = len(ordered_paths)
    projections = np.empty((n_views, det_rows, det_cols), dtype=np.float32)
    for idx, path in enumerate(ordered_paths):
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept
        # Transpose to (rows_z, cols_fan)
        if arr.shape[0] == det_cols:
            arr = arr.T
        projections[idx] = arr
        if (idx + 1) % 500 == 0:
            print(f" {idx+1}", end='', flush=True)
    print(f" done ({n_views} loaded)")

    geom = {
        'sod': float(sod),
        'sdd': float(sdd),
        'det_rows': det_rows,
        'det_cols': det_cols,
        'delta_gamma': float(delta_gamma),
        'pixel_height': float(pixel_height),
        'angles': angles_unwrapped,
        'angle_range': total_angle_range,
        'pitch_mm_per_rad': float(pitch_mm_per_rad),
        'pitch_mm_per_turn': float(pitch_mm_per_turn),
        'table_positions': table_positions,
        'n_turns': float(n_turns),
    }

    return projections, geom


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Curved-detector Katsevich demo')
    parser.add_argument('--dicom-dir', required=True, help='Path to DICOM-CT-PD directory')
    parser.add_argument('--rows', type=int, default=128, help='Recon grid X/Y size')
    parser.add_argument('--cols', type=int, default=128, help='Recon grid X/Y size (alias)')
    parser.add_argument('--slices', type=int, default=64, help='Recon grid Z size')
    parser.add_argument('--voxel-size', type=float, default=1.0, help='Voxel size in mm')
    parser.add_argument('--pitch-mm-turn', type=float, default=None, help='Override pitch (mm/turn)')
    parser.add_argument('--decimate', type=int, default=10, help='Use every Nth projection')
    parser.add_argument('--rebin-rows', type=int, default=128, help='Rebin rows')
    parser.add_argument('--save-npy', type=str, default=None, help='Save .npy output')
    parser.add_argument('--show', action='store_true', help='Show result slices')
    args = parser.parse_args()

    # Load data
    sinogram, geom = load_dicom_projections(args.dicom_dir, decimate=args.decimate)
    n_projs, n_rows, n_cols = sinogram.shape

    # Pitch (always positive for algorithm)
    if args.pitch_mm_turn is not None:
        pitch = abs(args.pitch_mm_turn)
        print(f"  Using override pitch: {pitch:.1f} mm/turn")
    else:
        pitch = geom['pitch_mm_per_turn']  # already abs
        print(f"  Pitch from DICOM: {pitch:.2f} mm/turn")

    # Normalize helix direction: Katsevich requires source_pos to increase.
    # If angles decrease (clockwise rotation), negate angles and flip columns.
    angles = geom['angles']
    if angles[-1] < angles[0]:
        print("  Normalizing: negating angles + flipping columns (clockwise -> CCW)")
        angles = -angles
        sinogram = sinogram[:, :, ::-1].copy()

    xy_size = max(args.rows, args.cols)

    # Create configuration
    conf = create_curved_conf(
        sod=geom['sod'],
        sdd=geom['sdd'],
        det_rows=n_rows,
        det_cols=n_cols,
        pixel_span_rad=geom['delta_gamma'],
        pixel_height_mm=geom['pixel_height'],
        progress_per_turn=pitch,
        x_voxels=xy_size,
        y_voxels=xy_size,
        z_voxels=args.slices,
        voxel_size=args.voxel_size,
        detector_rebin_rows=args.rebin_rows,
    )
    conf = finalize_conf(conf, n_projs, actual_angles=angles)

    print(f"\nConfiguration:")
    print(f"  Volume: {xy_size}x{xy_size}x{args.slices}, voxel={args.voxel_size}mm")
    print(f"  FOV: {xy_size * args.voxel_size:.0f}mm")
    print(f"  Projections: {n_projs}")
    print(f"  Projs/turn: {conf['projs_per_turn']}")
    print(f"  Pitch: {pitch:.1f} mm/turn")
    print(f"  s_range: [{conf['s_min']:.2f}, {conf['s_max']:.2f}] rad")

    # Run reconstruction
    recon = reconstruct_curved(sinogram, conf)

    print(f"\nResult: shape={recon.shape}, range=[{recon.min():.6f}, {recon.max():.6f}]")

    if args.save_npy:
        np.save(args.save_npy, recon)
        print(f"Saved to {args.save_npy}")

    if args.show:
        from matplotlib import pyplot as plt
        mid = recon.shape[2] // 2
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, idx in enumerate([mid - 5, mid, mid + 5]):
            idx = min(max(idx, 0), recon.shape[2] - 1)
            axes[i].imshow(recon[:, :, idx].T, cmap='gray')
            axes[i].set_title(f'Slice {idx}')
        plt.suptitle('Curved Katsevich Reconstruction')
        plt.tight_layout()
        plt.savefig('curved_recon_demo.png', dpi=150)
        print("Saved curved_recon_demo.png")
        plt.show()


if __name__ == '__main__':
    main()
