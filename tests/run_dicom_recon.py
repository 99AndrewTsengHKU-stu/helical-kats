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
One-off driver to reconstruct a projection DICOM series with Pykatsevich.
Includes Auto-Focus capability to detect correct rotation parameters.

Example:
    python tests/run_dicom_recon.py ^
        --dicom-dir D:\\1212_High_Pitch_argparse\\C001\\C001_bundle_nview_div4_pitch0.6\\dcm_proj ^
        --rows 256 --cols 256 --slices 256 --voxel-size 0.5 ^
        --save-npy rec_c001.npy --auto-focus
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import copy

import astra
import numpy as np

# Make sure we import the local package, not a stale installed one.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pykatsevich import load_dicom_projections
from pykatsevich.geometry import astra_helical_views
from pykatsevich.initialize import create_configuration
from pykatsevich.reconstruct import reconstruct


def make_vol_geom(rows: int, cols: int, slices: int, voxel_size: float):
    half_x = cols * voxel_size * 0.5
    half_y = rows * voxel_size * 0.5
    half_z = slices * voxel_size * 0.5
    return astra.create_vol_geom(
        rows,
        cols,
        slices,
        -half_x,
        half_x,
        -half_y,
        half_y,
        -half_z,
        half_z,
    )


def reverse_views_per_turn(angles, sino, z_shift):
    """
    Reverse projection order within each 2*pi rotation while keeping rotation order.
    """
    angles_arr = np.asarray(angles)
    sino_arr = np.asarray(sino)
    z_arr = None if z_shift is None else np.asarray(z_shift)

    unwrapped = np.unwrap(angles_arr)
    turn_ids = np.floor((unwrapped - unwrapped[0]) / (2 * np.pi)).astype(int)
    unique_turns = np.unique(turn_ids)
    reorder_idx = np.concatenate([np.flatnonzero(turn_ids == turn)[::-1] for turn in unique_turns])

    angles_out = angles_arr[reorder_idx]
    sino_out = sino_arr[reorder_idx]
    z_out = None if z_arr is None else z_arr[reorder_idx]
    return angles_out, sino_out, z_out


def run_reconstruction(
    sino,
    meta,
    args,
    rows=None,
    cols=None,
    slices=None,
    override_opts=None
):
    """
    Helper to run reconstruction with given parameters and optional overrides.
    """
    # Defaults
    if rows is None: rows = args.rows
    if cols is None: cols = args.cols
    if slices is None: slices = args.slices
    if override_opts is None: override_opts = {}

    # merge args and overrides
    def get_opt(name, default):
        return override_opts.get(name, getattr(args, name))

    pitch_signed = meta["pitch_mm_per_rad_signed"]
    pitch_used = pitch_signed if get_opt("pitch_signed", False) else abs(pitch_signed)
    
    scan_geom = copy.deepcopy(meta["scan_geometry"]) # Deepcopy to avoid mutating original
    angles = meta["angles_rad"].copy()
    if meta["table_positions_mm"] is not None:
        z_shift = meta["table_positions_mm"] - meta["table_positions_mm"].mean()
    else:
        z_shift = np.zeros(len(angles), dtype=np.float32)

    # Transforms
    if get_opt("reverse_per_turn", False):
        angles, sino, z_shift = reverse_views_per_turn(angles, sino, z_shift)
    
    if get_opt("reverse_angle_order", False):
        angles = angles[::-1]
        sino = sino[::-1]
        z_shift = z_shift[::-1]

    if get_opt("negate_angles", False):
        angles = -angles

    if get_opt("flip_cols", False):
        sino = sino[:, :, ::-1]

    if get_opt("flip_rows", False):
        sino = sino[:, ::-1, :]

    scan_geom["helix"]["pitch_mm_rad"] = float(pitch_used)

    # Inject the actual scan angle range into scan_geom so that
    # create_configuration uses s_len = angles_range instead of
    # the z-based fallback (which breaks badly when slices==1).
    scan_geom["helix"]["angles_range"] = float(abs(angles[-1] - angles[0]))

    # Geometry construction
    views = astra_helical_views(
        scan_geom["SOD"],
        scan_geom["SDD"],
        scan_geom["detector"]["detector psize"],
        angles,
        meta["pitch_mm_per_angle"],
        vertical_shifts=z_shift,
    )
    proj_geom = astra.create_proj_geom(
        "cone_vec",
        scan_geom["detector"]["detector rows"],
        scan_geom["detector"]["detector cols"],
        views,
    )

    vol_geom = make_vol_geom(rows, cols, slices, args.voxel_size)
    conf = create_configuration(scan_geom, vol_geom)

    # ── FIX: replace synthetic source_pos with actual DICOM angles ──────────
    # backproject_a() uses source_pos[i] as the angle θ in the GPU
    # scale_integrate kernel:  1 / (R - x·cosθ - y·sinθ)
    # The synthetic array is centered at 0, but real DICOM angles may be
    # centered elsewhere (e.g. ~1.96 rad for L067 quarter_2000), causing
    # bow-shaped artifacts. Using actual angles removes this offset.
    conf['source_pos'] = angles.astype(np.float32)
    # delta_s must carry the sign of the angular step:
    #   differentiate() computes d_proj = (sino[i+1] - sino[i]) / (4*delta_s)
    #   If angles decrease (CW scan), delta_s < 0 → correct derivative sign.
    #   Using abs() gives the wrong sign and mirrors/corrupts the Hilbert output.
    _angle_step = float(np.mean(np.diff(angles)))
    conf['delta_s'] = _angle_step
    print(f"[Angle] step={_angle_step:.5f} rad  direction={'CW (decreasing)' if _angle_step < 0 else 'CCW (increasing)'}")
    # ────────────────────────────────────────────────────────────────────────


    # Print pitch / TD window diagnostics
    ppr  = conf['progress_per_radian']
    ppt  = conf['progress_per_turn']
    dw   = conf['detector rows'] * conf['pixel_height']
    w_top_max  = float(conf['proj_row_maxs'].max())
    w_bot_min  = float(conf['proj_row_mins'].min())
    print(f"[Pitch] progress_per_radian={ppr:.3f} mm/rad  per_turn={ppt:.2f} mm")
    print(f"[TD]    detector height={dw:.2f} mm  w_top_max={w_top_max:.2f} mm  w_bot_min={w_bot_min:.2f} mm")
    print(f"[TD]    window covers {(w_top_max-w_bot_min)/dw*100:.1f}% of detector height")

    # Override T-D smoothing if requested
    td_smoothing = override_opts.get('td_smoothing', None)
    if td_smoothing is not None:
        conf['T-D smoothing'] = float(td_smoothing)
        print(f"[TD]    smoothing overridden to {td_smoothing}")
    else:
        print(f"[TD]    smoothing = {conf['T-D smoothing']} (default)")

    # Reconstruct
    # Use quiet mode for optimization runs
    print_time = override_opts.get("verbose", True)
    
    rec = reconstruct(
        sino,
        conf,
        vol_geom,
        proj_geom,
        {
            "Diff": {"Print time": print_time},
            "FwdRebin": {"Print time": print_time},
            "BackRebin": {"Print time": print_time},
            "BackProj": {"Print time": print_time},
        },
    )

    # Normalize/Windowing for better contrast
    if getattr(args, "windowing", False):
        # simple clipping to Hu-like range or just relative contrast boost
        # For AAPM, typical range is -1000 to 1000 or so. 
        # But we don't have HU calibration here, so let's just do a percentile stretch.
        vmin, vmax = np.percentile(rec, [1, 99])
        if vmax > vmin:
            rec = np.clip(rec, vmin, vmax)
            rec = (rec - vmin) / (vmax - vmin)
            print(f"Applied windowing: range [{vmin:.4f}, {vmax:.4f}] scaled to [0, 1]")

    return rec


def measure_sharpness(img):
    """
    Estimate image sharpness using gradient magnitude variance.
    Robust and dependency-free (numpy only).
    """
    # Gradient in two dimensions
    gy, gx = np.gradient(img)
    gnorm = np.sqrt(gx**2 + gy**2)
    return np.mean(gnorm)


def auto_focus(sino, meta, args):
    print("\n--- Starting Auto-Focus (Sharpness Detection) ---")
    
    # Modes to test:
    # 1. Default (No change)
    # 2. Negate Angles (Common issue: CW vs CCW)
    # 3. Reverse Angle Order (Common issue: Superior-Inferior vs Inferior-Superior)
    # 4. Reverse Per Turn (Less common, but possible)
    
    modes = [
        ("Normal", {}),
        ("Negate Angles", {"negate_angles": True}),
        ("Reverse Order", {"reverse_angle_order": True}),
        ("Reverse Per Turn", {"reverse_per_turn": True}),
        ("Flip Cols", {"flip_cols": True}),
        ("Flip Rows", {"flip_rows": True}),
        ("Flip Cols + Negate", {"flip_cols": True, "negate_angles": True}),
        ("Flip Rows + Negate", {"flip_rows": True, "negate_angles": True}),
        ("Flip Cols + RevOrder", {"flip_cols": True, "reverse_angle_order": True}),
    ]

    best_score = -1.0
    best_mode_name = "Normal"
    best_opts = {}

    # Use a central ROI for speed
    # Central slice, 1 slice.
    # Reduce resolution slightly for speed if needed, but 256x256 is fast enough for 1 slice.
    
    test_rows, test_cols = 256, 256
    
    for name, opts in modes:
        print(f"Testing mode: {name} ... ", end="", flush=True)
        try:
            # Reconstruct 1 slice
            rec = run_reconstruction(
                sino, meta, args,
                rows=test_rows, cols=test_cols, slices=1,
                override_opts={**opts, "verbose": False}
            )
            # Remove NaNs if any (reconstruction outside FOV)
            rec = np.nan_to_num(rec)
            
            score = measure_sharpness(rec[:,:,0]) # rec is (slices, rows, cols) -> (1, rows, cols)
            print(f"Score: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_mode_name = name
                best_opts = opts
                
        except Exception as e:
            print(f"Failed ({e})")

    print(f"--- Auto-Focus Selected: {best_mode_name} (Score: {best_score:.4f}) ---\n")
    return best_opts


def main():
    parser = argparse.ArgumentParser(description="Reconstruct helical projections stored as DICOM.")
    parser.add_argument(
        "--dicom-dir",
        required=True,
        help="Folder containing projection DICOM files.",
    )
    parser.add_argument("--rows", type=int, default=256, help="Volume rows (Y).")
    parser.add_argument("--cols", type=int, default=256, help="Volume cols (X).")
    parser.add_argument("--slices", type=int, default=256, help="Volume slices (Z).")
    parser.add_argument("--voxel-size", type=float, default=0.5, help="Voxel size in mm.")
    parser.add_argument("--save-npy", type=str, default=None, help="Path to save reconstruction as .npy.")
    parser.add_argument("--save-tiff", type=str, default=None, help="Path to save reconstruction as TIFF stack.")
    parser.add_argument("--pitch-signed", action="store_true", help="Use signed pitch from DICOM (default uses absolute value to avoid orientation artifacts).")
    parser.add_argument("--reverse-per-turn", action="store_true", help="Reverse projection order within each rotation, keeping rotation order.")
    parser.add_argument("--reverse-angle-order", action="store_true", help="Reverse projection order (angles and sinogram).")
    parser.add_argument("--negate-angles", action="store_true", help="Multiply all projection angles by -1.")
    parser.add_argument("--flip-cols", action="store_true", help="Flip detector columns (U-axis) in the sinogram.")
    parser.add_argument("--flip-rows", action="store_true", help="Flip detector rows (V-axis) in the sinogram.")
    parser.add_argument("--auto-focus", action="store_true", help="Automatically detect rotation parameters by maximizing sharpness.")
    parser.add_argument("--decimate", type=int, default=1, help="Use only every N-th projection (e.g. 10) to save memory/time.")
    parser.add_argument("--max-views", type=int, default=None, help="Keep only the first N projections (consecutive, not decimated). Useful to extract ~1 full turn from a long scan.")
    parser.add_argument("--td-smoothing", type=float, default=None, help="Override Tam-Danielsson boundary smoothing factor (default=0.025). Try 0.05-0.10 to reduce bow-shaped artifacts.")
    parser.add_argument("--windowing", action="store_true", help="Apply 1%-99% percentile windowing for better contrast.")

    args = parser.parse_args()

    dicom_dir = Path(args.dicom_dir)
    if not dicom_dir.exists():
        raise FileNotFoundError(f"{dicom_dir} not found")

    print(f"Loading DICOM projections from {dicom_dir} ...")
    sino, meta = load_dicom_projections(dicom_dir)

    # Trim to first N consecutive views (for ~1-turn coverage tests)
    if args.max_views is not None and args.max_views < len(sino):
        print(f"Trimming to first {args.max_views} consecutive projections ...")
        sino = sino[:args.max_views].copy()
        meta["angles_rad"] = meta["angles_rad"][:args.max_views]
        if meta["table_positions_mm"] is not None:
            meta["table_positions_mm"] = meta["table_positions_mm"][:args.max_views]
        meta["scan_geometry"]["helix"]["angles_count"] = args.max_views
        meta["pitch_mm_per_angle"] = float(np.mean(np.abs(np.diff(meta["angles_rad"]))))
        print(f"Angle range after trim: {np.degrees(meta['angles_rad'][-1] - meta['angles_rad'][0]):.1f} deg ({(meta['angles_rad'][-1] - meta['angles_rad'][0])/(2*np.pi):.2f} turns)")

    # Apply decimation to save memory/VRAM if requested
    if args.decimate > 1:
        print(f"Decimating projections by factor {args.decimate} ...")
        sino = sino[::args.decimate].copy()
        meta["angles_rad"] = meta["angles_rad"][::args.decimate]
        if meta["table_positions_mm"] is not None:
            meta["table_positions_mm"] = meta["table_positions_mm"][::args.decimate]
        meta["scan_geometry"]["helix"]["angles_count"] = len(sino)
        # pitch_mm_per_angle changes, but pitch_mm_per_rad stays same.
        # pykatsevich/reconstruct.py uses pitch_mm_rad which is correct.
        # But astra_helical_views uses pitch_per_angle.
        meta["pitch_mm_per_angle"] *= args.decimate
        print(f"New projection count: {len(sino)}")
    
    # Auto-Focus Logic
    final_opts = {}
    if args.auto_focus:
        best_opts = auto_focus(sino, meta, args)
        final_opts.update(best_opts)
    if args.td_smoothing is not None:
        final_opts['td_smoothing'] = args.td_smoothing

    print("Running final reconstruction ...")
    rec = run_reconstruction(sino, meta, args, override_opts=final_opts)


    print("Reconstruction finished. Volume shape:", rec.shape, "min/max:", rec.min(), rec.max())

    if args.save_npy:
        np.save(args.save_npy, rec)
        print(f"Saved reconstruction to {os.path.abspath(args.save_npy)}")

    if args.save_tiff:
        try:
            import tifffile
        except ImportError as exc:
            raise RuntimeError("Saving TIFF requires the 'tifffile' package (pip install tifffile).") from exc
        # Reorder to (slices, rows, cols) for TIFF stacks
        tiff_data = np.moveaxis(rec, 2, 0).astype(np.float32, copy=False)
        tifffile.imwrite(args.save_tiff, tiff_data)
        print(f"Saved reconstruction TIFF stack to {os.path.abspath(args.save_tiff)}")


if __name__ == "__main__":
    main()
