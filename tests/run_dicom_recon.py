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

Example:
    python tests/run_dicom_recon.py ^
        --dicom-dir D:\\1212_High_Pitch_argparse\\C001\\C001_bundle_nview_div4_pitch0.6\\dcm_proj ^
        --rows 256 --cols 256 --slices 256 --voxel-size 0.5 ^
        --save-npy rec_c001.npy
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

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
    args = parser.parse_args()

    dicom_dir = Path(args.dicom_dir)
    if not dicom_dir.exists():
        raise FileNotFoundError(f"{dicom_dir} not found")

    print(f"Loading DICOM projections from {dicom_dir} ...")
    sino, meta = load_dicom_projections(dicom_dir)
    pitch_signed = meta["pitch_mm_per_rad_signed"]
    # Katsevich configuration expects a positive pitch magnitude; ASTRA geometry still uses the real z_shift.
    pitch_used = pitch_signed if args.pitch_signed else abs(pitch_signed)
    scan_geom = meta["scan_geometry"]
    angles = meta["angles_rad"].copy()
    z_shift = meta["table_positions_mm"] - meta["table_positions_mm"].mean()
    # Optional debugging transforms for suspected orientation issues:
    if args.reverse_per_turn:
        angles, sino, z_shift = reverse_views_per_turn(angles, sino, z_shift)
    if args.reverse_angle_order:
        angles = angles[::-1]
        sino = sino[::-1]
        z_shift = z_shift[::-1]
    if args.negate_angles:
        angles = -angles
    if args.flip_cols:
        sino = sino[:, :, ::-1]
    if args.flip_rows:
        sino = sino[:, ::-1, :]
    scan_geom["helix"]["pitch_mm_rad"] = float(pitch_used)

    print("Building ASTRA projection geometry ...")
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

    print("Building ASTRA volume geometry ...")
    vol_geom = make_vol_geom(args.rows, args.cols, args.slices, args.voxel_size)

    print("Preparing Katsevich configuration ...")
    conf = create_configuration(scan_geom, vol_geom)

    print("Running reconstruction ...")
    rec = reconstruct(
        sino,
        conf,
        vol_geom,
        proj_geom,
        {
            "Diff": {"Print time": True},
            "FwdRebin": {"Print time": True},
            "BackRebin": {"Print time": True},
            "BackProj": {"Print time": True},
        },
    )

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
