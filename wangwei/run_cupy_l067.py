"""Run the Wang Wei Katsevich pipeline with CuPy on the L067 HDF5 data.

The default command reconstructs the same 450-slice grid used by
``rec_L067_h5_prefiltered_20260504.m``.  For the first validation run, use
``--slice-indices`` to reconstruct a few slices while still pre-filtering the
complete scan exactly once.

Examples
--------
Build/reuse the full filtered cache and reconstruct three baseline slices::

    python -m wangwei.run_cupy_l067 --slice-indices 0,224,449

Run all 450 slices from an existing cache::

    python -m wangwei.run_cupy_l067 --reuse-prefilter
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import cupy as cp
import h5py
import numpy as np

from .cupy_recon import prefilter_h5_to_memmap, reconstruct_prefiltered_cupy


DSD = 1085.6
DSO = 595.0
PITCH_MM = 22.97
HF = 0.0192
VIEWS_PER_ROTATION = 2304
VOXEL_XY_MM = 0.664
DEFAULT_FULL_NZ = 450
DEFAULT_VOXEL_Z_MM = 0.8


def _parse_slice_indices(text: str | None, full_nz: int) -> np.ndarray:
    if not text:
        return np.arange(full_nz, dtype=np.int32)
    values = np.asarray([int(part.strip()) for part in text.split(",")], dtype=np.int32)
    if values.size == 0 or np.any(values < 0) or np.any(values >= full_nz):
        raise ValueError(f"slice indices must be within [0, {full_nz - 1}]")
    if len(np.unique(values)) != len(values):
        raise ValueError("slice indices must be unique")
    return values


def _load_scan_metadata(h5_path: Path, decimate: int) -> tuple[np.ndarray, float, int]:
    with h5py.File(h5_path, "r") as handle:
        angles = handle["angles_rad"][::decimate].astype(np.float32)
        table_positions = handle["table_positions_mm"]
        # The validated MATLAB script computes z_center before decimating
        # z_all, so it uses the two endpoints of the complete scan.
        z_center = (float(table_positions[0]) + float(table_positions[-1])) / 2.0
        nraw = int(handle["projections"].shape[0])
    nviews = math.ceil(nraw / decimate)
    if len(angles) != nviews:
        raise RuntimeError("angle/projection view count mismatch")
    return angles, z_center, nviews


def _save_preview(hu: np.ndarray, z_cor: np.ndarray, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    nslices = hu.shape[2]
    columns = min(nslices, 6)
    chosen = np.linspace(0, nslices - 1, columns, dtype=int)
    fig, axes = plt.subplots(3, columns, figsize=(3.2 * columns, 9.2), squeeze=False)
    windows = [(-200, 300, "soft tissue"), (-1000, 200, "lung"), (-500, 1500, "bone")]
    for row, (vmin, vmax, label) in enumerate(windows):
        for column, slice_index in enumerate(chosen):
            axes[row, column].imshow(hu[:, :, slice_index], cmap="gray", vmin=vmin, vmax=vmax)
            axes[row, column].set_title(
                f"z={z_cor[slice_index]:.1f} mm\n{label} [{vmin},{vmax}]", fontsize=9
            )
            axes[row, column].axis("off")
    fig.suptitle(f"L067 Wang Wei Katsevich CuPy ({nslices} reconstructed slices)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5",
        type=Path,
        default=Path("D:/AAPM-Data/L067/L067/quarter_projections.h5"),
    )
    parser.add_argument("--decimate", type=int, default=2)
    parser.add_argument("--chunk-theta", type=int, default=300)
    parser.add_argument(
        "--pitch-mm",
        type=float,
        default=PITCH_MM,
        help="Table travel per complete rotation in mm.",
    )
    parser.add_argument(
        "--views-per-rotation",
        type=int,
        default=VIEWS_PER_ROTATION,
        help="Projection views per complete rotation before --decimate.",
    )
    parser.add_argument("--full-nz", type=int, default=DEFAULT_FULL_NZ)
    parser.add_argument("--voxel-z", type=float, default=DEFAULT_VOXEL_Z_MM)
    parser.add_argument(
        "--slice-indices",
        default=None,
        help="Comma-separated zero-based indices from the full z grid; default reconstructs all.",
    )
    parser.add_argument(
        "--prefilter-cache",
        type=Path,
        default=Path("C:/Temp/helical_kats_cupy/L067_prefilter_dec2.npy"),
    )
    parser.add_argument(
        "--reuse-prefilter",
        action="store_true",
        help="Reuse --prefilter-cache after validating its shape and metadata.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("D:/Github/helical-kats/out_cupy_2026-08-01/rec_L067_cupy.npy"),
    )
    parser.add_argument("--save-tiff", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    if args.decimate < 1 or args.chunk_theta < 3:
        parser.error("--decimate must be >=1 and --chunk-theta must be >=3")
    if args.pitch_mm <= 0 or args.views_per_rotation < 3:
        parser.error("--pitch-mm must be >0 and --views-per-rotation must be >=3")
    if not args.h5.is_file():
        parser.error(f"HDF5 file not found: {args.h5}")

    slice_indices = _parse_slice_indices(args.slice_indices, args.full_nz)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.prefilter_cache.parent.mkdir(parents=True, exist_ok=True)

    angles, z_center, nviews = _load_scan_metadata(args.h5, args.decimate)
    delt_theta = (2.0 * np.pi / args.views_per_rotation) * args.decimate
    theta = np.arange(nviews + 1, dtype=np.float32) * np.float32(delt_theta)
    theta_offset = float(angles[-1]) - np.pi / 2.0

    delt_alpha = 1.2858 / DSD
    alpha_cor = (np.arange(1, 737) - 369.625) * delt_alpha
    alpha_cor = -alpha_cor[::-1].astype(np.float32)
    w_cor = ((np.arange(1, 65) - 32.5) * 1.0947).astype(np.float32)
    x_cor = ((np.arange(1, 513) - 256.5) * VOXEL_XY_MM).astype(np.float32)
    y_cor = ((np.arange(1, 513) - 256.5) * VOXEL_XY_MM).astype(np.float32)

    full_z_cor = (
        z_center
        + (np.arange(1, args.full_nz + 1) - (args.full_nz + 1) / 2.0) * args.voxel_z
    ).astype(np.float32)
    z_cor = full_z_cor[slice_indices]

    h = args.pitch_mm / (2.0 * np.pi)
    half_fan = np.arcsin(float(np.max(np.abs(x_cor))) / DSO)
    delt_phi = delt_alpha * 4.0
    rdphi = int(np.ceil((np.pi / 2.0 + half_fan) / delt_phi))
    phi_cor = (
        np.arange(-rdphi, rdphi + 1) * delt_phi + 0.25 * delt_phi
    ).astype(np.float32)

    device = cp.cuda.Device(0)
    properties = cp.cuda.runtime.getDeviceProperties(0)
    gpu_name = properties["name"].decode() if isinstance(properties["name"], bytes) else properties["name"]
    print(f"GPU: {gpu_name}; free={device.mem_info[0] / 1e9:.2f} GB")
    print(f"HDF5: {args.h5}")
    print(f"Views: {nviews} (decimate={args.decimate}); detector=736x64")
    print(
        f"Helix: pitch={args.pitch_mm:.6g} mm/rotation; "
        f"views/rotation={args.views_per_rotation}"
    )
    print(f"theta_offset={theta_offset:.6f} rad")
    print(f"Selected full-grid slices: {slice_indices.tolist()}")
    print(f"Selected z range: [{z_cor.min():.1f}, {z_cor.max():.1f}] mm")

    expected_cache_shape = (nviews, len(alpha_cor), len(w_cor))
    prefilter_seconds = 0.0
    cache_sidecar = args.prefilter_cache.with_suffix(args.prefilter_cache.suffix + ".json")
    reuse_ok = False
    if args.reuse_prefilter and args.prefilter_cache.is_file() and cache_sidecar.is_file():
        metadata = json.loads(cache_sidecar.read_text(encoding="utf-8"))
        cached = np.load(args.prefilter_cache, mmap_mode="r")
        reuse_ok = (
            tuple(cached.shape) == expected_cache_shape
            and int(metadata.get("decimate", -1)) == args.decimate
            and Path(metadata.get("h5_path", "")).resolve() == args.h5.resolve()
            and bool(np.isclose(float(metadata.get("h", np.nan)), h))
            and bool(np.isclose(float(metadata.get("delt_theta", np.nan)), delt_theta))
        )
        del cached
        if not reuse_ok:
            raise RuntimeError("Existing prefilter cache metadata does not match this run")

    if reuse_ok:
        print(f"Reusing filtered cache: {args.prefilter_cache}")
    else:
        print(f"Building filtered cache: {args.prefilter_cache}")
        started = time.perf_counter()
        prefilter_h5_to_memmap(
            args.h5,
            args.prefilter_cache,
            decimate=args.decimate,
            chunk_theta=args.chunk_theta,
            DSD=DSD,
            h=h,
            DSO=DSO,
            alpha_cor=alpha_cor,
            w_cor=w_cor,
            phi_cor=phi_cor,
            delt_alpha=delt_alpha,
            delt_theta=delt_theta,
        )
        prefilter_seconds = time.perf_counter() - started

    g_filt = np.load(args.prefilter_cache, mmap_mode="r")
    if tuple(g_filt.shape) != expected_cache_shape:
        raise RuntimeError(f"Filtered cache shape {g_filt.shape} != {expected_cache_shape}")

    print(f"Reconstructing {len(z_cor)} selected slices ...")
    bp_started = time.perf_counter()
    rf, slice_records = reconstruct_prefiltered_cupy(
        g_filt,
        theta,
        theta_offset,
        args.pitch_mm,
        DSD,
        DSO,
        x_cor,
        y_cor,
        z_cor,
        alpha_cor,
        w_cor,
    )
    backprojection_seconds = time.perf_counter() - bp_started
    hu = (np.float32(1000.0) * (rf - np.float32(HF)) / np.float32(HF)).astype(
        np.float32, copy=False
    )

    np.save(args.out, rf)
    hu_path = args.out.with_name(args.out.stem + "_hu.npy")
    np.save(hu_path, hu)
    z_path = args.out.with_name(args.out.stem + "_z_cor.npy")
    np.save(z_path, z_cor)

    if args.save_tiff:
        import tifffile

        tiff_path = args.out.with_name(args.out.stem + "_hu.tif")
        tifffile.imwrite(
            tiff_path,
            np.moveaxis(hu, 2, 0),
            metadata={"axes": "ZYX"},
            compression="zlib",
        )
        print(f"Saved TIFF: {tiff_path}")

    preview_path = args.out.with_name(args.out.stem + "_preview.png")
    if not args.no_preview:
        _save_preview(hu, z_cor, preview_path)

    total_seconds = prefilter_seconds + backprojection_seconds
    run_metadata = {
        "h5_path": str(args.h5.resolve()),
        "prefilter_cache": str(args.prefilter_cache.resolve()),
        "reused_prefilter": reuse_ok,
        "output_rf": str(args.out.resolve()),
        "output_hu": str(hu_path.resolve()),
        "output_z": str(z_path.resolve()),
        "output_preview": None if args.no_preview else str(preview_path.resolve()),
        "gpu": gpu_name,
        "decimate": args.decimate,
        "chunk_theta": args.chunk_theta,
        "pitch_mm_per_rotation": args.pitch_mm,
        "views_per_rotation": args.views_per_rotation,
        "delt_theta_rad": delt_theta,
        "slice_indices": slice_indices.tolist(),
        "z_cor_mm": z_cor.tolist(),
        "rf_shape": list(rf.shape),
        "rf_range": [float(rf.min()), float(rf.max())],
        "hu_range": [float(hu.min()), float(hu.max())],
        "prefilter_seconds_this_run": prefilter_seconds,
        "backprojection_seconds": backprojection_seconds,
        "total_seconds_this_run": total_seconds,
        "slice_records": slice_records,
    }
    metadata_path = args.out.with_name(args.out.stem + "_run.json")
    metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    print("\nTIMING SUMMARY")
    print(f"  Pre-filter this run: {prefilter_seconds:.1f}s")
    print(f"  Backprojection:      {backprojection_seconds:.1f}s")
    print(f"  Total this run:      {total_seconds:.1f}s ({total_seconds / 60.0:.1f} min)")
    print(f"  RF range:            [{rf.min():.7f}, {rf.max():.7f}]")
    print(f"  HU range:            [{hu.min():.0f}, {hu.max():.0f}]")
    print(f"Saved RF:       {args.out}")
    print(f"Saved HU:       {hu_path}")
    print(f"Saved metadata: {metadata_path}")
    if not args.no_preview:
        print(f"Saved preview:  {preview_path}")


if __name__ == "__main__":
    main()
