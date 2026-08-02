"""
example_L109.py — Reproduce wangwei's rec_L109.m in Python

Usage
-----
    cd d:/Github/helical-kats
    python -m wangwei.example_L109 --dicom_dir /path/to/L109/DICOM-CT-PD_FD \
                                   --out_dir /tmp/recon_l109

Geometry constants are taken directly from rec_L109.m.
"""

import argparse
import pathlib
import sys

import numpy as np

# Allow running as `python -m wangwei.example_L109` from project root
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from wangwei import recon_helical


def read_dicom_sinogram(dicom_dir):
    """
    Read helical CT sinogram from a folder of DICOM projections.
    Returns (sino, theta, theta_offset, z_positions).

    Requires pydicom.
    """
    import pydicom, glob, re

    files = sorted(glob.glob(str(dicom_dir / "*.dcm")))
    if not files:
        raise FileNotFoundError(f"No .dcm files in {dicom_dir}")

    phi_list, z_list, slabs = [], [], []
    for f in files:
        ds = pydicom.dcmread(f)
        phi_list.append(float(ds[0x7031, 0x1001].value))   # DetectorFocalCenterAngularPosition
        z_list.append(  float(ds[0x7031, 0x1002].value))   # DetectorFocalCenterAxialPosition
        raw = ds.pixel_array.astype(np.float32)
        slabs.append(raw)

    phi = np.array(phi_list)
    z   = np.array(z_list)
    sino = np.stack(slabs, axis=0)   # (ntheta, nrow, ncol) — adjust to (ntheta, nalpha, nw)

    return sino, phi, z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom_dir", required=True)
    parser.add_argument("--out_dir",   default=".")
    args = parser.parse_args()

    dicom_dir = pathlib.Path(args.dicom_dir)
    out_dir   = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Scanner geometry (from rec_L109.m) ────────────────────────────────
    DSD       = 1085.6          # mm
    DSO       = 595.0           # mm
    p         = 23.0            # mm per 2π
    DefTimes  = 2304            # views per rotation
    HF        = 0.0192          # HU calibration factor

    delt_theta = 2 * np.pi / DefTimes

    # Detector grid
    delt_alpha = 1.2858 / DSD
    alpha_cor  = (np.arange(1, 737) - 369.625) * delt_alpha
    alpha_cor  = -alpha_cor[::-1]   # MATLAB: alpha_cor=-alpha_cor(end:-1:1)
    w_cor      = (np.arange(1, 65) - 32.5) * 1.0947

    # Reconstruction grid
    x_cor = (np.arange(1, 513) - 257.0) * 0.7813     # ≈ linspace(-264,247,512)*0.7813
    y_cor = (np.arange(1, 513) - 257.0) * 0.7813

    # ── Load sinogram ──────────────────────────────────────────────────────
    print("Loading DICOM projections …")
    sino_raw, phi, z = read_dicom_sinogram(dicom_dir)

    # MATLAB: sin=sin(end:-1:1,:,end:-1:1)  → flip theta and w axes
    sino = sino_raw[::-1, :, ::-1].copy()

    theta = np.arange(sino.shape[0] + 1) * delt_theta
    theta_offset = float(phi[-1]) - np.pi / 2.0

    # z_cor from rec_L109.m: z(end)-[99:2:353]
    z_cor = float(z[-1]) - np.arange(99, 354, 2).astype(np.float32)

    alpha_cor = alpha_cor.astype(np.float32)
    w_cor     = w_cor.astype(np.float32)
    x_cor     = x_cor.astype(np.float32)
    y_cor     = y_cor.astype(np.float32)

    # ── Reconstruct ────────────────────────────────────────────────────────
    print("Reconstructing …")
    rf = recon_helical(sino, theta, theta_offset, p, DSD, DSO,
                       x_cor, y_cor, z_cor, alpha_cor, w_cor)

    hu = 1000.0 * (rf - HF) / HF

    out_path = out_dir / "recon_L109.npy"
    np.save(str(out_path), hu)
    print(f"Saved → {out_path}   shape={hu.shape}  dtype={hu.dtype}")
    print(f"HU range: [{hu.min():.0f}, {hu.max():.0f}]")


if __name__ == "__main__":
    main()
