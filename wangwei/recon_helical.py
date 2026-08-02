"""
recon_helical.py — Top-level helical Katsevich reconstruction driver

Port of wangwei-cmd/Katsevich-algorithm/helical_curve/recon_helical.m

Usage
-----
    from wangwei.recon_helical import recon_helical
    vol = recon_helical(sino, theta, theta_offset, p, DSD, DSO,
                        x_cor, y_cor, z_cor, alpha_cor, w_cor)
    hu  = 1000 * (vol - HF) / HF      # convert to HU

Parameters
----------
sino         : float32 ndarray (ntheta, nalpha, nw)
               Raw sinogram, source-angle first.
theta        : 1D array — source angles (rad), uniformly spaced
theta_offset : float — angular offset  φ(last) - π/2  from DICOM header
p            : float — table pitch  (mm per 2π rotation)
DSD          : float — source-to-detector distance (mm)
DSO          : float — source-to-isocenter distance (mm)
x_cor        : 1D array — reconstruction x coordinates (mm)
y_cor        : 1D array — reconstruction y coordinates (mm)
z_cor        : 1D array — reconstruction z coordinates (mm)
alpha_cor    : 1D array — detector column fan angles (rad)
w_cor        : 1D array — detector row positions (mm)

Returns
-------
rf : float32 ndarray (nx, ny, nz)
"""

import numpy as np

from .solve_pi          import solve_pi
from .index_theta       import index_theta
from .compute_alpha_v_w import compute_alpha_v_w
from .backproject       import backproject_helical


def recon_helical(sino, theta, theta_offset, p, DSD, DSO,
                  x_cor, y_cor, z_cor, alpha_cor, w_cor,
                  verbose=True):
    """
    Full helical Katsevich reconstruction.

    Reconstructs one z-slice per outer loop iteration (matches wangwei).
    """
    h         = p / (2.0 * np.pi)       # pitch constant (mm/rad)
    delt_alpha = float(alpha_cor[1] - alpha_cor[0])
    delt_theta = float(theta[1]     - theta[0])

    rFOV      = float(np.max(np.abs(x_cor)))
    half_fan  = np.arcsin(rFOV / DSO)
    delt_phi  = delt_alpha * 4.0
    rDphi     = int(np.ceil((np.pi / 2.0 + half_fan) / delt_phi))
    phi_cor   = np.arange(-rDphi, rDphi + 1) * delt_phi + 0.25 * delt_phi

    # Account for angular offset of the first view
    z_cor_rec = np.asarray(z_cor, dtype=np.float32) + h * theta_offset

    nx, ny, nz = len(x_cor), len(y_cor), len(z_cor)
    rf = np.zeros((nx, ny, nz), dtype=np.float32)

    for iz in range(nz):
        z_val = float(z_cor_rec[iz])

        # ── PI-line endpoints ──────────────────────────────────────────────
        s_b, s_t = solve_pi(x_cor, y_cor, z_val,
                             h=h / DSO,   # normalised pitch h/R
                             R=DSO,
                             tol=1e-3, maxiter=100)
        # s_b, s_t: (nx, ny)

        s_min = float(s_b.min())
        s_max = float(s_t.max())

        # Index range in theta array (1-based → 0-based)
        t1 = int(np.floor((s_min - theta[0] - theta_offset) / delt_theta))
        t2 = int(np.ceil ((s_max - theta[0] - theta_offset) / delt_theta)) + 1
        t1 = max(t1, 0)
        t2 = min(t2, len(theta) - 1)

        t_need  = theta[t1:t2] + theta_offset    # (Ltheta,)
        Ltheta  = len(t_need)
        if Ltheta < 2:
            if verbose:
                print(f"  z[{iz}] = {z_val:.1f} mm  — skipped (Ltheta={Ltheta})")
            continue

        # ── T-D weights ───────────────────────────────────────────────────
        ind = index_theta(s_b, s_t, t_need)   # (Ltheta, nx, ny)

        # ── Voxel-to-detector mapping ─────────────────────────────────────
        alpha_s, v_s, w_s = compute_alpha_v_w(
            DSO, DSD, h, t_need, x_cor, y_cor, z_val
        )
        # each: (Ltheta, nx, ny)

        # ── Sinogram window ───────────────────────────────────────────────
        sino_slice = sino[t1:t2]   # (Ltheta, nalpha, nw)

        # ── Backprojection ────────────────────────────────────────────────
        rf[:, :, iz] = backproject_helical(
            sino_slice, DSD, h, DSO,
            phi_cor, alpha_cor, w_cor,
            delt_alpha, delt_theta,
            alpha_s, v_s, w_s, ind, Ltheta,
        )

        if verbose:
            print(f"  z[{iz+1}/{nz}] = {z_val:.1f} mm  Ltheta={Ltheta}", flush=True)

    return rf
