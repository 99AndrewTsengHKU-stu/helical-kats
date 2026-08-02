"""
compute_alpha_v_w.py — Voxel-to-detector mapping for each source angle

Port of wangwei-cmd/Katsevich-algorithm/helical_curve/compue_alpha_v_w.m

For each source angle θ and voxel (x, y, z), computes:
  α  : fan angle  α = atan2((-x·sinθ + y·cosθ), (DSO - x·cosθ - y·sinθ))
  v  : virtual SDD  v = DSO - x·cosθ - y·sinθ   (used as distance divisor)
  w  : detector row  w = DSD·cos(α)·(z - h·θ) / v
"""

import numpy as np


def compute_alpha_v_w(DSO, DSD, h, theta, x, y, z_val):
    """
    Parameters
    ----------
    DSO   : source-to-isocenter distance (mm)
    DSD   : source-to-detector distance  (mm)
    h     : pitch constant (mm/rad), i.e. table_feed / (2*pi)
    theta : 1D array length Ltheta — source angles for this slice window
    x, y  : 1D coordinate arrays (mm)
    z_val : scalar z coordinate of this slice (mm)

    Returns
    -------
    alpha : float32 array (Ltheta, nx, ny)
    v     : float32 array (Ltheta, nx, ny)
    w     : float32 array (Ltheta, nx, ny)
    """
    nx, ny = len(x), len(y)
    Ltheta = len(theta)

    # Build 2-D voxel grid — shape (nx, ny)
    xg, yg = np.meshgrid(x, y, indexing='ij')   # (nx, ny)

    alpha = np.zeros((Ltheta, nx, ny), dtype=np.float32)
    v     = np.zeros((Ltheta, nx, ny), dtype=np.float32)
    w     = np.zeros((Ltheta, nx, ny), dtype=np.float32)

    for i, th in enumerate(theta):
        ct, st = np.cos(th), np.sin(th)
        tmp  = DSO - xg * ct - yg * st                  # v_i
        tmp1 = np.arctan2(-xg * st + yg * ct, tmp)      # α_i
        tmp2 = DSD * np.cos(tmp1) * (z_val - h * th) / tmp  # w_i

        v[i]     = tmp.astype(np.float32)
        alpha[i] = tmp1.astype(np.float32)
        w[i]     = tmp2.astype(np.float32)

    return alpha, v, w
