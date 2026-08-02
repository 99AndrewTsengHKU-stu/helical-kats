"""
index_theta.py — T-D (theta-delta) windowing weights

Port of wangwei-cmd/Katsevich-algorithm/helical_curve/index_theta.m

For each source angle θ and each voxel (x,y,z), computes the angular weight:
  - delt_theta   if θ is strictly inside [s_b, s_t]
  - partial arc  if θ is at the boundary (fractional cells)
  - 0            otherwise

This encodes the PI-arc measure used in Katsevich backprojection.
"""

import numpy as np


def index_theta(s_b, s_t, theta):
    """
    Compute per-theta voxel weights from PI-line arc endpoints.

    Parameters
    ----------
    s_b   : ndarray shape (nx, ny)  — bottom PI-line angle
    s_t   : ndarray shape (nx, ny)  — top PI-line angle
    theta : 1D array of source angles for this z-slice window

    Returns
    -------
    index : ndarray shape (ntheta, nx, ny), float32
    """
    delt_theta = float(theta[1] - theta[0])
    nx, ny = s_b.shape
    ntheta = len(theta)

    # Cell-boundary grids on the theta axis
    index_R = (s_b - theta[0]) / delt_theta      # float index of s_b
    s_b_R   = np.ceil(index_R) * delt_theta + theta[0]   # next grid point ≥ s_b

    index_L = (s_t - theta[0]) / delt_theta      # float index of s_t
    s_t_L   = np.floor(index_L) * delt_theta + theta[0]  # prev grid point ≤ s_t

    index = np.zeros((ntheta, nx, ny), dtype=np.float32)

    for i, th in enumerate(theta):
        tmp = np.zeros((nx, ny), dtype=np.float32)

        # Fully inside arc
        mask_full = (s_b_R <= th) & (th <= s_t_L)
        tmp[mask_full] = delt_theta

        # Bottom boundary (partial cell)
        mask_bot = (s_b <= th) & (th < s_b_R)
        tmp[mask_bot] = (s_b_R - s_b)[mask_bot]

        # Top boundary (partial cell)
        mask_top = (s_t_L < th) & (th <= s_t)
        tmp[mask_top] = (s_t - s_t_L)[mask_top]

        index[i] = tmp

    return index
