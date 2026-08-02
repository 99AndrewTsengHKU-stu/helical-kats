"""
solve_pi.py — PI-line endpoint solver (Newton iteration)

Port of wangwei-cmd/Katsevich-algorithm/helical_curve/solve_pi.m

For every voxel (x,y,z) inside the helix cylinder, finds angles s_b, s_t
such that the source positions a(s_b) and a(s_t) define the PI-line passing
through (x,y,z).

Reference:
  Noo et al., "A fast algorithm to compute the pi-line through points inside
  a helix cylinder," Med. Phys. 2004.
"""

import numpy as np


def solve_pi(x, y, z_val, h, R, tol=1e-3, maxiter=100):
    """
    Compute PI-line endpoints for a single z-slice of voxels.

    Parameters
    ----------
    x, y   : 1D arrays of voxel x/y coordinates (mm)
    z_val  : scalar z coordinate for this slice (mm)
    h      : reduced pitch  h = p / (2*pi*R)  (dimensionless)
             where p = table feed per 2*pi rotation
    R      : source-to-isocenter distance DSO (mm)
    tol    : convergence tolerance (sum of absolute residuals)
    maxiter: maximum Newton iterations

    Returns
    -------
    s_b : ndarray shape (nx, ny)  — bottom PI-line angle (rad)
    s_t : ndarray shape (nx, ny)  — top    PI-line angle (rad)
    """
    # Normalise to unit helix radius (R=1)
    xn = x / R          # shape (nx,)
    yn = y / R          # shape (ny,)
    zn = z_val / R      # scalar

    # Build 2-D grids — convention matches MATLAB meshgrid then permute([2,1,3])
    # result shape: (nx, ny)
    xg, yg = np.meshgrid(xn, yn, indexing='ij')  # (nx, ny)
    rho = np.sqrt(xg**2 + yg**2)                  # (nx, ny)

    # Azimuth angle gamma  ∈ [0, 2π)
    gamma = np.real(np.arccos(np.clip(xg / np.where(rho == 0, 1.0, rho), -1, 1)).astype(complex))
    gamma[yg < 0] = 2 * np.pi - gamma[yg < 0]
    gamma[rho == 0] = 0.0

    beta = gamma - zn / h
    beta = np.mod(beta, 2 * np.pi)
    beta[beta > np.pi] -= 2 * np.pi

    # Newton iteration to solve:
    #   g(θ) = θ - ρ·sin(θ)·arccos(ρ·cos(θ)) / sqrt(1 - ρ²·cos²(θ)) = β
    theta = np.zeros_like(rho)
    for _ in range(maxiter):
        c = np.cos(theta)
        s = np.sin(theta)
        denom2 = 1.0 - rho**2 * c**2            # 1 - ρ²cos²θ
        denom2 = np.maximum(denom2, 1e-12)
        sq = np.sqrt(denom2)
        ac = np.real(np.arccos(np.clip(rho * c, -1, 1)).astype(complex))

        g  = theta - rho * s * ac / sq
        g1 = (1.0 - rho**2) / (denom2 * sq) * (sq - rho * c * ac)

        theta = theta - (g - beta) / np.where(np.abs(g1) < 1e-15, 1e-15, g1)
        theta = np.clip(theta, -np.pi, np.pi)

        if np.sum(np.abs(g - beta)) < tol:
            break

    # Recover PI-line angles
    c = np.cos(theta)
    rho_c = rho * c
    alpha_half = np.real(np.arccos(np.clip(rho_c, -1, 1)).astype(complex))
    denom2 = np.maximum(1.0 - rho_c**2, 1e-12)

    lambda_c = zn / h - rho * np.sin(theta) * alpha_half / np.sqrt(denom2)

    s_b = lambda_c - alpha_half   # bottom endpoint angle
    s_t = lambda_c + alpha_half   # top    endpoint angle

    return s_b, s_t
