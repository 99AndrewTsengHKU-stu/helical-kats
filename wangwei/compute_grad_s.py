"""
compute_grad_s.py — Mixed partial derivative ∂g/∂s of the sinogram

Port of wangwei-cmd/Katsevich-algorithm/helical_curve/compute_grad_s.m

Computes the Katsevich differentiation step:
    ∂g/∂s ≈ (∂/∂θ)g + (∂/∂α)g

using a central difference along θ (averaged over the two w columns) and
a forward difference along α.

Sinogram convention throughout this package:
    g[ntheta, nalpha, nw]

MATLAB original used shape [nalpha, nw, ntheta] and permuted before/after;
we work directly in [ntheta, nalpha, nw] to avoid confusion.
"""

import numpy as np


def compute_grad_s(pf, delt_alpha, delt_theta):
    """
    Parameters
    ----------
    pf          : float32 array (ntheta, nalpha, nw)  — raw sinogram slice
    delt_alpha  : float — detector column spacing (rad)
    delt_theta  : float — source angle step (rad)

    Returns
    -------
    g1 : float32 array (ntheta, nalpha, nw)  — differentiated sinogram
    """
    # Work in MATLAB dimension order [ntheta, nw, nalpha] for the derivative
    # so that axes match the original code comments exactly.
    # MATLAB: pf=permute(pf,[3,2,1])  with pf originally [nalpha,nw,ntheta]
    #         → [ntheta, nw, nalpha]
    # Our pf is already [ntheta, nalpha, nw], so we permute to [ntheta, nw, nalpha]:
    p = np.transpose(pf, (0, 2, 1)).astype(np.float32)  # (ntheta, nw, nalpha)

    ntheta, nw, nalpha = p.shape

    d_proj = np.zeros_like(p)
    d_col  = np.zeros_like(p)

    # ∂/∂θ  — central difference averaged over adjacent w rows (staggered)
    # MATLAB: d_proj(2:end-1,1:end-1,:) = (p(3:end,1:end-1,:)-p(1:end-2,1:end-1,:))/4/delt_theta
    #                                    + (p(3:end,2:end,:)  -p(1:end-2,2:end,:)  )/4/delt_theta
    d_proj[1:-1, :-1, :] = (
        (p[2:, :-1, :] - p[:-2, :-1, :]) / 4.0 / delt_theta
        + (p[2:, 1:,  :] - p[:-2, 1:,  :]) / 4.0 / delt_theta
    )

    # ∂/∂α  — forward difference along nalpha (last axis)
    # MATLAB: d_col(:,:,1:end-1) = (p(:,:,2:end)-p(:,:,1:end-1))/delt_alpha
    d_col[:, :, :-1] = (p[:, :, 1:] - p[:, :, :-1]) / delt_alpha

    g = d_proj + d_col  # (ntheta, nw, nalpha)

    # Permute back to [ntheta, nalpha, nw]
    # MATLAB: g1=permute(g1,[1,3,2])  on [ntheta,nw,nalpha] → [ntheta,nalpha,nw]
    return np.transpose(g, (0, 2, 1)).astype(np.float32)
