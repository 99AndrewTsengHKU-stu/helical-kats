"""
solve_k_line.py — K-line interpolation index and weight table

Port of wangwei-cmd/Katsevich-algorithm/helical_curve/solve_k_line.m

After H-transform the data lives on (α, φ) grid.  Before backprojection we
need to map back from φ to w via:
    w_φ(α, φ) = (DSD·h/DSO) · [φ·cosα + φ/tanφ · sinα]

This function pre-computes, for every (α, w) detector cell, which φ-index
(k_index) it falls into and the fractional weight for linear interpolation.

The search direction follows wangwei's original:
  α ≥ 0 → scan φ from small to large
  α < 0 → scan φ from large to small

Output
------
k_index : int32  array (nalpha, nw)  — lower φ-index  (0-based)
weight  : float32 array (nalpha, nw) — fractional weight toward upper index
"""

import numpy as np


def solve_k_line(cor_alpha, cor_w, cor_phi, DSD, h, DSO):
    """
    Parameters
    ----------
    cor_alpha : 1D float array — α grid (rad)
    cor_w     : 1D float array — w grid (mm)
    cor_phi   : 1D float array — φ grid (rad)
    DSD, h, DSO : scanner geometry

    Returns
    -------
    k_index : int32  (nalpha, nw)
    weight  : float32 (nalpha, nw)
    """
    nalpha = len(cor_alpha)
    nw     = len(cor_w)
    nphi   = len(cor_phi)

    # w_phi(alpha, phi) — shape (nalpha, nphi)
    alpha2d, phi2d = np.meshgrid(cor_alpha, cor_phi, indexing='ij')  # (nalpha, nphi)
    safe_phi = np.where(phi2d == 0, np.finfo(float).eps, phi2d)
    w_phi = (DSD * h / DSO) * (
        phi2d * np.cos(alpha2d)
        + safe_phi / np.tan(safe_phi) * np.sin(alpha2d)
    )
    w_phi[phi2d == 0] = (DSD * h / DSO) * np.sin(alpha2d[phi2d == 0])
    # shape: (nalpha, nphi)

    weight  = np.zeros((nalpha, nw), dtype=np.float32)
    k_index = np.zeros((nalpha, nw), dtype=np.int32)

    for i in range(nalpha):
        row_w = w_phi[i]   # (nphi,)

        if cor_alpha[i] >= 0:
            phi_range = range(nphi - 1)           # forward search
        else:
            phi_range = range(nphi - 2, -1, -1)   # backward search

        for j in range(nw):
            wj = cor_w[j]
            for k in phi_range:
                if row_w[k] <= wj <= row_w[k + 1]:
                    dw = row_w[k + 1] - row_w[k]
                    weight[i, j]  = (wj - row_w[k]) / dw if dw != 0 else 0.0
                    k_index[i, j] = k          # 0-based
                    break

    return k_index, weight
