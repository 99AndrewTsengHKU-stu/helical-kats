"""
rebin_cor.py — Remap detector w → tilted-plane φ coordinate

Port of wangwei-cmd/Katsevich-algorithm/helical_curve/rebin_cor.m

After the ∂/∂s step the sinogram lives on (θ, α, w) axes.  Katsevich's
Hilbert filter must be applied along tilted lines parameterised by φ (the
PI-line tilt).  This function:
  1. Applies the cos-weighting  DSD / sqrt(DSD² + w²)  (flat-to-arc cosine correction)
  2. For each (α, φ) pair, looks up the w-value that lies on the tilted line:
        w_φ(α, φ) = (DSD·h/DSO) · [φ·cos α + φ/tan(φ)·sin α]
     and linearly interpolates the sinogram there.

Output shape: (ntheta, nalpha, nphi)

Sinogram convention: g[ntheta, nalpha, nw]
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def rebin_cor(g1, DSD, h, DSO, cor_phi, cor_alpha, cor_w):
    """
    Parameters
    ----------
    g1        : float32 array (ntheta, nalpha, nw)
    DSD       : source-to-detector distance (mm)
    h         : pitch constant  p/(2π)  (mm/rad)
    DSO       : source-to-isocenter distance (mm)
    cor_phi   : 1D array — φ grid (rad)
    cor_alpha : 1D array — α grid (rad)  [shifted by half-cell vs. input]
    cor_w     : 1D array — w grid (mm)

    Returns
    -------
    g3 : float32 array (ntheta, nalpha, nphi)
    """
    ntheta, nalpha, nw = g1.shape
    assert len(cor_w) == nw
    assert len(cor_alpha) == nalpha

    # Step 1: cos-weighting along w
    dist = np.sqrt(DSD**2 + cor_w**2)           # (nw,)
    # broadcast over (ntheta, nalpha, nw)
    g = g1 * (DSD / dist)                        # (ntheta, nalpha, nw)

    # Step 2: build w_phi(alpha, phi) look-up table
    # MATLAB: [alpha_cor, phi_cor] = meshgrid(cor_alpha, cor_phi) then alpha_cor=alpha_cor'
    # → alpha_cor shape (nalpha, nphi), phi_cor shape (nalpha, nphi)
    alpha2d, phi2d = np.meshgrid(cor_alpha, cor_phi, indexing='ij')  # (nalpha, nphi)

    # Avoid 0/tan(0) = inf:  lim_{φ→0} φ/tan(φ)·sin(α) = sin(α)
    safe_phi = np.where(phi2d == 0, np.finfo(float).eps, phi2d)
    w_phi = (DSD * h / DSO) * (
        phi2d * np.cos(alpha2d)
        + safe_phi / np.tan(safe_phi) * np.sin(alpha2d)
    )
    # edge correction for exactly-zero phi
    w_phi[phi2d == 0] = (DSD * h / DSO) * np.sin(alpha2d[phi2d == 0])
    # shape: (nalpha, nphi)

    nphi = len(cor_phi)
    g3 = np.zeros((ntheta, nalpha, nphi), dtype=np.float32)

    for i in range(ntheta):
        # tmp: (nalpha, nw)
        tmp = g[i]   # (nalpha, nw)

        # MATLAB interp2(alpha_cor, w_cor, tmp', alpha_cor_1, w_phi', 'linear', 0)
        # X=alpha (columns of V), Y=w (rows of V), V=tmp' shape [nw, nalpha]
        # Queries: (alpha_cor_1, w_phi') both shape (nalpha, nphi)
        interp_fn = RegularGridInterpolator(
            (cor_w, cor_alpha),     # (Y, X) axes of V
            tmp.T,                  # V shape [nw, nalpha]
            method='linear',
            bounds_error=False,
            fill_value=0.0,
        )

        # Query points: shape (nalpha*nphi, 2)  order (w, alpha)
        pts = np.stack([w_phi.ravel(), alpha2d.ravel()], axis=-1)
        vals = interp_fn(pts).reshape(nalpha, nphi)   # (nalpha, nphi)

        # MATLAB result: tmp1' → g3(i,:,:)
        # tmp1 = interp result at (alpha_cor_1, w_phi') shape (nphi,nalpha) after MATLAB
        # Then g3(i,:,:)=tmp1'  → (nalpha, nphi)
        # Our vals is already (nalpha, nphi)
        g3[i] = vals.astype(np.float32)

    return g3
