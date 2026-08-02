"""
backproject.py — Helical Katsevich backprojection (one z-slice)

Ports two MATLAB functions:
  • backproj_cor.m   — voxel-driven interpolation + weighted accumulation
  • backproject_helical.m — orchestrates the full backprojection pipeline

Pipeline for a single z-slice:
    1. compute_grad_s        :  ∂g/∂s
    2. rebin_cor             :  (θ, α, w) → (θ, α, φ)
    3. htransform_matrix     :  Hilbert filter along α
    4. solve_k_line          :  φ → w inverse-map table
    5. g[α, φ, θ] → g[α, w, θ]  via k-line table
    6. cos(α) weighting
    7. backproj_cor          :  interpolate + accumulate into voxel grid
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .compute_grad_s import compute_grad_s
from .rebin_cor       import rebin_cor
from .htransform      import htransform_matrix
from .solve_k_line    import solve_k_line


# ---------------------------------------------------------------------------
# backproj_cor
# ---------------------------------------------------------------------------

def backproj_cor(g_c, alpha_star, v_star, w_star, cor_alpha, cor_w, index):
    """
    Voxel-driven backprojection accumulator.

    Parameters
    ----------
    g_c        : float32 (Ltheta, nalpha, nw) — filtered sinogram
    alpha_star : float32 (Ltheta, nx, ny)     — voxel fan-angles
    v_star     : float32 (Ltheta, nx, ny)     — virtual SDD per voxel
    w_star     : float32 (Ltheta, nx, ny)     — detector row per voxel
    cor_alpha  : 1D (nalpha,)
    cor_w      : 1D (nw,)
    index      : float32 (Ltheta, nx, ny)     — T-D arc weights

    Returns
    -------
    img : float32 (nx, ny)
    """
    Ltheta, nalpha, nw = g_c.shape
    _, nx, ny = alpha_star.shape

    img = np.zeros((nx, ny), dtype=np.float32)

    for i in range(Ltheta):
        tmp = g_c[i]   # (nalpha, nw)

        # MATLAB interp2(alpha_cor, w_cor, tmp', alpha_q, w_q)
        # X=alpha (columns of V), Y=w (rows), V=tmp' shape [nw, nalpha]
        interp_fn = RegularGridInterpolator(
            (cor_w, cor_alpha),
            tmp.T,                  # (nw, nalpha)
            method='linear',
            bounds_error=False,
            fill_value=0.0,
        )
        alpha_q = alpha_star[i].ravel()   # (nx*ny,)
        w_q     = w_star[i].ravel()

        pts   = np.stack([w_q, alpha_q], axis=-1)
        temp1 = interp_fn(pts).reshape(nx, ny).astype(np.float32)

        img += temp1 * index[i] / v_star[i]

    return img / (2.0 * np.pi)


# ---------------------------------------------------------------------------
# backproject_helical
# ---------------------------------------------------------------------------

def backproject_helical(
    sino_slice,       # (Ltheta, nalpha, nw)  raw sinogram window
    DSD, h, DSO,
    phi_cor,          # (nphi,)
    alpha_cor,        # (nalpha,)
    w_cor,            # (nw,)
    delt_alpha,
    delt_theta,
    alpha_s,          # (Ltheta, nx, ny)  — voxel fan-angles
    v_s,              # (Ltheta, nx, ny)  — virtual SDD
    w_s,              # (Ltheta, nx, ny)  — detector rows
    ind,              # (Ltheta, nx, ny)  — T-D weights
    Ltheta,
):
    """
    Full one-slice Katsevich backprojection.

    Returns
    -------
    rf : float32 (nx, ny)
    """
    # Step 1: ∂/∂s
    g1 = compute_grad_s(sino_slice, delt_alpha, delt_theta)

    # Step 2: rebin w → φ   (use half-cell-shifted alpha for the filter)
    alpha_shift = alpha_cor + 0.5 * delt_alpha
    g1 = rebin_cor(g1, DSD, h, DSO, phi_cor, alpha_shift, w_cor)
    # g1: (Ltheta, nalpha, nphi)

    # Step 3: Hilbert filter along α
    g1 = htransform_matrix(g1, alpha_shift)
    # g1: (Ltheta, nalpha, nphi)

    # Step 4: k-line table (φ → w)
    k_index, k_weight = solve_k_line(alpha_cor, w_cor, phi_cor, DSD, h, DSO)
    # k_index, k_weight: (nalpha, nw)

    # Step 5: remap (Ltheta, nalpha, nphi) → (Ltheta, nalpha, nw) via k-line
    # MATLAB:
    #   g1 = permute(g1,[2,3,1])   → (nalpha, nphi, Ltheta)
    #   g1 = padarray(g1,[2,2,0])  → pad 2 zeros along nalpha and nphi dims
    #   for i in range(nalpha):
    #     g_5[i,:,:] = g1[i, k_index_1[i,:], :Ltheta]*(1-w) + g1[i, k_index_1[i,:]+1, :Ltheta]*w
    #   g_5: (nalpha, nw, Ltheta)
    nalpha, nw_out = k_index.shape
    nphi = g1.shape[2]

    # permute to (nalpha, nphi, Ltheta) then pad
    g_perm = np.transpose(g1, (1, 2, 0))        # (nalpha, nphi, Ltheta)
    # pad 2 zeros at end of nalpha and nphi dims
    g_pad  = np.pad(g_perm, ((0, 2), (0, 2), (0, 0)))  # (nalpha+2, nphi+2, Ltheta)

    # handle k_index==0 → use last valid phi row (same as MATLAB k_index_1)
    k_index_1 = k_index.copy()
    k_index_1[k_index == 0] = nphi - 1    # point to padded zeros region safely

    g_5 = np.zeros((nalpha, nw_out, Ltheta), dtype=np.float32)
    for i in range(nalpha):
        ki  = k_index_1[i]                # (nw_out,)
        kw  = k_weight[i]                 # (nw_out,)
        # lower and upper phi rows for this alpha
        lo  = g_pad[i, ki,     :Ltheta]   # (nw_out, Ltheta)
        hi  = g_pad[i, ki + 1, :Ltheta]   # (nw_out, Ltheta)
        g_5[i] = (lo * (1.0 - kw[:, None]) + hi * kw[:, None]).astype(np.float32)

    # permute to (Ltheta, nalpha, nw)
    g_5 = np.transpose(g_5, (2, 0, 1))    # (Ltheta, nalpha, nw)

    # Step 6: cos(α) weighting
    cos_a = np.cos(alpha_cor).astype(np.float32)   # (nalpha,)
    g_5 = g_5 * cos_a[None, :, None]              # broadcast over (Ltheta, nalpha, nw)

    # Step 7: backproject
    rf = backproj_cor(g_5, alpha_s, v_s, w_s, alpha_cor, w_cor, ind)
    return rf
