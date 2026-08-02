"""
htransform.py — Hilbert-like filter applied along the α (fan-angle) axis

Port of wangwei-cmd/Katsevich-algorithm/helical_curve/Htransform_matrix.m

The Katsevich filter kernel in the rebinned domain is:

    h(t) = (1 - cos(2π·b·t)) / (π·t)      b = 1/(2·Δα)

with optional windowing (Hamming applied here, matching wangwei's default).

The filter is applied as a matrix–vector multiply for each (θ, φ) column,
which is equivalent to a 1-D finite convolution along the α dimension.

Input/output shape: (ntheta, nalpha, nphi)
"""

import numpy as np
from scipy.signal import windows


def _make_hilbert_kernel(u_cor):
    """
    Build the bandlimited Hilbert kernel evaluated on the shifted grid.

    MATLAB: uu = [-L+1 : L-1]*delt_u - 0.5*delt_u
            h_k = (1-cos(2π·b·uu))/(π·uu)    b = 1/(2·delt_u)
            then Hamming-windowed via FFT.
    """
    delt_u = float(u_cor[1] - u_cor[0])
    L = len(u_cor)
    b = 0.5 / delt_u

    # Shifted lag grid: length 2L-1
    lags = np.arange(-(L - 1), L) * delt_u - 0.5 * delt_u  # (2L-1,)

    # Raw kernel
    h_k = np.where(lags == 0, 0.0, (1.0 - np.cos(2 * np.pi * b * lags)) / (np.pi * lags))

    # Hamming window in frequency domain (matching wangwei)
    fht = np.fft.fft(h_k)
    win = np.fft.fftshift(windows.hamming(len(h_k)))
    h_k = np.real(np.fft.ifft(fht * win))

    return h_k.astype(np.float32)


def _make_filter_matrix(u_cor):
    """
    Build the convolution matrix  H  of shape (L, L).

    Row i of H contains h_k[i+L-1 : i-1 : -1]  (i.e. reversed lags centred at i).
    Multiplying H @ v gives the causal Hilbert-filtered vector v.
    """
    h_k = _make_hilbert_kernel(u_cor)
    L = len(u_cor)
    H = np.zeros((L, L), dtype=np.float32)
    for i in range(L):
        # MATLAB: h_matrix(i,:) = h_k(i+L-1 : -1 : i)   (1-indexed)
        # Python (0-indexed): slice from index (i+L-1) down to i, step -1
        H[i, :] = h_k[i + L - 1: i - 1 if i > 0 else None: -1]
    return H


def htransform_matrix(g3, u_cor):
    """
    Apply Hilbert filter along α for every (θ, φ) pair.

    Parameters
    ----------
    g3    : float32 array (ntheta, nalpha, nphi)
    u_cor : 1D array — α grid used to build the kernel  (length nalpha)

    Returns
    -------
    g3_filtered : float32 array (ntheta, nalpha, nphi)
    """
    delt_u = float(u_cor[1] - u_cor[0])
    H = _make_filter_matrix(u_cor)    # (nalpha, nalpha)

    # MATLAB: pagemtimes(H, permute(g3,[2,1,3])) * delt_u
    #   permute(g3,[2,1,3]):  (ntheta,nalpha,nphi) → (nalpha, ntheta, nphi)
    #   pagemtimes(H, ...)  : (nalpha, ntheta, nphi)
    #   permute([2,1,3])    : → (ntheta, nalpha, nphi)
    #
    # In NumPy: einsum('ij, jkl -> ikl', H, g3.T(1,0,2)) then .T(1,0,2)
    perm = np.transpose(g3, (1, 0, 2))            # (nalpha, ntheta, nphi)
    out  = np.einsum('ij,jkl->ikl', H, perm, optimize=True) * delt_u
    return np.transpose(out, (1, 0, 2)).astype(np.float32)   # (ntheta, nalpha, nphi)
