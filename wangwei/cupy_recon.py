"""CuPy implementation of the Wang Wei Katsevich reconstruction pipeline.

The implementation follows the validated MATLAB functions in
``wangwei-katsevich/helical_curve`` while keeping the two largest arrays out
of host RAM:

* projection data are read from HDF5 in reversed/decimated view chunks;
* the pre-filtered sinogram is written directly to a NumPy ``.npy`` memmap.

The per-slice CUDA kernel fuses detector geometry, T-D weights, bilinear
interpolation, and accumulation.  This avoids materialising four
``(n_theta, 512, 512)`` arrays for every reconstructed slice.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable

import cupy as cp
import h5py
import numpy as np

from .solve_pi import solve_pi


_BACKPROJ_GEOMETRY_KERNEL = cp.RawKernel(
    r'''
extern "C" __global__ void backproj_geometry_fused(
    const float* __restrict__ g_c,       // (ntheta, nalpha, nw)
    const float* __restrict__ s_b,       // (nx, ny), x-major
    const float* __restrict__ s_t,       // (nx, ny), x-major
    const float* __restrict__ x_cor,
    const float* __restrict__ y_cor,
    float* __restrict__ output,          // (ny, nx), MATLAB image order
    const int ntheta, const int nalpha, const int nw,
    const int nx, const int ny,
    const float theta0, const float dtheta,
    const float z_value, const float h,
    const float DSO, const float DSD,
    const float alpha0, const float d_alpha,
    const float w0, const float d_w
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;

    const int xy_index = ix * ny + iy;
    const float sb = s_b[xy_index];
    const float st = s_t[xy_index];
    const float sb_right = ceilf((sb - theta0) / dtheta) * dtheta + theta0;
    const float st_left  = floorf((st - theta0) / dtheta) * dtheta + theta0;
    const float x = x_cor[ix];
    const float y = y_cor[iy];
    const int theta_stride = nalpha * nw;

    float acc = 0.0f;
    for (int it = 0; it < ntheta; ++it) {
        const float theta = theta0 + ((float)it) * dtheta;
        float td_weight = 0.0f;

        if (sb_right <= theta && theta <= st_left) {
            td_weight = dtheta;
        } else if (sb <= theta && theta < sb_right) {
            td_weight = sb_right - sb;
        } else if (st_left < theta && theta <= st) {
            td_weight = st - st_left;
        }
        if (td_weight <= 0.0f) continue;

        float sin_theta, cos_theta;
        sincosf(theta, &sin_theta, &cos_theta);
        const float v = DSO - x * cos_theta - y * sin_theta;
        if (v <= 0.0f) continue;

        const float alpha = atan2f(-x * sin_theta + y * cos_theta, v);
        const float w = DSD * cosf(alpha) * (z_value - h * theta) / v;
        const float alpha_index = (alpha - alpha0) / d_alpha;
        const float w_index = (w - w0) / d_w;

        if (alpha_index < 0.0f || alpha_index > (float)(nalpha - 1) ||
            w_index < 0.0f || w_index > (float)(nw - 1)) {
            continue;
        }

        int ia = (int)floorf(alpha_index);
        int iw = (int)floorf(w_index);
        float fa = alpha_index - (float)ia;
        float fw = w_index - (float)iw;

        // MATLAB interp2 includes points exactly on the upper grid boundary.
        if (ia == nalpha - 1) { ia = nalpha - 2; fa = 1.0f; }
        if (iw == nw - 1)     { iw = nw - 2;     fw = 1.0f; }

        const int base = it * theta_stride + ia * nw + iw;
        const float value =
              g_c[base]              * (1.0f - fa) * (1.0f - fw)
            + g_c[base + nw]         * fa          * (1.0f - fw)
            + g_c[base + 1]          * (1.0f - fa) * fw
            + g_c[base + nw + 1]     * fa          * fw;

        acc += value * td_weight / v;
    }

    // MATLAB arrays are stored as image rows=y, columns=x.
    output[iy * nx + ix] = acc * 0.15915494309189535f;  // 1/(2*pi)
}
''',
    "backproj_geometry_fused",
)


def compute_grad_s_cupy(
    sino: cp.ndarray, delt_alpha: float, delt_theta: float
) -> cp.ndarray:
    """Port of ``compute_grad_s.m`` for ``(theta, alpha, w)`` input."""
    pf = sino.transpose(0, 2, 1)  # MATLAB intermediate: (theta, w, alpha)
    d_proj = cp.zeros_like(pf)
    d_col = cp.zeros_like(pf)

    d_proj[1:-1, :-1, :] = (
        (pf[2:, :-1, :] - pf[:-2, :-1, :]) / (4.0 * delt_theta)
        + (pf[2:, 1:, :] - pf[:-2, 1:, :]) / (4.0 * delt_theta)
    )
    d_col[:, :, :-1] = (pf[:, :, 1:] - pf[:, :, :-1]) / delt_alpha
    return (d_proj + d_col).transpose(0, 2, 1)


def make_hilbert_matrix(u_cor: np.ndarray) -> np.ndarray:
    """Build the exact finite Hilbert matrix used by ``Htransform_matrix.m``."""
    # Preserve the spacing of the incoming single-precision detector grid.
    # The bandlimited kernel is sensitive to this value; recomputing the
    # spacing after a float64 cast does not match MATLAB's single GPU path.
    u_cor = np.asarray(u_cor)
    delt_u = float(u_cor[1] - u_cor[0])
    length = len(u_cor)
    uu = np.arange(-length + 1, length, dtype=np.float64) * delt_u - 0.5 * delt_u
    sin_u = np.sin(uu)
    b = 1.0 / (2.0 * delt_u)

    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = (1.0 - np.cos(2.0 * np.pi * b * sin_u)) / (np.pi * sin_u)
    kernel[sin_u == 0] = 0.0
    kernel = np.real(
        np.fft.ifft(np.fft.fft(kernel) * np.fft.fftshift(np.hamming(len(kernel))))
    )
    kernel[~np.isfinite(kernel)] = 0.0

    matrix = np.empty((length, length), dtype=np.float32)
    for index in range(length):
        matrix[index] = kernel[index : index + length][::-1]
    return matrix


def _phi_over_tan(phi: np.ndarray) -> np.ndarray:
    result = np.ones_like(phi, dtype=np.float64)
    nonzero = np.abs(phi) > 1e-12
    result[nonzero] = phi[nonzero] / np.tan(phi[nonzero])
    return result


def solve_k_line_lut(
    alpha_cor: np.ndarray,
    w_cor: np.ndarray,
    phi_cor: np.ndarray,
    DSD: float,
    h: float,
    DSO: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return zero-based k-line indices with an explicit validity mask.

    The MATLAB implementation uses ``0`` as the no-match sentinel while valid
    indices are one-based.  A direct zero-based port therefore cannot also use
    ``0`` as the sentinel.  This function uses ``-1`` for no match and returns
    ``valid`` separately.
    """
    alpha_cor = np.asarray(alpha_cor, dtype=np.float64)
    w_cor = np.asarray(w_cor, dtype=np.float64)
    phi_cor = np.asarray(phi_cor, dtype=np.float64)
    alpha_grid, phi_grid = np.meshgrid(alpha_cor, phi_cor, indexing="ij")
    w_phi = (DSD * h / DSO) * (
        phi_grid * np.cos(alpha_grid)
        + _phi_over_tan(phi_grid) * np.sin(alpha_grid)
    )

    nalpha, nphi = w_phi.shape
    nw = len(w_cor)
    k_index = np.full((nalpha, nw), -1, dtype=np.int32)
    weight = np.zeros((nalpha, nw), dtype=np.float32)

    for ia in range(nalpha):
        row = w_phi[ia]
        for iw, w_value in enumerate(w_cor):
            matches = np.flatnonzero((row[:-1] <= w_value) & (w_value <= row[1:]))
            if matches.size == 0:
                continue
            k = int(matches[0] if alpha_cor[ia] >= 0 else matches[-1])
            width = row[k + 1] - row[k]
            k_index[ia, iw] = k
            if width != 0:
                weight[ia, iw] = np.float32((w_value - row[k]) / width)

    valid = k_index >= 0
    return k_index, weight, valid


def _build_prefilter_constants(
    DSD: float,
    h: float,
    DSO: float,
    alpha_cor: np.ndarray,
    w_cor: np.ndarray,
    phi_cor: np.ndarray,
    delt_alpha: float,
) -> dict[str, cp.ndarray | float]:
    alpha_shift = np.asarray(alpha_cor, dtype=np.float64) + 0.5 * delt_alpha
    w_cor64 = np.asarray(w_cor, dtype=np.float64)
    phi_cor64 = np.asarray(phi_cor, dtype=np.float64)
    d_w = float(w_cor64[1] - w_cor64[0])

    alpha_mesh, phi_mesh = np.meshgrid(alpha_shift, phi_cor64, indexing="ij")
    w_phi = (DSD * h / DSO) * (
        phi_mesh * np.cos(alpha_mesh)
        + _phi_over_tan(phi_mesh) * np.sin(alpha_mesh)
    )
    w_index = (w_phi - w_cor64[0]) / d_w
    valid_rebin = (w_index >= 0.0) & (w_index <= len(w_cor64) - 1)
    lower_w = np.clip(np.floor(w_index).astype(np.int32), 0, len(w_cor64) - 2)
    fraction_w = (w_index - lower_w).astype(np.float32)

    k_index, k_weight, valid_k = solve_k_line_lut(
        alpha_cor, w_cor, phi_cor, DSD, h, DSO
    )
    safe_k = np.where(valid_k, k_index, 0).astype(np.int32)

    return {
        "lower_w": cp.asarray(lower_w),
        "fraction_w": cp.asarray(fraction_w),
        "valid_rebin": cp.asarray(valid_rebin.astype(np.float32)),
        "hilbert": cp.asarray(make_hilbert_matrix(alpha_shift)),
        "delt_u": float(alpha_shift[1] - alpha_shift[0]),
        "safe_k": cp.asarray(safe_k),
        "k_weight": cp.asarray(k_weight),
        "valid_k": cp.asarray(valid_k.astype(np.float32)),
        "cos_alpha": cp.asarray(np.cos(alpha_cor).astype(np.float32)),
        "cos_w": cp.asarray(
            (DSD / np.sqrt(DSD**2 + w_cor64**2)).astype(np.float32)
        ),
    }


def prefilter_chunk_cupy(
    sino_chunk: np.ndarray,
    constants: dict[str, cp.ndarray | float],
    delt_alpha: float,
    delt_theta: float,
) -> cp.ndarray:
    """Apply MATLAB pre-filter steps 1-6 to one view chunk."""
    chunk = cp.asarray(np.ascontiguousarray(sino_chunk), dtype=cp.float32)
    ntheta, nalpha, _ = chunk.shape

    g1 = compute_grad_s_cupy(chunk, delt_alpha, delt_theta)
    del chunk
    g1 *= constants["cos_w"][None, None, :]

    alpha_index = cp.arange(nalpha, dtype=cp.int32)[:, None]
    lower_w = constants["lower_w"]
    fraction_w = constants["fraction_w"]
    g_lo = g1[:, alpha_index, lower_w]
    g_hi = g1[:, alpha_index, lower_w + 1]
    g3 = (g_lo * (1.0 - fraction_w) + g_hi * fraction_w)
    g3 *= constants["valid_rebin"][None, :, :]
    del g1, g_lo, g_hi

    nphi = g3.shape[2]
    g3_flat = cp.ascontiguousarray(g3.transpose(0, 2, 1)).reshape(-1, nalpha)
    del g3
    filtered_flat = g3_flat @ constants["hilbert"].T
    del g3_flat
    filtered = filtered_flat.reshape(ntheta, nphi, nalpha).transpose(0, 2, 1)
    filtered *= constants["delt_u"]
    del filtered_flat

    safe_k = constants["safe_k"]
    k_weight = constants["k_weight"]
    g_lo = filtered[:, alpha_index, safe_k]
    g_hi = filtered[:, alpha_index, safe_k + 1]
    remapped = (g_lo * (1.0 - k_weight) + g_hi * k_weight)
    remapped *= constants["valid_k"][None, :, :]
    del filtered, g_lo, g_hi

    remapped *= constants["cos_alpha"][None, :, None]
    return remapped


def _read_reversed_decimated_chunk(
    projection_dataset: h5py.Dataset,
    start: int,
    end: int,
    decimate: int,
    nviews: int,
) -> np.ndarray:
    """Read transformed views matching MATLAB's decimate-then-flip sequence."""
    selected_start = nviews - end
    selected_end = nviews - start
    raw_start = selected_start * decimate
    raw_stop = selected_end * decimate
    block = projection_dataset[raw_start:raw_stop:decimate]
    expected = end - start
    if len(block) != expected:
        raise RuntimeError(
            f"HDF5 chunk length mismatch: expected {expected}, got {len(block)}"
        )
    # HDF5: (theta, detector_row, detector_col)
    # MATLAB after h5read+flip: (detector_col, detector_row, theta), flipped
    # along detector columns and theta.  Return Python (theta, alpha, w).
    return np.ascontiguousarray(block[::-1, :, ::-1].transpose(0, 2, 1))


def prefilter_h5_to_memmap(
    h5_path: str | Path,
    cache_path: str | Path,
    *,
    decimate: int,
    chunk_theta: int,
    DSD: float,
    h: float,
    DSO: float,
    alpha_cor: np.ndarray,
    w_cor: np.ndarray,
    phi_cor: np.ndarray,
    delt_alpha: float,
    delt_theta: float,
    progress: Callable[[str], None] = print,
) -> np.memmap:
    """Stream HDF5 projections through CuPy into an on-disk filtered cache."""
    h5_path = Path(h5_path)
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as handle:
        projections = handle["projections"]
        nraw, nw, nalpha = projections.shape
        nviews = math.ceil(nraw / decimate)
        if nalpha != len(alpha_cor) or nw != len(w_cor):
            raise ValueError(
                f"Detector mismatch: HDF5 {(nalpha, nw)} vs grids "
                f"{(len(alpha_cor), len(w_cor))}"
            )

        output = np.lib.format.open_memmap(
            cache_path,
            mode="w+",
            dtype=np.float32,
            shape=(nviews, nalpha, nw),
        )
        constants = _build_prefilter_constants(
            DSD, h, DSO, alpha_cor, w_cor, phi_cor, delt_alpha
        )
        margin = 1
        nchunks = math.ceil(nviews / chunk_theta)
        started = time.perf_counter()

        for chunk_index in range(nchunks):
            start = chunk_index * chunk_theta
            end = min(start + chunk_theta, nviews)
            local_start = max(start - margin, 0)
            local_end = min(end + margin, nviews)
            keep_start = start - local_start
            keep_end = end - local_start

            raw_chunk = _read_reversed_decimated_chunk(
                projections, local_start, local_end, decimate, nviews
            )
            filtered = prefilter_chunk_cupy(
                raw_chunk, constants, delt_alpha, delt_theta
            )
            output[start:end] = cp.asnumpy(filtered[keep_start:keep_end])
            output.flush()
            del raw_chunk, filtered
            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()

            elapsed = time.perf_counter() - started
            progress(
                f"  prefilter {chunk_index + 1}/{nchunks}: views "
                f"{start + 1}..{end}  elapsed={elapsed:.1f}s"
            )

    metadata = {
        "h5_path": str(h5_path.resolve()),
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "decimate": decimate,
        "chunk_theta": chunk_theta,
        "DSD": DSD,
        "DSO": DSO,
        "h": h,
        "delt_alpha": delt_alpha,
        "delt_theta": delt_theta,
    }
    cache_path.with_suffix(cache_path.suffix + ".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return output


def backproject_slice_cupy(
    g_window: np.ndarray,
    s_b: np.ndarray,
    s_t: np.ndarray,
    x_cor: np.ndarray,
    y_cor: np.ndarray,
    *,
    theta0: float,
    delt_theta: float,
    z_value: float,
    h: float,
    DSO: float,
    DSD: float,
    alpha_cor: np.ndarray,
    w_cor: np.ndarray,
) -> np.ndarray:
    """Run fused geometry, T-D weighting, interpolation, and accumulation."""
    g_gpu = cp.asarray(np.ascontiguousarray(g_window), dtype=cp.float32)
    sb_gpu = cp.asarray(np.ascontiguousarray(s_b), dtype=cp.float32)
    st_gpu = cp.asarray(np.ascontiguousarray(s_t), dtype=cp.float32)
    x_gpu = cp.asarray(x_cor, dtype=cp.float32)
    y_gpu = cp.asarray(y_cor, dtype=cp.float32)

    ntheta, nalpha, nw = g_gpu.shape
    nx, ny = len(x_cor), len(y_cor)
    output = cp.zeros((ny, nx), dtype=cp.float32)
    block = (16, 16)
    grid = ((nx + block[0] - 1) // block[0], (ny + block[1] - 1) // block[1])

    _BACKPROJ_GEOMETRY_KERNEL(
        grid,
        block,
        (
            g_gpu,
            sb_gpu,
            st_gpu,
            x_gpu,
            y_gpu,
            output,
            np.int32(ntheta),
            np.int32(nalpha),
            np.int32(nw),
            np.int32(nx),
            np.int32(ny),
            np.float32(theta0),
            np.float32(delt_theta),
            np.float32(z_value),
            np.float32(h),
            np.float32(DSO),
            np.float32(DSD),
            np.float32(alpha_cor[0]),
            np.float32(alpha_cor[1] - alpha_cor[0]),
            np.float32(w_cor[0]),
            np.float32(w_cor[1] - w_cor[0]),
        ),
    )
    result = cp.asnumpy(output)
    del g_gpu, sb_gpu, st_gpu, x_gpu, y_gpu, output
    cp.get_default_memory_pool().free_all_blocks()
    return result


def reconstruct_prefiltered_cupy(
    g_filt: np.ndarray,
    theta: np.ndarray,
    theta_offset: float,
    p: float,
    DSD: float,
    DSO: float,
    x_cor: np.ndarray,
    y_cor: np.ndarray,
    z_cor: np.ndarray,
    alpha_cor: np.ndarray,
    w_cor: np.ndarray,
    progress: Callable[[str], None] = print,
) -> tuple[np.ndarray, list[dict[str, float | int | bool]]]:
    """Reconstruct requested z coordinates from a pre-filtered sinogram."""
    h = p / (2.0 * np.pi)
    delt_theta = float(theta[1] - theta[0])
    z_cor_rec = np.asarray(z_cor, dtype=np.float32) + h * theta_offset
    nviews = g_filt.shape[0]

    rf = np.zeros((len(y_cor), len(x_cor), len(z_cor)), dtype=np.float32)
    records: list[dict[str, float | int | bool]] = []

    for iz, z_value_np in enumerate(z_cor_rec):
        slice_started = time.perf_counter()
        z_value = float(z_value_np)
        s_b, s_t = solve_pi(
            x_cor,
            y_cor,
            z_value,
            h=h / DSO,
            R=DSO,
            tol=1e-3,
            maxiter=100,
        )

        raw_t1 = int(
            np.floor((float(np.min(s_b)) - theta[0] - theta_offset) / delt_theta)
        )
        raw_t2 = int(
            np.ceil((float(np.max(s_t)) - theta[0] - theta_offset) / delt_theta)
        ) + 1
        t1 = max(raw_t1, 0)
        t2 = min(raw_t2, nviews)
        clipped = t1 != raw_t1 or t2 != raw_t2
        if t2 - t1 < 2:
            raise RuntimeError(f"Slice z={z_value:.3f} has only {t2 - t1} usable views")

        g_window = np.asarray(g_filt[t1:t2])
        rf[:, :, iz] = backproject_slice_cupy(
            g_window,
            s_b,
            s_t,
            x_cor,
            y_cor,
            theta0=float(theta[t1] + theta_offset),
            delt_theta=delt_theta,
            z_value=z_value,
            h=h,
            DSO=DSO,
            DSD=DSD,
            alpha_cor=alpha_cor,
            w_cor=w_cor,
        )
        elapsed = time.perf_counter() - slice_started
        record: dict[str, float | int | bool] = {
            "slice": iz,
            "z_input_mm": float(z_cor[iz]),
            "z_reconstruction_mm": z_value,
            "t1": t1,
            "t2": t2,
            "nviews": t2 - t1,
            "clipped": clipped,
            "elapsed_s": elapsed,
        }
        records.append(record)
        progress(
            f"  z[{iz + 1}/{len(z_cor)}]={z_value:.1f} mm  "
            f"views={t2 - t1}  {elapsed:.2f}s"
            + ("  [CLIPPED]" if clipped else "")
        )

    return rf, records
