# -----------------------------------------------------------------------
# GPU-accelerated Katsevich filter using CuPy.
# Drop-in replacement for filter.py CPU functions.
# -----------------------------------------------------------------------

import numpy as np
import cupy as cp
from time import time


# ── Differentiation CUDA kernel ─────────────────────────────────────────
_differentiate_kernel = cp.RawKernel(r'''
extern "C" __global__
void differentiate_kernel(
    const float* sino,      // (V, R, C)
    float* out,             // (V, R, C)
    int V, int R, int C,
    float delta_s, float pixel_height, float pixel_span,
    float dia, float dia_sqr,
    const float* col_coords,  // (C-1,)
    const float* row_coords   // (R-1,)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (V - 1) * (R - 1) * (C - 1);
    if (idx >= total) return;

    int c = idx % (C - 1);
    int tmp = idx / (C - 1);
    int r = tmp % (R - 1);
    int p = tmp / (R - 1);

    // Helper macros for sinogram indexing
    #define S(pp, rr, cc) sino[(pp)*R*C + (rr)*C + (cc)]

    float d_proj = (S(p+1,r,c) - S(p,r,c) + S(p+1,r+1,c) - S(p,r+1,c)
                  + S(p+1,r,c+1) - S(p,r,c+1) + S(p+1,r+1,c+1) - S(p,r+1,c+1))
                  / (4.0f * delta_s);

    float d_row = (S(p,r+1,c) - S(p,r,c) + S(p,r+1,c+1) - S(p,r,c+1)
                 + S(p+1,r+1,c) - S(p+1,r,c) + S(p+1,r+1,c+1) - S(p+1,r,c+1))
                 / (4.0f * pixel_height);

    float d_col = (S(p,r,c+1) - S(p,r,c) + S(p,r+1,c+1) - S(p,r+1,c)
                 + S(p+1,r,c+1) - S(p+1,r,c) + S(p+1,r+1,c+1) - S(p+1,r+1,c))
                 / (4.0f * pixel_span);

    #undef S

    float cc = col_coords[c];
    float rr = row_coords[r];
    float cc2 = cc * cc;
    float rr2 = rr * rr;

    float val = d_proj + d_col * (cc2 + dia_sqr) / dia + d_row * (cc * rr) / dia;
    val *= dia / sqrtf(cc2 + dia_sqr + rr2);

    out[p * R * C + r * C + c] = val;
}
''', 'differentiate_kernel')


# ── Forward rebinning CUDA kernel ───────────────────────────────────────
_fw_rebin_kernel = cp.RawKernel(r'''
extern "C" __global__
void fw_rebin_kernel(
    const float* input,       // (V, det_rows, det_cols)
    float* output,            // (V, rebin_rows, det_cols)
    const float* fwd_rebin_row, // (rebin_rows, det_cols)
    int V, int det_rows, int det_cols, int rebin_rows,
    float pixel_height, float det_row_offset
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = V * rebin_rows * det_cols;
    if (idx >= total) return;

    int c = idx % det_cols;
    int tmp = idx / det_cols;
    int rb = tmp % rebin_rows;
    int p = tmp / rebin_rows;

    float row_scaled = fwd_rebin_row[rb * det_cols + c] / pixel_height
                     + 0.5f * det_rows - det_row_offset;

    // Clamp
    if (row_scaled < 0.0f) row_scaled = 0.0f;
    if (row_scaled > (float)(det_rows - 2)) row_scaled = (float)(det_rows - 2);

    int row_i = (int)floorf(row_scaled);
    float frac = row_scaled - (float)row_i;

    int base = p * det_rows * det_cols;
    output[p * rebin_rows * det_cols + rb * det_cols + c] =
        (1.0f - frac) * input[base + row_i * det_cols + c]
        + frac * input[base + (row_i + 1) * det_cols + c];
}
''', 'fw_rebin_kernel')


# ── Reverse rebinning CUDA kernel ───────────────────────────────────────
_rev_rebin_kernel = cp.RawKernel(r'''
extern "C" __global__
void rev_rebin_kernel(
    const float* src,         // (V, rebin_rows, det_cols)
    float* dst,               // (V, det_rows, det_cols)
    const int* rebin_row_idx, // (det_rows, det_cols) - integer indices
    const float* fracs_0,     // (det_rows, det_cols)
    const float* fracs_1,     // (det_rows, det_cols)
    int V, int det_rows, int det_cols, int rebin_rows,
    int pos_start
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = V * det_rows * det_cols;
    if (idx >= total) return;

    int c = idx % det_cols;
    int tmp = idx / det_cols;
    int r = tmp % det_rows;
    int p = tmp / det_rows;

    int map_idx = r * det_cols + c;
    int rb = rebin_row_idx[map_idx];
    float f0 = fracs_0[map_idx];
    float f1 = fracs_1[map_idx];

    int src_base = p * rebin_rows * det_cols;
    float val;
    if (c >= pos_start) {
        // fracs_1 * src[rb] + fracs_0 * src[rb+1]
        val = f1 * src[src_base + rb * det_cols + c]
            + f0 * src[src_base + (rb + 1) * det_cols + c];
    } else {
        // fracs_1 * src[rb-1] + fracs_0 * src[rb]
        val = f1 * src[src_base + (rb - 1) * det_cols + c]
            + f0 * src[src_base + rb * det_cols + c];
    }

    dst[p * det_rows * det_cols + r * det_cols + c] = val;
}
''', 'rev_rebin_kernel')


# ── T-D weighting CUDA kernel ──────────────────────────────────────────
_td_weight_kernel = cp.RawKernel(r'''
extern "C" __global__
void td_weight_kernel(
    const float* input,     // (V, R, C)
    float* output,          // (V, R, C)
    const float* td_mask,   // (R, C)
    int V, int R, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = V * R * C;
    if (idx >= total) return;

    int rc = idx % (R * C);
    output[idx] = input[idx] * td_mask[rc];
}
''', 'td_weight_kernel')


# ═══════════════════════════════════════════════════════════════════════
# Python wrapper functions
# ═══════════════════════════════════════════════════════════════════════

def _launch(kernel, total, *args):
    block = 256
    grid = (total + block - 1) // block
    kernel((grid,), (block,), args)


def differentiate_gpu(sinogram_gpu, conf):
    """GPU differentiation. sinogram_gpu: CuPy array (V, R, C)."""
    V, R, C = sinogram_gpu.shape
    out = cp.zeros_like(sinogram_gpu)

    col_coords_gpu = cp.asarray(conf['col_coords'][:-2], dtype=cp.float32)
    row_coords_gpu = cp.asarray(conf['row_coords'][:-2], dtype=cp.float32)

    total = (V - 1) * (R - 1) * (C - 1)
    _launch(_differentiate_kernel, total,
            sinogram_gpu, out,
            np.int32(V), np.int32(R), np.int32(C),
            np.float32(conf['delta_s']),
            np.float32(conf['pixel_height']),
            np.float32(conf['pixel_span']),
            np.float32(conf['scan_diameter']),
            np.float32(conf['scan_diameter'] ** 2),
            col_coords_gpu, row_coords_gpu)
    return out


def fw_height_rebinning_gpu(input_gpu, conf):
    """GPU forward height rebinning. input_gpu: CuPy array (V, R, C)."""
    V = input_gpu.shape[0]
    det_rows = conf['detector rows']
    det_cols = conf['detector cols']
    rebin_rows = conf['detector_rebin_rows']

    fwd_rebin_row_gpu = cp.asarray(conf['fwd_rebin_row'], dtype=cp.float32)
    out = cp.zeros((V, rebin_rows, det_cols), dtype=cp.float32)

    total = V * rebin_rows * det_cols
    _launch(_fw_rebin_kernel, total,
            input_gpu, out, fwd_rebin_row_gpu,
            np.int32(V), np.int32(det_rows), np.int32(det_cols),
            np.int32(rebin_rows),
            np.float32(conf['pixel_height']),
            np.float32(0.5 - conf.get('detector_row_offset', 0.0)))
    return out


def hilbert_conv_gpu(input_gpu, conf):
    """GPU Hilbert convolution using cuFFT-based approach."""
    V, rebin_rows, C = input_gpu.shape

    # Compute Hilbert kernel on GPU
    kernel_width = conf['kernel_width']
    kernel_radius = conf['kernel_radius']
    k = cp.arange(kernel_width, dtype=cp.float32)
    hilbert_gpu = (1.0 - cp.cos(cp.float32(np.pi) * (k - kernel_radius - 0.5))) \
                / (cp.float32(np.pi) * (k - kernel_radius - 0.5))

    # FFT-based convolution: pad to (kernel_width + C - 1), use FFT
    conv_len = kernel_width + C - 1
    # Round up to power of 2 for efficiency
    fft_len = 1
    while fft_len < conv_len:
        fft_len *= 2

    # Pre-compute kernel FFT (same for all rows)
    h_padded = cp.zeros(fft_len, dtype=cp.float32)
    h_padded[:kernel_width] = hilbert_gpu
    H = cp.fft.rfft(h_padded)

    # Process in batches to avoid GPU OOM
    # Each row is fft_len floats; estimate batch size from free memory
    free_mem = cp.cuda.Device().mem_info[0]
    # Each signal needs: padded input (fft_len*4) + complex FFT ((fft_len//2+1)*8) + output (fft_len*4)
    bytes_per_signal = fft_len * 4 + (fft_len // 2 + 1) * 8 + fft_len * 4
    max_batch = max(1, int(free_mem * 0.5 / bytes_per_signal))

    total_signals = V * rebin_rows
    output_gpu = cp.zeros((V, rebin_rows, C), dtype=cp.float32)
    out_flat = output_gpu.reshape(total_signals, C)

    for start in range(0, total_signals, max_batch):
        end = min(start + max_batch, total_signals)
        batch = end - start

        x_padded = cp.zeros((batch, fft_len), dtype=cp.float32)
        x_padded[:, :C] = input_gpu.reshape(total_signals, C)[start:end]

        X = cp.fft.rfft(x_padded, axis=1)
        del x_padded
        Y = X * H[None, :]
        del X
        y = cp.fft.irfft(Y, n=fft_len, axis=1)
        del Y

        out_flat[start:end] = y[:, C - 1: 2 * C - 1]
        del y

    return output_gpu


def rev_rebin_vec_gpu(input_gpu, conf):
    """GPU reverse height rebinning. input_gpu: CuPy array (V, rebin_rows, C)."""
    V = input_gpu.shape[0]
    det_rows = conf['detector rows']
    det_cols = conf['detector cols']
    rebin_rows = conf['detector_rebin_rows']
    col_offset = conf.get('detector_col_offset', 0.0)
    pos_start = int(round(0.5 * (det_cols - 1) + col_offset))

    rebin_row_gpu = cp.asarray(conf['rebin_row'], dtype=cp.int32)
    fracs_0_gpu = cp.asarray(conf['rebin_fracs_0'], dtype=cp.float32)
    fracs_1_gpu = cp.asarray(conf['rebin_fracs_1'], dtype=cp.float32)

    out = cp.zeros((V, det_rows, det_cols), dtype=cp.float32)

    total = V * det_rows * det_cols
    _launch(_rev_rebin_kernel, total,
            input_gpu, out, rebin_row_gpu, fracs_0_gpu, fracs_1_gpu,
            np.int32(V), np.int32(det_rows), np.int32(det_cols),
            np.int32(rebin_rows), np.int32(pos_start))
    return out


def sino_weight_td_gpu(input_gpu, conf):
    """GPU T-D weighting. input_gpu: CuPy array (V, R, C)."""
    V, R, C = input_gpu.shape

    # Compute T-D mask on GPU (same logic as CPU version)
    w_bottom = cp.asarray(conf['proj_row_mins'][:-1].reshape(1, -1), dtype=cp.float32)
    w_top = cp.asarray(conf['proj_row_maxs'][:-1].reshape(1, -1), dtype=cp.float32)
    dw = conf['detector rows'] * conf['pixel_height']
    a = conf['T-D smoothing']

    row_c = cp.asarray(conf['row_coords'][:-1], dtype=cp.float32)
    W = row_c[:, None] * cp.ones(C, dtype=cp.float32)[None, :]  # (R, C)

    W_top_high = w_top + a * dw
    W_top_low = w_top - a * dw
    W_bot_high = w_bottom + a * dw
    W_bot_low = w_bottom - a * dw

    mask = cp.zeros((R, C), dtype=cp.float32)
    # Upper transition
    m1 = (W_top_low < W) & (W < W_top_high)
    mask[m1] = ((W_top_high - W) / (2 * a * dw))[m1]
    # Fully weighted
    m2 = (W_bot_high < W) & (W < W_top_low)
    mask[m2] = 1.0
    # Lower transition
    m3 = (W_bot_low < W) & (W < W_bot_high)
    mask[m3] = ((W - W_bot_low) / (2 * a * dw))[m3]

    out = cp.zeros_like(input_gpu)
    total = V * R * C
    _launch(_td_weight_kernel, total,
            input_gpu, out, mask,
            np.int32(V), np.int32(R), np.int32(C))
    return out


# ═══════════════════════════════════════════════════════════════════════
# Main entry point — drop-in replacement for filter_katsevich + sino_weight_td
# ═══════════════════════════════════════════════════════════════════════

def filter_katsevich_gpu(sinogram, conf, verbosity_options=None):
    """
    GPU-accelerated Katsevich filter pipeline.

    Drop-in replacement for filter_katsevich().
    Input: numpy array (V, R, C), returns: numpy array (V, R, C).
    """
    if verbosity_options is None:
        verbosity_options = {}
    print_time = any(v.get('Print time', False) for v in verbosity_options.values()) if verbosity_options else False

    t0 = time()
    sino_gpu = cp.asarray(sinogram, dtype=cp.float32)
    if print_time:
        cp.cuda.Stream.null.synchronize()
        print(f"  H2D transfer: {time()-t0:.2f}s")

    # Step 1: Differentiation
    t1 = time()
    diff_gpu = differentiate_gpu(sino_gpu, conf)
    del sino_gpu
    cp.cuda.Stream.null.synchronize()
    if print_time:
        print(f"  Differentiate: {time()-t1:.2f}s")

    # Step 2: Forward height rebinning
    t2 = time()
    rebin_gpu = fw_height_rebinning_gpu(diff_gpu, conf)
    del diff_gpu
    cp.cuda.Stream.null.synchronize()
    if print_time:
        print(f"  Fwd rebinning: {time()-t2:.2f}s")

    # Step 3: Hilbert convolution (cuFFT)
    t3 = time()
    hilbert_gpu = hilbert_conv_gpu(rebin_gpu, conf)
    del rebin_gpu
    cp.cuda.Stream.null.synchronize()
    if print_time:
        print(f"  Hilbert conv:  {time()-t3:.2f}s")

    # Step 4: Reverse height rebinning
    t4 = time()
    filtered_gpu = rev_rebin_vec_gpu(hilbert_gpu, conf)
    del hilbert_gpu
    cp.cuda.Stream.null.synchronize()
    if print_time:
        print(f"  Rev rebinning: {time()-t4:.2f}s")

    # Transfer back to CPU
    t5 = time()
    result = cp.asnumpy(filtered_gpu)
    del filtered_gpu
    # Force free GPU memory pool to prevent accumulation across chunks
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    if print_time:
        print(f"  D2H transfer: {time()-t5:.2f}s")
        print(f"  Total filter:  {time()-t0:.2f}s")

    return result


def sino_weight_td_gpu_np(sinogram, conf, show_td_window=False):
    """
    T-D weighting — compute mask on CPU (tiny 64x736), broadcast multiply.
    Drop-in replacement for sino_weight_td().
    """
    R = conf['detector rows']
    C = conf['detector cols']
    w_bottom = conf['proj_row_mins'][:-1].reshape(1, -1)
    w_top = conf['proj_row_maxs'][:-1].reshape(1, -1)
    dw = R * conf['pixel_height']
    a = conf['T-D smoothing']

    W = conf['row_coords'][:-1].reshape(-1, 1) * np.ones(C, dtype=np.float32).reshape(1, -1)
    mask = np.zeros((R, C), dtype=np.float32)

    m1 = (w_top - a*dw < W) & (W < w_top + a*dw)
    mask[m1] = ((w_top + a*dw - W) / (2*a*dw))[m1]
    m2 = (w_bottom + a*dw < W) & (W < w_top - a*dw)
    mask[m2] = 1.0
    m3 = (w_bottom - a*dw < W) & (W < w_bottom + a*dw)
    mask[m3] = ((W - (w_bottom - a*dw)) / (2*a*dw))[m3]

    return sinogram * mask[None, :, :]
