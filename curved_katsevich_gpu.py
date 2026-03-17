"""
GPU-accelerated curved-detector Katsevich reconstruction.

Filtering steps: vectorized NumPy (no Python loops).
Backprojection: CuPy CUDA kernel.

Based on CPH CT Toolbox curved-detector formulas, converted to Python 3.
"""

import numpy as np
from numpy import pi
from time import time
import argparse
import sys


# ============================================================================
# GPU Backprojection CUDA kernel
# ============================================================================

CURVED_BP_KERNEL = r'''
extern "C" __global__
void curved_backproject(
    const float* __restrict__ filtered,   // (n_projs, n_rows, n_cols)
    float* __restrict__ recon,            // (x_voxels, y_voxels, z_voxels)
    const float* __restrict__ source_pos, // (n_projs,)
    const float* __restrict__ row_mins,   // (n_cols+1,)
    const float* __restrict__ row_maxs,   // (n_cols+1,)
    int n_projs, int n_rows, int n_cols,
    int x_voxels, int y_voxels, int z_voxels,
    float scan_radius, float scan_diameter,
    float x_min, float y_min, float z_min,
    float delta_x, float delta_y, float delta_z,
    float pixel_span, float pixel_height,
    float det_row_shift, float det_col_offset,
    float progress_per_radian, float fov_radius_sq)
{
    // Each thread handles one (x, y) voxel column
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= x_voxels || y >= y_voxels) return;

    float x_coord = x_min + x * delta_x;
    float y_coord = y_min + y * delta_y;

    // Skip voxels outside FOV cylinder
    if (x_coord * x_coord + y_coord * y_coord > fov_radius_sq) return;

    for (int proj = 0; proj < n_projs; proj++) {
        float s = source_pos[proj];
        float cos_s, sin_s;
        sincosf(s, &sin_s, &cos_s);

        // Denominator U = R - x*cos(s) - y*sin(s)
        float U = scan_radius - x_coord * cos_s - y_coord * sin_s;
        float inv_U = 1.0f / U;

        // CURVED: column projection = arctan(t / U)
        float t = -x_coord * sin_s + y_coord * cos_s;
        float gamma = atan2f(t, U);
        float cos_gamma = cosf(gamma);

        // Column index (angular pixel units)
        float col_real = gamma / pixel_span + 0.5f * n_cols - det_col_offset;
        int col_int = (int)floorf(col_real);
        if (col_int < 0 || col_int >= n_cols - 1) continue;
        float col_frac = col_real - col_int;

        // CURVED: row projection with cos(gamma)
        float row_scale = scan_diameter * cos_gamma * inv_U;
        float row_ind_z_min = row_scale * (z_min - progress_per_radian * s)
                              / pixel_height + det_row_shift;
        float row_step = row_scale * delta_z / pixel_height;

        // Tam-Danielsson z limits
        float rm = (1.0f - col_frac) * row_mins[col_int]
                   + col_frac * row_mins[col_int + 1];
        float rM = (1.0f - col_frac) * row_maxs[col_int]
                   + col_frac * row_maxs[col_int + 1];
        float z_lo_coord = s * progress_per_radian
                           + rm * U / (scan_diameter * cos_gamma);
        float z_hi_coord = s * progress_per_radian
                           + rM * U / (scan_diameter * cos_gamma);

        int z_first = (int)ceilf((z_lo_coord - z_min) / delta_z);
        int z_last  = (int)floorf((z_hi_coord - z_min) / delta_z);
        if (z_first < 0) z_first = 0;
        if (z_last >= z_voxels) z_last = z_voxels - 1;
        if (z_first > z_last) continue;

        int base_idx = proj * n_rows * n_cols;
        for (int z = z_first; z <= z_last; z++) {
            float row_real = row_ind_z_min + z * row_step;
            int row_int = (int)floorf(row_real);
            if (row_int < 0 || row_int >= n_rows - 1) continue;
            float row_frac = row_real - row_int;

            // Bilinear interpolation
            int idx00 = base_idx + row_int * n_cols + col_int;
            float val = (1.0f - row_frac) * (1.0f - col_frac) * filtered[idx00]
                      + row_frac          * (1.0f - col_frac) * filtered[idx00 + n_cols]
                      + (1.0f - row_frac) * col_frac          * filtered[idx00 + 1]
                      + row_frac          * col_frac          * filtered[idx00 + n_cols + 1];

            int out_idx = (x * y_voxels + y) * z_voxels + z;
            recon[out_idx] += val * inv_U;
        }
    }
}
'''


# ============================================================================
# Vectorized filtering steps (no Python loops over projections)
# ============================================================================

def curved_differentiate_fast(sinogram, conf):
    """Vectorized curved-detector differentiation (all projections at once)."""
    dia = conf['scan_diameter']
    delta_s = conf['delta_s']
    pixel_span = conf['detector_pixel_span']

    row_coords = conf['row_coords'][:-2]
    row_sqr = row_coords.reshape(-1, 1) ** 2
    length_corr = dia / np.sqrt(dia ** 2 + row_sqr)

    print("  Differentiating (vectorized)...", end='', flush=True)
    t0 = time()

    # All projections at once using array slicing
    d_proj = (sinogram[1:, :-1, :-1] - sinogram[:-1, :-1, :-1]
              + sinogram[1:, :-1, 1:] - sinogram[:-1, :-1, 1:]) / (2 * delta_s)
    d_col = (sinogram[:-1, :-1, 1:] - sinogram[:-1, :-1, :-1]
             + sinogram[1:, :-1, 1:] - sinogram[1:, :-1, :-1]) / (2 * pixel_span)

    result = (d_proj + d_col) * length_corr

    # Pad back to full detector size (last row/col = 0)
    n_projs = sinogram.shape[0] - 1
    n_rows, n_cols = sinogram.shape[1], sinogram.shape[2]
    out = np.zeros((n_projs, n_rows, n_cols), dtype=sinogram.dtype)
    out[:, :-1, :-1] = result

    print(f" {time()-t0:.1f}s")
    return out


def forward_rebin_fast(diff_data, conf):
    """Vectorized forward height rebinning using pre-computed index mapping."""
    n_projs, n_rows, n_cols = diff_data.shape
    n_rebin = conf['detector_rebin_rows']
    pixel_height = conf['detector_pixel_height']
    fwd_rebin_row = conf['fwd_rebin_row']
    row_coords = conf['row_coords'][:-1]  # non-extended

    print(f"  Forward rebinning (vectorized) to {n_rebin} rows...", end='', flush=True)
    t0 = time()

    # Pre-compute fractional row indices for all (rebin, col) pairs
    # fwd_rebin_row[rebin, col] gives the physical row coordinate
    # Convert to fractional row index in original detector
    rebin_scaled = fwd_rebin_row / pixel_height  # (n_rebin, n_cols)

    # For each column, find the fractional indices into row_coords
    # row_coords is monotonically increasing
    row_indices = np.zeros((n_rebin, n_cols), dtype=np.float32)
    for col in range(n_cols):
        row_indices[:, col] = np.interp(
            rebin_scaled[:, col], row_coords, np.arange(n_rows, dtype=np.float32))

    # Clip and compute floor/frac
    row_lo = np.clip(np.floor(row_indices).astype(np.int32), 0, n_rows - 2)
    row_frac = np.clip(row_indices - row_lo, 0, 1)

    # Gather using advanced indexing for all projections at once
    col_idx = np.arange(n_cols)[np.newaxis, :]  # (1, n_cols)
    col_idx = np.broadcast_to(col_idx, (n_rebin, n_cols))

    out = np.zeros((n_projs, n_rebin, n_cols), dtype=diff_data.dtype)
    for p in range(n_projs):
        val_lo = diff_data[p, row_lo, col_idx]
        val_hi = diff_data[p, row_lo + 1, col_idx]
        out[p] = (1 - row_frac) * val_lo + row_frac * val_hi

    print(f" {time()-t0:.1f}s")
    return out


def hilbert_convolve_fft(rebin_data, conf):
    """FFT-based Hilbert convolution (much faster than direct convolution)."""
    from scipy.signal import fftconvolve

    n_projs, n_rebin, n_cols = rebin_data.shape
    hilbert = conf['hilbert_filter']

    print("  Hilbert convolution (FFT)...", end='', flush=True)
    t0 = time()

    out = np.zeros_like(rebin_data)
    # Vectorize over rebin rows: convolve all projections for each row
    for r in range(n_rebin):
        # Batch convolve all projections for this rebin row
        for p in range(n_projs):
            conv_full = fftconvolve(hilbert, rebin_data[p, r, :], mode='full')
            out[p, r, :] = conv_full[n_cols - 1: 2 * n_cols - 1]

    print(f" {time()-t0:.1f}s")
    return out


def reverse_rebin_fast(conv_data, conf):
    """Vectorized reverse rebinning with pre-computed index tables."""
    n_projs, n_rebin, n_cols = conv_data.shape
    n_rows = conf['detector_rows']
    fwd_rebin_row = conf['fwd_rebin_row']
    row_coords = conf['row_coords'][:-1]
    col_coords = conf['col_coords'][:-1]
    det_col_offset = conf.get('detector_column_offset', 0.0)
    pos_start = int(0.5 * n_cols - det_col_offset)

    print(f"  Reverse rebinning (vectorized) to {n_rows} rows...", end='', flush=True)
    t0 = time()

    # Pre-compute rebin index mapping: (n_rows, n_cols) -> rebin_row, frac
    rebin_row_idx = np.zeros((n_rows, n_cols), dtype=np.int32)
    fracs_0 = np.zeros((n_rows, n_cols), dtype=np.float32)

    for row in range(n_rows):
        # Positive column range
        for col in range(pos_start, n_cols):
            rebin_row_idx[row, col] = 0
            for rebin in range(n_rebin - 1):
                if (row_coords[row] >= fwd_rebin_row[rebin, col] and
                        row_coords[row] <= fwd_rebin_row[rebin + 1, col]):
                    rebin_row_idx[row, col] = rebin
                    fracs_0[row, col] = ((row_coords[row] - fwd_rebin_row[rebin, col])
                                         / (fwd_rebin_row[rebin + 1, col]
                                            - fwd_rebin_row[rebin, col]))
                    break

        # Negative column range
        for col in range(pos_start):
            rebin_row_idx[row, col] = 1
            for rebin in range(n_rebin - 1, 0, -1):
                if (row_coords[row] >= fwd_rebin_row[rebin - 1, col] and
                        row_coords[row] <= fwd_rebin_row[rebin, col]):
                    rebin_row_idx[row, col] = rebin
                    fracs_0[row, col] = ((row_coords[row] - fwd_rebin_row[rebin - 1, col])
                                         / (fwd_rebin_row[rebin, col]
                                            - fwd_rebin_row[rebin - 1, col]))
                    break

    fracs_1 = 1.0 - fracs_0

    # Now apply to all projections using advanced indexing
    col_idx = np.arange(n_cols)[np.newaxis, :]
    col_idx = np.broadcast_to(col_idx, (n_rows, n_cols))

    # For positive columns: interpolate between rebin_row and rebin_row+1
    # For negative columns: interpolate between rebin_row-1 and rebin_row
    # Build unified lower/upper index arrays
    lo_idx = rebin_row_idx.copy()
    hi_idx = rebin_row_idx.copy()

    # Positive side: lo=rebin_row, hi=rebin_row+1
    lo_idx[:, pos_start:] = rebin_row_idx[:, pos_start:]
    hi_idx[:, pos_start:] = rebin_row_idx[:, pos_start:] + 1

    # Negative side: lo=rebin_row-1, hi=rebin_row
    lo_idx[:, :pos_start] = rebin_row_idx[:, :pos_start] - 1
    hi_idx[:, :pos_start] = rebin_row_idx[:, :pos_start]

    np.clip(lo_idx, 0, n_rebin - 1, out=lo_idx)
    np.clip(hi_idx, 0, n_rebin - 1, out=hi_idx)

    out = np.zeros((n_projs, n_rows, n_cols), dtype=conv_data.dtype)
    cos_weight = np.cos(col_coords)  # CURVED-SPECIFIC

    for p in range(n_projs):
        val_lo = conv_data[p, lo_idx, col_idx]
        val_hi = conv_data[p, hi_idx, col_idx]
        out[p] = (fracs_1 * val_lo + fracs_0 * val_hi) * cos_weight

    print(f" {time()-t0:.1f}s")
    return out


def curved_backproject_gpu(filtered, conf):
    """Curved-detector backprojection using CuPy CUDA kernel."""
    import cupy as cp

    n_projs, n_rows, n_cols = filtered.shape
    x_vox = conf['x_voxels']
    y_vox = conf['y_voxels']
    z_vox = conf['z_voxels']

    print(f"  GPU Backprojection: {n_projs} projs -> ({x_vox},{y_vox},{z_vox})...", flush=True)
    t0 = time()

    # Compile CUDA kernel
    kernel = cp.RawKernel(CURVED_BP_KERNEL, 'curved_backproject')

    # Transfer data to GPU
    d_filtered = cp.asarray(filtered.astype(np.float32), dtype=cp.float32)
    d_recon = cp.zeros((x_vox, y_vox, z_vox), dtype=cp.float32)
    d_source_pos = cp.asarray(conf['source_pos'].astype(np.float32))
    d_row_mins = cp.asarray(conf['proj_row_mins'].astype(np.float32))
    d_row_maxs = cp.asarray(conf['proj_row_maxs'].astype(np.float32))

    # Launch config: one thread per (x,y) voxel
    block = (16, 16, 1)
    grid = ((x_vox + block[0] - 1) // block[0],
            (y_vox + block[1] - 1) // block[1],
            1)

    fov_radius_sq = float(conf['fov_radius']) ** 2

    kernel(grid, block, (
        d_filtered, d_recon, d_source_pos, d_row_mins, d_row_maxs,
        np.int32(n_projs), np.int32(n_rows), np.int32(n_cols),
        np.int32(x_vox), np.int32(y_vox), np.int32(z_vox),
        np.float32(conf['scan_radius']),
        np.float32(conf['scan_diameter']),
        np.float32(conf['x_min']),
        np.float32(conf['y_min']),
        np.float32(conf['z_min']),
        np.float32(conf['delta_x']),
        np.float32(conf['delta_y']),
        np.float32(conf['delta_z']),
        np.float32(conf['detector_pixel_span']),
        np.float32(conf['detector_pixel_height']),
        np.float32(conf['detector_row_shift']),
        np.float32(conf.get('detector_column_offset', 0.0)),
        np.float32(conf['progress_per_radian']),
        np.float32(fov_radius_sq),
    ))

    cp.cuda.Device().synchronize()
    recon = d_recon.get()

    elapsed = time() - t0
    print(f"  GPU Backprojection done in {elapsed:.1f}s")
    return recon


# ============================================================================
# Configuration (imported from CPU version)
# ============================================================================

from curved_katsevich_demo import (
    create_curved_conf, finalize_conf, load_dicom_projections
)


# ============================================================================
# Full GPU pipeline
# ============================================================================

def reconstruct_curved_gpu(sinogram, conf):
    """
    Full curved-detector Katsevich pipeline with GPU acceleration.

    Steps 1-4: Vectorized NumPy (fast CPU).
    Step 5: CuPy CUDA kernel (GPU).
    """
    print("=== Curved Katsevich Pipeline (GPU) ===")
    print(f"  Input: {sinogram.shape}")
    print(f"  detector_pixel_span (rad): {conf['detector_pixel_span']:.6f}")
    print(f"  detector_pixel_height (mm): {conf['detector_pixel_height']:.4f}")
    print(f"  half_fan_angle: {np.degrees(conf['half_fan_angle']):.2f} deg")

    t_total = time()

    # Step 1: Differentiate (vectorized)
    diff = curved_differentiate_fast(sinogram, conf)

    # Step 2: Forward rebin (vectorized)
    rebin = forward_rebin_fast(diff, conf)
    del diff  # free memory

    # Step 3: Hilbert convolution (FFT)
    conv = hilbert_convolve_fft(rebin, conf)
    del rebin

    # Step 4: Reverse rebin (vectorized + cos weighting)
    filtered = reverse_rebin_fast(conv, conf)
    del conv

    # Step 5: Backprojection (CUDA GPU)
    recon = curved_backproject_gpu(filtered, conf)
    del filtered

    # Final scaling
    recon *= conf['delta_s'] / (2 * pi)

    print(f"\n  Total pipeline: {time()-t_total:.1f}s")
    return recon


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Curved Katsevich GPU reconstruction')
    parser.add_argument('--dicom-dir', required=True, help='Path to DICOM-CT-PD directory')
    parser.add_argument('--rows', type=int, default=512, help='Recon grid X/Y size')
    parser.add_argument('--cols', type=int, default=512, help='Recon grid X/Y size (alias)')
    parser.add_argument('--slices', type=int, default=560, help='Recon grid Z size')
    parser.add_argument('--voxel-size', type=float, default=0.664, help='Voxel size in mm')
    parser.add_argument('--pitch-mm-turn', type=float, default=None, help='Override pitch (mm/turn)')
    parser.add_argument('--decimate', type=int, default=2, help='Use every Nth projection')
    parser.add_argument('--rebin-rows', type=int, default=128, help='Rebin rows')
    parser.add_argument('--save-npy', type=str, default=None, help='Save .npy output')
    parser.add_argument('--show', action='store_true', help='Show result slices')
    args = parser.parse_args()

    sinogram, geom = load_dicom_projections(args.dicom_dir, decimate=args.decimate)
    n_projs, n_rows, n_cols = sinogram.shape

    if args.pitch_mm_turn is not None:
        pitch = abs(args.pitch_mm_turn)
        print(f"  Using override pitch: {pitch:.1f} mm/turn")
    else:
        pitch = geom['pitch_mm_per_turn']
        print(f"  Pitch from DICOM: {pitch:.2f} mm/turn")

    # Normalize helix direction: Katsevich requires source_pos to increase.
    angles = geom['angles']
    if angles[-1] < angles[0]:
        print("  Normalizing: negating angles + flipping columns (clockwise -> CCW)")
        angles = -angles
        sinogram = sinogram[:, :, ::-1].copy()

    xy_size = max(args.rows, args.cols)

    conf = create_curved_conf(
        sod=geom['sod'], sdd=geom['sdd'],
        det_rows=n_rows, det_cols=n_cols,
        pixel_span_rad=geom['delta_gamma'],
        pixel_height_mm=geom['pixel_height'],
        progress_per_turn=pitch,
        x_voxels=xy_size, y_voxels=xy_size, z_voxels=args.slices,
        voxel_size=args.voxel_size,
        detector_rebin_rows=args.rebin_rows,
    )
    conf = finalize_conf(conf, n_projs, actual_angles=angles)

    print(f"\nConfiguration:")
    print(f"  Volume: {xy_size}x{xy_size}x{args.slices}, voxel={args.voxel_size}mm")
    print(f"  FOV: {xy_size * args.voxel_size:.0f}mm")
    print(f"  Projections: {n_projs}")
    print(f"  Projs/turn: {conf['projs_per_turn']}")
    print(f"  Pitch: {pitch:.1f} mm/turn")

    recon = reconstruct_curved_gpu(sinogram, conf)

    print(f"\nResult: shape={recon.shape}, range=[{recon.min():.6f}, {recon.max():.6f}]")

    if args.save_npy:
        np.save(args.save_npy, recon)
        print(f"Saved to {args.save_npy}")

    if args.show:
        from matplotlib import pyplot as plt
        mid = recon.shape[2] // 2
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, idx in enumerate([mid - 5, mid, mid + 5]):
            idx = min(max(idx, 0), recon.shape[2] - 1)
            axes[i].imshow(recon[:, :, idx].T, cmap='gray')
            axes[i].set_title(f'Slice {idx}')
        plt.suptitle('Curved Katsevich GPU Reconstruction')
        plt.tight_layout()
        plt.savefig('curved_recon_gpu.png', dpi=150)
        print("Saved curved_recon_gpu.png")
        plt.show()


if __name__ == '__main__':
    main()
