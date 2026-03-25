"""
Fused CuPy Katsevich backprojection kernel.

Replaces backproject_safe's per-projection ASTRA loop with a single GPU kernel
that does voxel-driven projection + bilinear interpolation + distance weighting
all in one pass. Each kernel launch processes a batch of projections to avoid
Windows TDR timeout (~2s).

Drop-in replacement: same function signature and output format as backproject_safe.
"""
import numpy as np
import astra

_KERNEL_SRC = r'''
extern "C" __global__
void fused_bp(
    float* __restrict__ rec,
    const float* __restrict__ sino,
    const float* __restrict__ angles,
    const float* __restrict__ src_z,
    int det_rows, int det_cols,
    float SOD, float SDD, float psize_col, float psize_row,
    float x_min, float y_min, float z_min,
    float dx, float dy, float dz,
    int nx, int ny, int nz,
    float inv_scale,
    float col_offset, float row_offset,
    int p_start, int p_end)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int iz = blockDim.z * blockIdx.z + threadIdx.z;

    if (ix >= nx || iy >= ny || iz >= nz) return;

    float X = x_min + ((float)ix + 0.5f) * dx;
    float Y = y_min + ((float)iy + 0.5f) * dy;
    float Z = z_min + ((float)iz + 0.5f) * dz;

    float val = 0.0f;
    float SDD_over_psize_col = SDD / psize_col;
    float SDD_over_psize_row = SDD / psize_row;
    float u_center = (float)(det_cols - 1) * 0.5f + col_offset;
    float v_center = (float)(det_rows - 1) * 0.5f + row_offset;

    for (int p = p_start; p < p_end; p++) {
        float cs, sn;
        sincosf(angles[p], &sn, &cs);

        float L = SOD - X * cs - Y * sn;
        if (L <= 0.0f) continue;

        float inv_L = 1.0f / L;
        float u = SDD_over_psize_col * (Y * cs - X * sn) * inv_L + u_center;
        float v = SDD_over_psize_row * (Z - src_z[p]) * inv_L + v_center;

        // Bounds check
        if (u < 0.0f || u > (float)(det_cols - 1) ||
            v < 0.0f || v > (float)(det_rows - 1)) continue;

        // Bilinear interpolation indices
        int u0 = (int)u;
        int v0 = (int)v;
        if (u0 >= det_cols - 1) u0 = det_cols - 2;
        if (v0 >= det_rows - 1) v0 = det_rows - 2;
        if (u0 < 0) u0 = 0;
        if (v0 < 0) v0 = 0;

        float fu = u - (float)u0;
        float fv = v - (float)v0;

        // sino layout: [n_projs, det_rows, det_cols] row-major
        long long base = (long long)p * det_rows * det_cols;
        float s00 = __ldg(&sino[base + v0 * det_cols + u0]);
        float s01 = __ldg(&sino[base + v0 * det_cols + u0 + 1]);
        float s10 = __ldg(&sino[base + (v0 + 1) * det_cols + u0]);
        float s11 = __ldg(&sino[base + (v0 + 1) * det_cols + u0 + 1]);

        float interp = (1.0f - fv) * ((1.0f - fu) * s00 + fu * s01)
                      + fv * ((1.0f - fu) * s10 + fu * s11);

        val += interp * inv_L * inv_L * inv_L;
    }

    // Volume layout: [nz, ny, nx] row-major
    long long idx = (long long)iz * ny * nx + (long long)iy * nx + ix;
    rec[idx] += val * inv_scale;
}
'''


def backproject_cupy(input_array, conf, vol_geom, proj_geom,
                     tqdm_bar=False, batch_size=128):
    """
    Fused CuPy backprojection - drop-in replacement for backproject_safe.

    Instead of calling ASTRA BP3D_CUDA per projection (creating/destroying
    objects each time), this runs a single CUDA kernel that does
    projection + bilinear interpolation + 1/L distance weighting in one pass.

    Parameters match backproject_safe / backproject_a exactly.
    Returns (Y, X, Z) ordered volume, same as backproject_a.

    Parameters
    ----------
    input_array : np.ndarray, shape (n_projs, det_rows, det_cols)
        Filtered + TD-weighted sinogram.
    conf : dict
        Helical configuration from create_configuration().
    vol_geom : dict
        ASTRA volume geometry.
    proj_geom : dict
        ASTRA cone_vec projection geometry.
    tqdm_bar : bool
        Show progress bar.
    batch_size : int
        Projections per kernel launch (avoid Windows TDR timeout).
    """
    import cupy as cp

    kernel = cp.RawKernel(_KERNEL_SRC, 'fused_bp',
                          options=('--use_fast_math',))

    # Geometry from conf
    delta_x = float(conf['delta_x'])
    delta_y = float(conf['delta_y'])
    delta_z = float(conf['delta_z'])
    x_min   = float(conf['x_min'])
    y_min   = float(conf['y_min'])
    z_min   = float(conf['z_min'])
    SOD       = float(conf['scan_radius'])         # source-object distance
    SDD       = float(conf['scan_diameter'])       # source-detector distance
    psize_col = float(conf['pixel_span'])          # detector pixel size (fan/col direction)
    psize_row = float(conf['pixel_height'])        # detector pixel size (z/row direction)

    # Scale coefficient.
    # ASTRA BP3D_CUDA applies implicit 1/L^2 weighting where L = SOD - X*cos - Y*sin,
    # with constant C = delta_x^3 * SDD^2 / psize^2.
    # backproject_safe then divides by (astra_bp_scaling * projs_per_turn * L).
    # Combined: rec += interp * SOD^2 / (projs_per_turn * L^3).
    # Our kernel does val += interp / L^3, then rec += val * inv_scale.
    inv_scale = np.float32(SOD ** 2 / conf['projs_per_turn'])

    # Sinogram -> GPU: (n_projs, det_rows, det_cols)
    sino_gpu = cp.asarray(np.ascontiguousarray(input_array, dtype=np.float32))
    n_projs = sino_gpu.shape[0]
    det_rows = int(proj_geom['DetectorRowCount'])
    det_cols = int(proj_geom['DetectorColCount'])

    # Detector center offsets (isocenter projection vs geometric center, in pixels)
    col_offset = np.float32(conf.get('detector_col_offset', 0.0))
    row_offset = np.float32(conf.get('detector_row_offset', 0.0))

    # Angles (source rotational position) and source z-positions -> GPU
    angles_gpu = cp.asarray(
        np.ascontiguousarray(conf['source_pos'], dtype=np.float32))
    src_z_gpu = cp.asarray(
        np.ascontiguousarray(proj_geom['Vectors'][:, 2].astype(np.float32)))

    # Output volume
    vol_shape = astra.geom_size(vol_geom)   # (Z, Y, X)
    nz, ny, nx = vol_shape
    rec_gpu = cp.zeros(vol_shape, dtype=cp.float32)

    # Launch configuration
    block = (8, 8, 4)    # 256 threads per block
    grid = (
        (nx + block[0] - 1) // block[0],
        (ny + block[1] - 1) // block[1],
        (nz + block[2] - 1) // block[2],
    )

    n_batches = (n_projs + batch_size - 1) // batch_size

    try:
        from tqdm import tqdm
        it = tqdm(range(n_batches), "BP-CuPy") if tqdm_bar else range(n_batches)
    except ImportError:
        it = range(n_batches)

    for b in it:
        p_start = b * batch_size
        p_end = min(p_start + batch_size, n_projs)
        kernel(
            grid, block,
            (
                rec_gpu, sino_gpu, angles_gpu, src_z_gpu,
                np.int32(det_rows), np.int32(det_cols),
                np.float32(SOD), np.float32(SDD), np.float32(psize_col), np.float32(psize_row),
                np.float32(x_min), np.float32(y_min), np.float32(z_min),
                np.float32(delta_x), np.float32(delta_y), np.float32(delta_z),
                np.int32(nx), np.int32(ny), np.int32(nz),
                inv_scale,
                col_offset, row_offset,
                np.int32(p_start), np.int32(p_end),
            ),
        )
        cp.cuda.Device().synchronize()

    # Convert from ASTRA layout (Z, Y, X) to pykatsevich layout (Y, X, Z)
    rec_volume = np.asarray(np.moveaxis(rec_gpu.get(), 0, 2), order='C')
    return rec_volume
