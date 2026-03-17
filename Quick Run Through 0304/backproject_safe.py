"""
Katsevich backprojection WITHOUT GPULink.
Uses standard ASTRA data management + CuPy for the per-voxel distance weighting.
Drop-in replacement for pykatsevich.filter.backproject_a.

IMPORTANT: ASTRA must initialize its CUDA context BEFORE CuPy is imported,
otherwise CuPy's CUDA 11.8 runtime corrupts ASTRA's texture object creation.
Call ensure_astra_cuda_init() early in your script before any CuPy import.
"""
import numpy as np
import astra


def ensure_astra_cuda_init():
    """Warm up ASTRA's CUDA context. Call BEFORE importing CuPy."""
    vg = astra.create_vol_geom(2, 2, 2)
    pg = astra.create_proj_geom(
        'cone_vec', 2, 2,
        np.array([[1, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 1]], dtype=np.float64),
    )
    sid = astra.data3d.create('-sino', pg, np.zeros((2, 1, 2), dtype=np.float32))
    rid = astra.data3d.create('-vol', vg, 0)
    cfg = astra.astra_dict('BP3D_CUDA')
    cfg['ReconstructionDataId'] = rid
    cfg['ProjectionDataId'] = sid
    alg = astra.algorithm.create(cfg)
    astra.algorithm.run(alg)
    astra.algorithm.delete([alg])
    astra.data3d.delete([sid, rid])


def backproject_safe(input_array, conf, vol_geom, proj_geom, tqdm_bar=False):
    """
    Katsevich backprojection using standard ASTRA data + CuPy weighting.

    Parameters match pykatsevich.filter.backproject_a exactly.
    Returns (Y, X, Z) ordered volume, same as backproject_a.
    """
    import cupy as cp

    # CuPy distance-weighting kernel
    scale_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void scale_integrate(
        const float* bp_volume, float* rec_volume,
        float xsize, float xmin, float ysize, float ymin,
        float angle, float scan_radius, float scale_const,
        unsigned int nx, unsigned int ny, unsigned int nz)
    {
        unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
        unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
        unsigned int k = blockDim.z * blockIdx.z + threadIdx.z;

        if (i >= nx || j >= ny || k >= nz) return;

        float ang_sin, ang_cos;
        sincosf(angle, &ang_sin, &ang_cos);

        float X = xsize * k + xmin + 0.5f * xsize;
        float Y = ysize * j + ymin + 0.5f * ysize;
        float w = scale_const * (scan_radius - X * ang_cos - Y * ang_sin);

        unsigned long long tid = (unsigned long long)i * ny * nz + j * nz + k;
        rec_volume[tid] += bp_volume[tid] / w;
    }
    ''', 'scale_integrate')

    try:
        from tqdm import tqdm
        _has_tqdm = True
    except ImportError:
        _has_tqdm = False

    delta_x = conf['delta_x']
    delta_y = conf['delta_y']
    x_min = conf['x_min']
    y_min = conf['y_min']
    scan_radius = conf['scan_radius']
    sdd = conf['scan_diameter']
    pixel_size = conf['pixel_span']
    source_pos = conf['source_pos']

    astra_bp_scaling = (delta_x ** 3) / ((pixel_size / (sdd / scan_radius)) ** 2)
    scale_coeff = float(astra_bp_scaling * conf['projs_per_turn'])

    sino_for_astra = np.asarray(np.swapaxes(input_array, 1, 0), dtype=np.float32, order='C')

    vol_shape = astra.geom_size(vol_geom)  # (Z, Y, X)
    rec_cp = cp.zeros(vol_shape, dtype=cp.float32)

    n_projs = proj_geom['Vectors'].shape[0]
    it = tqdm(range(n_projs), "Backprojection") if _has_tqdm and tqdm_bar else range(n_projs)

    nz, ny, nx = vol_shape
    block = (min(nz, 4), min(ny, 4), min(nx, 16))
    grid = (
        (nz + block[0] - 1) // block[0],
        (ny + block[1] - 1) // block[1],
        (nx + block[2] - 1) // block[2],
    )

    cfg = astra.astra_dict("BP3D_CUDA")

    for proj_idx in it:
        single_pg = astra.create_proj_geom(
            "cone_vec",
            proj_geom["DetectorRowCount"],
            proj_geom["DetectorColCount"],
            proj_geom["Vectors"][proj_idx : proj_idx + 1],
        )

        sino_slice = np.ascontiguousarray(sino_for_astra[:, proj_idx : proj_idx + 1, :])
        sino_id = astra.data3d.create("-sino", single_pg, sino_slice)
        bp_id = astra.data3d.create("-vol", vol_geom, 0)

        cfg["ReconstructionDataId"] = bp_id
        cfg["ProjectionDataId"] = sino_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)

        bp_np = astra.data3d.get(bp_id)
        bp_cp = cp.asarray(bp_np)

        astra.algorithm.delete([alg_id])
        astra.data3d.delete([sino_id, bp_id])

        angle = float(source_pos[proj_idx])
        scale_kernel(
            grid, block,
            (
                bp_cp, rec_cp,
                cp.float32(delta_x), cp.float32(x_min),
                cp.float32(delta_y), cp.float32(y_min),
                cp.float32(angle), cp.float32(scan_radius),
                cp.float32(scale_coeff),
                np.uint32(nz), np.uint32(ny), np.uint32(nx),
            ),
        )

    rec_volume = np.asarray(np.moveaxis(rec_cp.get(), 0, 2), order='C')
    return rec_volume
