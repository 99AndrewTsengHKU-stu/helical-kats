"""Numerical checks for the CuPy Wang Wei Katsevich implementation."""

from __future__ import annotations

import unittest

import cupy as cp
import numpy as np

from wangwei.backproject import backproj_cor
from wangwei.compute_alpha_v_w import compute_alpha_v_w
from wangwei.compute_grad_s import compute_grad_s
from wangwei.cupy_recon import (
    _build_prefilter_constants,
    backproject_slice_cupy,
    compute_grad_s_cupy,
    make_hilbert_matrix,
    prefilter_chunk_cupy,
    solve_k_line_lut,
)
from wangwei.index_theta import index_theta
from wangwei.rebin_cor import rebin_cor
from wangwei.solve_pi import solve_pi


class WangWeiCuPyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise unittest.SkipTest("CUDA GPU is required")

    def tearDown(self) -> None:
        cp.get_default_memory_pool().free_all_blocks()

    def test_compute_grad_matches_numpy_port(self) -> None:
        rng = np.random.default_rng(20260801)
        sino = rng.normal(size=(9, 8, 5)).astype(np.float32)
        expected = compute_grad_s(sino, delt_alpha=0.03, delt_theta=0.07)
        actual = cp.asnumpy(
            compute_grad_s_cupy(cp.asarray(sino), delt_alpha=0.03, delt_theta=0.07)
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

    def test_hilbert_matrix_matches_numpy_port(self) -> None:
        u_cor = np.linspace(-0.3, 0.3, 12, dtype=np.float32)
        actual = make_hilbert_matrix(u_cor)
        # Independent transcription of Htransform_matrix.m.  The older
        # wangwei/htransform.py draft incorrectly applies hilbert_fun to uu
        # instead of sin(uu), so it is intentionally not the reference here.
        delt_u = float(u_cor[1] - u_cor[0])
        length = len(u_cor)
        uu = np.arange(-length + 1, length) * delt_u - 0.5 * delt_u
        sin_u = np.sin(uu)
        b = 1.0 / (2.0 * delt_u)
        with np.errstate(divide="ignore", invalid="ignore"):
            kernel = (1.0 - np.cos(2.0 * np.pi * b * sin_u)) / (np.pi * sin_u)
        kernel[sin_u == 0] = 0.0
        kernel = np.real(
            np.fft.ifft(np.fft.fft(kernel) * np.fft.fftshift(np.hamming(len(kernel))))
        )
        expected = np.empty((length, length), dtype=np.float32)
        for index in range(length):
            expected[index] = kernel[index : index + length][::-1]
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

    def test_prefilter_chunk_matches_numpy_reference(self) -> None:
        rng = np.random.default_rng(7)
        sino = rng.normal(size=(9, 12, 7)).astype(np.float32)
        DSO, DSD, h = 100.0, 180.0, 3.5
        alpha = np.linspace(-0.18, 0.18, 12, dtype=np.float32)
        w_cor = np.linspace(-10.0, 10.0, 7, dtype=np.float32)
        phi = np.linspace(-1.2, 1.2, 19, dtype=np.float32) + np.float32(0.013)
        delt_alpha = float(alpha[1] - alpha[0])
        delt_theta = 0.05

        g1 = compute_grad_s(sino, delt_alpha, delt_theta)
        g3 = rebin_cor(
            g1,
            DSD,
            h,
            DSO,
            phi,
            alpha + 0.5 * delt_alpha,
            w_cor,
        )
        hilbert = make_hilbert_matrix(alpha + 0.5 * delt_alpha)
        g3 = np.einsum("ij,tjp->tip", hilbert, g3, optimize=True) * delt_alpha
        k_index, k_weight, valid = solve_k_line_lut(
            alpha, w_cor, phi, DSD, h, DSO
        )
        expected = np.zeros_like(sino)
        for ia in range(len(alpha)):
            for iw in range(len(w_cor)):
                if not valid[ia, iw]:
                    continue
                k = k_index[ia, iw]
                weight = k_weight[ia, iw]
                expected[:, ia, iw] = (
                    g3[:, ia, k] * (1.0 - weight) + g3[:, ia, k + 1] * weight
                ) * np.cos(alpha[ia])

        constants = _build_prefilter_constants(
            DSD, h, DSO, alpha, w_cor, phi, delt_alpha
        )
        actual = cp.asnumpy(
            prefilter_chunk_cupy(sino, constants, delt_alpha, delt_theta)
        )
        np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-4)

    def test_k_line_uses_negative_invalid_sentinel(self) -> None:
        alpha = np.linspace(-0.2, 0.2, 9)
        w_cor = np.array([-1000.0, 0.0, 1000.0])
        phi = np.linspace(-1.0, 1.0, 21) + 0.01
        index, _, valid = solve_k_line_lut(alpha, w_cor, phi, 180.0, 3.0, 100.0)
        self.assertTrue(np.any(index == -1))
        self.assertTrue(np.any(valid))
        self.assertTrue(np.all((index >= 0) == valid))

    def test_fused_backprojection_matches_cpu_reference(self) -> None:
        DSO, DSD, h = 100.0, 180.0, 4.0
        x_cor = np.linspace(-8.0, 8.0, 5, dtype=np.float32)
        y_cor = np.linspace(-6.0, 6.0, 4, dtype=np.float32)
        z_value = 0.5
        s_b, s_t = solve_pi(x_cor, y_cor, z_value, h / DSO, DSO)
        dtheta = 0.04
        theta0 = np.floor(float(s_b.min()) / dtheta) * dtheta
        theta1 = np.ceil(float(s_t.max()) / dtheta) * dtheta
        theta = np.arange(theta0, theta1 + 0.5 * dtheta, dtheta, dtype=np.float32)

        alpha_cor = np.linspace(-0.4, 0.4, 33, dtype=np.float32)
        w_cor = np.linspace(-60.0, 60.0, 41, dtype=np.float32)
        rng = np.random.default_rng(123)
        g_window = rng.normal(
            scale=0.05, size=(len(theta), len(alpha_cor), len(w_cor))
        ).astype(np.float32)

        index = index_theta(s_b, s_t, theta)
        alpha_s, v_s, w_s = compute_alpha_v_w(
            DSO, DSD, h, theta, x_cor, y_cor, z_value
        )
        expected_xy = backproj_cor(
            g_window, alpha_s, v_s, w_s, alpha_cor, w_cor, index
        )
        actual_yx = backproject_slice_cupy(
            g_window,
            s_b,
            s_t,
            x_cor,
            y_cor,
            theta0=float(theta[0]),
            delt_theta=dtheta,
            z_value=z_value,
            h=h,
            DSO=DSO,
            DSD=DSD,
            alpha_cor=alpha_cor,
            w_cor=w_cor,
        )
        np.testing.assert_allclose(actual_yx, expected_xy.T, rtol=2e-4, atol=2e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
