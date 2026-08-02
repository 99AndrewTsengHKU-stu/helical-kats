"""Create a source-data comparison panel for CuPy versus MATLAB L067 slices."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cupy-hu", type=Path, required=True)
    parser.add_argument("--cupy-z", type=Path, required=True)
    parser.add_argument("--matlab-reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cupy_hu = np.load(args.cupy_hu)
    cupy_z = np.load(args.cupy_z)
    matlab = loadmat(args.matlab_reference, squeeze_me=True)
    matlab_hu = np.asarray(matlab["hu_reference"], dtype=np.float32)
    matlab_z = np.ravel(matlab["z_reference"])
    if cupy_hu.shape != matlab_hu.shape:
        raise ValueError(f"shape mismatch: {cupy_hu.shape} vs {matlab_hu.shape}")

    nslices = cupy_hu.shape[2]
    fig, axes = plt.subplots(3, nslices, figsize=(4.2 * nslices, 11), squeeze=False)
    for index in range(nslices):
        diff = cupy_hu[:, :, index] - matlab_hu[:, :, index]
        axes[0, index].imshow(matlab_hu[:, :, index], cmap="gray", vmin=-200, vmax=300)
        axes[0, index].set_title(f"MATLAB z={matlab_z[index]:.3f} mm")
        axes[1, index].imshow(cupy_hu[:, :, index], cmap="gray", vmin=-200, vmax=300)
        axes[1, index].set_title(f"CuPy z={cupy_z[index]:.3f} mm")
        image = axes[2, index].imshow(diff, cmap="RdBu_r", vmin=-20, vmax=20)
        axes[2, index].set_title(
            f"CuPy - MATLAB\nMAE={np.mean(np.abs(diff)):.2f} HU; "
            f"RMSE={np.sqrt(np.mean(diff**2)):.2f} HU"
        )
        for row in range(3):
            axes[row, index].axis("off")

    axes[0, 0].set_ylabel("MATLAB", fontsize=12)
    axes[1, 0].set_ylabel("CuPy", fontsize=12)
    axes[2, 0].set_ylabel("Difference", fontsize=12)
    fig.colorbar(image, ax=axes[2, :].tolist(), shrink=0.65, label="HU difference")
    fig.suptitle("L067 Wang Wei Katsevich: validated MATLAB baseline vs CuPy")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.94, bottom=0.03, wspace=0.04, hspace=0.08)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
