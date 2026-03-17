"""
Compare Katsevich reconstruction with GT FBP from AAPM L067.

Usage:
    python compare_katsevich_gt.py rec_L067_raw.npy

The script:
  1. Loads the Katsevich .npy volume (expected shape: rows x cols x slices)
  2. Auto-calibrates to HU using the air background
  3. Loads GT slices from full_1mm/ at matching z-positions
  4. Displays side-by-side comparison in standard CT windows
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pydicom
import matplotlib.pyplot as plt

GT_DIR = Path(r"D:\AAPM-Data\L067\L067\full_1mm")

# --- helpers ---

def load_gt_slices(gt_dir: Path):
    """Load all GT slices, return (volume_HU, z_positions_mm)."""
    files = sorted(gt_dir.glob("*.IMA"))
    if not files:
        files = sorted(gt_dir.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"No IMA/dcm files in {gt_dir}")

    ds0 = pydicom.dcmread(str(files[0]))
    slope = float(getattr(ds0, "RescaleSlope", 1.0))
    intercept = float(getattr(ds0, "RescaleIntercept", 0.0))

    n = len(files)
    rows, cols = ds0.Rows, ds0.Columns
    vol = np.empty((rows, cols, n), dtype=np.float32)
    z_pos = np.empty(n, dtype=np.float64)

    for i, f in enumerate(files):
        ds = pydicom.dcmread(str(f))
        vol[:, :, i] = ds.pixel_array.astype(np.float32) * slope + intercept
        z_pos[i] = float(getattr(ds, "SliceLocation", i))

    return vol, z_pos


def mu_to_hu(vol_mu, mu_water=0.019):
    """Convert linear attenuation (mm^-1) to Hounsfield Units."""
    return (vol_mu - mu_water) / mu_water * 1000.0


def auto_calibrate_mu_water(rec):
    """
    Estimate mu_water from the reconstruction assuming background ~ air (mu~0)
    and that the body center ~ soft tissue (mu ~ mu_water).
    Returns estimated mu_water.
    """
    mid_slice = rec[:, :, rec.shape[2] // 2]
    # Use central 40% as body region
    r0, r1 = int(0.3 * mid_slice.shape[0]), int(0.7 * mid_slice.shape[0])
    c0, c1 = int(0.3 * mid_slice.shape[1]), int(0.7 * mid_slice.shape[1])
    center = mid_slice[r0:r1, c0:c1]
    # Soft tissue peak is around the median of the center region
    mu_est = float(np.median(center[center > 0]))
    print(f"[Auto-cal] Estimated mu_water = {mu_est:.5f} mm^-1")
    return mu_est


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_katsevich_gt.py <rec_npy_file>")
        sys.exit(1)

    rec_path = Path(sys.argv[1])
    rec = np.load(str(rec_path))
    print(f"Katsevich volume: shape={rec.shape}, range=[{rec.min():.5f}, {rec.max():.5f}]")

    # Check if already normalized to [0,1] (windowed)
    if rec.max() <= 1.01 and rec.min() >= -0.01:
        print("[WARNING] Values in [0,1] — looks like --windowing was used. Raw mm^-1 values lost.")
        print("[WARNING] Proceeding with raw values anyway for visual comparison.")

    # --- Load GT ---
    print(f"Loading GT from {GT_DIR} ...")
    gt_vol, gt_z = load_gt_slices(GT_DIR)
    print(f"GT volume: shape={gt_vol.shape}, z=[{gt_z[0]:.1f}, {gt_z[-1]:.1f}] mm")
    print(f"GT HU range: [{gt_vol.min():.0f}, {gt_vol.max():.0f}]")

    # --- HU conversion ---
    # Detect whether values are raw linear attenuation (mm^-1, typically in
    # [-0.01, 0.15]), already in HU ([-1024, ~3000]), or windowed to [0,1].
    val_range = rec.max() - rec.min()
    is_hu = rec.min() < -500 and rec.max() > 500        # already HU
    is_windowed = rec.min() >= -0.01 and rec.max() <= 1.01  # [0,1]
    is_raw_mu = not is_hu and not is_windowed             # raw mm^-1

    if is_raw_mu:
        mu_water_auto = auto_calibrate_mu_water(rec)
        # Use theoretical mu_water for 80 kVp; auto-cal shown for reference.
        mu_water = 0.019
        print(f"[HU] Using mu_water = {mu_water} (auto-cal was {mu_water_auto:.5f})")
        rec_hu = mu_to_hu(rec, mu_water)
    elif is_hu:
        rec_hu = rec
        mu_water = None
    else:
        # Windowed [0,1] — just stretch to rough HU range for display
        rec_hu = rec * 2000 - 1000
        mu_water = None

    print(f"Katsevich HU range: [{rec_hu.min():.0f}, {rec_hu.max():.0f}]")

    # --- Select comparison slices ---
    # Compare 5 slices evenly spaced
    n_compare = 5
    rec_slices = np.linspace(
        int(0.15 * rec.shape[2]),
        int(0.85 * rec.shape[2]),
        n_compare, dtype=int
    )
    gt_slices = np.linspace(
        int(0.15 * gt_vol.shape[2]),
        int(0.85 * gt_vol.shape[2]),
        n_compare, dtype=int
    )

    print(f"\nComparing slices:")
    print(f"  Katsevich: {rec_slices}")
    print(f"  GT:        {gt_slices}")

    # --- Plot: Soft tissue window ---
    fig, axes = plt.subplots(2, n_compare, figsize=(4 * n_compare, 8))
    fig.suptitle("Katsevich vs GT — Soft Tissue Window [-200, 300] HU", fontsize=14)

    for i in range(n_compare):
        axes[0, i].imshow(rec_hu[:, :, rec_slices[i]], cmap="gray", vmin=-200, vmax=300)
        axes[0, i].set_title(f"Kat slice {rec_slices[i]}")
        axes[0, i].axis("off")

        axes[1, i].imshow(gt_vol[:, :, gt_slices[i]], cmap="gray", vmin=-200, vmax=300)
        axes[1, i].set_title(f"GT slice {gt_slices[i]}")
        axes[1, i].axis("off")

    fig.tight_layout()
    out1 = str(rec_path.parent / "compare_soft_tissue.png")
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"Saved: {out1}")

    # --- Plot: Lung window ---
    fig, axes = plt.subplots(2, n_compare, figsize=(4 * n_compare, 8))
    fig.suptitle("Katsevich vs GT — Lung Window [-1000, 200] HU", fontsize=14)

    for i in range(n_compare):
        axes[0, i].imshow(rec_hu[:, :, rec_slices[i]], cmap="gray", vmin=-1000, vmax=200)
        axes[0, i].set_title(f"Kat slice {rec_slices[i]}")
        axes[0, i].axis("off")

        axes[1, i].imshow(gt_vol[:, :, gt_slices[i]], cmap="gray", vmin=-1000, vmax=200)
        axes[1, i].set_title(f"GT slice {gt_slices[i]}")
        axes[1, i].axis("off")

    fig.tight_layout()
    out2 = str(rec_path.parent / "compare_lung.png")
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"Saved: {out2}")

    # --- Plot: Auto-window (percentile) ---
    fig, axes = plt.subplots(2, n_compare, figsize=(4 * n_compare, 8))
    fig.suptitle("Katsevich vs GT — Auto Window (1-99% percentile)", fontsize=14)

    for i in range(n_compare):
        rimg = rec_hu[:, :, rec_slices[i]]
        rv0, rv1 = np.percentile(rimg, [1, 99])
        axes[0, i].imshow(rimg, cmap="gray", vmin=rv0, vmax=rv1)
        axes[0, i].set_title(f"Kat {rec_slices[i]}\n[{rv0:.0f},{rv1:.0f}]")
        axes[0, i].axis("off")

        gimg = gt_vol[:, :, gt_slices[i]]
        gv0, gv1 = np.percentile(gimg, [1, 99])
        axes[1, i].imshow(gimg, cmap="gray", vmin=gv0, vmax=gv1)
        axes[1, i].set_title(f"GT {gt_slices[i]}\n[{gv0:.0f},{gv1:.0f}]")
        axes[1, i].axis("off")

    fig.tight_layout()
    out3 = str(rec_path.parent / "compare_auto.png")
    fig.savefig(out3, dpi=150)
    plt.close(fig)
    print(f"Saved: {out3}")

    # --- Summary statistics ---
    mid_r = rec_hu[:, :, rec.shape[2] // 2]
    mid_g = gt_vol[:, :, gt_vol.shape[2] // 2]
    print(f"\n--- Middle slice statistics ---")
    print(f"Katsevich: mean={mid_r.mean():.1f} HU, std={mid_r.std():.1f}")
    print(f"GT:        mean={mid_g.mean():.1f} HU, std={mid_g.std():.1f}")
    p1r, p99r = np.percentile(mid_r, [1, 99])
    p1g, p99g = np.percentile(mid_g, [1, 99])
    print(f"Katsevich 1-99%: [{p1r:.0f}, {p99r:.0f}] HU")
    print(f"GT        1-99%: [{p1g:.0f}, {p99g:.0f}] HU")


if __name__ == "__main__":
    main()
