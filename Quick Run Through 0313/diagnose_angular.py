"""
Angular Correction Diagnostic Script for Katsevich Reconstruction.

Tests from the 0313 checklist:
  A. Does adding π/2 to angles eliminate the need for post-recon rot90?
  B. Angular uniformity — quantify jitter in DICOM angle spacing
  C. Couch stability — track couch ROI across z slices
  D. T-D window sanity — check bounds use negated angles correctly

Usage:
  python "Quick Run Through 0313/diagnose_angular.py"

Requires L067 DICOM data at the standard path.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Quick Run Through 0304"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

import numpy as np
from matplotlib import pyplot as plt
from time import time

from pykatsevich import load_dicom_projections

# ── Settings ─────────────────────────────────────────────────────────────
DICOM_DIR = r"D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

ROWS = 256
COLS = 256
SLICES = 1          # single-slice for quick A/B tests
VOXEL_SIZE = 0.664

DECIMATE = 10       # fast iteration


def load_data():
    """Load and decimate DICOM projections."""
    print(f"Loading DICOM from {DICOM_DIR} ...")
    sino, meta = load_dicom_projections(DICOM_DIR)
    if DECIMATE > 1:
        sino = sino[::DECIMATE].copy()
        meta["angles_rad"] = meta["angles_rad"][::DECIMATE]
        if meta["table_positions_mm"] is not None:
            meta["table_positions_mm"] = meta["table_positions_mm"][::DECIMATE]
        meta["scan_geometry"]["helix"]["angles_count"] = len(sino)
        meta["pitch_mm_per_angle"] *= DECIMATE
    return sino, meta


# ═════════════════════════════════════════════════════════════════════════
# Test B: Angular Uniformity
# ═════════════════════════════════════════════════════════════════════════
def test_angular_uniformity(meta):
    """Quantify angle spacing jitter in DICOM data."""
    angles = meta["angles_rad"]
    diffs = np.diff(angles)
    mean_step = np.mean(diffs)
    std_step = np.std(diffs)
    rel_jitter = std_step / abs(mean_step) * 100

    print("\n" + "=" * 60)
    print("TEST B: Angular Uniformity")
    print("=" * 60)
    print(f"  N views:          {len(angles)}")
    print(f"  Angle range:      {np.degrees(angles[-1] - angles[0]):.2f} deg "
          f"({(angles[-1] - angles[0]) / (2 * np.pi):.3f} turns)")
    print(f"  Mean step:        {np.degrees(mean_step):.5f} deg ({mean_step:.6f} rad)")
    print(f"  Std step:         {np.degrees(std_step):.5f} deg ({std_step:.6f} rad)")
    print(f"  Relative jitter:  {rel_jitter:.3f}%")
    print(f"  Min step:         {np.degrees(diffs.min()):.5f} deg")
    print(f"  Max step:         {np.degrees(diffs.max()):.5f} deg")

    if rel_jitter > 1.0:
        print("  [!] Significant jitter! Per-view delta_s should be used instead of constant.")
    else:
        print("  [OK] Jitter is small -- constant delta_s is acceptable.")

    # Plot angle spacing
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(np.degrees(diffs), linewidth=0.5)
    axes[0].axhline(np.degrees(mean_step), color='r', linestyle='--', label=f'mean={np.degrees(mean_step):.4f}°')
    axes[0].set_xlabel("View index")
    axes[0].set_ylabel("Angle step (deg)")
    axes[0].set_title("Angle step per view")
    axes[0].legend()

    axes[1].hist(np.degrees(diffs), bins=50)
    axes[1].set_xlabel("Angle step (deg)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Step distribution (jitter={rel_jitter:.2f}%)")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "angular_uniformity.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved -> {path}")
    return rel_jitter


# ═════════════════════════════════════════════════════════════════════════
# Test A: 90° Angle Offset vs Post-Recon rot90
# ═════════════════════════════════════════════════════════════════════════
def test_angle_offset(sino, meta):
    """
    Compare:
      Mode 1: negate + flip_rows, then rot90 post-recon (current approach)
      Mode 2: negate + flip_rows + angles += π/2, NO rot90
      Mode 3: negate + flip_rows + angles -= π/2, NO rot90

    If mode 2 or 3 matches mode 1, the rot90 is just compensating for a
    start-angle offset between DICOM's 0° axis and the code's X-axis.
    """
    # Lazy imports (need CUDA)
    from backproject_safe import ensure_astra_cuda_init
    ensure_astra_cuda_init()

    import copy
    import astra
    from pykatsevich.geometry import astra_helical_views
    from pykatsevich.initialize import create_configuration
    from pykatsevich.filter import filter_katsevich, sino_weight_td
    from backproject_safe import backproject_safe

    def recon_single_slice(sino_in, meta_in, negate, flip_rows, angle_offset, apply_rot90):
        """Reconstruct 1 central slice with given options."""
        angles = meta_in["angles_rad"].copy()
        scan_geom = copy.deepcopy(meta_in["scan_geometry"])
        z_shift = meta_in["table_positions_mm"] - meta_in["table_positions_mm"].mean() \
            if meta_in["table_positions_mm"] is not None else np.zeros(len(angles), dtype=np.float32)

        s = sino_in.copy()
        if negate:
            angles = -angles
        if angle_offset != 0:
            angles = angles + angle_offset
        if flip_rows:
            s = s[:, ::-1, :].copy()

        scan_geom["helix"]["pitch_mm_rad"] = float(abs(meta_in["pitch_mm_per_rad_signed"]))
        scan_geom["helix"]["angles_range"] = float(abs(angles[-1] - angles[0]))

        views = astra_helical_views(
            scan_geom["SOD"], scan_geom["SDD"],
            scan_geom["detector"]["detector psize"],
            angles, meta_in["pitch_mm_per_angle"],
            vertical_shifts=z_shift,
        )

        det_rows_n = scan_geom["detector"]["detector rows"]
        det_cols_n = scan_geom["detector"]["detector cols"]

        proj_geom = astra.create_proj_geom("cone_vec", det_rows_n, det_cols_n, views)
        half_x = COLS * VOXEL_SIZE * 0.5
        half_y = ROWS * VOXEL_SIZE * 0.5
        vol_geom = astra.create_vol_geom(ROWS, COLS, 1, -half_x, half_x, -half_y, half_y, -0.5 * VOXEL_SIZE, 0.5 * VOXEL_SIZE)

        conf = create_configuration(scan_geom, vol_geom)
        conf['source_pos'] = angles.astype(np.float32)
        conf['delta_s'] = float(np.mean(np.diff(angles)))

        sino_f32 = np.asarray(s, dtype=np.float32, order="C")
        filtered = filter_katsevich(sino_f32, conf,
            {"Diff": {"Print time": False}, "FwdRebin": {"Print time": False}, "BackRebin": {"Print time": False}})
        sino_td = sino_weight_td(filtered, conf, False)
        rec = backproject_safe(sino_td, conf, vol_geom, proj_geom, tqdm_bar=False)

        if apply_rot90:
            rec = np.rot90(rec, k=1, axes=(0, 1))
        return rec[:, :, 0]

    print("\n" + "=" * 60)
    print("TEST A: 90° Angle Offset vs Post-Recon rot90")
    print("=" * 60)

    modes = [
        ("Current (negate+flip+rot90)",    True, True, 0.0,       True),
        ("negate+flip+offset(+π/2)",       True, True, np.pi/2,   False),
        ("negate+flip+offset(-π/2)",       True, True, -np.pi/2,  False),
        ("negate+flip (no rot, no offset)", True, True, 0.0,       False),
    ]

    results = {}
    for name, neg, flr, offset, rot in modes:
        print(f"  Reconstructing: {name} ...", end="", flush=True)
        t0 = time()
        rec = recon_single_slice(sino, meta, neg, flr, offset, rot)
        print(f" {time()-t0:.1f}s")
        results[name] = rec

    # Compare each mode to the current approach using SSIM-like metric
    ref = results["Current (negate+flip+rot90)"]
    ref_norm = (ref - ref.mean()) / max(ref.std(), 1e-12)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for ax, (name, img) in zip(axes.flat, results.items()):
        vmin, vmax = np.percentile(img, [1, 99])
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)

        if name != "Current (negate+flip+rot90)":
            img_norm = (img - img.mean()) / max(img.std(), 1e-12)
            corr = float(np.mean(ref_norm * img_norm))
            ax.set_title(f"{name}\ncorr={corr:.4f}", fontsize=9)
        else:
            ax.set_title(f"{name} (reference)", fontsize=9)
        ax.axis("off")

    plt.suptitle("Test A: Does angle offset eliminate rot90?", fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "test_angle_offset.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved -> {path}")


# ═════════════════════════════════════════════════════════════════════════
# Test C: Couch Stability
# ═════════════════════════════════════════════════════════════════════════
def test_couch_stability_from_volume(vol_path=None):
    """
    Track couch position across z-slices in an existing reconstruction volume.
    The couch is the bottom-most high-density horizontal structure in each slice.

    If no volume exists, print instructions and skip.
    """
    if vol_path is None:
        vol_path = os.path.join(os.path.dirname(__file__), "..", "Quick Run Through 0304", "L067_rec_560.npy")

    if not os.path.exists(vol_path):
        print("\n" + "=" * 60)
        print("TEST C: Couch Stability -- SKIPPED (volume not found)")
        print(f"  Expected: {vol_path}")
        print("  Run full reconstruction first, then re-run this test.")
        print("=" * 60)
        return

    print("\n" + "=" * 60)
    print("TEST C: Couch Stability")
    print("=" * 60)
    print(f"  Loading volume: {vol_path}")

    rec = np.load(vol_path)
    print(f"  Volume shape: {rec.shape}")

    # Sample every N-th slice
    n_slices = rec.shape[2]
    step = max(n_slices // 50, 1)
    slice_indices = list(range(0, n_slices, step))

    # For each slice, find the couch: look at the bottom 30% of the image,
    # find the row with the strongest horizontal edge (the couch surface)
    couch_rows = []
    couch_strengths = []

    for si in slice_indices:
        img = rec[:, :, si]
        rows_n = img.shape[0]
        # Look at bottom 40% of image
        bottom_start = int(rows_n * 0.6)
        bottom = img[bottom_start:, :]

        # Vertical gradient (row-wise)
        gy = np.diff(bottom, axis=0)
        # Sum gradient magnitude across columns for each row
        row_gradient_sum = np.sum(np.abs(gy), axis=1)
        # The couch top edge is where we see the strongest horizontal gradient
        couch_row_local = np.argmax(row_gradient_sum)
        couch_row_global = bottom_start + couch_row_local
        couch_rows.append(couch_row_global)
        couch_strengths.append(row_gradient_sum[couch_row_local])

    couch_rows = np.array(couch_rows)
    couch_strengths = np.array(couch_strengths)

    # Filter out weak detections (no couch visible)
    threshold = np.median(couch_strengths) * 0.3
    valid = couch_strengths > threshold
    valid_indices = np.array(slice_indices)[valid]
    valid_rows = couch_rows[valid]

    if len(valid_rows) < 5:
        print("  [!] Could not reliably detect couch in enough slices.")
        return

    drift = valid_rows.max() - valid_rows.min()
    std_drift = np.std(valid_rows)

    print(f"  Couch detected in {len(valid_rows)}/{len(slice_indices)} slices")
    print(f"  Row range:  [{valid_rows.min()}, {valid_rows.max()}]  (drift={drift} pixels)")
    print(f"  Row std:    {std_drift:.2f} pixels")

    if drift <= 3:
        print("  [OK] Couch is stable -- geometry alignment looks correct.")
    elif drift <= 10:
        print("  [!] Moderate couch drift -- possible residual angular error.")
    else:
        print("  [FAIL] Significant couch drift -- angular/geometric mismatch likely.")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    axes[0].plot(valid_indices, valid_rows, 'b.-', markersize=3)
    axes[0].set_xlabel("Slice index (z)")
    axes[0].set_ylabel("Couch row position (pixels)")
    axes[0].set_title(f"Couch stability: drift={drift}px, std={std_drift:.1f}px")
    axes[0].axhline(np.median(valid_rows), color='r', linestyle='--', alpha=0.5, label=f'median={np.median(valid_rows):.0f}')
    axes[0].legend()

    # Show 5 representative slices with couch line marked
    sample_idx = np.linspace(0, len(valid_indices) - 1, min(5, len(valid_indices)), dtype=int)
    for i, si_idx in enumerate(sample_idx):
        si = valid_indices[si_idx]
        row = valid_rows[si_idx]
        # small inset in axes[1]
        ax = fig.add_axes([0.05 + i * 0.19, 0.02, 0.17, 0.35])
        img = rec[:, :, si]
        vmin, vmax = np.percentile(img, [2, 98])
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.axhline(row, color='r', linewidth=0.8)
        ax.set_title(f"z={si}", fontsize=8)
        ax.axis("off")

    axes[1].axis("off")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "couch_stability.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved -> {path}")


# ═════════════════════════════════════════════════════════════════════════
# Test D: T-D Window Sanity Check
# ═════════════════════════════════════════════════════════════════════════
def test_td_window(meta):
    """
    Check that Tam-Danielsson bounds are computed correctly with negated angles.

    The T-D window depends on progress_per_turn and scan geometry, NOT on angles
    directly. But the z-coordinate mapping in backprojection uses
    source_pos * progress_per_radian. If angles are negated, the z-mapping
    reverses — check if this is handled correctly.
    """
    import copy
    from pykatsevich.initialize import create_configuration
    import astra

    print("\n" + "=" * 60)
    print("TEST D: Tam-Danielsson Window Sanity")
    print("=" * 60)

    angles_orig = meta["angles_rad"].copy()
    angles_neg = -angles_orig
    scan_geom = copy.deepcopy(meta["scan_geometry"])
    scan_geom["helix"]["pitch_mm_rad"] = abs(meta["pitch_mm_per_rad_signed"])

    # Config with original angles
    scan_geom_o = copy.deepcopy(scan_geom)
    scan_geom_o["helix"]["angles_range"] = float(abs(angles_orig[-1] - angles_orig[0]))
    half = 10 * VOXEL_SIZE * 0.5
    vol_geom = astra.create_vol_geom(ROWS, COLS, 10,
        -COLS*VOXEL_SIZE*0.5, COLS*VOXEL_SIZE*0.5,
        -ROWS*VOXEL_SIZE*0.5, ROWS*VOXEL_SIZE*0.5,
        -half, half)
    conf_orig = create_configuration(scan_geom_o, vol_geom)

    # Config with negated angles
    scan_geom_n = copy.deepcopy(scan_geom)
    scan_geom_n["helix"]["angles_range"] = float(abs(angles_neg[-1] - angles_neg[0]))
    conf_neg = create_configuration(scan_geom_n, vol_geom)

    # The T-D window should be the same since it depends on geometry, not angle sign
    td_min_orig = conf_orig['proj_row_mins']
    td_max_orig = conf_orig['proj_row_maxs']
    td_min_neg = conf_neg['proj_row_mins']
    td_max_neg = conf_neg['proj_row_maxs']

    print(f"  Original angles: range {np.degrees(angles_orig[0]):.1f}° to {np.degrees(angles_orig[-1]):.1f}°")
    print(f"  Negated angles:  range {np.degrees(angles_neg[0]):.1f}° to {np.degrees(angles_neg[-1]):.1f}°")
    print(f"  T-D row_mins identical: {np.allclose(td_min_orig, td_min_neg)}")
    print(f"  T-D row_maxs identical: {np.allclose(td_max_orig, td_max_neg)}")

    # Check the z-mapping: source_pos * progress_per_radian
    # With original angles, source_pos is centered; with negated, it's also centered
    # but in the opposite direction
    sp_orig = conf_orig['source_pos']
    sp_neg = conf_neg['source_pos']
    ppr = conf_orig['progress_per_radian']

    print(f"\n  source_pos (original): [{sp_orig[0]:.4f}, {sp_orig[-1]:.4f}]")
    print(f"  source_pos (negated):  [{sp_neg[0]:.4f}, {sp_neg[-1]:.4f}]")
    print(f"  progress_per_radian:   {ppr:.4f} mm/rad")

    # The critical line in backprojection:
    #   z_coord = source_pos * progress_per_radian + row * scale_helpers / dia
    # With negated angles, source_pos changes sign → z mapping reverses!
    # But we OVERRIDE source_pos with actual (negated) angles in the pipeline.
    # Let's check if the override is consistent.

    # After override: source_pos = negated DICOM angles
    delta_s_orig = float(np.mean(np.diff(angles_orig)))
    delta_s_neg = float(np.mean(np.diff(angles_neg)))
    print(f"\n  delta_s (original): {delta_s_orig:.6f} rad (direction: {'CCW' if delta_s_orig > 0 else 'CW'})")
    print(f"  delta_s (negated):  {delta_s_neg:.6f} rad (direction: {'CCW' if delta_s_neg > 0 else 'CW'})")

    # z_shift from table positions vs from angles
    if meta["table_positions_mm"] is not None:
        tp = meta["table_positions_mm"]
        z_from_table = tp - tp.mean()
        z_from_angles_orig = angles_orig * ppr
        z_from_angles_neg = angles_neg * ppr

        # The ASTRA views use explicit vertical_shifts (from table positions),
        # so the backprojection source z comes from the views, NOT from source_pos.
        # But filter.py's backproject_a uses: source_pos * progress_per_radian
        # for the z-coordinate mapping.
        # After angle negation: source_pos (=negated angles) * ppr gives REVERSED z.
        # Meanwhile, ASTRA views still use the ORIGINAL table positions.
        # This is a POTENTIAL INCONSISTENCY if using filter.py's CPU backprojection.

        # The CuPy kernel uses src_z from proj_geom['Vectors'][:,2] (= ASTRA views),
        # so it's correct. But source_pos*ppr is only used in filter.py's CPU path
        # and in the T-D z_coord_mins/maxs calculation.

        corr_table_orig = np.corrcoef(z_from_table[:100], z_from_angles_orig[:100])[0, 1]
        corr_table_neg = np.corrcoef(z_from_table[:100], z_from_angles_neg[:100])[0, 1]

        print(f"\n  Z-mapping consistency check:")
        print(f"    corr(table_z, original_angles * ppr): {corr_table_orig:.6f}")
        print(f"    corr(table_z, negated_angles * ppr):  {corr_table_neg:.6f}")

        if corr_table_neg < 0:
            print("    [!] NEGATED angles * ppr gives REVERSED z-direction vs table positions!")
            print("    → This means filter.py's T-D z_coord_mins/maxs (line 534) may be wrong")
            print("    → The CuPy kernel is NOT affected (uses ASTRA views for src_z)")
            print("    → But the T-D weighting window (sino_weight_td) may apply wrong bounds")
        else:
            print("    [OK] Z-direction is consistent after negation.")

    print()


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    sino, meta = load_data()

    # Test B: Angular uniformity (no GPU needed)
    test_angular_uniformity(meta)

    # Test D: T-D window sanity (no GPU needed)
    test_td_window(meta)

    # Test A: 90° offset (needs GPU)
    try:
        test_angle_offset(sino, meta)
    except Exception as e:
        print(f"\nTest A failed: {e}")
        import traceback; traceback.print_exc()

    # Test C: Couch stability (from existing volume, no GPU needed)
    test_couch_stability_from_volume()

    print("\n" + "=" * 60)
    print("ALL DIAGNOSTICS COMPLETE")
    print("=" * 60)
