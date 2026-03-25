"""
Phase 6: Test pykatsevich on synthetic phantoms (test01/02/03).

No flip_rows, no negate_angles, no angle_offset — pure vanilla pipeline.
Measures flicker on the reconstruction to determine if the algorithm
itself inherently produces flicker, or if it's our DICOM pipeline.

Also tests our CuPy backprojection vs the library's ASTRA backprojection.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Quick Run Through 0304"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from time import time
import yaml
import astra

from common import phantom_objects_3d, project
from pykatsevich.initialize import create_configuration
from pykatsevich.filter import (
    differentiate, fw_height_rebinning, compute_hilbert_kernel,
    hilbert_conv, rev_rebin_vec, sino_weight_td, backproject_a
)
from backproject_cupy import backproject_cupy

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "tests"))


def flicker_metric(vol):
    """Per-slice-pair flicker: mean|s[i+1]-s[i]| / mean|s[i]|"""
    n = vol.shape[2]
    m = []
    for i in range(n - 1):
        s0 = vol[:, :, i].astype(np.float64)
        s1 = vol[:, :, i + 1].astype(np.float64)
        denom = 0.5 * (np.mean(np.abs(s0)) + np.mean(np.abs(s1)))
        if denom > 1e-12:
            m.append(np.mean(np.abs(s1 - s0)) / denom)
        else:
            m.append(0.0)
    return np.array(m)


def run_phantom_test(yaml_file, label):
    """Run full Katsevich pipeline on a phantom and return flicker metrics."""
    print(f"\n{'='*60}")
    print(f"PHANTOM TEST: {label} ({yaml_file})")
    print(f"{'='*60}", flush=True)

    # Load settings
    yaml_path = os.path.join(TEST_DIR, yaml_file)
    with open(yaml_path, "r") as f:
        settings = yaml.safe_load(f)

    ps = settings['phantom']
    geom = settings['geometry']

    # Generate phantom
    print(f"  Generating phantom: {ps['rows']}x{ps['columns']}x{ps['slices']}, "
          f"voxel={ps['voxel_size']}mm", flush=True)
    phantom = phantom_objects_3d(
        ps['rows'], ps['columns'], ps['slices'],
        voxel_size=ps['voxel_size'],
        objects_list=ps['objects'],
    )
    print(f"    phantom range: [{phantom.min():.3f}, {phantom.max():.3f}]")

    # Phantom flicker (ground truth — should be very low for binary spheres)
    fm_phantom = flicker_metric(phantom)
    print(f"    phantom flicker: mean={fm_phantom.mean():.6f}")

    # Forward project
    print("  Forward projecting...", flush=True)
    t0 = time()
    sinogram, vol_geom, proj_geom = project(phantom, ps['voxel_size'], geom)
    print(f"    Done in {time()-t0:.1f}s, sino shape={sinogram.shape}")

    # Sinogram comes from ASTRA as (det_rows, n_projs, det_cols)
    # pykatsevich expects (n_projs, det_rows, det_cols)
    sinogram_swapped = np.asarray(np.swapaxes(sinogram, 0, 1), order='C')
    print(f"    sinogram (swapped): {sinogram_swapped.shape}")

    # Create configuration
    conf = create_configuration(geom, vol_geom, geom.get('options', {}))

    # Filter pipeline: Diff → FwdRebin → Hilbert → BackRebin → TD
    print("  Filtering...", flush=True)
    t0 = time()
    sino_diff = differentiate(sinogram_swapped, conf)
    sino_rebin = fw_height_rebinning(sino_diff, conf)
    hilbert_array = compute_hilbert_kernel(conf)
    sino_hilbert = hilbert_conv(sino_rebin, hilbert_array, conf)
    sino_rev = rev_rebin_vec(sino_hilbert, conf)
    sino_td = sino_weight_td(sino_rev, conf, False)
    t_filt = time() - t0
    print(f"    Filter done in {t_filt:.1f}s")

    # Backproject with ASTRA (library's own BP)
    print("  Backprojecting (ASTRA)...", flush=True)
    t0 = time()
    rec_astra = backproject_a(sino_td, conf, vol_geom, proj_geom)
    t_bp_astra = time() - t0
    fm_astra = flicker_metric(rec_astra)
    print(f"    ASTRA BP done in {t_bp_astra:.1f}s, range=[{rec_astra.min():.4f}, {rec_astra.max():.4f}]")
    print(f"    ASTRA flicker: mean={fm_astra.mean():.6f}, max={fm_astra.max():.6f}")

    # Backproject with CuPy (our kernel)
    print("  Backprojecting (CuPy)...", flush=True)
    t0 = time()
    rec_cupy = backproject_cupy(sino_td, conf, vol_geom, proj_geom, tqdm_bar=False)
    t_bp_cupy = time() - t0
    fm_cupy = flicker_metric(rec_cupy)
    print(f"    CuPy BP done in {t_bp_cupy:.1f}s, range=[{rec_cupy.min():.4f}, {rec_cupy.max():.4f}]")
    print(f"    CuPy flicker: mean={fm_cupy.mean():.6f}, max={fm_cupy.max():.6f}")

    return {
        'label': label,
        'phantom': phantom,
        'rec_astra': rec_astra,
        'rec_cupy': rec_cupy,
        'fm_phantom': fm_phantom,
        'fm_astra': fm_astra,
        'fm_cupy': fm_cupy,
    }


# ══════════════════════════════════════════════════════════════════════
# RUN ALL THREE TESTS
# ══════════════════════════════════════════════════════════════════════
results = []
for yaml_file, label in [
    ("test03.yaml", "test03 (54x56x58, small)"),
    ("test01.yaml", "test01 (256x256x256, medium)"),
    ("test02.yaml", "test02 (384x600x116, clinical-like)"),
]:
    try:
        r = run_phantom_test(yaml_file, label)
        results.append(r)
    except Exception as e:
        print(f"  FAILED: {e}", flush=True)
        import traceback
        traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("SUMMARY: Phantom Flicker Tests")
print(f"{'='*60}")
print(f"  {'Test':<40} {'Phantom':>10} {'ASTRA BP':>10} {'CuPy BP':>10}")
print(f"  {'-'*70}")
for r in results:
    print(f"  {r['label']:<40} {r['fm_phantom'].mean():>10.6f} {r['fm_astra'].mean():>10.6f} {r['fm_cupy'].mean():>10.6f}")
print(f"  {'Our L067 DICOM pipeline':<40} {'':>10} {'':>10} {'0.229':>10}")
print(f"  {'GT L067':<40} {'':>10} {'':>10} {'~0.040':>10}")

# ══════════════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════════════
n_tests = len(results)
if n_tests > 0:
    fig, axes = plt.subplots(3, n_tests, figsize=(6 * n_tests, 15))
    if n_tests == 1:
        axes = axes.reshape(-1, 1)

    for col, r in enumerate(results):
        mid = r['phantom'].shape[2] // 2

        # Row 0: phantom
        ax = axes[0, col]
        ax.imshow(r['phantom'][:, :, mid], cmap='gray')
        ax.set_title(f"Phantom (fl={r['fm_phantom'].mean():.4f})", fontsize=9)
        ax.axis('off')

        # Row 1: ASTRA reconstruction
        ax = axes[1, col]
        ax.imshow(r['rec_astra'][:, :, mid], cmap='gray')
        ax.set_title(f"ASTRA BP (fl={r['fm_astra'].mean():.4f})", fontsize=9)
        ax.axis('off')

        # Row 2: CuPy reconstruction
        ax = axes[2, col]
        ax.imshow(r['rec_cupy'][:, :, mid], cmap='gray')
        ax.set_title(f"CuPy BP (fl={r['fm_cupy'].mean():.4f})", fontsize=9)
        ax.axis('off')

    axes[0, 0].set_ylabel("Phantom", fontsize=11)
    axes[1, 0].set_ylabel("ASTRA BP", fontsize=11)
    axes[2, 0].set_ylabel("CuPy BP", fontsize=11)

    plt.suptitle("Phantom Flicker: Pure pykatsevich pipeline (no flip/negate)", fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "phantom_flicker_comparison.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved -> {out_path}")

    # Flicker curves
    fig2, axes2 = plt.subplots(n_tests, 1, figsize=(14, 4 * n_tests))
    if n_tests == 1:
        axes2 = [axes2]
    for i, r in enumerate(results):
        ax = axes2[i]
        ax.plot(r['fm_phantom'], 'g-', linewidth=0.8, label=f"Phantom ({r['fm_phantom'].mean():.4f})")
        ax.plot(r['fm_astra'], 'b-', linewidth=0.8, label=f"ASTRA BP ({r['fm_astra'].mean():.4f})")
        ax.plot(r['fm_cupy'], 'r-', linewidth=0.8, label=f"CuPy BP ({r['fm_cupy'].mean():.4f})")
        ax.axhline(0.229, color='purple', linestyle='--', linewidth=0.8, label="Our L067 (0.229)")
        ax.axhline(0.040, color='orange', linestyle='--', linewidth=0.8, label="GT L067 (~0.040)")
        ax.set_xlabel("Slice pair")
        ax.set_ylabel("Flicker")
        ax.set_title(r['label'])
        ax.legend(fontsize=8)
    plt.suptitle("Flicker per slice-pair", fontsize=13)
    plt.tight_layout()
    out_path2 = os.path.join(OUT_DIR, "phantom_flicker_curves.png")
    plt.savefig(out_path2, dpi=150)
    print(f"Saved -> {out_path2}")

print("\nDone.", flush=True)
