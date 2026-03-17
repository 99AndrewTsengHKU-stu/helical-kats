
import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image

# Ensure local pykatsevich is importable
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.run_dicom_recon import main as recon_main
from pykatsevich import load_dicom_projections

def run_diagnostic():
    dicom_dir = r"D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD_2000"
    if not os.path.exists(dicom_dir):
        print("Dicom dir not found.")
        return

    # Modes to test explicitly
    # 1. Default (Negative angles, Positive Pitch)
    # 2. Negate Angles (Positive angles, Positive Pitch) -> CCW
    # 3. Flip Cols (Reverse Fan)
    # 4. Negate Angles + Flip Cols
    
    test_modes = [
        ("Normal", []),
        ("Negate", ["--negate-angles"]),
        ("FlipCols", ["--flip-cols"]),
        ("Negate_FlipCols", ["--negate-angles", "--flip-cols"])
    ]

    print("--- Running Multi-Mode Geometry Diagnostic ---")
    
    results = []
    
    for name, extra_args in test_modes:
        print(f"\n>>> Mode: {name}")
        out_npy = f"diag_{name}.npy"
        
        # Build command-like argv
        sys.argv = [
            "run_dicom_recon.py",
            "--dicom-dir", dicom_dir,
            "--rows", "512",
            "--cols", "512",
            "--slices", "1",
            "--voxel-size", "0.664",
            "--save-npy", out_npy,
            "--windowing"
        ] + extra_args
        
        try:
            recon_main()
            results.append((name, out_npy))
        except Exception as e:
            print(f"FAILED: {e}")

    # Create a comparison grid
    print("\n--- Creating Comparison Grid ---")
    imgs = []
    for name, npy in results:
        data = np.load(npy)
        # data is (512, 512, 1) -> normalized for PNG by windowing
        img_data = (data[:, :, 0] * 255).astype(np.uint8)
        imgs.append((name, img_data))
    
    if len(imgs) >= 1:
        # Create a 2x2 grid if we have 4 results
        grid = Image.new('L', (1024, 1024))
        for i, (name, img_data) in enumerate(imgs):
            r, c = divmod(i, 2)
            grid.paste(Image.fromarray(img_data), (c*512, r*512))
            # Just some console output for the user
            print(f"Result for {name} saved in grid at position {r},{c}")
        
        grid_path = "aapm_diag_grid.png"
        grid.save(grid_path)
        print(f"Diagnostic grid saved: {grid_path}")

if __name__ == "__main__":
    run_diagnostic()
