
import numpy as np
import sys

def measure(path):
    try:
        img = np.load(path)
        # Handle 3D (slices, rows, cols) or 2D
        if img.ndim == 3:
            if img.shape[2] == 1:
                img = img[:, :, 0]
            elif img.shape[0] == 1:
                img = img[0]
            else:
                # Fallback to middle slice
                img = img[:, :, img.shape[2]//2]
            
        gy, gx = np.gradient(img)
        gnorm = np.sqrt(gx**2 + gy**2)
        score = np.mean(gnorm)
        print(f"Sharpness Score for {path}: {score:.6f}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        measure(sys.argv[1])
