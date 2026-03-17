
import numpy as np
import sys
from PIL import Image

def save_png(path):
    try:
        img = np.load(path)
        if img.ndim == 3:
            if img.shape[2] == 1:
                img = img[:, :, 0]
            elif img.shape[0] == 1:
                img = img[0]
            else:
                img = img[:, :, img.shape[2]//2]
        
        # Normalize to 0-255
        # Robust normalization (clip outliers)
        vmin = np.percentile(img, 1)
        vmax = np.percentile(img, 99)
        img = np.clip(img, vmin, vmax)
        img = (img - vmin) / (vmax - vmin) * 255.0
        img = img.astype(np.uint8)
        
        out_path = path.replace('.npy', '.png')
        Image.fromarray(img).save(out_path)
        print(f"Saved {out_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        save_png(sys.argv[1])
