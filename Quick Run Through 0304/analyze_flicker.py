"""
Analyze slice-to-slice flickering in the existing 560-slice reconstruction.
Focus on: periodicity, correlation with detector rows (64) and chunk boundaries.
"""
import numpy as np
from matplotlib import pyplot as plt
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
npy_path = os.path.join(OUT_DIR, "L067_rec_560.npy")

print(f"Loading {npy_path} ...")
rec = np.load(npy_path)
print(f"Shape: {rec.shape}, dtype: {rec.dtype}")

ROWS, COLS, SLICES = rec.shape
cx, cy = ROWS // 2, COLS // 2

# ── Per-slice statistics ────────────────────────────────────────────────
roi_r = 80  # larger ROI for better stats
means = np.array([rec[cx-roi_r:cx+roi_r, cy-roi_r:cy+roi_r, s].mean() for s in range(SLICES)])
stds  = np.array([rec[cx-roi_r:cx+roi_r, cy-roi_r:cy+roi_r, s].std()  for s in range(SLICES)])

# Slice-to-slice differences
diffs = np.diff(means)

# ── FFT of diffs ────────────────────────────────────────────────────────
fft_vals = np.fft.rfft(diffs - diffs.mean())
freqs = np.fft.rfftfreq(len(diffs), d=1.0)  # in cycles/slice
power = np.abs(fft_vals) ** 2
periods = np.zeros_like(freqs)
periods[1:] = 1.0 / freqs[1:]

# Top 10 peaks
top_idx = np.argsort(power[1:])[-10:][::-1] + 1  # skip DC
print("\nTop 10 FFT peaks in slice-diff signal:")
print(f"{'Rank':>4}  {'Period (slices)':>16}  {'Power':>12}")
for rank, idx in enumerate(top_idx):
    print(f"{rank+1:4d}  {periods[idx]:16.1f}  {power[idx]:12.6f}")

# ── Check chunk boundary effects ───────────────────────────────────────
CHUNK_Z = 64
boundaries = [i * CHUNK_Z for i in range(1, SLICES // CHUNK_Z)]
print(f"\nChunk boundaries at slices: {boundaries}")

boundary_diffs = [abs(diffs[b-1]) for b in boundaries if b-1 < len(diffs)]
all_diffs_abs = np.abs(diffs)
print(f"Mean |diff| at boundaries: {np.mean(boundary_diffs):.6f}")
print(f"Mean |diff| overall:       {np.mean(all_diffs_abs):.6f}")
print(f"Max  |diff| overall:       {np.max(all_diffs_abs):.6f}")
print(f"Boundary diffs are {np.mean(boundary_diffs)/np.mean(all_diffs_abs):.2f}x average")

# ── Multi-pixel trace at different radii ────────────────────────────────
# Check if the flicker pattern varies with distance from center
radii = [0, 50, 100, 150, 200]
traces = {}
for r in radii:
    px, py = cx + r, cy
    if px < ROWS:
        traces[r] = rec[px, py, :]

# ── Visualization ───────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(16, 14))

# 1. Slice means with chunk boundaries
ax = axes[0, 0]
ax.plot(means, 'b-', linewidth=0.5)
for b in boundaries:
    ax.axvline(b, color='r', alpha=0.3, linewidth=0.5)
ax.set_title("Center ROI mean per slice")
ax.set_xlabel("Slice")
ax.set_ylabel("Mean")

# 2. Slice-to-slice diffs
ax = axes[0, 1]
ax.plot(diffs, 'b-', linewidth=0.5)
for b in boundaries:
    ax.axvline(b, color='r', alpha=0.3, linewidth=0.5)
ax.set_title("Slice-to-slice mean difference")
ax.set_xlabel("Slice")
ax.set_ylabel("Diff")

# 3. FFT power spectrum
ax = axes[1, 0]
# Plot only meaningful range (period > 2 slices)
mask = (freqs > 0) & (freqs < 0.5)
ax.plot(periods[mask], power[mask], 'b-')
ax.set_xlim(2, 200)
ax.set_xlabel("Period (slices)")
ax.set_ylabel("Power")
ax.set_title("FFT of slice-diff signal")
# Mark key periods
for p_mark in [64, 32, 128, 16]:
    ax.axvline(p_mark, color='r', alpha=0.3, linestyle='--')
    ax.text(p_mark, ax.get_ylim()[1]*0.9, f"{p_mark}", color='r', fontsize=8, ha='center')

# 4. FFT power (log scale, vs frequency)
ax = axes[1, 1]
ax.semilogy(freqs[1:], power[1:], 'b-')
ax.set_xlabel("Frequency (cycles/slice)")
ax.set_ylabel("Power (log)")
ax.set_title("FFT power spectrum (log)")
# Mark 1/64 frequency
for p_mark in [64, 32, 128]:
    f_mark = 1.0 / p_mark
    ax.axvline(f_mark, color='r', alpha=0.3, linestyle='--')
    ax.text(f_mark, ax.get_ylim()[1]*0.5, f"1/{p_mark}", color='r', fontsize=8)

# 5. Pixel traces at different radii
ax = axes[2, 0]
for r in radii:
    if r in traces:
        t = traces[r]
        # Normalize for comparison
        t_norm = (t - t.mean()) / max(t.std(), 1e-10)
        ax.plot(t_norm, linewidth=0.5, label=f"r={r}")
ax.set_title("Pixel traces at different radii (normalized)")
ax.set_xlabel("Slice")
ax.legend(fontsize=7)
ax.set_ylim(-5, 5)

# 6. Autocorrelation of diffs
ax = axes[2, 1]
autocorr = np.correlate(diffs - diffs.mean(), diffs - diffs.mean(), mode='full')
autocorr = autocorr[len(autocorr)//2:]  # positive lags only
autocorr /= autocorr[0]  # normalize
ax.plot(autocorr[:200], 'b-')
ax.set_xlabel("Lag (slices)")
ax.set_ylabel("Autocorrelation")
ax.set_title("Autocorrelation of slice diffs")
for p_mark in [64, 32, 128]:
    ax.axvline(p_mark, color='r', alpha=0.3, linestyle='--')
    ax.text(p_mark, 0.8, f"{p_mark}", color='r', fontsize=8, ha='center')

fig.suptitle("L067 Flicker Analysis (560 slices)", fontsize=14)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "flicker_fft_detail.png")
plt.savefig(out_path, dpi=150)
print(f"\nSaved -> {out_path}")
plt.close()

# ── Also compute: what fraction of the flicker is at the 64-slice period? ──
# Band around 1/64 frequency
f_target = 1.0 / 64
band = (freqs > f_target * 0.8) & (freqs < f_target * 1.2)
power_at_64 = power[band].sum()
power_total = power[1:].sum()
print(f"\nPower at ~64-slice period: {power_at_64:.6f} ({100*power_at_64/power_total:.1f}% of total)")

# Projs per turn (from metadata): 48590 projs / 21.09 turns ≈ 2304 projs/turn
# Detector rows: 64
# Both match the 64-slice period
print(f"\nKey numbers:")
print(f"  Detector rows: 64")
print(f"  Chunk size: 64 slices")
print(f"  ~Projs per turn: 48590/21.09 ≈ {48590/21.09:.0f}")
print(f"  Voxel size: 0.664 mm -> 64 slices = {64*0.664:.1f} mm")
print(f"  Pitch per turn: {abs(-3.6564)*2*np.pi:.1f} mm")
