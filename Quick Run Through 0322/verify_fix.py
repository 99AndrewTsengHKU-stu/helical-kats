"""
Verification: forward-project GT with CORRECTED vs WRONG geometry,
compare both with actual DICOM projection.
"""
import sys, os, struct
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pydicom
import astra
from pathlib import Path

DICOM_DIR = Path(r"D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD")
GT_DIR = Path(r"D:\AAPM-Data\L067\L067\full_1mm")

def decode_f32(ds, tag, count=1):
    if tag not in ds:
        return None
    raw = bytes(ds[tag].value)
    vals = struct.unpack("<" + "f" * count, raw[:4*count])
    return vals if count > 1 else vals[0]

# ── Read geometry ──
dcm_files = sorted(DICOM_DIR.glob("*.dcm"))
ds0 = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)

sod = decode_f32(ds0, (0x7031, 0x1003))
sdd = decode_f32(ds0, (0x7031, 0x1031))
psize_col_det = decode_f32(ds0, (0x7029, 0x1002))
psize_row_det = decode_f32(ds0, (0x7029, 0x1006))
central_elem = decode_f32(ds0, (0x7031, 0x1033), count=2)
det_rows = int(ds0.Columns)
det_cols = int(ds0.Rows)

# Corrected geometry
delta_gamma_correct = psize_col_det / sdd
j_center_correct = central_elem[0]
gamma_c = (np.arange(det_cols, dtype=np.float64) - j_center_correct) * delta_gamma_correct
u_flat_c = sdd * np.tan(gamma_c)
flat_psize_correct = float((u_flat_c[-1] - u_flat_c[0]) / (det_cols - 1))

# Wrong geometry (old code interpretation)
delta_gamma_wrong = (369.625 / det_cols) / sod
j_center_wrong = (det_cols - 1) / 2.0
gamma_w = (np.arange(det_cols, dtype=np.float64) - j_center_wrong) * delta_gamma_wrong
u_flat_w = sdd * np.tan(gamma_w)
flat_psize_wrong = float((u_flat_w[-1] - u_flat_w[0]) / (det_cols - 1))

print(f"SOD={sod}, SDD={sdd}, det={det_rows}x{det_cols}")
print(f"Correct: delta_gamma={delta_gamma_correct:.6e}, flat_psize={flat_psize_correct:.4f}, center={j_center_correct}")
print(f"Wrong:   delta_gamma={delta_gamma_wrong:.6e}, flat_psize={flat_psize_wrong:.4f}, center={j_center_wrong}")

# ── Load GT volume ──
gt_files = sorted(GT_DIR.glob("*.IMA")) or sorted(GT_DIR.glob("*.dcm"))
ds_gt0 = pydicom.dcmread(str(gt_files[len(gt_files)//2]))
gt_recon_diam = float(getattr(ds_gt0, 'ReconstructionDiameter', 340))
half_xy = gt_recon_diam / 2.0

n_slices = 40
mid = len(gt_files) // 2
start = mid - n_slices // 2
gt_vol = np.zeros((n_slices, 512, 512), dtype=np.float32)
gt_z_positions = []
for i in range(n_slices):
    ds_i = pydicom.dcmread(str(gt_files[start + i]))
    img = ds_i.pixel_array.astype(np.float32)
    slope = float(getattr(ds_i, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds_i, 'RescaleIntercept', 0.0))
    hu = img * slope + intercept
    mu_cal = float(ds_i[(0x0018, 0x0061)].value) if (0x0018, 0x0061) in ds_i else 0.02
    if mu_cal > 1:
        mu_cal /= 1000.0
    mu = mu_cal * (1.0 + hu / 1000.0)
    mu[hu < -900] = 0.0
    gt_vol[i] = mu
    z_pos = float(getattr(ds_i, 'SliceLocation',
                  float(getattr(ds_i, 'ImagePositionPatient', [0,0,0])[2])))
    gt_z_positions.append(z_pos)

gt_z_lo = min(gt_z_positions)
gt_z_hi = max(gt_z_positions)
gt_z_center = (gt_z_lo + gt_z_hi) / 2.0
# Ensure ascending order for ASTRA
if gt_z_positions[0] > gt_z_positions[-1]:
    gt_vol = gt_vol[::-1]
    gt_z_positions = gt_z_positions[::-1]
gt_z_min, gt_z_max = gt_z_positions[0], gt_z_positions[-1]
print(f"\nGT: {n_slices} slices, z=[{gt_z_min:.1f}, {gt_z_max:.1f}], center={gt_z_center:.1f}")

# ── Find DICOM projection matching GT z-center ──
print("Scanning DICOM z-positions...")
best_idx, best_dist = 0, 1e9
sample_step = max(1, len(dcm_files) // 200)
for idx in range(0, len(dcm_files), sample_step):
    ds_tmp = pydicom.dcmread(str(dcm_files[idx]), stop_before_pixels=True)
    z_tmp = decode_f32(ds_tmp, (0x7031, 0x1002))
    if z_tmp is not None:
        dist = abs(z_tmp - gt_z_center)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx

# Fine search around best
for idx in range(max(0, best_idx - sample_step), min(len(dcm_files), best_idx + sample_step)):
    ds_tmp = pydicom.dcmread(str(dcm_files[idx]), stop_before_pixels=True)
    z_tmp = decode_f32(ds_tmp, (0x7031, 0x1002))
    if z_tmp is not None:
        dist = abs(z_tmp - gt_z_center)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx

ds_match = pydicom.dcmread(str(dcm_files[best_idx]), stop_before_pixels=True)
angle_match = decode_f32(ds_match, (0x7031, 0x1001))
z_match = decode_f32(ds_match, (0x7031, 0x1002))
print(f"Best match: idx={best_idx}, z={z_match:.1f} (GT center={gt_z_center:.1f}, dist={best_dist:.1f})")
print(f"Angle = {np.degrees(angle_match):.1f} deg")

# ── Forward project ──
vol_geom = astra.create_vol_geom(512, 512, n_slices,
    -half_xy, half_xy, -half_xy, half_xy, gt_z_min, gt_z_max)

cos_a, sin_a = np.cos(angle_match), np.sin(angle_match)
src = np.array([sod * cos_a, sod * sin_a, z_match])
det_ctr = np.array([-(sdd - sod) * cos_a, -(sdd - sod) * sin_a, z_match])

def do_fp(fp_psize, row_psize):
    u_vec = np.array([-fp_psize * sin_a, fp_psize * cos_a, 0.0])
    v_vec = np.array([0.0, 0.0, row_psize])
    vectors = np.array([[*src, *det_ctr, *u_vec, *v_vec]], dtype=np.float64)
    pg = astra.create_proj_geom('cone_vec', det_rows, det_cols, vectors)
    pid = astra.data3d.create('-sino', pg, 0)
    vid = astra.data3d.create('-vol', vol_geom, gt_vol)
    cfg = astra.astra_dict('FP3D_CUDA')
    cfg['ProjectionDataId'] = pid
    cfg['VolumeDataId'] = vid
    aid = astra.algorithm.create(cfg)
    astra.algorithm.run(aid)
    result = np.squeeze(astra.data3d.get(pid).copy())
    astra.algorithm.delete(aid)
    astra.data3d.delete(pid)
    astra.data3d.delete(vid)
    return result

print("Forward projecting with CORRECT geometry...")
fp_correct = do_fp(flat_psize_correct, psize_row_det)
print(f"  FP correct shape={fp_correct.shape}, range=[{fp_correct.min():.4f}, {fp_correct.max():.4f}]")
print("Forward projecting with WRONG geometry...")
fp_wrong = do_fp(flat_psize_wrong, float(369.625 / det_cols * sdd / sod))
print(f"  FP wrong shape={fp_wrong.shape}, range=[{fp_wrong.min():.4f}, {fp_wrong.max():.4f}]")

# ── Load DICOM projection (raw equi-angular) ──
ds_dcm = pydicom.dcmread(str(dcm_files[best_idx]))
dcm_raw = ds_dcm.pixel_array.astype(np.float32)
slope = float(getattr(ds_dcm, 'RescaleSlope', 1.0))
intercept = float(getattr(ds_dcm, 'RescaleIntercept', 0.0))
dcm_raw = (dcm_raw * slope + intercept).T  # → (det_rows, det_cols)

# Resample DICOM to flat with CORRECT geometry
from scipy.ndimage import map_coordinates
u_target_c = u_flat_c[0] + np.arange(det_cols, dtype=np.float64) * flat_psize_correct
j_src_c = j_center_correct + np.arctan(u_target_c / sdd) / delta_gamma_correct
col_coords = np.broadcast_to(j_src_c[np.newaxis, :], (det_rows, det_cols))
row_coords = np.broadcast_to(np.arange(det_rows, dtype=np.float64)[:, np.newaxis], (det_rows, det_cols))
dcm_flat_correct = map_coordinates(dcm_raw.astype(np.float64), [row_coords, col_coords],
                                    order=3, mode='constant', cval=0.0).astype(np.float32)

# Also resample with WRONG geometry
u_target_w = u_flat_w[0] + np.arange(det_cols, dtype=np.float64) * flat_psize_wrong
j_src_w = j_center_wrong + np.arctan(u_target_w / sdd) / delta_gamma_wrong
col_coords_w = np.broadcast_to(j_src_w[np.newaxis, :], (det_rows, det_cols))
dcm_flat_wrong = map_coordinates(dcm_raw.astype(np.float64), [row_coords, col_coords_w],
                                  order=3, mode='constant', cval=0.0).astype(np.float32)

# ── Profiles ──
cr = det_rows // 2
fp_c_prof = fp_correct[cr, :]
fp_w_prof = fp_wrong[cr, :]
dcm_c_prof = dcm_flat_correct[cr, :]
dcm_w_prof = dcm_flat_wrong[cr, :]

def measure_width(profile, thresh=0.10):
    pos = profile[profile > 0]
    if len(pos) == 0:
        return 0, 0, 0
    p5, p95 = np.percentile(pos, [5, 95])
    norm = np.clip((profile - p5) / max(p95 - p5, 1e-10), 0, 1)
    above = norm > thresh
    if not above.any():
        return 0, 0, 0
    left = int(np.argmax(above))
    right = int(len(above) - 1 - np.argmax(above[::-1]))
    return right - left, left, right

fp_c_w, fp_c_l, fp_c_r = measure_width(fp_c_prof)
fp_w_w, fp_w_l, fp_w_r = measure_width(fp_w_prof)
dcm_c_w, dcm_c_l, dcm_c_r = measure_width(dcm_c_prof)
dcm_w_w, dcm_w_l, dcm_w_r = measure_width(dcm_w_prof)

print(f"\n{'='*60}")
print(f"PROFILE WIDTH COMPARISON (center row)")
print(f"{'='*60}")
print(f"  FP correct geom:   width={fp_c_w} px  [{fp_c_l}..{fp_c_r}]")
print(f"  FP wrong geom:     width={fp_w_w} px  [{fp_w_l}..{fp_w_r}]")
print(f"  DICOM correct res: width={dcm_c_w} px [{dcm_c_l}..{dcm_c_r}]")
print(f"  DICOM wrong res:   width={dcm_w_w} px [{dcm_w_l}..{dcm_w_r}]")
if fp_c_w > 0 and dcm_c_w > 0:
    print(f"\n  CORRECT geom ratio (DICOM/FP) = {dcm_c_w/fp_c_w:.4f}  (target: ~1.0)")
if fp_w_w > 0 and dcm_w_w > 0:
    print(f"  WRONG geom ratio   (DICOM/FP) = {dcm_w_w/fp_w_w:.4f}  (was: ~0.72)")

# ── Plot ──
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def norm01(x):
    mx = x.max()
    return x / mx if mx > 0 else x

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Top-left: sinogram images
axes[0, 0].imshow(fp_correct[:, :], aspect='auto', cmap='gray')
axes[0, 0].set_title('FP (correct geometry)')
axes[0, 1].imshow(dcm_flat_correct[:, :], aspect='auto', cmap='gray')
axes[0, 1].set_title('DICOM (correct resampling)')

# Bottom-left: CORRECT geometry comparison
axes[1, 0].plot(norm01(fp_c_prof), 'b-', label=f'FP correct (w={fp_c_w})', linewidth=1.5)
axes[1, 0].plot(norm01(dcm_c_prof), 'r-', label=f'DICOM correct (w={dcm_c_w})', linewidth=1.5)
if fp_c_w > 0 and dcm_c_w > 0:
    axes[1, 0].set_title(f'CORRECT geometry: ratio={dcm_c_w/fp_c_w:.3f} (target=1.0)')
else:
    axes[1, 0].set_title('CORRECT geometry')
axes[1, 0].legend(fontsize=10)
axes[1, 0].set_xlabel('Detector column')

# Bottom-right: WRONG geometry comparison
axes[1, 1].plot(norm01(fp_w_prof), 'b--', label=f'FP wrong (w={fp_w_w})', linewidth=1.5)
axes[1, 1].plot(norm01(dcm_w_prof), 'r--', label=f'DICOM wrong (w={dcm_w_w})', linewidth=1.5)
if fp_w_w > 0 and dcm_w_w > 0:
    axes[1, 1].set_title(f'WRONG geometry: ratio={dcm_w_w/fp_w_w:.3f} (was ~0.72)')
else:
    axes[1, 1].set_title('WRONG geometry')
axes[1, 1].legend(fontsize=10)
axes[1, 1].set_xlabel('Detector column')

fig.suptitle('Geometry Fix Verification: FP(GT) vs DICOM Projection', fontsize=14, fontweight='bold')
plt.tight_layout()
out = Path(__file__).parent / "verify_fix_fp.png"
plt.savefig(str(out), dpi=150)
print(f"\nSaved -> {out}")
