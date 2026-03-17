============================================================
Py-Kat DICOM Diagnostic — 0228
Data dir : D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD_2000
Total files: 2000
============================================================

--- First file: L067_4L_100kv_quarterdose.1.00001.dcm ---
InstanceNumber : 1
Rows × Cols : 736 × 64
BitsAllocated : 16
PixelRepresentation: 0
RescaleSlope : 0.00014143614681
RescaleIntercept : -0.2077793795498
PixelSpacing : N/A
SOD : 595.0 mm
SDD : 1085.5999755859375 mm
Angle (private) : 4.679664134979248 rad
TablePosition : N/A mm
DetectorExtents : (369.625, 32.5) (col_mm, row_mm)

--- Pixel value analysis (5 files sampled) ---
L067_4L_100kv_quarterdose.1.00001.dcm: min=-0.04 max=5.68 mean=2.02
L067_4L_100kv_quarterdose.1.00401.dcm: min=-0.08 max=6.59 mean=1.94
L067_4L_100kv_quarterdose.1.00801.dcm: min=-0.08 max=4.67 mean=1.93
L067_4L_100kv_quarterdose.1.01201.dcm: min=-0.08 max=5.04 mean=1.93
L067_4L_100kv_quarterdose.1.01601.dcm: min=-0.07 max=6.45 mean=1.92

Overall calibrated range: [-0.08, 6.59]

--- Angle / Pitch analysis (first vs last 3 views) ---
inst= 1 angle=4.67966 rad table=None mm
inst= 2 angle=4.67694 rad table=None mm
inst= 3 angle=4.67421 rad table=None mm
...
inst= 1998 angle=-0.76631 rad table=None mm
inst= 1999 angle=-0.76904 rad table=None mm
inst= 2000 angle=-0.77176 rad table=None mm

--- 诊断结论 ---
[判断] 值域在 (-5, 50) → 大概率已经是线积分（-log 形式），可直接输入 Katsevich

============================================================
Step 2 额外发现
============================================================

- TablePosition tag (0018,9327) 不存在 → table=None
- DetectorExtents = (369.625, 32.5) mm → psize_cols ≈ 0.502 mm, psize_rows ≈ 0.508 mm
- Angle 顺序: inst=1 → 4.68 rad 递减到 inst=2000 → -0.77 rad (顺时针/负向)
- SOD=595.0 mm, SDD=1085.6 mm
- Pitch 需从 SpiralPitchFactor 推算（table position 不可用）

============================================================
Step 3: 单层重建结果 (2026-02-28)
============================================================
脚本: Quick Run Through 0228/run_single_slice_recon.py
参数: --rows 512 --cols 512 --slices 1 --voxel-size 1.0

[结果] 图像可见胸腔解剖结构（肺、心脏轮廓），这是显著进步！
[问题1] 图像整体偏右下，不在 512×512 FOV 中心
[问题2] 从右上角方向有扇形条纹伪影
[值域] 重建值约在 [-0.05, 0.17]，量级偏低（FBP的HU级别应在-1000到1000）

============================================================
根本原因分析
============================================================
图像偏移 → source_pos (s_min/s_max) 与实际角度范围不匹配

问题根源：initialize.py 中当 angles_range 不在 scan_geometry 里时，
使用以下公式推算：
s_min = -π + z_min / pitch
s_max = π + z_max / pitch
但 slices=1 时 z_min≈z_max≈0，导致 s_len = 2π，
而实际扫描范围是 4.68 rad 到 -0.77 rad ≈ 5.45 rad ≠ 2π

============================================================
下一步建议
============================================================

1. 在 run_dicom_recon.py 里把角度范围显式传给 scan_geometry:
   scan_geometry['helix']['angles_range'] = float(angles_unwrapped[-1] - angles_unwrapped[0])
   这样 create_configuration 会走正确的 s_len = angles_range 路径

2. 同时确认 pitch 是否正确被推算（从 SpiralPitchFactor × collimation）

3. 修复后重跑单层重建，预期图像应该居中且无扇形条纹
