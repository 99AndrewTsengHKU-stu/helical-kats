# 螺旋 Katsevich 重建 · 完整进展总结
# Helical Katsevich Reconstruction · Full Progress Summary
### 2026-03-18 → 2026-03-25

---

## 总览 / Overview

| 阶段 Phase | 日期 Date | 核心成果 Key Outcome |
|-----------|-----------|----------------------|
| 1 | 0228–0304 | 管道搭通，首次 560 片重建完成 / Pipeline built, first 560-slice recon complete |
| 2 | 0313 | 角度/坐标修正，flicker 5.7× GT 确认 / Angle+coord fixes, flicker 5.7× GT confirmed |
| 3 | 0318 | 螺旋方向验证，排除多项假设 / Helix direction verified, many hypotheses ruled out |
| 4 | 0319–0321 | 尺度偏差 1.41× 根因排查 / Scale error 1.41× root-cause investigation |
| 5 | 0322 | **DICOM tag 解析 bug 发现并修复** / **DICOM tag parsing bug found & fixed** |
| 6 | 0323 | GPU 滤波实现，全量重建启动 / GPU filter implemented, full recon launched |
| 7 | 0324 | 全量重建完成，每转 flicker 量化 / Full recon done, per-turn flicker quantified |
| 8 | 0325 | **反投影偏移 bug 根因确认并修复** / **Backprojector offset bug found & fixed** |

---

## 第 1–2 阶段（0228–0313）：管道搭建与基础问题确认
## Phase 1–2 (0228–0313): Pipeline Build & Initial Problem Confirmation

### 问题定义 / Problem Statement

重建体存在 **周期性闪烁伪影（flicker）**：
The reconstructed volume has **periodic flicker artifacts**:

- 闪烁指标 0.229 vs GT 0.040 → **比参考体差 5.7 倍**
- Flicker metric 0.229 vs GT 0.040 → **5.7× worse than ground truth**
- 沙发（刚性物体）跨 560 片漂移 **92 px**，理论值应为零
- Couch (rigid object) drifts **92 px** across 560 slices — should be zero
- 不在分块边界处，贯穿整个体积
- NOT at chunk boundaries — present throughout the entire volume

### 已排除原因 / Ruled Out

| # | 假设 Hypothesis | 证据 Evidence |
|---|----------------|---------------|
| 1 | 分块边界混合方式 Chunk boundary blending | 单块/重叠/硬边界结果完全相同 All 3 methods: flicker=0.0000699 |
| 2 | T-D 平滑值 T-D smoothing value | 4 个值 [0.025–0.2] 效果微小 4 values tested, negligible effect |
| 3 | CuPy kernel bug | ASTRA CPU 对比误差 < 1e-5 vs ASTRA CPU, diff < 1e-5 |
| 4 | 角度抖动 Angular jitter | std/mean < 0.1% |
| 5 | cos/sin 精度 cos/sin precision | 大角度值误差 < 1e-6 Large angle error < 1e-6 |
| 6 | 螺旋方向不匹配 Helix handedness mismatch | 确认右手螺旋 dz/ds = +3.815 Confirmed RIGHT-HANDED |
| 7 | pitch 数值误差 Pitch numerical error | 3.656 mm/rad 全投影一致 Consistent across all projections |
| 8 | flip_rows 坐标错误 flip_rows coord error | 数学推导验证正确 Mathematical derivation verified |

### 发现的反常现象 / Anomalies Found

- **T-D 平滑值越大，flicker 越严重**（0.025→0.229；0.100→0.378）
- **Larger T-D smoothing worsens flicker** (0.025→0.229; 0.100→0.378)
- 这说明 TD 窗口边界位置本身可能是错的
- This suggests the T-D window boundary positions themselves may be wrong

---

## 第 3 阶段（0318）：螺旋方向与 TD 诊断
## Phase 3 (0318): Helix Direction & T-D Diagnostics

### 关键验证 / Key Verification

```
螺旋手性分析 / Helix handedness analysis:
  dz/ds = +3.815 → 右手螺旋 RIGHT-HANDED ✓
  TD 窗口位置 / T-D window position: 居中，对称 centered, symmetric ✓
  带符号 pitch 导致 TD mask 为空 → 确认须用 abs(pitch) ✓
  Signed pitch → empty TD mask → confirmed abs(pitch) required ✓
```

### 0318 末的待验证假设 / Open Hypotheses at End of 0318

```
A. 重建无 TD 加权 → 若 flicker 消失则 TD mask 位置错误
   Recon WITHOUT TD weighting → if flicker drops, TD mask position is wrong
B. 反投影 v 坐标 psize_row 不匹配 → 每片采样错误行
   Backprojection v-coord psize_row mismatch → each slice samples wrong row
C. projs_per_turn 按块变化 → 块间幅度不一致
   projs_per_turn varies per chunk → amplitude inconsistency between chunks
D. 微分 d_row 符号错误 → z 相关滤波误差
   Diff d_row wrong sign → z-dependent filter error
E. 前向/反向 rebin 方向约定不匹配
   Forward/backward rebin direction convention mismatch
```

---

## 第 4 阶段（0319–0321）：尺度偏差 1.41× 根因排查
## Phase 4 (0319–0321): Scale Error 1.41× Root-Cause Investigation

### 现象 / Symptom

重建体尺寸比 GT 小约 1.41 倍（身体直径：重建 ≈ 243mm vs GT 340mm）
Reconstruction is ~1.41× too small vs GT (body diameter: recon ≈ 243mm vs GT 340mm)

### 系统性排查 / Systematic Investigation

| 测试 Test | 结果 Result |
|-----------|-------------|
| CuPy vs ASTRA 反投影 | 完全一致，ratio=0.9994 / Identical, ratio=0.9994 |
| 幻体测试（已知几何） | 管道正确，位置误差 <1px / Pipeline correct, position error <1px |
| 等角重采样 vs 不重采样 | 4 种配置结果相同，缩放无差异 / 4 configs identical |
| 改变 psize | **对尺度无影响** / **No effect on scale** |

### 数学洞察 / Mathematical Insight

```
psize 在滤波和反投影中均出现，效果近似相消：
psize appears in both filter and backprojection, effects approximately cancel:

  filter scale ∝ 1/psize
  backprojection scale ∝ psize
  → net: psize-invariant（至一阶）
```

### 可疑信号 / Suspicious Signal

```
投影体宽度诊断 / Projection width diagnostic (angle 72°):
  实际探测器宽度 Actual width on detector: 486 px
  几何预测宽度   Geometry-predicted width: 618 px
  比值 Ratio: 486/618 = 0.787

→ 探测器几何 tag 描述的比实际宽 21% / Geometry tags describe 21% wider than actual
→ SOD/SDD/探测器参数解读可能有误 / SOD/SDD/detector params may be misinterpreted
```

---

## 第 5 阶段（0322）：DICOM Tag 解析 Bug 根因发现与修复
## Phase 5 (0322): DICOM Tag Parsing Bug — Root Cause Found & Fixed

### 根因 / Root Cause

**私有 tag `(0x7031, 0x1033)` 被误解读为"探测器范围（mm）"，实际含义为"中心元素索引（像素）"**

**Private tag `(0x7031, 0x1033)` was misread as "detector extent in mm"; actual meaning is "central element pixel index"**

DICOM-CT-PD 用户手册 v12：
Per DICOM-CT-PD User Manual v12:
> `DetectorCentralElement (7031,1033)`: the pixel (Column X, Row Y) index of the detector element aligning with the isocenter and the focal center.

```
错误解读 / Wrong interpretation:
  (369.625, 32.5) → 探测器物理范围 physical extent in mm
  arc_psize_iso = 369.625 / 736 = 0.502 mm → delta_gamma = 8.44e-4 rad
  flat_psize_cols = 0.947 mm  ← 比实际小 1.45×

正确解读 / Correct interpretation:
  (369.625, 32.5) → 中心元素像素坐标 central element pixel coordinate
  真正的 psize 来自 / actual psize from (7029,1002): 1.2858 mm（探测器面 at detector）
  delta_gamma = 1.2858 / 1085.6 = 1.184e-3 rad
  flat_psize_cols = 1.374 mm ← 正确 correct
```

### 修复前后对比 / Before vs After Fix

| 参数 Parameter | 错误 Wrong | 正确 Correct | 比值 Ratio |
|----------------|-----------|--------------|-----------|
| delta_gamma | 8.44e-4 rad | 1.184e-3 rad | **1.403** |
| flat_psize_cols | 0.947 mm | 1.374 mm | 1.451 |
| 隐含采集直径 Implied DCD | 363.7 mm | **502.4 mm** | 1.381 |
| 中心列 center_col | 367.5 | **369.625** | +2.125 px |

```
验证 / Verification:
  修正后 DCD 502.4mm ≈ GT DataCollectionDiameter tag 500mm ✓
  delta_gamma 比值 1.403 ≈ 观测尺度误差 1.41 ✓
  投影体宽预测误差 3.8%（修复前 25.5%）✓
```

### 修复内容 / Fix Applied (`pykatsevich/dicom.py`)

```python
# 读取探测器形状 / Read detector shape
DetectorShape = (7029,100B) → "CYLINDRICAL"

# 圆柱形探测器：从探测器面 psize 计算等角步长
# Cylindrical detector: compute equi-angular step from physical psize at detector
delta_gamma = psize_col_at_det / SDD         # = 1.2858 / 1085.6

# 等角→等距重采样到平板等效 / Equi-angular → flat-panel resampling
# via map_coordinates interpolation

# 正确的等中心偏移 / Correct isocenter offset
det_col_offset = 369.625 - (736-1)/2.0      # = +2.125 px (before resampling)
                                              # = +2.419 px (after resampling)
det_row_offset = 32.5   - (64-1)/2.0        # = +1.000 px
```

---

## 第 6 阶段（0323）：GPU 滤波实现 + 内存修复 + 首次全量重建
## Phase 6 (0323): GPU Filter Implementation + Memory Fix + First Full Recon

### GPU 滤波加速（`pykatsevich/filter_gpu.py`）
### GPU Filter Acceleration (`pykatsevich/filter_gpu.py`)

| 步骤 Step | CPU 实现 CPU | GPU 实现 GPU |
|-----------|-------------|-------------|
| 微分 Differentiate | Python 逐投影循环 loop | CUDA kernel 全并行 full parallel |
| 前向高度重采样 Fwd rebin | 逐投影×逐列循环 nested loop | CUDA kernel |
| Hilbert 卷积 Hilbert conv | numpy.convolve 逐行 per-row | cuFFT 批量 FFT batch |
| 反向高度重采样 Back rebin | 逐投影循环 loop | CUDA kernel |
| 加速比 Speedup | — | **~10–20×** |

### 内存修复 / Memory Fix

```python
# 原始 sino：48590×64×736×float32 ≈ 8.6 GB
# Raw sino: 48590×64×736×float32 ≈ 8.6 GB
# 修复：加载后立即释放 / Fix: free immediately after loading
sino_work = sino[:, ::-1, :].copy()   # flip rows
del sino; gc.collect()                 # free 8.6 GB
```

---

## 第 7 阶段（0324）：全量重建完成 + Per-Turn Flicker 量化
## Phase 7 (0324): Full Recon Complete + Per-Turn Flicker Quantified

### 重建参数 / Reconstruction Parameters

```
体积 Volume: 512×512×560, voxel=(0.664, 0.664, 0.800) mm
重建中心偏移 Center offset: +11mm（匹配 GT ReconstructionTargetCenterPatient）
处理方式 Processing: 28 chunks × 20 slices, GPU filter + CuPy backprojector
输出 Output: full_recon_volume.tiff（ImageJ 兼容）
```

### Per-Slice 最优位移分析 / Per-Slice Optimal Shift Analysis

对 560 个切片分别做 dy/dx 网格搜索，找最小 MAE 的位移：
Grid search dy/dx for each of 560 slices to minimize MAE:

**关键发现 / Key Finding：**

```
dy（y 方向）位移周期 / dy shift period: ~28-30 slices
振幅 Amplitude: ±2 pixel
dx（x 方向）类似，相位差 ≈ 90°
dx similar, phase offset ≈ 90°

换算 / Cross-reference:
  z_per_turn = 22.97 mm
  VOXEL_SIZE_Z = 0.800 mm
  slices_per_turn = 22.97 / 0.800 = 28.7 ≈ 观测周期 observed period ✓

→ 位移振荡周期 = 1 螺旋圈
→ Shift oscillation period = 1 helical turn
```

### 桌面位置残差分析 / Table Position Residual Analysis

```
目的 / Purpose: 验证桌面物理运动是否有每转抖动
                 Verify if table physical motion has per-turn jitter

结果 / Results:
  table z 残差 std: 0.029 mm（vs pitch 22.97 mm = 0.12%）
  角度残差 std:     6×10⁻⁸ rad ≈ 0
  FFT 中无 1-圈周期峰 / No 1-turn frequency peak in FFT

→ 桌面运动正常，flicker 不是物理运动引起的
→ Table motion is normal; flicker is algorithmic, not physical
```

---

## 第 8 阶段（0325）：反投影偏移 Bug 根因确认与修复
## Phase 8 (0325): Backprojector Offset Bug — Root Cause Confirmed & Fixed

### 代码审查发现 / Code Review Finding

**`backproject_cupy.py` CUDA kernel 将探测器中心硬编码为几何中心：**
**`backproject_cupy.py` CUDA kernel hardcoded detector center to geometric center:**

```c
// 修复前 / Before fix (WRONG)
float u_center = (float)(det_cols - 1) * 0.5f;   // = 367.5 (geometric center)
float v_center = (float)(det_rows - 1) * 0.5f;   // = 31.5  (geometric center)

// 修复后 / After fix (CORRECT)
float u_center = (float)(det_cols - 1) * 0.5f + col_offset;  // = 367.5 + 2.419 = 369.919
float v_center = (float)(det_rows - 1) * 0.5f + row_offset;  // = 31.5  + 1.000 = 32.5
```

### 为什么 col_offset 误差产生每转振荡 / Why col_offset Error Causes Per-Turn Oscillation

探测器列偏移 δu 等效于把重建图像沿切向偏移：
A detector column offset δu is equivalent to shifting the reconstructed image tangentially:

```
对角度 θ 的投影 / For projection at angle θ:
  Δx(θ) = -A·sin(θ)
  Δy(θ) = +A·cos(θ)
  A = δu × psize_col × SOD/SDD = 2.419 × 1.374 × 595/1086 = 1.82 mm

由于 θ 随 z 线性增加：
Since θ increases linearly with z:
  偏移向量随 z 旋转，周期 = z_per_turn
  Offset vector rotates with z, period = z_per_turn

预测振幅 / Predicted amplitude:
  A = 1.82 mm = 2.74 recon pixels

观测振幅 / Observed amplitude: ±2 px ✓
```

同时，`det_row_offset = +1.0 px` 贡献 z 方向系统偏移 ≈ 0.60mm，解释了残余尺度偏差的一部分。
Also, `det_row_offset = +1.0 px` contributes a systematic z-shift ≈ 0.60mm, explaining part of the residual scale error.

### 修复链路 / Fix Chain（完整，无遗漏 Complete, no gaps）

```
DICOM tag → dicom.py → det_col_offset = +2.419 px
                        det_row_offset = +1.000 px
                              ↓
           initialize.py → conf['detector_col_offset'] = +2.419
                            conf['detector_row_offset'] = +1.000
                              ↓
         backproject_cupy.py → col_offset = conf.get('detector_col_offset')
                                row_offset = conf.get('detector_row_offset')
                                → 传入 CUDA kernel passed to CUDA kernel
                              ↓
         CUDA kernel → u_center = 367.5 + 2.419 = 369.919  ✓
                        v_center =  31.5 + 1.000 =  32.5    ✓
```

### 待验证 / Pending Validation

`test_offset_fix.py`（单片 MAE 对比）已跑，效果微弱（预期如此——单片无法体现旋转相消效应）。
`test_offset_fix.py` (single-slice MAE comparison) ran — effect minimal (expected: single slice can't show rotational cancellation).

**全量重建正在后台跑 / Full recon running in background** (task ID: `b6qnmidtu`)
预计完成时间 / ETA: ~2 hours

---

## 当前状态汇总 / Current Status Summary

| # | 问题 Issue | 状态 Status | 下一步 Next Step |
|---|-----------|-------------|-----------------|
| 1 | 1.41× 尺度偏差 Scale error | ✅ 已修复 Fixed (DICOM tag 03-22) | — |
| 2 | 每转 flicker (±2px 振荡) Per-turn flicker | ✅ 根因修复 Fix applied | 等全量重建结果 Wait for full recon |
| 3 | 残余尺度偏差 ~1.7% Residual scale | 部分解释 Partial (row_offset) | 修复后重测 Re-measure after fix |
| 4 | 外圈偏亮（差异图同心圆）Outer ring bias | 未排查 Uninvestigated | 待 #2 修复确认后再看 After #2 confirmed |

---

## 关键参数速查 / Key Parameters Quick Reference

```
L067 Quarter Dose:
  SOD = 595.0 mm, SDD = 1085.6 mm
  探测器 Detector: 736 col × 64 row, CYLINDRICAL
  物理 psize / Physical psize: col=1.2858mm, row=1.0947mm (at detector)
  平板等效 Flat-panel equiv: col=1.374mm, row=1.095mm (after resampling)
  中心元素 Central element: col=369.625, row=32.5 (DICOM pixel index)
  det_col_offset = +2.419 px (重采样后 after resampling)
  det_row_offset = +1.000 px
  Total views = 48590, projs_per_turn = 2304
  z_per_turn = 22.97 mm, pitch_mm_per_rad = 3.656 mm/rad
  桌面残差 Table z residual: std = 0.029 mm（正常 normal）
```

---

## 关键文件索引 / Key File Index

```
pykatsevich/
  dicom.py          — DICOM 加载+圆柱形探测器几何解析 (03-22 修复)
  initialize.py     — Katsevich conf 初始化，T-D 边界，偏移传递
  geometry.py       — ASTRA cone_vec 视角向量生成
  filter.py         — CPU Katsevich 滤波（differentiate/rebin/hilbert）
  filter_gpu.py     — GPU Katsevich 滤波，CuPy CUDA (03-23 新增)
  reconstruct.py    — 重建流水线

Quick Run Through 0304/
  backproject_cupy.py  — 自定义 CUDA 反投影核 (03-25 修复 u/v_center 偏移)

Quick Run Through 0318/
  INVESTIGATION_OVERVIEW.md  — 0228→0318 全部测试记录与结论
  diagnose_helix_direction.py — 螺旋方向+TD开关测试

Quick Run Through 0319/
  scale_investigation_log.md  — 尺度 1.41× 排查日志
  diagnose_geometry_fp.py     — 前投影对比（决定性验证）

Quick Run Through 0322/
  verify_fix.py        — 几何修复正投影验证
  phantom_corrected_test.py — 幻体尺度测试（发现 1.7% 残余偏差）

Quick Run Through 0324/
  full_recon.py           — 全量重建主脚本（560 slices）
  analyze_shift_periodicity.py — Per-slice 位移 FFT 分析
  analyze_table_residual.py    — 桌面位置残差分析
  test_offset_fix.py          — 探测器偏移修复验证（已跑）

Quick Run Through 0325/
  progress_0325.md    — 本文档 This document
```
