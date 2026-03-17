# Progress Report — 2026-03-13
# 项目进度报告 — 2026年3月13日

**Project / 项目**: helical-kats (Katsevich Helical CT Reconstruction)
**Dataset / 数据集**: AAPM L067 Quarter-Dose DICOM-CT-PD (48,590 projections)

---

## Phase 1: Initial Diagnostics / 初始诊断 (Feb 28)

**Directory / 目录**: `Quick Run Through 0228/`

### What was done / 完成的工作
- Loaded L067 quarter-dose DICOM series (2000 projections, 736×64 detector)
- Extracted geometry from DICOM private tags: SOD=595mm, SDD=1085.6mm
- Ran first single-slice and 10-slice Katsevich reconstructions
- Identified and fixed `initialize.py` fallback formula bug (`s_len = 2π` → actual angle range)

加载了 L067 quarter-dose DICOM 序列（2000 投影，736×64 探测器），从 DICOM 私有标签中提取几何参数（SOD=595mm, SDD=1085.6mm），完成了首次单层和10层 Katsevich 重建，修复了 `initialize.py` 中 `s_len` 回退公式的 bug。

### Key findings / 关键发现
- Reconstruction showed recognizable anatomy (lungs, heart, ribs)
- Fan-shaped streak artifacts present; body offset from center
- Value range too low: [-0.05, 0.17] vs expected HU scale

重建结果可辨识人体解剖结构，但存在扇形条状伪影、体位偏移、值域偏低等问题。

### Results / 结果图

First single-slice reconstruction vs ground truth / 首次单层重建与 GT 对比：

![rec_vs_gt](Quick%20Run%20Through%200228/rec_vs_gt.png)

10-slice reconstruction / 10层重建结果：

![rec_10slices](Quick%20Run%20Through%200228/rec_10slices_all.png)

Bug fix before/after comparison / Bug 修复前后对比：

![fix_comparison](Quick%20Run%20Through%200228/fix_comparison.png)

T-D smoothing comparison / T-D 平滑参数对比：

![td_smoothing](Quick%20Run%20Through%200228/td_smoothing_compare.png)

---

## Phase 2: Detector Pixel Size Fix / 探测器像素尺寸修正 (Mar 2)

**Directory / 目录**: `Quick Run Through 0302/`

### Critical discovery / 关键发现

DICOM private tag `(0x7031, 0x1033)` stores detector extents **at isocenter**, not at the physical detector plane. This caused the pixel size to be underestimated by a factor of SDD/SOD = 1.824.

DICOM 私有标签 `(0x7031, 0x1033)` 存储的是**等中心处**的探测器范围，不是物理探测器平面的。导致像素尺寸被低估了 SDD/SOD = 1.824 倍。

| Parameter / 参数 | Before / 修正前 | After / 修正后 |
|:---|:---:|:---:|
| Pixel size / 像素尺寸 | 0.502 mm (isocenter) | 0.916 mm (physical) |
| Detector height / 探测器高度 | 32.3 mm | 59.0 mm |
| Body width in image / 体位宽度 | ~60% FOV | ~75-80% FOV |

### Implementation / 实现

Rewrote `pykatsevich/dicom.py` — complete DICOM loader with:
- Private tag extraction for angles, detector extents, SOD/SDD
- **Magnification correction**: `pixel_size_physical = pixel_size_iso × SDD/SOD`
- Equiangular→flat detector resampling via bicubic interpolation
- Pitch extraction priority: table position tags → focal center tag → pitch factor fallback

重写了 `pykatsevich/dicom.py`，实现了完整的 DICOM 加载器：私有标签解析、放大倍率校正、等角→平面重采样、螺距优先级提取。

### Results / 结果图

Before vs after pixel size fix — auto window / 修正前后对比（自动窗）：

![compare_auto](Quick%20Run%20Through%200302/compare_auto.png)

After fix — soft tissue window / 修正后软组织窗：

![compare_fixed_soft_tissue](Quick%20Run%20Through%200302/compare_fixed_soft_tissue.png)

After fix — lung window / 修正后肺窗：

![compare_fixed_lung](Quick%20Run%20Through%200302/compare_fixed_lung.png)

---

## Phase 3: Full-Volume Reconstruction / 全体重建 (Mar 4–10)

**Directory / 目录**: `Quick Run Through 0304/`

### Chunked Z-reconstruction pipeline / 分块 Z 轴重建流水线

Built `run_L067.py` — end-to-end pipeline for full 512×512×560 volume:

构建了 `run_L067.py`，实现了完整的 512×512×560 体积重建流水线：

1. **DICOM loading**: 48,590 projections from full dataset
   **DICOM 加载**：从完整数据集读取 48,590 个投影

2. **Angle convention fix**: L067 angles decrease (clockwise); negate + flip columns for Katsevich
   **角度约定修正**：L067 角度递减（顺时针）；取反+翻转列以适配 Katsevich

3. **Z-culling**: Skip projections that cannot illuminate a given z-slab (cone geometry check)
   **Z 轴裁剪**：根据锥束几何跳过无法照射目标层的投影

4. **64-slice chunks**: Process independently, each with its own culled projection set
   **64 层分块**：独立处理每块，各自使用裁剪后的投影子集

5. **Post-rotation**: CCW 90° to correct coordinate system
   **后旋转**：逆时针90°校正坐标系

### GPU backprojection / GPU 反投影

Built `backproject_cupy.py` — CuPy CUDA kernel replacing ASTRA:
- Batch-based launches (128 projections/batch) to avoid Windows TDR timeout
- 16×16 thread blocks, fused per-voxel accumulation

构建了 `backproject_cupy.py`，用 CuPy CUDA 核函数替代 ASTRA 反投影。分批启动（每批128投影）避免 Windows TDR 超时。

### Result / 结果
- **Full 560-slice reconstruction completed** → `L067_rec_560.npy` (561 MB), `L067_rec_560.tiff`
- Reconstruction time: ~50 min (9 chunks × ~6 min each)
- Body anatomy clearly visible; voxel size 0.664mm matches ground truth FOV

**560 层全体重建完成**，重建时间约50分钟，解剖结构清晰可见，体素尺寸 0.664mm 与 GT FOV 匹配。

Autofocus parameter sweep / 自动聚焦参数扫描：

![autofocus](Quick%20Run%20Through%200304/L067_autofocus.png)

Final reconstruction result / 最终重建结果：

![L067_result](Quick%20Run%20Through%200304/L067_result.png)

CuPy vs ASTRA backprojection validation / CuPy 与 ASTRA 反投影验证：

![validate_cupy](Quick%20Run%20Through%200304/validate_cupy_bp.png)

---

## Phase 4: Artifact Investigation / 伪影调查 (Mar 10–13)

### Flicker analysis / 闪烁分析

**Script / 脚本**: `analyze_flicker.py`

Observed slice-to-slice intensity variations when scrolling through TIFF in viewer. Investigated root cause:

在 TIFF 浏览器中滚动时观察到层间亮度波动，进行了根因调查：

- Per-slice ROI mean extracted → computed inter-slice differences
- FFT analysis of flicker pattern → no dominant frequency (broadband)
- Flicker magnitude: ~0.07% relative variation

Flicker analysis / 闪烁分析：

![flicker_analysis](Quick%20Run%20Through%200304/flicker_analysis.png)

Flicker detail & FFT / 闪烁细节与频谱分析：

![flicker_detail](Quick%20Run%20Through%200304/flicker_detail.png)

![flicker_fft](Quick%20Run%20Through%200304/flicker_fft_detail.png)

### T-D smoothing study / 时域平滑研究

**Script / 脚本**: `test_td_smoothing.py`

Tested `sino_weight_td()` smoothing values: 0.025, 0.05, 0.1, 0.2

测试了不同 T-D 平滑参数值对闪烁的影响。

**Result / 结果**: Smoothing reduces flicker marginally but does not eliminate it. Best value: default (0.05).

平滑可轻微减少闪烁但无法消除，最佳值为默认 0.05。

T-D smoothing comparison / T-D 平滑参数对比：

![td_smoothing](Quick%20Run%20Through%200304/td_smoothing_compare.png)

### Overlap blending study / 重叠混合研究

**Script / 脚本**: `test_overlap.py`

Compared three chunk boundary strategies on 80-slice test region [240:320]:

在80层测试区域上比较了三种分块边界策略：

| Method / 方法 | Flicker | Boundary max diff |
|:---|:---:|:---:|
| A: Single chunk (no boundary) / 单块无边界 | 0.0000699 | 0.0000637 |
| B: Overlap + feather blend / 重叠+羽化混合 | 0.0000699 | 0.0000637 |
| C: Hard boundary / 硬边界 | 0.0000699 | 0.0000634 |

Overlap comparison (A: single chunk, B: overlap+feather, C: hard boundary) / 重叠对比：

![overlap_compare](Quick%20Run%20Through%200304/overlap_compare.png)

### Conclusion / 结论

**All three methods produce identical results.** Chunk boundaries do NOT cause flickering. The flicker is an intrinsic property of the Katsevich algorithm — different z-positions use different helical projection subsets, causing natural micro-variations in reconstructed intensity.

**三种方法结果完全相同。** 分块边界不是闪烁的原因。闪烁是 Katsevich 算法的固有特性——不同 z 位置使用不同的螺旋投影子集，导致重建亮度的自然微小波动。

Overlap code was removed from `run_L067.py` for simplicity.

已从 `run_L067.py` 中移除 overlap 代码。

---

## Current File Structure / 当前文件结构

```
helical-kats/
├── pykatsevich/
│   ├── __init__.py          # exports load_dicom_projections
│   ├── dicom.py             # DICOM loader (magnification fix, equiangular resampling)
│   ├── geometry.py          # astra_helical_views() for ASTRA cone_vec geometry
│   ├── filter.py            # Katsevich filter pipeline (diff, rebin, hilbert, T-D)
│   ├── initialize.py        # Configuration setup
│   └── pykat.ipynb          # Jupyter notebook demo
├── tests/
│   ├── run_dicom_recon.py   # CLI driver with autofocus
│   └── t03.py               # Test case 3
├── Quick Run Through 0228/  # Phase 1: initial diagnostics
├── Quick Run Through 0302/  # Phase 2: pixel size fix
├── Quick Run Through 0304/  # Phase 3-4: full recon + artifact study
│   ├── run_L067.py          # Main 560-slice reconstruction pipeline
│   ├── backproject_cupy.py  # GPU backprojection (CuPy CUDA)
│   ├── backproject_safe.py  # CPU backprojection (ASTRA)
│   ├── analyze_flicker.py   # Flicker pattern analysis
│   ├── test_td_smoothing.py # T-D smoothing parameter sweep
│   ├── test_overlap.py      # Overlap blending comparison
│   ├── L067_rec_560.npy     # Full reconstruction (561 MB)
│   └── L067_rec_560.tiff    # Full reconstruction (TIFF stack)
└── progress_0313.md         # This report
```

---

## Key Geometry Parameters / 关键几何参数

```
SOD:                595.0 mm
SDD:                1085.6 mm
Magnification:      1.824×

Detector:           736 cols × 64 rows (equiangular)
Pixel size:         0.916 mm (physical, after magnification correction)
Row height:         0.927 mm (physical)

Pitch:              22.97 mm/turn
Total projections:  48,590
Total turns:        21.09
Angular range:      132.4 rad (negated to increasing for Katsevich)

Reconstruction:     512 × 512 × 560, voxel = 0.664 mm
```

---

## Known Issues / 已知问题

1. **HU calibration offset / HU 标定偏移**: μ_water ≈ 0.026 mm⁻¹ (expected 0.019). Not yet calibrated to Hounsfield units.
   水的衰减系数偏高，尚未标定为 HU 单位。

2. **Streak artifacts at edges / 边缘条状伪影**: Visible near body boundary, typical of Katsevich with finite cone angle.
   体表附近可见条状伪影，为 Katsevich 有限锥角的典型表现。

3. **Curved detector formulas not yet integrated / 弯曲探测器公式尚未集成**: Current pipeline uses flat-detector Katsevich; curved-detector port exists (`curved_katsevich_gpu.py`) but not yet integrated into main pipeline.
   当前流水线使用平面探测器 Katsevich；弯曲探测器版本已移植但尚未集成。

---

## Next Steps / 下一步计划

1. Integrate curved-detector Katsevich formulas into main pipeline
   将弯曲探测器 Katsevich 公式集成到主流水线

2. HU calibration: convert μ values to Hounsfield units
   HU 标定：将衰减系数转换为 Hounsfield 单位

3. Quantitative comparison with ground truth (SSIM, RMSE, body dimension matching)
   与 Ground Truth 定量对比（SSIM、RMSE、体位尺寸匹配）

4. Merge validated code into main `pykatsevich` library
   将验证通过的代码合并入 `pykatsevich` 主库
