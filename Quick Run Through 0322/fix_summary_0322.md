# DICOM 几何标签误读修复总结 — 2026-03-22

## 问题

Katsevich 重建 AAPM L067 DICOM 数据时，重建体比 Ground Truth **小 ~1.41 倍**。

## 根因

`pykatsevich/dicom.py` 第 49 行将私有标签 `(0x7031, 0x1033)` 误读为**探测器物理尺寸 (mm)**，
实际上该标签是 **DetectorCentralElement**——中心像素的**索引** (column, row)。

```
标签值: (369.625, 32.5)
旧代码理解: 探测器弧长 = 369.625 mm, z 方向覆盖 = 32.5 mm
实际含义:   中心像素列号 = 369.625, 中心像素行号 = 32.5
```

来源: DICOM-CT-PD User Manual v12, Table 1

## 错误的几何计算链

```
arc_psize_iso = 369.625 / 736 = 0.5022 mm    ← 像素索引当成了 mm
delta_gamma   = 0.5022 / 595  = 8.44e-4 rad  ← 偏小 1.4 倍
flat_psize    = 0.947 mm
DCD           = 363.7 mm                      ← 远小于 GT 标签 500 mm
```

## 正确的几何参数 (来自 DICOM-CT-PD 标签)

| 标签 | 名称 | 值 | 用途 |
|------|------|-----|------|
| `(7029,1002)` | DetectorElementTransverseSpacing | **1.2858 mm** | 像素宽度 (at detector) |
| `(7029,1006)` | DetectorElementAxialSpacing | **1.0947 mm** | 像素高度 (at detector) |
| `(7029,100B)` | DetectorShape | **CYLINDRICAL** | 探测器形状 |
| `(7031,1033)` | DetectorCentralElement | **(369.625, 32.5)** | 中心像素索引 |
| `(7031,1003)` | DetectorFocalCenterRadialDistance | **595.0 mm** | SOD |
| `(7031,1031)` | ConstantRadialDistance | **1085.6 mm** | SDD |

正确计算:

```
delta_gamma = psize_col_at_det / SDD = 1.2858 / 1085.6 = 1.184e-3 rad
flat_psize  = 1.374 mm
DCD         = 502.4 mm  ≈ GT DataCollectionDiameter = 500 mm ✓
```

## 修改内容 (`pykatsevich/dicom.py`)

### 1. 标签常量替换

```python
# 旧 (错误)
DETECTOR_EXTENTS_TAG = (0x7031, 0x1033)

# 新 (正确)
DETECTOR_CENTRAL_ELEMENT_TAG = (0x7031, 0x1033)  # 中心像素索引
DETECTOR_COL_SPACING_TAG = (0x7029, 0x1002)       # 像素宽度 (mm, at detector)
DETECTOR_ROW_SPACING_TAG = (0x7029, 0x1006)       # 像素高度 (mm, at detector)
DETECTOR_SHAPE_TAG = (0x7029, 0x100B)              # CYLINDRICAL / FLAT
```

### 2. 几何计算重写

旧逻辑:
- 把 `(7031,1033)` 的值当作探测器尺寸
- `arc_psize = extent / n_pixels`, `delta_gamma = arc_psize / SOD`

新逻辑:
- 从 `(7029,1002)` 读取真实像素尺寸 (at detector)
- 从 `(7029,100B)` 读取探测器形状
- 从 `(7031,1033)` 读取中心像素索引
- 对 CYLINDRICAL: `delta_gamma = psize_at_det / SDD`
- 使用实际中心索引 369.625 (而非假设的 367.5)

### 3. Pitch fallback 修复

旧代码在 pitch 回退计算中用 `det_extents[1] = 32.5` 作为 z-collimation (mm)，
但 32.5 是行号索引。修正为:
```python
collimation_mm = n_rows * psize_row_at_det * SOD / SDD
```

### 4. SOD/SDD 读取顺序调整

将 SOD/SDD 读取移到 pitch fallback 之前，避免引用未定义变量。

## 验证结果

| 指标 | 旧 (错误) | 新 (修正) |
|------|----------|----------|
| delta_gamma | 8.44e-4 rad | 1.184e-3 rad |
| DCD | 363.7 mm | 502.4 mm (GT=500) |
| 投影推算体径 | 242.9 mm (误差 -25.5%) | 338.5 mm (误差 +3.8%) |
| FP/DICOM 宽度比 | 0.778 | 1.111 |

## 待验证

- 用修正后的几何跑完整 Katsevich 重建，确认 optimal scale ≈ 1.0
