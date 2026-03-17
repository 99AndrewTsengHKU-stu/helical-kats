# Helical CT重建 - Katsevich算法

## 项目概述

本项目使用Katsevich算法重建螺旋锥束CT数据。已实现自动几何校正功能(Auto-Focus)来检测最佳重建参数。

## 当前状态

✅ **已完成**
- 修复了PixelSpacing bug
- 实现了Auto-Focus自动参数选择
- 测试了7种几何配置
- 生成了完整的560层重建结果

⚠️ **问题**
- 所有配置产生的"旋涡"伪影持续存在
- 清晰度分数普遍较低（0.0001-0.0003）
- 表明问题可能不是简单的几何错误

## 快速开始

### 基本重建
```bash
conda activate MNIST
python tests/run_dicom_recon.py \
  --dicom-dir "path/to/dicom" \
  --rows 512 --cols 512 --slices 560 \
  --voxel-size 0.5 \
  --save-tiff output.tiff
```

### 使用Auto-Focus
```bash
python tests/run_dicom_recon.py ... --auto-focus
```

### 快速参数测试
```bash
python quick_test.py  # 自动测试多种参数组合
```

## 工具

| 工具 | 用途 |
|------|------|
| `compare_with_gt.py` | 分析DICOM几何参数 |
| `measure_sharpness.py` | 评估重建质量 |
| `save_as_png.py` | NPY转PNG可视化 |
| `quick_test.py` | 批量测试参数组合 |

## 文档

- **USAGE_GUIDE.md** - 详细使用说明
- **final_summary.md** - 完整问题分析
- **verification_report.md** - 测试结果对比
- **walkthrough.md** - 开发记录

## 测试结果

| 配置 | 清晰度 | 说明 |
|------|--------|------|
| Normal | 0.0003 | 🏆 最高分数 |
| Negate Angles | 0.0003 | CW↔CCW |
| Flip Columns | 0.00015 | U轴翻转 |
| Flip Rows | 0.00015 | V轴翻转 |
| Pitch Signed | 0.00016 | 有符号pitch |

## 关键发现

从几何分析(`compare_with_gt.py`)：
- **角度步长**: -0.010908 rad (逆时针)
- **Pitch**: -3.67 mm/rad (负值)
- **SOD**: 595.0 mm
- **SDD**: 1085.6 mm

## 可能的问题来源

1. 🔴 **算法不匹配** - GT可能使用FDK而非Katsevich
2. 🔴 **参数差异** - SOD/SDD/Pitch与GT不一致
3. 🟡 **缺少COR校正** - 旋转中心偏移未校正
4. 🟡 **数据预处理** - Log校正/散射校正差异

## 下一步建议

### 立即可做
1. ✅ 运行 `python compare_with_gt.py` 查看参数
2. ✅ 打开 `rec_c001_final.tiff` 与GT对比
3. ✅ 查看 `verification_report.md` 的视觉对比

### 需要信息
- [ ] Ground Truth使用的重建算法？
- [ ] GT的确切几何参数（SOD/SDD/pitch）？
- [ ] GT是否应用了COR校正？
- [ ] GT的预处理步骤？

### 可以尝试
- [ ] 调整voxel-size (0.4, 0.6等)
- [ ] 运行 `quick_test.py` 批量测试
- [ ] 如果有GT参数，手动设置对比

## 已修改的文件

### 核心代码
- `pykatsevich/dicom.py` - PixelSpacing修复
- `tests/run_dicom_recon.py` - Auto-Focus实现

### 新增工具
- `compare_with_gt.py`
- `measure_sharpness.py`
- `save_as_png.py`
- `quick_test.py`

## 生成的重建结果

- `rec_c001_final.tiff` - 完整560层（Normal模式）
- `rec_c001_final.npy` - NumPy格式
- `rec_c001_auto.tiff` - Auto-Focus结果
- 多个测试切片PNG文件

---

**更新时间**: 2026-01-03  
**版本**: 1.0 with Auto-Focus
