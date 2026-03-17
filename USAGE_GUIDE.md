# Helical CT重建快速使用指南

## 快速开始

### 1. 基本重建（使用Auto-Focus）
```bash
conda activate MNIST
cd D:\Github\helical-kats

python tests/run_dicom_recon.py \
  --dicom-dir "D:\1212_High_Pitch_argparse\C001\C001_bundle_nview_div4_pitch0.6\dcm_proj" \
  --rows 512 --cols 512 --slices 560 \
  --voxel-size 0.5 \
  --save-npy output.npy \
  --save-tiff output.tiff \
  --auto-focus
```

### 2. 指定几何配置重建
```bash
# Normal模式（当前最佳）
python tests/run_dicom_recon.py --dicom-dir <path> --rows 512 --cols 512 --slices 560 --voxel-size 0.5 --save-tiff output.tiff

# 使用有符号pitch
python tests/run_dicom_recon.py ... --pitch-signed --save-tiff output_signed.tiff

# 翻转探测器列
python tests/run_dicom_recon.py ... --flip-cols --save-tiff output_flipcols.tiff

# 翻转探测器行
python tests/run_dicom_recon.py ... --flip-rows --save-tiff output_fliprows.tiff

# 角度取反
python tests/run_dicom_recon.py ... --negate-angles --save-tiff output_negate.tiff

# 组合使用
python tests/run_dicom_recon.py ... --flip-cols --negate-angles --save-tiff output_combo.tiff
```

## 诊断工具

### 查看几何参数
```bash
python compare_with_gt.py
```
输出包括：SOD, SDD, 探测器尺寸, 像素大小, 角度步长, Pitch值

### 测量图像清晰度
```bash
python measure_sharpness.py output.npy
```

### 转换NPY为PNG
```bash
python save_as_png.py output.npy
# 生成 output.png
```

## 可用的几何选项

| 选项 | 说明 |
|------|------|
| `--auto-focus` | 自动测试多种配置并选择最佳 |
| `--pitch-signed` | 使用有符号pitch（而非绝对值） |
| `--negate-angles` | 角度取反（CW↔CCW） |
| `--flip-cols` | 翻转探测器U轴（左右） |
| `--flip-rows` | 翻转探测器V轴（上下） |
| `--reverse-angle-order` | 反转投影顺序 |
| `--reverse-per-turn` | 每圈内部反转 |

## 测试结果总结

当前测试显示（单层清晰度分数）：
- ✅ Normal: 0.0003（最高）
- Negate Angles: 0.0003
- Flip Columns: 0.00015
- Flip Rows: 0.00015
- Flip+Negate: 0.00014
- Pitch Signed: 0.00016

## 下一步调试建议

### 1. 参数对比
运行 `python compare_with_gt.py` 并将输出与Ground Truth的参数对比：
- SOD (Source-Object Distance)
- SDD (Source-Detector Distance)  
- Pitch值和符号
- 探测器像素大小

### 2. 尝试不同体素大小
```bash
python tests/run_dicom_recon.py ... --voxel-size 0.4  # 更小
python tests/run_dicom_recon.py ... --voxel-size 0.6  # 更大
```

### 3. 视觉检查
- 打开 `rec_c001_final.tiff` 与Ground Truth对比
- 使用ImageJ/Fiji或其他DICOM查看器
- 关注伪影的形态：旋涡、条纹、模糊等

### 4. 确认算法
**最重要**：Ground Truth使用的是Katsevich还是FDK算法？
- Katsevich：精确的螺旋锥束重建
- FDK：近似算法，对pitch不敏感

## 生成的文件

### 重建结果
- `rec_c001_final.tiff` - 完整560层重建
- `rec_c001_final.npy` - NumPy格式
- `rec_c001_auto.tiff` - Auto-Focus选择的结果

### 测试文件（单层）
- `test_flip.npy/.png` - Flip Columns
- `test_flip_rows.npy/.png` - Flip Rows
- `test_flip_neg.npy/.png` - Flip+Negate
- `test_pitch_signed.npy/.png` - Pitch Signed

### 文档
- `final_summary.md` - 完整分析报告
- `verification_report.md` - 视觉对比报告
- `walkthrough.md` - 工作记录

## 常见问题

**Q: 为什么所有配置的清晰度分数都很低？**
A: 可能原因：
1. 算法不匹配（GT用FDK？）
2. 参数不正确
3. 数据质量问题
4. 需要额外的校正（如COR偏移）

**Q: Auto-Focus没有显著改善怎么办？**
A: 这表明问题不是简单的几何翻转。建议：
1. 确认GT使用的算法类型
2. 仔细对比所有几何参数
3. 检查是否需要预处理步骤

**Q: 如何知道重建是否正确？**
A: 
1. 视觉对比：解剖结构是否清晰可辨
2. 伪影减少：与GT对比旋涡/条纹是否减少
3. 清晰度：边缘是否锐利
4. HU值：软组织/骨骼的密度值是否合理

## 联系与支持

如果问题持续存在，建议：
1. 联系原始数据提供者确认扫描协议
2. 咨询Katsevich算法专家
3. 考虑使用商业重建软件作为对比

---
*生成于: 2026-01-03*
*版本: 1.0 with Auto-Focus*
