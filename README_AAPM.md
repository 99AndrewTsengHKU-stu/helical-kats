# AAPM真实CT数据重建指南

## 数据说明

### 投影数据（原始扫描）
**路径**: `D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD_2000`
- 这是真实CT扫描仪的原始投影数据
- 应该包含多个角度的投影视图

### Ground Truth（参考重建）
**路径**: `D:\AAPM-Data\L067\L067\full_1mm`
- 这是标准重建结果（可能是FBP或商业软件重建）
- 用作对比参考

## 快速开始

### 1. 分析数据结构
```bash
python analyze_aapm_data.py
```

这将显示：
- 投影数据的格式和参数
- GT数据的尺寸和重建参数
- 建议的重建命令

### 2. 运行重建

#### 使用Auto-Focus（推荐）
```bash
python tests/run_dicom_recon.py \
  --dicom-dir "D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD_2000" \
  --rows 512 --cols 512 --slices 300 \
  --voxel-size 1.0 \
  --save-tiff rec_L067_real.tiff \
  --auto-focus
```

#### 指定参数重建
```bash
# 根据analyze_aapm_data.py的输出调整参数
python tests/run_dicom_recon.py \
  --dicom-dir "D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD_2000" \
  --rows <GT_rows> --cols <GT_cols> --slices <GT_slices> \
  --voxel-size <GT_pixel_spacing> \
  --save-tiff rec_L067_real.tiff
```

### 3. 对比结果

将重建结果与GT对比：
```bash
# 转换为PNG便于查看
python save_as_png.py rec_L067_real.npy

# 在ImageJ或其他DICOM查看器中
# 并排对比 rec_L067_real.tiff 和 full_1mm 目录中的切片
```

## 重要区别：真实数据 vs 仿真数据

### 仿真数据（之前的C001）
- 路径: `D:\1212_High_Pitch_argparse\C001\...`
- 特点: 模拟生成，参数可能理想化
- 可能缺少真实扫描的噪声和伪影

### 真实数据（L067）
- 路径: `D:\AAPM-Data\L067\...`
- 特点: 真实CT扫描仪采集
- 包含真实的噪声、散射、beam hardening等
- **更可能需要预处理和校正**

## 常见问题

### Q: AAPM数据的投影格式是什么？
A: 运行 `analyze_aapm_data.py` 查看。通常可能是：
- 标准DICOM CT投影
- 或已经预处理过的sinogram数据

### Q: 如何知道用什么参数？
A: 
1. 先运行 `analyze_aapm_data.py`
2. 查看GT的 PixelSpacing, SliceThickness
3. 使用相同的参数重建

### Q: 如果格式不兼容怎么办？
A: AAPM数据可能需要专门的加载器。如果当前的 `load_dicom_projections` 不工作，可能需要：
1. 检查DICOM标签是否不同
2. 修改加载代码适配AAPM格式
3. 或使用AAPM提供的工具预处理

### Q: quarter vs full是什么意思？
A: 可能指：
- quarter: 1/4剂量扫描（低剂量）
- full: 正常剂量扫描
- 或 quarter: 1/4视图数（下采样）

## 下一步

1. ✅ 运行 `python analyze_aapm_data.py`
2. ✅ 根据输出调整重建参数
3. ✅ 运行重建
4. ✅ 与GT对比
5. ✅ 如果格式不兼容，报告具体错误信息

## 参考

- AAPM Low-Dose CT Grand Challenge: https://www.aapm.org/GrandChallenge/LowDoseCT/
- 数据格式可能需要参考AAPM官方文档

---
*更新时间*: 2026-01-03  
*针对真实AAPM数据*
