# AAPM L067真实数据重建 - 快速指南

## 数据情况

✅ **投影数据**: `D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD_2000`
- 2000个DICOM文件
- 分辨率: 736×64 (探测器行×列)
- 包含角度信息（私有标签）

✅ **Ground Truth**: `D:\AAPM-Data\L067\L067\full_1mm`
- 560个.IMA切片（DICOM格式）
- 这是标准的FBP重建结果（参考）

## 立即开始重建

### 1. 使用Auto-Focus（推荐）
```bash
conda activate MNIST
cd D:\Github\helical-kats

python tests/run_dicom_recon.py \
  --dicom-dir "D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD_2000" \
  --rows 512 --cols 512 --slices 560 \
  --voxel-size 1.0 \
  --save-npy rec_L067_auto.npy \
  --save-tiff rec_L067_auto.tiff \
  --auto-focus
```

### 2. 快速单层测试（验证格式）
```bash
# 先重建一层确认数据格式兼容
python tests/run_dicom_recon.py \
  --dicom-dir "D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD_2000" \
  --rows 512 --cols 512 --slices 1 \
  --voxel-size 1.0 \
  --save-npy test_L067.npy
```

### 3. 查看GT参数
```python
import pydicom
gt_file = r"D:\AAPM-Data\L067\L067\full_1mm\L067_FD_1_1.CT.0001.0001.2015.12.22.18.09.40.840353.358074219.IMA"
ds = pydicom.dcmread(gt_file)
print(f"GT尺寸: {ds.Rows} × {ds.Columns}")
print(f"像素间距: {ds.PixelSpacing}")
print(f"层厚: {ds.SliceThickness}")
```

## 对比结果

### 转换GT为可比较格式
```bash
# GT是.IMA格式，可以用pydicom读取
# 提取中间层对比
python -c "
import pydicom
import numpy as np
from PIL import Image

gt_file = r'D:\AAPM-Data\L067\L067\full_1mm\L067_FD_1_1.CT.0001.0280.*.IMA'
ds = pydicom.dcmread(gt_file)
img = ds.pixel_array
img_normalized = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
Image.fromarray(img_normalized).save('gt_middle_slice.png')
"
```

### 对比重建结果
```bash
# 转换重建结果为PNG
python save_as_png.py rec_L067_auto.npy  # 生成rec_L067_auto.png

# 并排对比
# gt_middle_slice.png vs rec_L067_auto.png
```

## 预期结果

由于这是**真实扫描数据**（不是仿真），可能会有：
- 真实的量子噪声
- 散射伪影
- Beam hardening效应
- 需要与GT的FBP重建对比

**关键指标**：
- 解剖结构是否清晰
- 噪声水平是否合理
- 伪影形态是否与GT相似

## 故障排查

### 如果加载失败
可能原因：
1. AAPM数据的DICOM标签与仿真数据不同
2. 需要检查私有标签的具体位置
3. 运行诊断：
```bash
python compare_with_gt.py  # 更新路径为AAPM数据
```

### 如果重建质量不佳
1. 检查SOD/SDD参数是否正确
2. 对比GT的重建参数
3. 考虑预处理需求（log校正等）

---

**下一步**: 运行单层测试，确认数据兼容性后再运行完整重建
