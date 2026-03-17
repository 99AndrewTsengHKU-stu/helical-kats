"""
AAPM真实CT数据分析脚本

分析真实投影数据和Ground Truth的特征
"""

import os
import pydicom
import numpy as np
from pathlib import Path

def analyze_projection_data(proj_dir):
    """分析投影数据"""
    print("="*60)
    print("投影数据分析")
    print("="*60)
    print(f"目录: {proj_dir}\n")
    
    if not os.path.exists(proj_dir):
        print(f"❌ 目录不存在: {proj_dir}")
        return None
    
    # 查找DICOM文件
    dcm_files = list(Path(proj_dir).rglob("*.dcm"))
    if not dcm_files:
        dcm_files = list(Path(proj_dir).rglob("*.DCM"))
    
    print(f"找到 {len(dcm_files)} 个DICOM文件")
    
    if len(dcm_files) == 0:
        # 可能是裸文件或其他格式
        all_files = list(Path(proj_dir).glob("*"))
        print(f"目录包含 {len(all_files)} 个文件")
        if all_files:
            print("前10个文件:")
            for f in all_files[:10]:
                print(f"  {f.name} ({f.stat().st_size} bytes)")
        return None
    
    # 读取第一个文件
    first_file = dcm_files[0]
    print(f"\n读取第一个文件: {first_file.name}")
    
    try:
        ds = pydicom.dcmread(first_file)
        
        print(f"\n基本信息:")
        print(f"  Modality: {getattr(ds, 'Modality', 'N/A')}")
        print(f"  Patient ID: {getattr(ds, 'PatientID', 'N/A')}")
        
        # 检查是否是投影数据还是重建数据
        if hasattr(ds, 'ImageType'):
            print(f"  Image Type: {ds.ImageType}")
        
        # 图像尺寸
        if hasattr(ds, 'Rows') and hasattr(ds, 'Columns'):
            print(f"\n图像尺寸:")
            print(f"  Rows: {ds.Rows}")
            print(f"  Columns: {ds.Columns}")
        
        # 检查私有标签（投影数据的角度信息）
        print(f"\n私有标签检查:")
        angle_tag = (0x7031, 0x1001)
        if angle_tag in ds:
            print(f"  ✅ 找到角度标签 {angle_tag}")
        else:
            print(f"  ❌ 未找到角度标签 {angle_tag}")
        
        # 几何参数
        print(f"\n扫描几何:")
        sod = ds.get((0x0018, 0x9402))
        sdd = ds.get((0x0018, 0x1110))
        if sod:
            print(f"  SOD: {sod}")
        if sdd:
            print(f"  SDD: {sdd}")
        
        return ds
        
    except Exception as e:
        print(f"❌ 读取DICOM失败: {e}")
        return None

def analyze_gt_data(gt_dir):
    """分析Ground Truth数据"""
    print("\n" + "="*60)
    print("Ground Truth数据分析")
    print("="*60)
    print(f"目录: {gt_dir}\n")
    
    if not os.path.exists(gt_dir):
        print(f"❌ 目录不存在: {gt_dir}")
        return None
    
    # 查找DICOM文件
    dcm_files = list(Path(gt_dir).rglob("*.dcm"))
    if not dcm_files:
        dcm_files = list(Path(gt_dir).rglob("*.DCM"))
    
    print(f"找到 {len(dcm_files)} 个DICOM文件（重建切片）")
    
    if len(dcm_files) == 0:
        all_files = list(Path(gt_dir).glob("*"))
        print(f"目录包含 {len(all_files)} 个文件")
        return None
    
    # 读取第一个和最后一个文件
    first_file = sorted(dcm_files)[0]
    print(f"\n读取第一个切片: {first_file.name}")
    
    try:
        ds = pydicom.dcmread(first_file)
        
        print(f"\n基本信息:")
        print(f"  Modality: {getattr(ds, 'Modality', 'N/A')}")
        
        # 图像尺寸
        if hasattr(ds, 'Rows') and hasattr(ds, 'Columns'):
            print(f"\n图像尺寸:")
            print(f"  Rows: {ds.Rows}")
            print(f"  Columns: {ds.Columns}")
        
        # 像素间距和层厚
        if hasattr(ds, 'PixelSpacing'):
            print(f"  Pixel Spacing: {ds.PixelSpacing} mm")
        
        if hasattr(ds, 'SliceThickness'):
            print(f"  Slice Thickness: {ds.SliceThickness} mm")
        
        # 重建参数
        print(f"\n重建参数:")
        if hasattr(ds, 'ConvolutionKernel'):
            print(f"  Convolution Kernel: {ds.ConvolutionKernel}")
        if hasattr(ds, 'ReconstructionDiameter'):
            print(f"  Reconstruction Diameter: {ds.ReconstructionDiameter} mm")
        
        # 窗宽窗位
        if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
            print(f"\n显示窗口:")
            print(f"  Window Center: {ds.WindowCenter}")
            print(f"  Window Width: {ds.WindowWidth}")
        
        return ds
        
    except Exception as e:
        print(f"❌ 读取DICOM失败: {e}")
        return None

def main():
    print("\n" + "="*60)
    print("AAPM真实CT数据分析工具")
    print("="*60)
    
    # 数据路径
    proj_dir = r"D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD_2000"
    gt_dir = r"D:\AAPM-Data\L067\L067\full_1mm"
    
    # 分析投影数据
    proj_ds = analyze_projection_data(proj_dir)
    
    # 分析GT数据
    gt_ds = analyze_gt_data(gt_dir)
    
    # 对比建议
    print("\n" + "="*60)
    print("重建建议")
    print("="*60)
    
    if proj_ds and gt_ds:
        print("\n✅ 找到投影数据和Ground Truth")
        print("\n建议的重建命令:")
        
        if hasattr(gt_ds, 'Rows'):
            rows = cols = gt_ds.Rows
            print(f"\n# 使用GT相同的分辨率 ({rows}×{cols})")
        else:
            rows = cols = 512
            print(f"\n# 使用默认分辨率 ({rows}×{cols})")
        
        if hasattr(gt_ds, 'PixelSpacing'):
            voxel_size = float(gt_ds.PixelSpacing[0])
        else:
            voxel_size = 1.0
        
        print(f"""
conda activate MNIST
cd D:\\Github\\helical-kats

python tests/run_dicom_recon.py \\
  --dicom-dir "{proj_dir}" \\
  --rows {rows} --cols {cols} --slices 300 \\
  --voxel-size {voxel_size} \\
  --save-npy rec_L067_real.npy \\
  --save-tiff rec_L067_real.tiff \\
  --auto-focus
""")
        
        print("注意事项:")
        print("1. slices数量需要根据实际扫描范围调整")
        print("2. 如果投影数据格式与之前不同，可能需要修改加载代码")
        print("3. 重建后与 full_1mm 目录中的GT切片对比")
        
    else:
        print("\n⚠️  数据分析未完全成功")
        print("请检查:")
        print("1. 路径是否正确")
        print("2. 是否有权限访问这些目录")
        print("3. DICOM文件格式是否正确")

if __name__ == "__main__":
    main()
