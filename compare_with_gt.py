"""
诊断脚本：比较重建参数与Ground Truth的差异

使用方法：
python compare_with_gt.py
"""

import pydicom
import struct
import numpy as np
import os

def decode_float32_tag(ds, tag, count=1):
    if tag not in ds:
        return None
    raw = bytes(ds[tag].value)
    needed = 4 * count
    if len(raw) < needed:
        return None
    values = struct.unpack("<" + "f" * count, raw[:needed])
    return values if count > 1 else values[0]

def analyze_geometry():
    """分析当前使用的几何参数"""
    dicom_dir = r"D:\1212_High_Pitch_argparse\C001\C001_bundle_nview_div4_pitch0.6\dcm_proj"
    
    if not os.path.exists(dicom_dir):
        print(f"错误：目录不存在 {dicom_dir}")
        return
    
    files = sorted([f for f in os.listdir(dicom_dir) if f.lower().endswith('.dcm')])
    if not files:
        print("未找到DICOM文件")
        return
    
    # 读取第一个文件获取几何参数
    first_file = os.path.join(dicom_dir, files[0])
    ds = pydicom.dcmread(first_file, stop_before_pixels=True)
    
    print("=" * 60)
    print("当前DICOM几何参数")
    print("=" * 60)
    
    # SOD/SDD
    sod_std = ds.get((0x0018, 0x9402))
    sdd_std = ds.get((0x0018, 0x1110))
    sod_priv = decode_float32_tag(ds, (0x7031, 0x1003))
    sdd_priv = decode_float32_tag(ds, (0x7031, 0x1031))
    
    print(f"SOD (标准): {sod_std}")
    print(f"SOD (私有): {sod_priv}")
    print(f"SDD (标准): {sdd_std}")
    print(f"SDD (私有): {sdd_priv}")
    
    # 探测器参数
    det_extents = decode_float32_tag(ds, (0x7031, 0x1033), count=2)
    print(f"\n探测器尺寸 (私有标签 0x7031,0x1033): {det_extents}")
    print(f"探测器行数 (Columns): {ds.Columns}")
    print(f"探测器列数 (Rows): {ds.Rows}")
    
    if det_extents:
        print(f"像素大小 U方向: {det_extents[0] / ds.Rows:.4f} mm")
        print(f"像素大小 V方向: {det_extents[1] / ds.Columns:.4f} mm")
    
    # Pitch参数
    pitch_factor = ds.get((0x0018, 0x9311))
    print(f"\nSpiral Pitch Factor: {pitch_factor}")
    
    # 读取多个文件获取角度信息
    print("\n=" * 60)
    print("角度和Table Position信息")
    print("=" * 60)
    
    angles = []
    table_pos = []
    for i, f in enumerate(files[:10]):  # 前10个文件
        ds = pydicom.dcmread(os.path.join(dicom_dir, f), stop_before_pixels=True)
        angle = decode_float32_tag(ds, (0x7031, 0x1001))
        tpos = float(ds[0x0018, 0x9327].value) if (0x0018, 0x9327) in ds else None
        if angle is not None:
            angles.append(angle)
        if tpos is not None:
            table_pos.append(tpos)
        if i < 5:  # 打印前5个
            print(f"文件 {i}: Angle={angle:.6f} rad, TablePos={tpos:.2f} mm" if tpos else f"文件 {i}: Angle={angle:.6f} rad")
    
    if len(angles) > 1:
        angle_step = np.mean(np.diff(angles))
        print(f"\n平均角度步长: {angle_step:.6f} rad ({np.degrees(angle_step):.4f} deg)")
    
    if len(table_pos) > 1:
        table_step = np.mean(np.diff(table_pos))
        print(f"平均Table步长: {table_step:.4f} mm")
        if len(angles) > 1:
            pitch_mm_per_rad = table_step / angle_step
            print(f"计算的Pitch (mm/rad): {pitch_mm_per_rad:.4f}")
    
    print("\n=" * 60)
    print("建议检查项")
    print("=" * 60)
    print("1. 将上述参数与Ground Truth重建使用的参数对比")
    print("2. 特别注意：")
    print("   - SOD/SDD是否完全一致")
    print("   - Pitch值和符号是否匹配")
    print("   - 探测器像素大小是否正确")
    print("3. 如果GT使用了不同的重建算法（如FDK），需要切换算法")
    print("4. 检查是否存在旋转中心(COR)偏移校正")

if __name__ == "__main__":
    analyze_geometry()
