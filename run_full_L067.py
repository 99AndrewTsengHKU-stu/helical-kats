"""
完整AAPM数据集（48590投影）重建脚本
"""

import subprocess
import sys

def main():
    print("="*60)
    print("AAPM L067 完整数据集重建")
    print("="*60)
    
    # 数据路径
    full_proj_dir = r"D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD"
    
    print(f"\n投影数据: {full_proj_dir}")
    print(f"预期投影数: 48590")
    print(f"目标: 重建560层")
    
    # 重建参数
    params = {
        "dicom_dir": full_proj_dir,
        "rows": 512,
        "cols": 512,
        "slices": 560,
        "voxel_size": 0.664,
        "output_npy": "rec_L067_FULL_FIXED.npy",
        "output_tiff": "rec_L067_FULL_FIXED.tiff"
    }
    
    print(f"\n重建参数:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    # 构建命令
    python_exe = r"C:\Users\Andrew\anaconda3\envs\MNIST\python.exe"
    script = "tests/run_dicom_recon.py"
    
    cmd = [
        python_exe, script,
        "--dicom-dir", params["dicom_dir"],
        "--rows", str(params["rows"]),
        "--cols", str(params["cols"]),
        "--slices", str(params["slices"]),
        "--voxel-size", str(params["voxel_size"]),
        "--save-npy", params["output_npy"],
        "--save-tiff", params["output_tiff"],
        "--decimate", "10",
        "--windowing"
    ]
    
    print(f"\n运行命令:")
    print(" ".join(cmd))
    
    print(f"\n预计时间: 15-30分钟（取决于GPU性能）")
    print(f"输出文件:\n  - {params['output_npy']}\n  - {params['output_tiff']}")
    
    confirm = input("\n按Enter开始重建，或Ctrl+C取消: ")
    
    print("\n开始重建...")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n" + "="*60)
        print("✅ 重建完成!")
        print("="*60)
        print(f"\n生成的文件:")
        print(f"  ✓ {params['output_npy']}")
        print(f"  ✓ {params['output_tiff']}")
        print(f"\n下一步:")
        print(f"  1. 与GT对比: D:\\AAPM-Data\\L067\\L067\\full_1mm")
        print(f"  2. 测量清晰度: python measure_sharpness.py {params['output_npy']}")
        print(f"  3. 可视化: python save_as_png.py {params['output_npy']}")
    else:
        print("\n" + "="*60)
        print("❌ 重建失败")
        print("="*60)
        sys.exit(1)

if __name__ == "__main__":
    main()
