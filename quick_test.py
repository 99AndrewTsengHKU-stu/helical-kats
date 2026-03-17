#!/usr/bin/env python
"""
快速测试脚本 - 尝试多种参数组合

使用方法:
    python quick_test.py

这将生成多个测试重建并报告清晰度分数，帮助您快速找到最佳配置。
"""

import subprocess
import os
import sys
import numpy as np

# 配置
DICOM_DIR = r"D:\1212_High_Pitch_argparse\C001\C001_bundle_nview_div4_pitch0.6\dcm_proj"
PYTHON_EXE = r"C:\Users\Andrew\anaconda3\envs\MNIST\python.exe"
SCRIPT = "tests/run_dicom_recon.py"

# 测试配置（单层快速测试）
TEST_CONFIGS = [
    ("normal", []),
    ("pitch_signed", ["--pitch-signed"]),
    ("voxel_04", ["--voxel-size", "0.4"]),
    ("voxel_06", ["--voxel-size", "0.6"]),
    ("flip_cols", ["--flip-cols"]),
    ("flip_rows", ["--flip-rows"]),
    ("negate", ["--negate-angles"]),
    ("flip_cols_negate", ["--flip-cols", "--negate-angles"]),
]

def measure_sharpness(npy_file):
    """测量NPY文件的清晰度"""
    try:
        img = np.load(npy_file)
        if img.ndim == 3:
            if img.shape[2] == 1:
                img = img[:, :, 0]
            elif img.shape[0] == 1:
                img = img[0]
            else:
                img = img[:, :, img.shape[2]//2]
        
        gy, gx = np.gradient(img)
        gnorm = np.sqrt(gx**2 + gy**2)
        return np.mean(gnorm)
    except Exception as e:
        print(f"  ⚠️  清晰度测量失败: {e}")
        return 0.0

def run_test(name, extra_args):
    """运行单个测试配置"""
    print(f"\n{'='*60}")
    print(f"测试配置: {name}")
    print(f"{'='*60}")
    
    output_file = f"test_quick_{name}.npy"
    
    cmd = [
        PYTHON_EXE, SCRIPT,
        "--dicom-dir", DICOM_DIR,
        "--rows", "512",
        "--cols", "512", 
        "--slices", "1",
        "--voxel-size", "0.5",
        "--save-npy", output_file
    ] + extra_args
    
    print(f"运行命令: {' '.join(cmd[-10:])}")  # 显示最后几个参数
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            if os.path.exists(output_file):
                score = measure_sharpness(output_file)
                print(f"✅ 完成! 清晰度分数: {score:.6f}")
                return score
            else:
                print(f"⚠️  重建完成但未找到输出文件")
                return 0.0
        else:
            print(f"❌ 重建失败")
            print(f"错误: {result.stderr[-200:]}")  # 显示最后200字符
            return 0.0
            
    except subprocess.TimeoutExpired:
        print(f"⏱️  超时（10分钟）")
        return 0.0
    except Exception as e:
        print(f"❌ 异常: {e}")
        return 0.0

def main():
    print("="*60)
    print("快速参数测试工具")
    print("="*60)
    print(f"DICOM目录: {DICOM_DIR}")
    print(f"Python: {PYTHON_EXE}")
    print(f"将测试 {len(TEST_CONFIGS)} 种配置...")
    
    if not os.path.exists(DICOM_DIR):
        print(f"\n❌ 错误: DICOM目录不存在: {DICOM_DIR}")
        print("请编辑脚本中的DICOM_DIR变量")
        return
    
    results = []
    
    for name, args in TEST_CONFIGS:
        score = run_test(name, args)
        results.append((name, score, args))
    
    # 排序并显示结果
    results.sort(key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*60)
    print("测试结果总结（按清晰度排序）")
    print("="*60)
    print(f"{'排名':<6} {'配置':<25} {'清晰度分数':<15} {'参数'}")
    print("-"*60)
    
    for rank, (name, score, args) in enumerate(results, 1):
        marker = "🏆" if rank == 1 else "  "
        args_str = " ".join(args) if args else "(默认)"
        print(f"{marker} {rank:<4} {name:<25} {score:<15.6f} {args_str}")
    
    print("\n" + "="*60)
    print("建议:")
    print("="*60)
    
    best_name, best_score, best_args = results[0]
    if best_score > 0.0005:
        print(f"✅ 最佳配置: {best_name} (分数: {best_score:.6f})")
        print(f"   使用参数: {' '.join(best_args) if best_args else '默认'}")
        print("\n   运行完整重建:")
        full_cmd = f"python {SCRIPT} --dicom-dir \"{DICOM_DIR}\" --rows 512 --cols 512 --slices 560 --voxel-size 0.5 --save-tiff final_best.tiff"
        if best_args:
            full_cmd += " " + " ".join(best_args)
        print(f"   {full_cmd}")
    else:
        print("⚠️  所有配置的分数都很低 (<0.0005)")
        print("   这表明问题可能不是简单的参数调整")
        print("\n   建议:")
        print("   1. 确认Ground Truth使用的重建算法")
        print("   2. 运行 compare_with_gt.py 对比几何参数")
        print("   3. 检查是否需要额外的校正步骤")
    
    print("\n生成的测试文件:")
    for name, _, _ in results:
        fname = f"test_quick_{name}.npy"
        if os.path.exists(fname):
            print(f"  - {fname}")

if __name__ == "__main__":
    main()
