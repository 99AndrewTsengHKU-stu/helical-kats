"""
检查两个目录的完整 DICOM 标签，寻找 table position / z 相关信息。
1. D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD_2000  (投影数据)
2. D:\AAPM-Data\L067\L067\full_1mm                  (GT 重建图像)
"""

import sys
from pathlib import Path
import pydicom
import struct
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent

def print_all_tags(ds, out_lines, max_private=True):
    """打印所有标签，特别标记含 position / table / z / pitch / location 的"""
    keywords = ["position", "table", "location", "pitch", "slice", "z", "height", "offset", "travel"]
    for elem in ds:
        tag_str = f"({elem.tag.group:04X},{elem.tag.element:04X})"
        try:
            name = elem.keyword or elem.name or "?"
        except:
            name = "?"
        try:
            val = repr(elem.value)[:120]
        except:
            val = "<unreadable>"
        
        line = f"  {tag_str} {name}: {val}"
        is_interesting = any(k in name.lower() for k in keywords) or any(k in val.lower() for k in keywords)
        if is_interesting:
            line = "⭐ " + line
        out_lines.append(line)

lines = []
def pr(s=""):
    print(s)
    lines.append(s)

# ---- 1. 投影 DICOM (quarter_2000) ----
proj_dir = Path(r"D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD_2000")
if proj_dir.exists():
    proj_paths = sorted(p for p in proj_dir.iterdir() if p.suffix.lower() == ".dcm")
    pr("=" * 70)
    pr(f"投影 DICOM: {proj_paths[0].name}")
    pr("=" * 70)
    ds = pydicom.dcmread(str(proj_paths[0]))
    out = []
    print_all_tags(ds, out)
    interesting = [l for l in out if l.startswith("⭐")]
    all_others = [l for l in out if not l.startswith("⭐")]
    pr("\n--- ⭐ 有趣的标签 ---")
    for l in interesting:
        pr(l)
    pr(f"\n--- 全部标签（共 {len(out)} 个）---")
    for l in out:
        pr(l)
else:
    pr(f"[ERROR] 找不到投影目录: {proj_dir}")

# ---- 2. GT DICOM (full_1mm) ----
gt_dir = Path(r"D:\AAPM-Data\L067\L067\full_1mm")
if gt_dir.exists():
    gt_paths = sorted(p for p in gt_dir.iterdir())
    pr("\n" + "=" * 70)
    pr(f"GT DICOM (full_1mm): {gt_paths[0].name}")
    pr("=" * 70)
    try:
        ds_gt = pydicom.dcmread(str(gt_paths[0]))
        out_gt = []
        print_all_tags(ds_gt, out_gt)
        interesting_gt = [l for l in out_gt if l.startswith("⭐")]
        pr("\n--- ⭐ 有趣的标签 ---")
        for l in interesting_gt:
            pr(l)
        pr(f"\n--- 全部标签（共 {len(out_gt)} 个）---")
        for l in out_gt:
            pr(l)
    except Exception as e:
        pr(f"[ERROR] 读取失败: {e}")
else:
    pr(f"[ERROR] 找不到 GT 目录: {gt_dir}")

# Save
with open(str(OUT_DIR / "all_dicom_tags.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\n[Done] 保存至: {OUT_DIR / 'all_dicom_tags.txt'}")
