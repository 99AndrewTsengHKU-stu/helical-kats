# 组会后排障计划 / Post-Meeting Debugging Plan
### 2026-03-25

已知已修复 / Already Fixed: 探测器中心偏移 detector center offset (+2.419 col, +1.000 row)

全量重建结果待出 / Full recon result pending: task b6qnmidtu

---

## 三个待验证假设 / Three Hypotheses to Investigate

| # | 导师原话 Advisor Quote | 物理含义 Physical Meaning | 优先级 Priority |
|---|----------------------|--------------------------|----------------|
| H2 | "quarter or half offset in angular direction" | 投影角度方向残余子像素偏移 Residual sub-pixel offset in fan direction | ⭐⭐⭐ 高 High |
| H3 | "one or two projections back-projected more than expected, weighting not correct" | 反投影幅度归一化不一致 BP amplitude normalization inconsistency | ⭐⭐ 中 Medium |
| H4 | "boundary projections contribute more than they should" | T-D 边界投影权重异常 T-D boundary projection overweighting | ⭐⭐ 中 Medium |

---

## H2：角度方向残余偏移
## H2: Angular Direction Residual Offset

### 物理含义 / Physical Meaning

导师所说的"angular direction offset"是指：在扇形束方向（即探测器列方向）上，X射线中心轴（过等中心的射线）投影到探测器上的位置，与代码中假设的位置之间的偏差。

The "angular direction offset" means: in the fan-beam direction (detector column direction), the position where the central ray (passing through the isocenter) hits the detector differs from what the code assumes by a sub-pixel amount.

```
已修复的部分 / Already fixed:
  DetectorCentralElement (7031,1033) → det_col_offset = +2.419 px

可能的残余 / Possible residual:
  扫描仪机械对准误差：探测器物理中心与等中心射线之间还差 ±0.5 px
  Scanner mechanical alignment error: detector physical center vs central ray ±0.5 px

  这在文献中有记录：GE/Siemens 螺旋 CT 通常有 0.25–0.5 像素的系统性几何误差
  Documented in literature: GE/Siemens helical CT typically has 0.25–0.5 px systematic error
```

### 症状预测 / Predicted Symptom

每半个像素的额外 col_offset → 等中心处的等效空间偏移：
Each 0.5px additional col_offset → equivalent spatial shift at isocenter:
```
Δ = 0.5 × psize_col × SOD/SDD = 0.5 × 1.374 × 595/1086 = 0.377 mm = 0.57 recon px
```

对于固定偏移，重建图像会整体横向漂移；对于 **随角度旋转** 的偏移残差，依然产生每转振荡。
A fixed offset shifts image laterally; **residual after partial correction** still produces per-turn oscillation.

### 测试方案 / Test Plan

**T2-A：条形模体角度偏移扫参（无需 GPU，可先做）**
**T2-A: Bar phantom angular offset sweep (CPU-only, do first)**

```python
# 原理 / Principle:
# 用圆柱幻体重建，对比不同额外 col_offset 下的 flicker 指标
# Reconstruct cylinder phantom, compare flicker metric for different extra col_offset values

# 扫参范围 / Sweep range:
extra_col_offsets = np.linspace(-1.0, 1.0, 21)  # -1.0 to +1.0 in steps of 0.1 px
# 对每个 offset，重建 3 片相隔 1 圈的切片
# For each offset, reconstruct 3 slices spaced ~1 turn apart
# 量化 flicker = std(mean_ROI_per_slice) / mean(mean_ROI_per_slice)
# Measure flicker = std(mean_ROI) / mean(mean_ROI)
```

**T2-B：L067 数据子像素偏移扫参（需 GPU，等全量重建结束后）**
**T2-B: L067 data sub-pixel offset sweep (GPU needed, after full recon)**

```python
# 在已有的 full_recon.py 基础上，增加 extra_col_offset 参数
# Build on full_recon.py, add extra_col_offset parameter
# 仅重建 5 片（等间距约 1 圈）
# Reconstruct only 5 slices (spaced ~1 turn)
extra_offsets = [-0.5, -0.25, 0.0, +0.25, +0.5]  # px, on top of existing 2.419
# 找使 flicker 最小的值
# Find value minimizing flicker
```

**确认判据 / Confirmation Criterion:**
- 若最优 extra_offset ≠ 0，则存在残余偏移 → fix 并全量重建
- If optimal extra_offset ≠ 0, residual offset exists → fix and rerun full recon
- 若最优 extra_offset = 0 且 flicker 不变，则角度偏移假设被排除
- If optimal = 0 and flicker unchanged, hypothesis H2 ruled out

---

## H3：反投影幅度归一化不一致
## H3: Backprojection Amplitude Normalization Inconsistency

### 物理含义 / Physical Meaning

`backproject_cupy.py` 中的 `inv_scale`：
`inv_scale` in `backproject_cupy.py`:
```python
inv_scale = np.float32(conf['scan_radius']**2 / conf['projs_per_turn'])
```

问题：`conf['projs_per_turn']` 是每块单独算的：
Issue: `conf['projs_per_turn']` is computed independently per chunk:

```python
# initialize.py line 99:
helical_conf['projs_per_turn'] = total_projs / s_len * 2*pi
#   where: total_projs = len(angles_c)  ← per chunk
#          s_len = angles_range          ← per chunk
```

若某块角度范围因 z-cull 边界不均匀导致 `s_len` 偏小，`projs_per_turn` 会偏大，`inv_scale` 偏小，该块重建幅度整体偏低。
If a chunk's angle range is shortened by z-cull asymmetry, `s_len` is underestimated → `projs_per_turn` overestimated → `inv_scale` too small → chunk amplitude too low.

注意：这是**每块**偏差，与**每转**偏差不同。但若每块恰好包含整数圈，则周期会与 per-turn 混叠。
Note: this is a **per-chunk** effect, not per-turn. But if chunks contain whole-turn multiples, effects alias with per-turn.

### 测试方案 / Test Plan

**T3-A：打印每块的 projs_per_turn（零成本诊断）**
**T3-A: Print projs_per_turn per chunk (zero-cost diagnostic)**

在 `full_recon.py` 中加 1 行打印：
Add 1 print line in `full_recon.py`:
```python
conf = create_configuration(sg_c, vol_geom)
print(f"  Chunk {chunk_idx}: projs_per_turn={conf['projs_per_turn']:.4f}, "
      f"total_projs={len(angles_c)}, s_len={abs(angles_c[-1]-angles_c[0]):.6f}")
```

**确认判据 / Confirmation Criterion:**
- 若所有块的 `projs_per_turn` 一致（差异 < 0.1%）→ H3 排除
- If all chunks have consistent `projs_per_turn` (variation < 0.1%) → H3 ruled out
- 若有块偏差 > 0.5% → 改为使用全局固定值：
- If any chunk differs > 0.5% → fix: use global constant:
  ```python
  conf['projs_per_turn'] = global_projs_per_turn  # precomputed from full dataset
  ```

---

## H4：T-D 边界投影权重异常（分区一致性检验）
## H4: T-D Boundary Projection Overweighting (Partition of Unity Check)

### 物理含义 / Physical Meaning

Katsevich 的 T-D 加权（`sino_weight_td`）应满足**分区一致性**：
The Katsevich T-D weighting should satisfy **partition of unity**:
```
对任意体素 (x, y, z):
For any voxel (x, y, z):
  Σ_{所有贡献投影 s} w_TD(s; x, y, z) = 1.0
```

若该和 ≠ 1，该体素的最终重建值就有系统性偏差。
If this sum ≠ 1, the final reconstructed value for that voxel has systematic bias.

若该和随 z 以每转为周期振荡 → 直接解释 flicker。
If the sum oscillates with z at per-turn period → directly explains flicker.

### T-D 权重的代码路径 / T-D Weight Code Path

```python
# filter.py / filter_gpu.py: sino_weight_td()
# 对每个投影 s，对每个列 col，计算该投影在 (col, row) 处的 T-D 权重：
# For each projection s, each col, compute T-D weight at (col, row):

w(col, row) = smoothstep(row - proj_row_mins[col], alpha)  # bottom boundary
            × smoothstep(proj_row_maxs[col] - row, alpha)  # top boundary

# 其中 proj_row_mins/maxs 来自 Noo et al. Eq.(78)，已包含 col_offset
# where proj_row_mins/maxs from Noo et al. Eq.(78), already include col_offset
```

### 测试方案 / Test Plan

**T4-A：中心体素分区一致性（无 GPU，纯 NumPy）**
**T4-A: Central voxel partition of unity (CPU-only, pure NumPy)**

```python
# 对中心体素 (x=0, y=0)，z=z_k（k=0,1,...,560），
# 找到所有对该 z 有贡献的投影，累积 T-D 权重
# For center voxel (x=0, y=0), z=z_k, find all contributing projections,
# accumulate T-D weight

# 每个投影 s 的贡献行位置：
# Row position for projection s at voxel (0, 0, z_k):
#   L = SOD (central voxel, no lateral displacement)
#   row_hit = (z_k - src_z[s]) / (psize_row * L/SDD)  -- detector row coordinate
#   col_hit = 0 + u_center  -- central column

# 检查 w_TD(col_hit, row_hit) 对所有 s 的累积和是否 = 1
# Check if Σ w_TD(col_hit, row_hit) over all s = 1

# 对 z_k = [0, 0.8, 1.6, ..., 447.2] mm，绘制累积和 vs z 曲线
# Plot accumulated sum vs z for z_k = [0, 0.8, ..., 447.2] mm
```

**T4-B：T-D 边界可视化（无 GPU）**
**T4-B: T-D boundary visualization (CPU-only)**

```python
# 对切片 z=200mm，绘制：
# For slice z=200mm, plot:
# 1. 哪些投影 s 的 proj_row_mins ≤ row_hit(s) ≤ proj_row_maxs
#    Which projections s have proj_row_mins ≤ row_hit(s) ≤ proj_row_maxs
# 2. 各投影的 T-D 权重值
#    T-D weight value for each projection
# 3. 累积权重随 s 增加的积分
#    Cumulative weight sum as function of s
```

**确认判据 / Confirmation Criterion:**
- 若累积和 ≈ 1.0 全程（偏差 < 1%）→ H4 排除
- If cumulative sum ≈ 1.0 throughout (< 1% deviation) → H4 ruled out
- 若累积和有每转振荡（幅度 > 1%）→ T-D 边界公式有问题，深查 `proj_row_mins/maxs` 的 `col_coords` 依赖
- If cumulative sum has per-turn oscillation (> 1% amplitude) → T-D boundary formula issue, investigate `proj_row_mins/maxs` vs `col_coords`

---

## 执行顺序 / Execution Order

```
立即可做（无需 GPU）:
Immediately (no GPU needed):

  Step 1: T3-A — 给 full_recon.py 加 1 行打印，等下次重建时自动输出
           Add 1 print to full_recon.py, auto-outputs next recon run

  Step 2: T4-A — 写分区一致性检验脚本（~50行 NumPy）
           Write partition-of-unity check script (~50 lines NumPy)

  Step 3: T4-B — T-D 边界可视化（~30行 matplotlib）
           T-D boundary visualization (~30 lines matplotlib)

等全量重建结果出来后（~2小时）:
After full recon result (ETA ~2h):

  Step 4: 检查 flicker 指标 —
           Check flicker metric:
           - 若 flicker 已消除（< 0.05）→ col_offset fix 有效，做 T4-A 验证理解
           - If flicker eliminated (< 0.05) → col_offset fix worked, do T4-A for understanding
           - 若 flicker 仍存在（> 0.05）→ 进入 T2-B 子像素扫参
           - If flicker persists (> 0.05) → proceed to T2-B sub-pixel sweep

  Step 5（若需要）: T2-B — 子像素 col_offset 扫参（单片，~20min GPU）
           T2-B: sub-pixel col_offset sweep (single-slice, ~20min GPU)
```

---

## 各假设的关键代码位置 / Key Code Locations Per Hypothesis

| 假设 | 关键位置 Key Location | 具体内容 |
|------|----------------------|---------|
| H2 | `dicom.py` → `det_col_offset` 计算 | 是否有圆柱→平板重采样引入的非整像素误差 |
| H2 | `initialize.py:135` | `col_coords` 中心对齐方式 |
| H3 | `initialize.py:99` | `projs_per_turn = total_projs / s_len * 2π`（每块算）|
| H3 | `backproject_cupy.py` | `inv_scale = SOD² / projs_per_turn` |
| H4 | `initialize.py:171–172` | `proj_row_mins/maxs` Noo Eq.(78) |
| H4 | `filter.py` / `filter_gpu.py` | `sino_weight_td` 中的 smoothstep |

---

## 本次组会新增排障项 vs 之前状态对比
## New vs Previously Known

| 状态 Status | 内容 Item |
|-------------|-----------|
| ✅ 已修复 Fixed | 探测器中心偏移（DICOM tag 误读）|
| ✅ 已排除 Ruled out | chunk 边界/TD平滑值/CuPy kernel/角度抖动/螺旋方向 |
| 🔄 刚修复待验证 Fixed, pending | backproject u/v_center 偏移（全量重建中）|
| 🆕 H2 新增 New | 角度方向残余子像素偏移（导师建议）|
| 🆕 H3 新增 New | per-chunk projs_per_turn 不一致 |
| 🆕 H4 新增 New | T-D 分区一致性（导师边界投影建议）|
