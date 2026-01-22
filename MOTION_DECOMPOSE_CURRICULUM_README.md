# Motion Decompose Curriculum - 使用说明

## 功能概述

`g1_stu_future` 现在支持自动的运动分解课程学习，可以在训练过程中根据迭代次数自动切换轨迹长度：

- **阶段1 (0-10次迭代)**: 使用10s分解的短轨迹
- **阶段2 (10-20次迭代)**: 使用30s分解的中等轨迹
- **阶段3 (20-30次迭代)**: 使用完整长轨迹（不分解）

## 配置文件

配置位于 `legged_gym/envs/g1/g1_mimic_future_config.py`:

```python
class G1MimicStuFutureCfgDAgger(G1MimicStuFutureCfg):
    class runner(G1MimicPrivCfgPPO.runner):
        # Motion decompose curriculum settings
        enable_motion_decompose_curriculum = True
        decompose_phase_1_iters = 10   # iterations for 10s decompose
        decompose_phase_2_iters = 20   # iterations for 30s decompose
        decompose_phase_3_iters = 30   # iterations for no decompose (full motions)
```

## Debug 日志输出

运行时会输出详细的debug信息，证明轨迹长度确实在切换：

### 1. 每个迭代的状态
```
[DEBUG] Iteration 0: Phase 1, Mode: 10.0s decompose
[DEBUG] Iteration 5: Phase 1, Mode: 10.0s decompose
[DEBUG] Iteration 10: Phase 2, Mode: 30.0s decompose
[DEBUG] Iteration 15: Phase 2, Mode: 30.0s decompose
[DEBUG] Iteration 20: Phase 3, Mode: FULL motions (no decompose)
[DEBUG] Iteration 25: Phase 3, Mode: FULL motions (no decompose)
```

### 2. 切换时的详细日志
```
================================================================================
MOTION DECOMPOSE CURRICULUM SWITCH at iteration 10
================================================================================
  Previous mode: 10.0s decompose
  New mode: 30.0s decompose (Phase 2)
================================================================================

[DEBUG] Reloading motion library...
  - Motion file: /path/to/twist2_dataset.yaml
  - Motion decompose: True
  - Decompose mode: 30.0s DECOMPOSE
  - Decompose length: 30.0 seconds

[DEBUG] Motion library reloaded successfully!
  - Total number of motions: XXX
  - Sample motion names (first 5): [...]

[DEBUG] Motion length statistics:
  - Min length: XX.XXs
  - Max length: XX.XXs
  - Mean length: XX.XXs
  - Median length: XX.XXs
  - Unique lengths: [30.0]

[DEBUG] ✓ Verified: Using 30.0s decomposed motion segments
  - Motions <= 30s: XXX/XXX (100.0%)
================================================================================
```

### 3. 完整轨迹的验证日志
```
[DEBUG] ✓ Verified: Using FULL original motion lengths (no decomposition)
  - Max length: 65.23s  (证明是原始长轨迹)
  - Unique lengths: [8.45, 12.23, ..., 65.23]  (多种长度)
```

## 使用方法

### 快速测试 (10/20/30次迭代)

直接使用标准的训练脚本，运行30次迭代后可以Ctrl+C停止观察日志：

```bash
bash train.sh test_curriculum cuda:0
```

这会演示三个阶段的切换，并输出详细的debug信息：
- 迭代 0-9: 使用10s分解轨迹
- 迭代 10-19: 使用30s分解轨迹
- 迭代 20-29: 使用完整长轨迹

### 正常训练

修改配置文件中的迭代次数为实际训练需要的值：

```python
# 在 legged_gym/envs/g1/g1_mimic_future_config.py 中修改
decompose_phase_1_iters = 10000  # 前10000次用10s
decompose_phase_2_iters = 20000  # 中间10000次用30s
decompose_phase_3_iters = 21000  # 最后1000次用完整轨迹
```

然后使用标准的 `train.sh` 脚本运行：

```bash
bash train.sh <experiment_id> <device>

# 例如：
bash train.sh 001_twist2_test cuda:0
```

### 3. 观察输出

训练时会自动：
1. 在每次迭代开始时输出当前阶段和模式
2. 当阶段切换时重新加载运动库
3. 显示详细的运动长度统计信息，证明使用了正确的轨迹长度

## 实现细节

### 修改的文件

1. **legged_gym/envs/g1/g1_mimic_future_config.py**
   - 添加了课程学习配置参数

2. **legged_gym/envs/g1/g1_mimic_future.py**
   - 添加了 `update_motion_decompose_curriculum()` 方法
   - 添加了 `_reload_motion_library_with_decompose()` 方法
   - 添加了详细的debug日志

3. **rsl_rl/runners/on_policy_dagger_runner.py**
   - 在训练循环中调用环境更新方法
   - 启用motion decompose curriculum

4. **pose/utils/motion_lib_pkl.py**
   - 添加了 `decompose_length_s` 参数支持可配置的分解长度

### 工作原理

1. 训练开始时，runner会启用环境的 `enable_motion_decompose_curriculum` 标志
2. 每个迭代开始时，runner调用 `env.update_motion_decompose_curriculum(iteration)`
3. 环境根据当前迭代次数判断应该使用哪种分解模式
4. 如果模式改变，重新创建 MotionLib 实例加载新的轨迹集
5. 输出详细的debug信息供验证

## 自定义配置

你可以轻松调整阶段切换的迭代次数和分解长度：

```python
# 修改 g1_mimic_future_config.py
decompose_phase_1_iters = 5000   # 改为5000次迭代
decompose_phase_2_iters = 15000  # 改为15000次迭代
decompose_phase_3_iters = 16000  # 改为16000次迭代
```

## 注意事项

- 运动库重新加载需要一些时间（取决于运动文件大小）
- 每次切换后motion_difficulty会被重置为初始值10
- 所有环境的motion_ids会重新采样
- 切换会在训练日志中清晰显示，不会默默发生
