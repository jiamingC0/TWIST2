# 离线评估系统使用说明

## 概述

本系统为 `g1_stu_future_cjm` 任务提供了完整的离线评估功能，用于 D0 基线复现实验。

## 核心文件

### 1. `eval_d0.sh` - 批量评估所有 checkpoint
**用途**: 评估训练过程中保存的所有 checkpoint，生成完整的性能曲线

**用法**:
```bash
bash eval_d0.sh <experiment_id> [device] [num_rollouts]

# 示例:
bash eval_d0.sh 01029_test cuda:0 10
```

**输出**:
- `logs/g1_stu_future_cjm/<exptid>/D0_evaluation/evaluation_results.json`
- `logs/g1_stu_future_cjm/<exptid>/D0_evaluation/evaluation_plots.png`

### 2. `legged_gym/legged_gym/scripts/offline_eval.py` - 核心评估脚本
**用途**: 实现 D0 协议的完整评估逻辑

**用法**:
```bash
cd legged_gym/legged_gym/scripts

# 评估所有 checkpoint
python offline_eval.py \
    --task g1_stu_future_cjm \
    --proj_name g1_stu_future_cjm \
    --exptid <experiment_id> \
    --device cuda:0 \
    --num_rollouts 10 \
    --seed 42

# 评估单个 checkpoint
python offline_eval.py \
    --task g1_stu_future_cjm \
    --proj_name g1_stu_future_cjm \
    --exptid <experiment_id> \
    --device cuda:0 \
    --num_rollouts 10 \
    --checkpoint 20000
```

### 3. `legged_gym/legged_gym/scripts/eval_checkpoint.py` - 单个 checkpoint 快速评估
**用途**: 快速评估单个 checkpoint，用于调试和详细分析

**用法**:
```bash
cd legged_gym/legged_gym/scripts

python eval_checkpoint.py \
    --exptid <experiment_id> \
    --checkpoint <checkpoint_num> \
    --device cuda:0 \
    --num_rollouts 10

# 示例:
python eval_checkpoint.py --exptid 01029_test --checkpoint 20000
```

## D0 评估协议

### 评估环境配置

评估使用独立的环境实例，配置如下：

| 配置项 | 设置 | 说明 |
|-------|------|------|
| Domain randomization | ❌ OFF | 关闭域随机化 |
| Observation noise | ❌ OFF | 关闭观测噪声 |
| External perturbation | ❌ OFF | 关闭外部扰动 |
| Motion curriculum | ❌ OFF | 固定动作难度 |
| Terrain curriculum | ❌ OFF | 固定地形 |
| Force curriculum | ❌ OFF | 关闭力干扰 |
| Randomized init | ❌ OFF | 固定初始状态 |
| Action sampling | ❌ OFF | 使用确定性策略 |

### 评估指标

1. **Tracking Error** (跟踪误差)
   - 关节位置误差均值
   - 主要指标，评估运动模仿质量

2. **Episode Length** (回合长度)
   - 回合持续的步数
   - 反映策略稳定性

3. **Energy** (能耗)
   - 扭矩与动作的乘积
   - 评估控制效率

4. **Success Rate** (成功率)
   - 非掉落的回合比例
   - 反映鲁棒性

5. **Fall Rate** (掉落率)
   - 掉落的回合比例
   - 与成功率互补

6. **Reward** (奖励)
   - 训练过程中的奖励值
   - 仅用于对比参考

### D0 结论类型

#### ✅ A: 无退化
- Deterministic tracking 性能不随训练下降
- 任何观察到的 reward 下降都是评估噪声或探索减少的副作用

#### ⚠️ B: Reward 退化但控制不退化
- Reward 下降但确定性控制质量保持
- 原因可能是 reward shaping 或 entropy 影响

#### ❌ C: 确定性控制退化
- 真正的学习退化
- 需要进入 D1/D2 进行诊断和修复

## 使用流程

### 完整评估流程

1. **训练模型**
```bash
bash train.sh 01029_test cuda:0
```

2. **批量评估所有 checkpoint**
```bash
bash eval_d0.sh 01029_test cuda:0 10
```

3. **查看结果**
```bash
# 查看可视化结果
ls logs/g1_stu_future_cjm/01029_test/D0_evaluation/

# 使用图片查看器打开
eog logs/g1_stu_future_cjm/01029_test/D0_evaluation/evaluation_plots.png
```

4. **分析结果**
- 评估脚本会自动分析并输出 D0 结论
- 查看 JSON 文件获取详细数据

### 快速验证流程

1. **评估单个 checkpoint**
```bash
cd legged_gym/legged_gym/scripts
python eval_checkpoint.py --exptid 01029_test --checkpoint 20000
```

2. **对比不同 checkpoint**
```bash
# 评估早期 checkpoint
python eval_checkpoint.py --exptid 01029_test --checkpoint 5000

# 评估中期 checkpoint
python eval_checkpoint.py --exptid 01029_test --checkpoint 15000

# 评估晚期 checkpoint
python eval_checkpoint.py --exptid 01029_test --checkpoint 30000
```

## 输出文件说明

### evaluation_results.json

JSON 格式保存所有评估结果：

```json
{
  "task_name": "g1_stu_future_cjm",
  "proj_name": "g1_stu_future_cjm",
  "exptid": "01029_test",
  "num_rollouts": 10,
  "seed": 42,
  "results": [
    {
      "iteration": 5000,
      "tracking_error_mean": 0.1234,
      "tracking_error_std": 0.0123,
      "episode_length_mean": 240.5,
      "energy_mean": 1.2345,
      "success_rate": 0.9,
      "fall_rate": 0.1,
      "reward_mean": 45.67
    },
    ...
  ]
}
```

### evaluation_plots.png

包含 6 个子图的可视化结果：

1. **Tracking Error** - 跟踪误差随训练迭代的变化
2. **Success Rate** - 成功率随训练迭代的变化
3. **Episode Length** - 回合长度随训练迭代的变化
4. **Energy** - 能耗随训练迭代的变化
5. **Fall Rate** - 掉落率随训练迭代的变化
6. **Reward** - 奖励值随训练迭代的变化（用于对比）

## 参数说明

### 通用参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--task` | g1_stu_future_cjm | 任务名称 |
| `--proj_name` | g1_stu_future_cjm | 项目名称 |
| `--exptid` | required | 实验ID |
| `--device` | cuda:0 | 使用的设备 |
| `--num_rollouts` | 10 | 每个 checkpoint 的评估回合数 |
| `--seed` | 42 | 随机种子（保证可重复性） |

### offline_eval.py 独有参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--checkpoint` | -1 | 评估特定 checkpoint（-1 表示全部） |
| `--output_dir` | None | 结果输出目录（None 使用默认路径） |

### eval_checkpoint.py 独有参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--checkpoint` | required | 要评估的 checkpoint 编号 |

## 常见问题

### Q: 如何选择 num_rollouts？

A: 通常使用 10 个回合即可获得稳定的结果。如果需要更高的统计精度，可以增加到 20 或更多。

### Q: 评估需要多长时间？

A: 取决于 checkpoint 数量和 num_rollouts。通常 10 个 checkpoint × 10 个回合，大约需要 10-20 分钟。

### Q: 可以在 CPU 上运行吗？

A: 可以，但速度会慢很多。建议使用 GPU。

### Q: 如何处理内存不足？

A: 减少 `--num_rollouts` 或减少 `env_cfg.env.num_envs` 的值。

### Q: 评估结果如何解释？

A: 查看 `doc/D0_Baseline_Guide.md` 获取详细的解释指南。

## 进阶用法

### 自定义评估指标

如果需要添加自定义评估指标，可以修改 `offline_eval.py` 中的 `evaluate_checkpoint` 方法。

### 修改评估环境配置

评估环境配置在 `set_eval_cfg` 函数中，可以根据需求修改。

### 批量对比多个实验

```bash
for exptid in exp1 exp2 exp3; do
    bash eval_d0.sh $exptid cuda:0 10
done
```

## 文档

- **完整 D0 指南**: `doc/D0_Baseline_Guide.md`
- **训练脚本**: `train.sh`
- **在线评估**: `eval.sh`

## 联系和支持

如有问题或建议，请参考 `doc/D0_Baseline_Guide.md` 或联系开发团队。

---

**最后更新**: 2026-01-29
**任务**: g1_stu_future_cjm D0 基线复现
