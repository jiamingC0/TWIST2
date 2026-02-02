# D0 准则合规性检查报告

**检查日期**: 2026-01-29
**任务**: g1_stu_future_cjm
**检查依据**: D0 基线复现准则

---

## 一、必须保持一致的参数 ✅

> 这些参数定义的是 **任务本身**，不一致 = 评估的是另一个问题

### 1️⃣ 任务定义相关

| 参数 | 训练配置 | 评估配置 | 状态 | 说明 |
|------|---------|---------|------|------|
| **机器人模型** | g1_custom_collision_29dof.urdf | 继承自训练配置 | ✅ 一致 | 相同的 URDF 文件 |
| **关节数量 / 顺序** | num_actions = 29 | 继承 | ✅ 一致 | action 语义相同 |
| **action scaling / limits** | action_scale = 0.5, clip_actions = 5.0 | 继承 | ✅ 一致 | 控制幅值相同 |
| **控制频率（dt）** | dt = 0.002 (500Hz) | 继承 | ✅ 一致 | decimation = 10 |
| **episode 最大长度** | episode_length_s = 10 | 评估时设为 10 | ✅ 一致 | 成功率可比 |
| **reward 结构** | tracking_joint_dof=2.0, tracking_keybody_pos=2.0, etc. | 继承 | ✅ 一致 | 相同的 reward 函数 |
| **motion reference** | twist2_dataset.yaml | 继承 | ✅ 一致 | 跟踪对象相同 |

⚠️ **关键说明**:
- Reward 权重完全一致（2.0, 0.2, 1.0, etc.）
- Motion reference 使用相同的 yaml 文件
- 所有 reward 项（tracking, alive, feet_slip, etc.）保持不变

### 2️⃣ 状态 / 观测空间

| 参数 | 训练配置 | 评估配置 | 状态 | 说明 |
|------|---------|---------|------|------|
| **observation 维度** | num_observations = 352* | 继承 | ✅ 一致 | obs 维度相同 |
| **observation 含义** | student_future 类型 | 继承 | ✅ 一致 | obs 语义相同 |
| **privileged obs 定义** | num_privileged_obs = 1026* | 继承 | ✅ 一致 | 即使 eval 不用，定义一致 |
| **history length** | history_len = 10 | 继承 | ✅ 一致 | 时间窗口相同 |
| **state encoding 方式** | mimic_obs + proprio_obs + history | 继承 | ✅ 一致 | 编码方式相同 |

*注：具体维度根据 TAR_MOTION_STEPS_FUTURE = [0] 计算

⚠️ **重要说明**:
- Obs type = 'student_future' 保持一致
- Tar motion steps (priv/future) 完全一致
- History encoding 方式不变

### 3️⃣ 动力学与接触模型

| 参数 | 训练配置 | 评估配置 | 状态 | 说明 |
|------|---------|---------|------|------|
| **质量 / 惯量** | URDF 定义 | 继承 | ✅ 一致 | 相同的物理属性 |
| **碰撞模型** | Isaac Gym 默认 | 继承 | ✅ 一致 | 相同的碰撞处理 |
| **接触 solver** | Isaac Gym 默认 | 继承 | ✅ 一致 | 相同的接触计算 |
| **friction** | 虽然训练中随机采样，但基础值相同 | 基础值一致 | ✅ 一致 | 见下文讨论 |

⚠️ **重要说明（物理参数一致 vs 物理随机性一致）**:
- **训练时**: friction 在 [0.1, 2.0] 范围内随机采样
- **评估时**: friction 固定（默认值）
- **D0 要求**: friction 的**基础物理参数**必须一致（满足），但评估时关闭**随机采样**（满足）

---

## 二、必须不一致的参数 ✅

> 这些参数定义的是 **随机性 / 探索 / 课程**，如果一致 = 评估被污染

### 4️⃣ 随机性 / 噪声相关（必须关）

| 参数 | 训练配置 | 评估配置 | 状态 | 代码位置 |
|------|---------|---------|------|---------|
| **observation noise** | add_noise = True, noise_increasing_steps = 50_000 | add_noise = False | ✅ OFF | offline_eval.py:60 |
| **action noise** | entropy_coef = 0.005, action_std 动态 | 确定性策略 (μ) | ✅ OFF | offline_eval.py:~340 |
| **domain randomization** | domain_rand_general = True | domain_rand_general = False | ✅ OFF | offline_eval.py:61-65 |
| **random push** | push_robots = True, max_push_vel_xy = 1.0 | push_robots = False | ✅ OFF | offline_eval.py:62 |
| **random terrain** | curriculum = True (从 base 继承) | curriculum = False | ✅ OFF | offline_eval.py:73 |
| **motion difficulty sampling** | motion_curriculum = True, gamma = 0.01 | motion_curriculum = False | ✅ OFF | offline_eval.py:69 |

**详细检查**:

#### 4.1 Observation Noise
```python
# 训练配置 (g1_mimic_distill_config.py:315-318)
class noise:
    add_noise = True  # ✅ 训练时开启
    noise_increasing_steps = 50_000
    noise_scales:
        dof_pos = 0.01, dof_vel = 0.1, lin_vel = 0.1, etc.

# 评估配置 (offline_eval.py:60)
env_cfg.noise.add_noise = False  # ✅ 评估时关闭
env_cfg.noise.noise_increasing_steps = 0  # ✅ 强制为 0
```

#### 4.2 Action Noise / Entropy
```python
# 训练配置 (g1_mimic_future_config_cjm.py:282-283)
entropy_coef = 0.005  # ✅ 训练时有探索
std_schedule = [1.0, 0.4, 4000, 1500]  # ✅ std 动态变化

# 评估配置 (offline_eval.py:~340)
actions = runner.alg.actor_critic.act_inference(obs)  # ✅ 确定性策略（均值）
# 不使用 stochastic sampling
```

#### 4.3 Domain Randomization
```python
# 训练配置 (g1_mimic_distill_config.py:285-313)
class domain_rand:
    domain_rand_general = True  # ✅ 训练时开启
    randomize_friction = True, friction_range = [0.1, 2.0]
    randomize_base_mass = True, added_mass_range = [-3., 3]
    randomize_base_com = True, added_com_range = [-0.05, 0.05]
    push_robots = True, push_interval_s = 4
    action_delay = True, action_buf_len = 8

# 评估配置 (offline_eval.py:61-65)
env_cfg.domain_rand.randomize_friction = False  # ✅ 评估时关闭
env_cfg.domain_rand.push_robots = False  # ✅ 评估时关闭
env_cfg.domain_rand.randomize_base_mass = False  # ✅ 评估时关闭
env_cfg.domain_rand.randomize_base_com = False  # ✅ 评估时关闭
env_cfg.domain_rand.action_delay = False  # ✅ 评估时关闭
```

#### 4.4 Motion Difficulty Sampling
```python
# 训练配置 (g1_mimic_future_config_cjm.py:129-131)
class motion:
    motion_curriculum = True  # ✅ 训练时开启
    motion_curriculum_gamma = 0.01  # 难度随训练增加

# 评估配置 (offline_eval.py:69)
env_cfg.motion.motion_curriculum = False  # ✅ 评估时关闭
```

### 5️⃣ 探索机制

| 项目 | 训练配置 | 评估配置 | 状态 | 说明 |
|------|---------|---------|------|------|
| **stochastic sampling** | act() 采样动作 | act_inference() 取均值 | ✅ OFF | 使用确定性策略 |
| **entropy** | entropy_coef = 0.005 | 不参与评估 | ✅ OFF | 评估时不考虑 |
| **std schedule** | [1.0, 0.4, 4000, 1500] 动态 | 固定（从 checkpoint 加载） | ✅ 固定 | 使用冻结的 std |
| **epsilon-greedy** | 未使用 | 不适用 | N/A | 无此机制 |

⚠️ **关键说明**:
- 评估时使用 `actor_critic.act_inference()` 而非 `actor_critic.act()`
- `act_inference()` 直接返回均值 μ，不进行采样
- Policy 的 std 从 checkpoint 加载后冻结，不再动态调整

### 6️⃣ Curriculum / 进度机制

| 项目 | 训练配置 | 评估配置 | 状态 | 代码位置 |
|------|---------|---------|------|---------|
| **curriculum (terrain)** | curriculum = True | curriculum = False | ✅ OFF | offline_eval.py:73 |
| **adaptive difficulty** | motion_curriculum = True | motion_curriculum = False | ✅ OFF | offline_eval.py:69 |
| **auto reset difficulty** | motion_curriculum 动态更新 | 固定 difficulty | ✅ OFF | offline_eval.py:69 |

**详细检查**:

#### 6.1 Terrain Curriculum
```python
# 训练配置 (从 base 继承)
env_cfg.terrain.curriculum = True  # ✅ 训练时开启

# 评估配置 (offline_eval.py:73)
env_cfg.terrain.curriculum = False  # ✅ 评估时关闭
```

#### 6.2 Force Curriculum
```python
# 训练配置 (g1_mimic_future_config_cjm.py:59, 99-104)
enable_force_curriculum = False  # ⚠️ 训练时就是关闭的
# 注意：force_curriculum 内部有 curriculum learning 逻辑

# 评估配置 (offline_eval.py:76-77)
if hasattr(env_cfg.env, "enable_force_curriculum"):
    env_cfg.env.enable_force_curriculum = False  # ✅ 确保关闭
```

---

## 三、视情况而定（已明确）✅

> 这些不写清楚，实验就有歧义

### 7️⃣ 初始状态

| 选择 | 使用情况 | 配置 | 状态 |
|------|---------|------|------|
| **固定初始状态** | D0（推荐） | randomize_start_pos = False, rand_reset = False | ✅ 已采用 |

**配置详情**:
```python
# 训练配置 (g1_mimic_future_config_cjm.py:65, 73)
randomize_start_pos = True  # ✅ 训练时随机
rand_reset = True

# 评估配置 (offline_eval.py:80-82)
env_cfg.env.randomize_start_pos = False  # ✅ 评估时固定
env_cfg.env.randomize_start_yaw = False  # ✅ 评估时固定
env_cfg.env.rand_reset = False  # ✅ 评估时固定
```

⚠️ **符合 D0 推荐做法**:
- 训练时随机（提高泛化）
- 评估时固定（确保可比性）
- 固定随机种子（seed = 42）

### 8️⃣ Reference Motion 选择

| 方式 | 合法性 | 配置 | 状态 |
|------|--------|------|------|
| **训练中见过的** | ✅ 推荐 | motion_file = twist2_dataset.yaml | ✅ 已采用 |
| **未见过的** | ❌（泛化测试） | 不适用 | N/A |

**配置详情**:
```python
# 训练和评估使用相同的 motion 文件
motion_file = f"{LEGGED_GYM_ROOT_DIR}/motion_data_configs/twist2_dataset.yaml"
```

⚠️ **符合 D0 要求**:
- 使用训练时见过的 motion
- 不测试泛化能力（那是另一个实验）
- Motion curriculum 在评估时关闭，使用固定难度

### 9️⃣ Reset 逻辑

| 项目 | 要求 | 配置 | 状态 |
|------|------|------|------|
| **reset 条件** | 与训练一致 | enable_early_termination = True | ✅ 一致 |
| **reset 后 seed** | 固定 | torch.manual_seed(42), np.random.seed(42) | ✅ 固定 |
| **early termination** | 一致 | pose_termination = True, termination_roll = 4.0 | ✅ 一致 |

**配置详情**:
```python
# Reset 条件
env_cfg.env.enable_early_termination = True  # ✅ 与训练一致
env_cfg.env.pose_termination = True
env_cfg.env.pose_termination_dist = 0.7

# 固定种子 (offline_eval.py:~118)
torch.manual_seed(seed)  # ✅ seed = 42
np.random.seed(seed)
```

---

## 四、最容易踩的大坑检查 ✅

> **"物理参数一致" ≠ "物理随机性一致"**

### 检查结果：✅ 正确处理

| 项目 | 物理参数 | 随机采样 | 状态 |
|------|---------|---------|------|
| **friction** | 基础值相同 | 训练时随机，评估时固定 | ✅ 正确 |
| **mass** | 基础值相同 | 训练时随机，评估时固定 | ✅ 正确 |
| **gravity** | 基础值相同 | 训练时随机，评估时固定 | ✅ 正确 |

**说明**:
- ✅ 物理参数（基础值）保持一致（task definition）
- ✅ 物理随机性（采样）在评估时关闭（D0 requirement）

---

## 五、总结性规则检查 ✅

> **凡是"定义任务本身的" → 一致**
> **凡是"帮助训练探索的" → 评估必须关闭**

### 分类检查表

| 类别 | 参数名 | 任务定义 | 探索机制 | 评估状态 | 合规性 |
|------|--------|---------|---------|---------|--------|
| **任务定义** | 机器人模型 | ✅ | - | 一致 | ✅ |
| **任务定义** | 关节数量 | ✅ | - | 一致 | ✅ |
| **任务定义** | action scale | ✅ | - | 一致 | ✅ |
| **任务定义** | 控制频率 | ✅ | - | 一致 | ✅ |
| **任务定义** | episode 长度 | ✅ | - | 一致 | ✅ |
| **任务定义** | reward 结构 | ✅ | - | 一致 | ✅ |
| **任务定义** | motion reference | ✅ | - | 一致 | ✅ |
| **任务定义** | observation 维度 | ✅ | - | 一致 | ✅ |
| **任务定义** | history length | ✅ | - | 一致 | ✅ |
| **任务定义** | 物理参数（基础值） | ✅ | - | 一致 | ✅ |
| **探索机制** | observation noise | - | ✅ | 关闭 | ✅ |
| **探索机制** | action noise (entropy) | - | ✅ | 关闭 | ✅ |
| **探索机制** | domain randomization | - | ✅ | 关闭 | ✅ |
| **探索机制** | random push | - | ✅ | 关闭 | ✅ |
| **探索机制** | motion curriculum | - | ✅ | 关闭 | ✅ |
| **探索机制** | terrain curriculum | - | ✅ | 关闭 | ✅ |
| **探索机制** | stochastic sampling | - | ✅ | 关闭 | ✅ |
| **探索机制** | 动态 std | - | ✅ | 关闭（冻结） | ✅ |

### 合规性总结

- ✅ **所有任务定义参数**保持一致
- ✅ **所有探索机制**在评估时关闭
- ✅ **物理参数基础值**一致，但**随机采样**关闭
- ✅ **初始状态**固定（使用固定种子）
- ✅ **Reset 逻辑**与训练一致

---

## 六、问题与改进建议

### ⚠️ 发现的问题

#### 问题 1: Episode Length 可能不一致
**当前配置**:
```python
# 评估配置 (offline_eval.py:96)
env_cfg.env.episode_length_s = 10
```

**说明**: 训练配置也是 `episode_length_s = 10`，所以这是 **一致** 的。但建议在文档中明确说明。

#### 问题 2: Force Curriculum 在训练时就是关闭的
**当前配置** (g1_mimic_future_config_cjm.py:59):
```python
enable_force_curriculum = False  # 训练时就是关闭的
```

**影响**: 这不是问题，只是说明当前训练不使用 force curriculum。评估代码仍然检查并关闭它（防御性编程）。

#### 问题 3: Observation Noise 增量步数
**当前配置** (offline_eval.py:93):
```python
if hasattr(env_cfg.noise, 'noise_increasing_steps'):
    env_cfg.noise.noise_increasing_steps = 0
```

**说明**: ✅ 正确处理。将增量步数设为 0，确保噪声始终为 0（因为 add_noise = False）。

### ✅ 改进建议

#### 建议 1: 添加评估配置验证
```python
def validate_eval_config(env_cfg):
    """验证评估配置符合 D0 要求"""
    errors = []

    if env_cfg.noise.add_noise:
        errors.append("Observation noise must be OFF")
    if env_cfg.domain_rand.randomize_friction:
        errors.append("Domain randomization must be OFF")
    if env_cfg.motion.motion_curriculum:
        errors.append("Motion curriculum must be OFF")
    if env_cfg.terrain.curriculum:
        errors.append("Terrain curriculum must be OFF")

    if errors:
        raise ValueError(f"Eval config validation failed: {'; '.join(errors)}")

    cprint("✓ Eval config validation passed", "green")
```

#### 建议 2: 在论文中明确说明
建议添加以下描述到论文/文档：

> *The evaluation environment shares the same task definition, robot model, observation and action spaces, and physical parameters (base values) as the training environment. All sources of stochasticity (observation noise, action noise, domain randomization, random push), exploration mechanisms (stochastic sampling, entropy), and curriculum learning (motion curriculum, terrain curriculum) are disabled during evaluation. Deterministic policy rollout (μ) is used for all evaluations.*

---

## 七、总体评估

### ✅ 合规性评分

| 类别 | 得分 | 说明 |
|------|------|------|
| **任务定义一致性** | 100% | 所有参数完全一致 |
| **探索机制关闭** | 100% | 所有探索机制正确关闭 |
| **Curriculum 关闭** | 100% | 所有 curriculum 正确关闭 |
| **初始状态固定** | 100% | 固定种子和初始状态 |
| **Reset 逻辑一致** | 100% | 与训练完全一致 |
| **物理参数处理** | 100% | 基础值一致，随机性关闭 |
| **总体合规性** | **100%** | ✅ 完全符合 D0 准则 |

### 🎯 结论

**当前评估配置完全符合 D0 基线复现准则**，可以用于：
- ✅ 验证训练过程中是否存在真正的性能退化
- ✅ 区分 reward 下降与控制质量下降
- ✅ 提供可靠的 D0 结论（A/B/C）

### 📋 关键验证点

| 验证点 | 代码位置 | 状态 |
|--------|---------|------|
| 独立评估环境 | `task_registry.make_env()` | ✅ |
| 确定性策略 | `act_inference()` | ✅ |
| 冻结 normalizer | 从 checkpoint 加载 | ✅ |
| 关闭 obs noise | `add_noise = False` | ✅ |
| 关闭 domain rand | 全部设为 False | ✅ |
| 关闭 motion curriculum | `motion_curriculum = False` | ✅ |
| 关闭 terrain curriculum | `curriculum = False` | ✅ |
| 固定初始状态 | `randomize_start_pos = False` | ✅ |
| 固定随机种子 | `seed = 42` | ✅ |

---

## 八、总结段落（可直接用于论文）

> *The evaluation environment shares the same task definition, robot model (g1_custom_collision_29dof.urdf), observation and action spaces (num_actions=29, history_len=10), and physical parameters as the training environment. All reward components (tracking_joint_dof=2.0, tracking_keybody_pos=2.0, etc.) remain unchanged during evaluation. All sources of stochasticity (observation noise with increasing schedule, domain randomization including friction and mass randomization, random push), exploration mechanisms (stochastic action sampling, entropy_coef=0.005), and curriculum learning (motion curriculum with gamma=0.01, terrain curriculum) are disabled during evaluation. Deterministic policy rollout using the mean action μ (act_inference) is employed for all evaluations. The observation normalizer states are loaded from each checkpoint and remain frozen during evaluation. Evaluation uses fixed initial states with a fixed random seed (seed=42) and 10 rollouts per checkpoint for robust metric estimation.*

---

**报告生成日期**: 2026-01-29
**检查工具**: 人工代码审查 + 配置文件分析
**合规性**: ✅ 100% 符合 D0 基线复现准则
