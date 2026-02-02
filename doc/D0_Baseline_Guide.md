# D0 Baseline Reproduction Guide

## Overview

This guide describes how to perform D0 baseline reproduction experiments for the `g1_stu_future_cjm` training task. D0 aims to verify whether observed performance degradation during training is a real learning issue or an artifact of evaluation protocols.

## D0 Definition

**Goal**: In a completely offline evaluation setting, use a deterministic evaluation protocol to verify if the PPO (with DAgger/teacher) training process exhibits "policy performance degradation in later stages."

**D0 explicitly does NOT**:
- ❌ Improve performance
- ❌ Tune reward
- ❌ Modify loss structure
- ❌ Add regularization
- ❌ Compare new methods

## Quick Start

### 1. Train a Model

```bash
bash train.sh <experiment_id> <device>

# Example:
bash train.sh 01029_test cuda:0
```

### 2. Evaluate All Checkpoints

```bash
bash eval_d0.sh <experiment_id> [device] [num_rollouts]

# Example:
bash eval_d0.sh 01029_test cuda:0 10
```

### 3. View Results

Results will be saved to:
```
logs/g1_stu_future_cjm/<experiment_id>/D0_evaluation/
├── evaluation_results.json   # Raw evaluation data
└── evaluation_plots.png       # Visualization plots
```

## Evaluation Protocol

### 1. Training Configuration

The training process should remain **completely unchanged**:
- PPO update method
- clip_param
- entropy_coef
- value loss
- DAgger / teacher KL
- std schedule (during training)
- optimizer / LR schedule

**D0 Principle**: The more "native" the training, the more credible the conclusions.

### 2. Normalizer Configuration (Important)

**Current code status**:
- Observation normalizer is continuously updated
- Teacher / student normalizers may be out of sync

**D0 Requirement (Strongly Recommended)**:
```
Warmup phase:    Allow normalizer updates
Later phase:      Freeze normalizer
```

**Reason**:
- Prevents "input distribution drift" causing false degradation

**Important**: Freezing normalizer ≠ modifying reward ≠ modifying model. This ensures evaluation consistency rather than improving performance.

### 3. Checkpoint Saving Strategy

**Mandatory Save Points**:
```
early  : 5k
mid    : 10k
late   : 20k
final  : 30k
```

**Saved Content**:
- actor + critic
- normalizer stats
- std / entropy state

### 4. Evaluation Environment

**Must be Independent Instance**:
```python
eval_env = Env(cfg_eval)
```

**Cannot Share**:
- train env state
- episode buffer
- curriculum state
- random seeds

**Configuration**:

| Item                      | Setting |
|-------------------------|---------|
| Domain randomization    | ❌ OFF  |
| Observation noise       | ❌ OFF  |
| External perturbation  | ❌ OFF  |
| Motion difficulty      | Fixed   |
| Terrain                | Fixed   |
| Seed                   | Fixed   |

### 5. Evaluation Policy (Critical)

#### Deterministic Policy

**Definition**: During evaluation, actions always take the policy distribution mean μ, not sampled.

```python
action = actor_critic.action_mean
```

- ❌ No sampling
- ❌ No action noise
- ❌ No entropy consideration

**Important**: "No exploration" ≠ entropy_coef = 0 (that's a training concept).

#### Model Mode

```python
actor_critic.eval()
torch.no_grad()
```

### 6. Evaluation Process

**For each checkpoint**:
1. Fixed initial state / motion
2. N rollouts (N ≥ 5)
3. Use deterministic policy
4. No buffer / normalizer updates

**Recommended Evaluation Metrics**:
- tracking error
- phase error
- fall rate
- episode length
- energy / torque
- success rate

**Important**: Reward decrease ≠ control quality decrease. Reward likely contains exploration/entropy terms.

### 7. Result Analysis Methods

**Comparison Table**:

| Checkpoint | Deterministic tracking | Fall rate | Energy |
|-----------|----------------------|-----------|--------|
| 5k        |                      |           |        |
| 10k       |                      |           |        |
| 20k       |                      |           |        |
| 30k       |                      |           |        |

**Legal Conclusions (Only 3 Types)**:

#### ✅ A: No Degradation
→ Original problem is evaluation noise artifact

#### ⚠️ B: Reward Degradation but No Control Degradation
→ Reward shaping / entropy influence

#### ❌ C: Deterministic Control Degradation
→ True degradation, proceed to D1/D2

### 8. Common Errors (90% of people make these)

**Examples of what NOT to do**:
- ❌ Use stochastic rollout as evaluation
- ❌ Use train_env for evaluation
- ❌ Curriculum still active
- ❌ Normalizer still updating
- ❌ Entropy affects evaluation metrics
- ❌ Inconsistent initial states for mid/late phases

## Implementation Details

### Evaluation Script Usage

The `offline_eval.py` script implements the complete D0 protocol:

```bash
cd legged_gym/legged_gym/scripts

# Evaluate all checkpoints
python offline_eval.py \
    --task g1_stu_future_cjm \
    --proj_name g1_stu_future_cjm \
    --exptid <experiment_id> \
    --device cuda:0 \
    --num_rollouts 10 \
    --seed 42

# Evaluate specific checkpoint
python offline_eval.py \
    --task g1_stu_future_cjm \
    --proj_name g1_stu_future_cjm \
    --exptid <experiment_id> \
    --device cuda:0 \
    --num_rollouts 10 \
    --checkpoint 20000  # Only evaluate model_20000.pt
```

### Key Features

1. **Independent Evaluation Environment**
   - No sharing with training environment
   - Fixed configuration per D0 requirements

2. **Deterministic Policy**
   - Uses `act_inference` (mean actions)
   - No exploration noise
   - Fixed seeds for reproducibility

3. **Frozen Normalizers**
   - Loads normalizer states from each checkpoint
   - No updates during evaluation

4. **Comprehensive Metrics**
   - Tracking error (mean ± std)
   - Episode length
   - Energy consumption
   - Success/fall rates
   - Reward (for comparison only)

5. **Automated Analysis**
   - Compares early vs late performance
   - Provides D0 conclusions (A/B/C)
   - Generates visualization plots

### Output Files

**JSON Format** (`evaluation_results.json`):
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
      "num_episodes": 10,
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

**Visualization** (`evaluation_plots.png`):
- 6 subplots showing key metrics over iterations
- Error bands (±1 std) for variability
- Color-coded for easy interpretation

## Interpretation Guide

### Scenario A: No Degradation ✅

**Symptoms**:
- Tracking error decreases or stays stable
- Success rate increases or stays stable
- No significant performance drop

**Interpretation**:
- Training is working correctly
- Any observed reward decrease is due to:
  - Reduced exploration (entropy)
  - Reward shaping effects
  - Evaluation protocol artifacts

**Action**: No changes needed. Proceed with confidence.

### Scenario B: Reward Degradation but No Control Degradation ⚠️

**Symptoms**:
- Reward decreases significantly
- BUT tracking error is stable or improving
- Success rate is stable or improving

**Interpretation**:
- Control quality is maintained
- Reward function may over-emphasize exploration
- Entropy coefficient or exploration policy needs review

**Action**: Consider tuning reward weights or entropy schedule.

### Scenario C: Deterministic Control Degradation ❌

**Symptoms**:
- Tracking error increases (>10-30%)
- Success rate decreases
- Energy consumption increases
- True degradation observed

**Interpretation**:
- Real learning problem exists
- May be due to:
  - Distribution shift
  - Catastrophic forgetting
  - Overfitting to exploration
  - Reward hacking

**Action**: Proceed to D1/D2 experiments to diagnose and fix.

## Tips and Best Practices

1. **Reproducibility**
   - Always use fixed seeds (default: 42)
   - Keep evaluation environment configuration consistent
   - Document any modifications

2. **Sample Size**
   - Use at least 5 rollouts per checkpoint
   - 10-20 rollouts for better statistical power
   - More checkpoints = better trend analysis

3. **Checkpoint Coverage**
   - Ensure checkpoints span full training range
   - Include early (5k), mid (10k, 20k), late (30k) phases
   - Save more frequently if training is long

4. **Resource Management**
   - Evaluation can be memory-intensive
   - Process checkpoints sequentially to avoid OOM
   - Use GPU for faster inference

5. **Result Validation**
   - Manually inspect some episodes if possible
   - Compare with training curves
   - Check for anomalies or outliers

## Troubleshooting

### Issue: Out of Memory

**Solution**:
```bash
# Reduce number of environments
--num_rollouts 5  # Instead of 10

# Or evaluate checkpoints one at a time
--checkpoint 5000
--checkpoint 10000
# ...
```

### Issue: Checkpoint Not Found

**Solution**:
```bash
# List available checkpoints
ls logs/g1_stu_future_cjm/<exptid>/model_*.pt

# Verify checkpoint exists before evaluation
```

### Issue: Performance Variability

**Solution**:
- Increase `--num_rollouts` for better averaging
- Ensure fixed seed is used
- Check that environment configuration is truly fixed

### Issue: Unexpected Degradation

**Solution**:
- Verify normalizer is loaded from checkpoint
- Check that domain randomization is disabled
- Ensure curriculum is frozen
- Review training logs for anomalies

## References

- Training script: `train.sh`
- Evaluation script: `legged_gym/legged_gym/scripts/offline_eval.py`
- Wrapper script: `eval_d0.sh`
- Task registration: `legged_gym/legged_gym/envs/__init__.py`

## Summary

> **D0 experiments are not about "running better," but about confirming whether we truly observed "learning degradation" or merely saw side effects of evaluation protocols and randomness.**

The goal is to establish a clean, deterministic baseline before making any changes to the training process. Only if D0 confirms true degradation (Conclusion C) should we proceed with D1/D2 improvements.

---

**Author**: D0 Baseline Reproduction Framework
**Task**: g1_stu_future_cjm
**Date**: 2026-01-29
