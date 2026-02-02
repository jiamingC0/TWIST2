"""
D0 Baseline Evaluation Configuration

This module contains default configuration parameters for D0 baseline evaluation.
"""

class D0EvalConfig:
    """Default configuration for D0 baseline evaluation."""

    # Task information
    TASK_NAME = "g1_stu_future_cjm"
    PROJ_NAME = "g1_stu_future_cjm"

    # Evaluation settings
    DEFAULT_NUM_ROLLOUTS = 10
    DEFAULT_SEED = 42
    DEFAULT_DEVICE = "cuda:0"

    # Checkpoint intervals for mandatory saves (as per D0 protocol)
    MANDATORY_CHECKPOINTS = [5000, 10000, 20000, 30000]

    # Environment configuration
    NUM_EVAL_ENVS = 50
    EPISODE_LENGTH_S = 10

    # Deterministic policy settings
    USE_STOCHASTIC_POLICY = False  # False = deterministic (D0 requirement)
    ENABLE_ACTION_NOISE = False      # D0 requirement

    # Domain randomization (all OFF for D0)
    ENABLE_NOISE = False
    ENABLE_DOMAIN_RANDOMIZATION = False
    ENABLE_MOTION_CURRICULUM = False
    ENABLE_FORCE_CURRICULUM = False
    ENABLE_TERRAIN_CURRICULUM = False

    # Initialization (fixed for D0)
    RANDOMIZE_START_POS = False
    RANDOMIZE_START_YAW = False
    RANDOMIZE_RESET = False

    # Normalizer settings (D0 requirement: frozen from checkpoint)
    NORMALIZER_UPDATE_DURING_EVAL = False

    # Evaluation mode settings
    EVALUATION_MODE = True
    FORCE_FULL_MASKING = True

    # Metrics to compute
    METRICS = {
        'tracking_error': True,
        'episode_length': True,
        'energy': True,
        'success_rate': True,
        'fall_rate': True,
        'reward': True,
        'phase_error': True,
        'keybody_error': True,
    }

    # D0 analysis thresholds
    ERROR_DEGRADATION_THRESHOLD_LOW = 0.1    # 10% degradation = acceptable (A)
    ERROR_DEGRADATION_THRESHOLD_HIGH = 0.3   # 30% degradation = moderate (B)
    SUCCESS_RATE_THRESHOLD = 0.9             # 90% success = acceptable (B)

    @classmethod
    def get_all(cls):
        """Get all configuration as dictionary."""
        return {
            'task_name': cls.TASK_NAME,
            'proj_name': cls.PROJ_NAME,
            'num_rollouts': cls.DEFAULT_NUM_ROLLOUTS,
            'seed': cls.DEFAULT_SEED,
            'device': cls.DEFAULT_DEVICE,
            'num_eval_envs': cls.NUM_EVAL_ENVS,
            'episode_length_s': cls.EPISODE_LENGTH_S,
            'use_stochastic_policy': cls.USE_STOCHASTIC_POLICY,
            'enable_action_noise': cls.ENABLE_ACTION_NOISE,
            'enable_noise': cls.ENABLE_NOISE,
            'enable_domain_randomization': cls.ENABLE_DOMAIN_RANDOMIZATION,
            'enable_motion_curriculum': cls.ENABLE_MOTION_CURRICULUM,
            'enable_force_curriculum': cls.ENABLE_FORCE_CURRICULUM,
            'enable_terrain_curriculum': cls.ENABLE_TERRAIN_CURRICULUM,
            'randomize_start_pos': cls.RANDOMIZE_START_POS,
            'randomize_start_yaw': cls.RANDOMIZE_START_YAW,
            'randomize_reset': cls.RANDOMIZE_RESET,
            'normalizer_update_during_eval': cls.NORMALIZER_UPDATE_DURING_EVAL,
            'evaluation_mode': cls.EVALUATION_MODE,
            'force_full_masking': cls.FORCE_FULL_MASKING,
        }


def get_eval_config():
    """Get default evaluation configuration."""
    return D0EvalConfig()


if __name__ == '__main__':
    # Print configuration
    config = D0EvalConfig()
    print("D0 Baseline Evaluation Configuration")
    print("=" * 50)
    for key, value in config.get_all().items():
        print(f"{key:40s}: {value}")
