from legged_gym.envs.g1.g1_mimic_distill_config import G1MimicPrivCfg, G1MimicPrivCfgPPO
from legged_gym.envs.base.humanoid_mimic_config import HumanoidMimicCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR


# TAR_MOTION_STEPS_FUTURE = [1,2,3,4,5]
TAR_MOTION_STEPS_FUTURE = [0]
class G1MimicStuFutureCJMCfg(G1MimicPrivCfg):
    """Student policy config with future motion support and curriculum masking.
    Extends existing G1MimicPrivCfg to add future motion capabilities."""
    
    class env:
        obs_type = 'student_future'
        
        tar_motion_steps_priv = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                         50, 55, 60, 65, 70, 75, 80, 85, 90, 95,]
        
        # Keep original student motion steps (current frame only)
        tar_motion_steps = [0]
        
        #from G1MimicPrivCfg.env
        num_envs = 4096  #4096
        num_actions = 29
        n_priv_latent = 4 + 1 + 2*num_actions
        extra_critic_obs = 3
        n_priv = 0
        n_priv_info = 3 + 3 + 4 + 3*9 + 2 + 4 + 1 + 2*num_actions # base lin vel, root height, key body pos, contact mask, priv latent
        history_len = 10
        
        n_proprio = 3 + 2 + 3*num_actions
        n_priv_mimic_obs = len(tar_motion_steps_priv) * (21 + num_actions + 3*9) # Hardcode for now, 9 is base, 9 is the number of key bodies
        
        # Future motion frames (additional input, not in history)
        # tar_motion_steps_future = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        # tar_motion_steps_future = [1,2,3,4,5,6,7,8,9,10]
        tar_motion_steps_future = TAR_MOTION_STEPS_FUTURE
        
        
        # Observation dimensions (keep original structure)
        n_mimic_obs_single = 6 + 29 # Modified: root_vel_xy(2) + root_pos_z(1) + roll_pitch(2) + yaw_ang_vel(1) + dof_pos(29)
        n_mimic_obs = len(tar_motion_steps) * n_mimic_obs_single  # Current frame only
        n_proprio = G1MimicPrivCfg.env.n_proprio
        
        

        # Future observation dimensions (same structure as student mimic obs)
        n_future_obs_single = 6 + 29  # Masking disabled -> no indicator channel
        n_future_obs = len(tar_motion_steps_future) * n_future_obs_single
        
        # Total observation size: maintain original structure + future observations (no force obs needed)
        n_obs_single = n_mimic_obs + n_proprio  # Current frame observation (for history)
        num_observations = n_obs_single * (G1MimicPrivCfg.env.history_len + 1) + n_future_obs
        #from G1MimicPrivCfg.env
        n_priv_obs_single = n_priv_mimic_obs + n_proprio + n_priv_info
        num_privileged_obs = n_priv_obs_single
        
        # FALCON-style curriculum force application (domain randomization)
        # enable_force_curriculum = True  # Enable force disturbances during training
        enable_force_curriculum = False  # Enable force disturbances during training

        #from G1MimicPrivCfg.env
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 10
        randomize_start_pos = True
        randomize_start_yaw = False
        history_encoding = True
        contact_buf_len = 10
        normalize_obs = True
        enable_early_termination = True
        pose_termination = True
        pose_termination_dist = 0.7
        rand_reset = True
        track_root = False
        root_tracking_termination_dist = 2.0
        dof_err_w = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # Left Leg
                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # Right Leg
                     1.0, 1.0, 1.0, # waist yaw, roll, pitch
                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # Left Arm
                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # Right Arm
                     ]
        
        global_obs = False
        
        # HumanoidMimicCfg.env
        enable_tar_obs = False
        ref_char_offset = 0.0
        
        # HumanoidCharCfg.env
        rand_yaw_range = 1.2
        record_video = False
        teleop_mode = False

        class force_curriculum:
            # Force application settings
            force_apply_links = ['left_rubber_hand', 'right_rubber_hand']  # Links to apply forces to
            
            # Force curriculum learning
            force_scale_curriculum = True
            force_scale_initial_scale = 1.0
            force_scale_up_threshold = 210    # Episode length threshold for scaling up force
            force_scale_down_threshold = 200  # Episode length threshold for scaling down force
            force_scale_up = 0.02            # Amount to increase force scale
            force_scale_down = 0.02          # Amount to decrease force scale
            force_scale_max = 1.0
            force_scale_min = 0.0
            
            # Force application ranges (in Newtons)
            apply_force_x_range = [-40.0, 40.0]
            apply_force_y_range = [-40.0, 40.0]
            apply_force_z_range = [-50.0, 5.0]
            
            # Force randomization
            zero_force_prob = [0.25, 0.25, 0.25]  # Probability of zeroing each force axis
            randomize_force_duration = [10, 50]  # Force duration range in steps (policy runs at 50Hz)
            
            # Advanced force settings
            max_force_estimation = True       # Use jacobian-based force estimation
            use_lpf = False                   # Low-pass filter applied forces
            force_filter_alpha = 0.05         # LPF coefficient
            
            # Task-specific force behavior
            only_apply_z_force_when_walking = False  # Restrict to Z-axis forces during walking
            only_apply_resistance_when_walking = True # Apply resistance forces against movement
    
    class motion:
        motion_file = f"{LEGGED_GYM_ROOT_DIR}/motion_data_configs/twist2_dataset.yaml"
        
        # Ensure motion curriculum is enabled for difficulty adaptation
        motion_curriculum = True
        motion_curriculum_gamma = 0.01
        motion_decompose = False

        # use_adaptive_pose_termination = Truee
        
        # Motion Domain Randomization - Enable for robustness
        motion_dr_enabled = False
        root_position_noise = [0.01, 0.05]  # ±1-5cm noise range for root position
        root_orientation_noise = [0.1, 0.2]  # ±5.7-11.4° noise range for root orientation (in rad)
        root_velocity_noise = [0.05, 0.1]   # ±0.05-0.1 noise range for root velocity
        joint_position_noise = [0.05, 0.1]  # ±0.05-0.1 rad noise range for joint positions
        motion_dr_resampling = True          # Resample noise each timestep
        
        # Error Aware Sampling parameters
        use_error_aware_sampling = False      # Enable error aware sampling based on max key body error
        error_sampling_power = 5.0          # Power exponent for error-based probability calculation
        error_sampling_threshold = 0.15     # Threshold for max key body error normalization
        
        # from G1MimicPrivCfg.motion
        reset_consec_frames = 30
        key_bodies = ["left_rubber_hand", "right_rubber_hand", "left_ankle_roll_link", "right_ankle_roll_link", "left_knee_link", "right_knee_link", "left_elbow_link", "right_elbow_link", "head_mocap"] # 9 key bodies
        upper_key_bodies = ["left_rubber_hand", "right_rubber_hand", "left_elbow_link", "right_elbow_link", "head_mocap"]
        sample_ratio = 1.0
        motion_smooth = True
        # from HumanoidMimicCfg.motion
        height_offset = 0.0
        use_adaptive_pose_termination = False  # True: use adaptive termination distance, False: use fixed termination distance
    
    class rewards:
        regularization_names = [
                        # "feet_stumble",
                        # "feet_contact_forces",
                        # "lin_vel_z",
                        # "ang_vel_xy",
                        # "orientation",
                        # "dof_pos_limits",
                        # "dof_torque_limits",
                        # "collision",
                        # "torque_penalty",
                        # "thigh_torque_roll_yaw",
                        # "thigh_roll_yaw_acc",
                        # "dof_acc",
                        # "dof_vel",
                        # "action_rate",
                        ]
        regularization_scale = 1.0
        regularization_scale_range = [0.8,2.0]
        regularization_scale_curriculum = False
        regularization_scale_gamma = 0.0001
        
        class scales:     
            # 0628 version  
            # tracking_joint_dof = 0.6
            
            tracking_joint_dof = 2.0
            # tracking_joint_dof2 = 0.3
            
            tracking_joint_vel = 0.2
            # tracking_joint_vel2 = 0.1
            # tracking_root_translation_xy = 1.0
            tracking_root_translation_z = 1.0
            tracking_root_rotation = 1.0
            tracking_root_linear_vel = 1.0
            tracking_root_angular_vel = 1.0
            tracking_keybody_pos = 2.0
            
            tracking_keybody_pos_global = 2.0
            alive = 0.5
            feet_slip = -0.1
            feet_contact_forces = -5e-4      
            feet_stumble = -1.25
            dof_pos_limits = -5.0
            dof_torque_limits = -1.0
            dof_vel = -1e-4
            dof_acc = -5e-8
            action_rate = -0.05
            # action_rate = -0.01
            feet_air_time = 5.0
            ang_vel_xy = -0.01            
            ankle_dof_acc = -5e-8 * 2
            ankle_dof_vel = -1e-4 * 2
            
        min_dist = 0.1
        max_dist = 0.4
        max_knee_dist = 0.4
        feet_height_target = 0.2
        feet_air_time_target = 0.5
        only_positive_rewards = False
        tracking_sigma = 0.2
        tracking_sigma_ang = 0.125
        max_contact_force = 500  # Forces above this value are penalized
        soft_torque_limit = 0.95
        torque_safety_limit = 0.9
        
        # =========================
        termination_roll = 4.0
        termination_pitch = 4.0
        root_height_diff_threshold = 0.3
        
        #from HumanoidMimicCfg.rewards
        clip_rewards = False
        soft_dof_pos_limit = 0.95 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.95
        termination_height = 0.5
        num_lower_body = 0
        target_feet_height = 0.1


class G1MimicStuFutureCJMCfgDAgger(G1MimicStuFutureCJMCfg):
    """DAgger training config for future motion student policy.
    Inherits from G1MimicStuFutureCJMCfg and extends G1MimicStuRLTrackingCfgDAgger."""
    
    seed = 1
    
    class teachercfg(G1MimicPrivCfgPPO):
        pass
    
    class runner:
        policy_class_name = 'ActorCriticFutureCJM'
        algorithm_class_name = 'DaggerPPOCJM'
        runner_class_name = 'OnPolicyDaggerRunnerCJM'
        max_iterations = 30_001
        warm_iters = 100

        # logging
        save_interval = 500
        experiment_name = 'test'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None

        teacher_experiment_name = 'test'
        teacher_proj_name = 'g1_priv_mimic'
        teacher_checkpoint = -1
        eval_student = False

        # Wandb model saving option
        save_to_wandb = False  # Set to False to disable wandb model saving

        # Evaluation settings for baseline experiments
        eval_interval = 200  # Evaluate every N iterations
        eval_num_episodes = 10  # Number of episodes for evaluation
        eval_save_metrics = True  # Save evaluation metrics

        # from HumanoidMimicCfgPPO.runner
        num_steps_per_env = 24 # per iteration

    class algorithm:
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]
        std_schedule = [1.0, 0.4, 4000, 1500]
        entropy_coef = 0.005
        
        dagger_coef_anneal_steps = 60000  # Total steps to anneal dagger_coef to dagger_coef_min
        dagger_coef = 0.2
        dagger_coef_min = 0.1
        
        # Future motion specific parameters
        future_weight_decay = 0.95      # Decay weight for older future frames
        future_consistency_loss = 0.1   # Weight for consistency loss between future predictions
        
        # training params from HumanoidMimicCfgPPO.algorithm
        grad_penalty_coef = 0.0
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 2e-4 #1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.008
        max_grad_norm = 1.
        dagger_update_freq = 20
        priv_reg_coef_schedual = [0, 0.1, 2000, 3000]
        priv_reg_coef_schedual_resume = [0, 0.1, 0, 1]
        normalizer_update_iterations = 3000

    class policy:
        action_std = [0.7] * 12 + [0.4] * 3 + [0.5] * 14
        init_noise_std = 1.0
        obs_context_len = 11
        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        activation = 'silu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        layer_norm = True
        motion_latent_dim = 128
        
        # Future motion encoder parameters
        future_encoder_dims = [256, 256, 128]  # Separate encoder for future motion
        future_attention_heads = 4              # Multi-head attention for future frames
        future_dropout = 0.1                   # Dropout for future encoder
        temporal_embedding_dim = 64            # Temporal position embedding
        future_latent_dim = 128                # Future motion latent dimension
        num_future_steps = len(TAR_MOTION_STEPS_FUTURE)                  # Number of future steps to expect
        
        # Explicit future observation dimensions (avoid miscalculations when tweaking configs)
        num_future_observations = G1MimicStuFutureCJMCfg.env.n_future_obs  # 360
        
        # MoE specific parameters
        num_experts = 4                        # Number of expert networks
        expert_hidden_dims = [256, 128]        # Hidden dimensions for each expert
        gating_hidden_dim = 128                # Hidden dimension for gating network
        moe_temperature = 1.0                  # Temperature for gating softmax
        moe_topk = None                        # Number of top experts to use (None = use all)
        load_balancing_loss_weight = 0.01      # Weight for load balancing loss
        
        #HumanoidMimicCfgPPO.policy
        priv_encoder_dims = [64, 20]
        tanh_encoder_output = False
        fix_action_std = False

