"""
Offline Evaluation Script for D0 Baseline Reproduction

This script implements the D0 baseline reproduction experiment:
- Evaluates all checkpoints deterministically
- Uses independent evaluation environment
- No exploration, no domain randomization
- Frozen normalizer states
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from termcolor import cprint
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from legged_gym.envs import *
from legged_gym.gym_utils import task_registry
from legged_gym import LEGGED_GYM_ROOT_DIR
from rsl_rl.runners import OnPolicyDaggerRunnerCJM


def get_policy_path(proj_name, exptid, checkpoint=-1):
    """Get path to policy checkpoint."""
    policy_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", proj_name, exptid)
    if checkpoint == -1:
        models = [file for file in os.listdir(policy_dir) if "model" in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)
    return os.path.join(policy_dir, model)


def get_all_checkpoints(proj_name, exptid):
    """Get all model checkpoints from training directory."""
    policy_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", proj_name, exptid)
    models = [file for file in os.listdir(policy_dir) if "model" in file and file.endswith(".pt")]
    # Sort by iteration number
    models.sort(key=lambda m: int(m.split("_")[1].split(".")[0]))
    return models


def set_eval_cfg(env_cfg):
    """Set evaluation environment configuration for D0 baseline experiment."""
    # Number of environments for evaluation
    env_cfg.env.num_envs = 50

    # Disable all domain randomization (D0 requirement)
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.action_delay = False

    # Disable motion domain randomization
    if hasattr(env_cfg, "motion"):
        env_cfg.motion.motion_curriculum = False  # Fixed motion difficulty
        env_cfg.motion.motion_dr_enabled = False

    # Disable terrain curriculum (fixed terrain)
    env_cfg.terrain.curriculum = False

    # Disable force curriculum if present
    if hasattr(env_cfg.env, "enable_force_curriculum"):
        env_cfg.env.enable_force_curriculum = False

    # Fixed initial states (critical for D0)
    env_cfg.env.randomize_start_pos = False
    env_cfg.env.randomize_start_yaw = False
    env_cfg.env.rand_reset = False

    # Evaluation mode settings
    if hasattr(env_cfg.env, 'evaluation_mode'):
        env_cfg.env.evaluation_mode = True

    if hasattr(env_cfg.env, 'force_full_masking'):
        env_cfg.env.force_full_masking = True

    # Disable observation noise
    if hasattr(env_cfg.noise, 'noise_increasing_steps'):
        env_cfg.noise.noise_increasing_steps = 0

    # Set fixed episode length for consistency (matches training config)
    env_cfg.env.episode_length_s = 10

    # Debug visualization disabled for batch evaluation
    env_cfg.env.debug_viz = False
    env_cfg.env.record_video = False

    cprint("Evaluation Configuration Set (D0 Baseline)", "cyan")
    cprint(f"  - Domain Randomization: OFF", "green")
    cprint(f"  - Motion Curriculum: OFF", "green")
    cprint(f"  - Force Curriculum: OFF", "green")
    cprint(f"  - Observation Noise: OFF", "green")
    cprint(f"  - Randomized Init: OFF", "green")

    # Validate evaluation configuration
    validate_eval_config(env_cfg)


def validate_eval_config(env_cfg):
    """Validate that evaluation configuration meets D0 baseline requirements."""
    errors = []
    warnings = []

    # Check task definition consistency (must be ON)
    if env_cfg.env.normalize_obs:
        pass  # OK: obs normalization is part of task definition
    else:
        warnings.append("Observation normalization is OFF")

    # Check exploration mechanisms (must be OFF)
    if env_cfg.noise.add_noise:
        errors.append("Observation noise must be OFF")

    if env_cfg.domain_rand.randomize_friction:
        errors.append("Domain randomization (friction) must be OFF")

    if env_cfg.domain_rand.push_robots:
        errors.append("Random push must be OFF")

    if env_cfg.domain_rand.randomize_base_mass:
        errors.append("Domain randomization (mass) must be OFF")

    if env_cfg.domain_rand.randomize_base_com:
        errors.append("Domain randomization (COM) must be OFF")

    if env_cfg.domain_rand.action_delay:
        errors.append("Action delay must be OFF")

    # Check curriculum (must be OFF)
    if hasattr(env_cfg, "motion") and env_cfg.motion.motion_curriculum:
        errors.append("Motion curriculum must be OFF")

    if hasattr(env_cfg, "motion") and env_cfg.motion.motion_dr_enabled:
        errors.append("Motion domain randomization must be OFF")

    if env_cfg.terrain.curriculum:
        errors.append("Terrain curriculum must be OFF")

    if hasattr(env_cfg.env, "enable_force_curriculum") and env_cfg.env.enable_force_curriculum:
        errors.append("Force curriculum must be OFF")

    # Check initial states (should be fixed)
    if env_cfg.env.randomize_start_pos:
        errors.append("Randomized start position must be OFF")

    if env_cfg.env.randomize_start_yaw:
        errors.append("Randomized start yaw must be OFF")

    if env_cfg.env.rand_reset:
        errors.append("Randomized reset must be OFF")

    # Report results
    if errors:
        cprint("\n❌ EVAL CONFIG VALIDATION FAILED", "red")
        cprint("The following D0 requirements are violated:", "red")
        for i, error in enumerate(errors, 1):
            cprint(f"  {i}. {error}", "red")
        raise ValueError(f"Eval config validation failed: {'; '.join(errors)}")

    if warnings:
        cprint("\n⚠️  EVAL CONFIG WARNINGS", "yellow")
        cprint("The following issues were detected:", "yellow")
        for i, warning in enumerate(warnings, 1):
            cprint(f"  {i}. {warning}", "yellow")

    if not errors:
        cprint("\n✓ Eval config validation passed (D0 compliant)", "green")


class OfflineEvaluator:
    """Offline evaluator for D0 baseline reproduction experiment."""

    def __init__(self, task_name, proj_name, exptid, device, num_rollouts=5, seed=42):
        self.task_name = task_name
        self.proj_name = proj_name
        self.exptid = exptid
        self.device = device
        self.num_rollouts = num_rollouts
        self.seed = seed

        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Get configurations
        env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
        self.env_cfg = env_cfg
        self.train_cfg = train_cfg

        # Set evaluation configuration
        set_eval_cfg(env_cfg)

        # Create evaluation environment (independent instance)
        cprint("\nCreating evaluation environment...", "yellow")
        self.env = task_registry.make_env(name=task_name, args=None, env_cfg=env_cfg)

        # Create runner instance
        self.runner = OnPolicyDaggerRunnerCJM(
            env=self.env,
            train_cfg=train_cfg,
            device=device
        )

        # Evaluation metrics storage
        self.all_results = {}

        cprint("Offline evaluator initialized successfully!", "green")

    def load_checkpoint(self, checkpoint_path):
        """Load a checkpoint for evaluation."""
        cprint(f"\nLoading checkpoint: {checkpoint_path}", "yellow")
        state_dict = torch.load(checkpoint_path, map_location=self.device)

        # Load model weights
        self.runner.alg.actor_critic.load_state_dict(state_dict['model_state_dict'])

        # Load normalizer states (D0 requirement: use frozen normalizer from checkpoint)
        if self.env_cfg.env.normalize_obs:
            self.runner.normalizer = state_dict['normalizer']
            if 'critic_normalizer' in state_dict and state_dict['critic_normalizer'] is not None:
                self.runner.critic_normalizer = state_dict['critic_normalizer']

        iteration = int(os.path.basename(checkpoint_path).split("_")[1].split(".")[0])
        cprint(f"Checkpoint loaded (iteration {iteration})", "green")

        return iteration

    def evaluate_checkpoint(self, checkpoint_path):
        """Evaluate a single checkpoint with deterministic policy."""
        iteration = self.load_checkpoint(checkpoint_path)

        # Switch to test mode (deterministic actions)
        self.runner.alg.actor_critic.test()

        # Reset environment with fixed seed
        self.env.reset_idx(torch.arange(self.env.num_envs))

        obs = self.env.get_observations()
        priv_obs = self.env.get_privileged_observations()
        critic_obs = priv_obs if priv_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)

        # Normalize with frozen normalizer (D0 requirement)
        if self.env_cfg.env.normalize_obs:
            obs = self.runner.normalizer.normalize(obs)
            critic_obs = self.runner.teacher_normalizer.normalize(critic_obs)

        # Metrics storage
        completed_episodes = 0
        episode_metrics = []

        # Per-episode tracking
        current_episode_tracking_error = torch.zeros(self.env.num_envs, device=self.device)
        current_episode_length = torch.zeros(self.env.num_envs, device=self.device)
        current_episode_energy = torch.zeros(self.env.num_envs, device=self.device)

        with torch.inference_mode():
            while completed_episodes < self.num_rollouts:
                # Deterministic action (D0 requirement: no sampling)
                actions = self.runner.alg.actor_critic.act_inference(obs)

                # Step environment
                next_obs, next_priv_obs, rewards, dones, infos = self.env.step(actions)
                critic_obs = next_priv_obs if next_priv_obs is not None else next_obs
                next_obs, critic_obs = next_obs.to(self.device), critic_obs.to(self.device)

                # Normalize observations (frozen normalizer)
                if self.env_cfg.env.normalize_obs:
                    next_obs = self.runner.normalizer.normalize(next_obs)
                    critic_obs = self.runner.teacher_normalizer.normalize(critic_obs)

                # Compute tracking error
                tracking_error = self.env._error_tracking_joint_dof().mean().item()

                # Compute energy (torque * action)
                energy = (torch.abs(self.env.torques) * torch.abs(actions)).sum(dim=-1).mean().item()

                # Accumulate metrics for each environment
                for env_id in range(self.env.num_envs):
                    current_episode_tracking_error[env_id] += tracking_error
                    current_episode_length[env_id] += 1
                    current_episode_energy[env_id] += energy

                # Check for completed episodes
                for env_id in range(self.env.num_envs):
                    if dones[env_id] and current_episode_length[env_id] > 0:
                        # Calculate episode metrics
                        episode_length = current_episode_length[env_id].item()
                        avg_tracking_error = (current_episode_tracking_error[env_id] / episode_length).item()
                        avg_energy = (current_episode_energy[env_id] / episode_length).item()

                        # Determine if episode ended due to fall or timeout
                        is_fall = True
                        if 'time_outs' in infos and infos['time_outs'][env_id]:
                            is_fall = False
                        elif 'episode' in infos and 'time_out' in infos['episode']:
                            if len(infos['episode']['time_out']) > env_id:
                                if infos['episode']['time_out'][env_id] > 0.5:
                                    is_fall = False

                        # Collect additional metrics
                        phase_error = 0.0
                        if hasattr(self.env, '_error_tracking_phase'):
                            phase_error = self.env._error_tracking_phase()[env_id].item()

                        # Store episode metrics
                        metrics = {
                            'tracking_error': avg_tracking_error,
                            'episode_length': episode_length,
                            'energy': avg_energy,
                            'is_fall': is_fall,
                            'phase_error': phase_error,
                            'reward': rewards[env_id].item(),
                        }

                        # Add key body errors if available
                        if hasattr(self.env, '_error_tracking_keybody_pos'):
                            keybody_error = self.env._error_tracking_keybody_pos()[env_id].item()
                            metrics['keybody_error'] = keybody_error

                        episode_metrics.append(metrics)

                        # Reset episode tracking
                        current_episode_tracking_error[env_id] = 0
                        current_episode_length[env_id] = 0
                        current_episode_energy[env_id] = 0
                        completed_episodes += 1

                        if completed_episodes >= self.num_rollouts:
                            break

                obs = next_obs

                if completed_episodes >= self.num_rollouts:
                    break

        # Compute aggregate metrics
        results = {
            'iteration': iteration,
            'num_episodes': len(episode_metrics),
            'tracking_errors': [m['tracking_error'] for m in episode_metrics],
            'tracking_error_mean': np.mean([m['tracking_error'] for m in episode_metrics]),
            'tracking_error_std': np.std([m['tracking_error'] for m in episode_metrics]),
            'episode_lengths': [m['episode_length'] for m in episode_metrics],
            'episode_length_mean': np.mean([m['episode_length'] for m in episode_metrics]),
            'energies': [m['energy'] for m in episode_metrics],
            'energy_mean': np.mean([m['energy'] for m in episode_metrics]),
            'energy_std': np.std([m['energy'] for m in episode_metrics]),
            'falls': sum([m['is_fall'] for m in episode_metrics]),
            'fall_rate': sum([m['is_fall'] for m in episode_metrics]) / len(episode_metrics),
            'success_rate': 1.0 - sum([m['is_fall'] for m in episode_metrics]) / len(episode_metrics),
            'rewards': [m['reward'] for m in episode_metrics],
            'reward_mean': np.mean([m['reward'] for m in episode_metrics]),
            'phase_errors': [m['phase_error'] for m in episode_metrics],
            'phase_error_mean': np.mean([m['phase_error'] for m in episode_metrics]),
        }

        if 'keybody_error' in episode_metrics[0]:
            results['keybody_errors'] = [m['keybody_error'] for m in episode_metrics]
            results['keybody_error_mean'] = np.mean([m['keybody_error'] for m in episode_metrics])

        return results

    def evaluate_all_checkpoints(self):
        """Evaluate all checkpoints from training."""
        checkpoint_files = get_all_checkpoints(self.proj_name, self.exptid)
        cprint(f"\nFound {len(checkpoint_files)} checkpoints to evaluate", "cyan")

        all_results = []

        for ckpt_file in tqdm(checkpoint_files, desc="Evaluating checkpoints"):
            checkpoint_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", self.proj_name, self.exptid, ckpt_file)

            try:
                results = self.evaluate_checkpoint(checkpoint_path)
                all_results.append(results)
                self.print_results(results)
            except Exception as e:
                cprint(f"Error evaluating {ckpt_file}: {e}", "red")
                import traceback
                traceback.print_exc()
                continue

        # Sort results by iteration
        all_results.sort(key=lambda x: x['iteration'])
        self.all_results = all_results

        return all_results

    def print_results(self, results):
        """Print evaluation results for a checkpoint."""
        cprint(f"\n{'='*70}", "cyan")
        cprint(f"Checkpoint: model_{results['iteration']:06d}.pt", "cyan")
        cprint(f"{'='*70}", "cyan")
        cprint(f"Tracking Error:  {results['tracking_error_mean']:.4f} ± {results['tracking_error_std']:.4f}", "green")
        cprint(f"Episode Length:  {results['episode_length_mean']:.2f} steps", "green")
        cprint(f"Energy:         {results['energy_mean']:.4f} ± {results['energy_std']:.4f}", "green")
        cprint(f"Success Rate:   {results['success_rate']*100:.1f}%", "green" if results['fall_rate'] < 0.5 else "yellow")
        cprint(f"Fall Rate:      {results['fall_rate']*100:.1f}%", "red" if results['fall_rate'] > 0.5 else "green")
        cprint(f"Reward:         {results['reward_mean']:.4f}", "yellow")
        if 'keybody_error_mean' in results:
            cprint(f"Keybody Error:  {results['keybody_error_mean']:.4f}", "green")
        cprint(f"{'='*70}", "cyan")

    def save_results(self, output_path):
        """Save evaluation results to JSON file."""
        # Convert numpy types to Python native types
        serializable_results = []
        for r in self.all_results:
            r_serializable = {}
            for k, v in r.items():
                if isinstance(v, np.ndarray):
                    r_serializable[k] = v.tolist()
                elif isinstance(v, (np.integer, np.floating)):
                    r_serializable[k] = float(v)
                else:
                    r_serializable[k] = v
            serializable_results.append(r_serializable)

        with open(output_path, 'w') as f:
            json.dump({
                'task_name': self.task_name,
                'proj_name': self.proj_name,
                'exptid': self.exptid,
                'num_rollouts': self.num_rollouts,
                'seed': self.seed,
                'results': serializable_results
            }, f, indent=2)

        cprint(f"\nResults saved to: {output_path}", "green")

    def plot_results(self, output_dir):
        """Generate and save evaluation plots."""
        os.makedirs(output_dir, exist_ok=True)

        iterations = [r['iteration'] for r in self.all_results]

        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"D0 Baseline: {self.task_name} ({self.exptid})", fontsize=16)

        # 1. Tracking Error
        tracking_errors_mean = [r['tracking_error_mean'] for r in self.all_results]
        tracking_errors_std = [r['tracking_error_std'] for r in self.all_results]
        axes[0, 0].plot(iterations, tracking_errors_mean, 'b-', linewidth=2, label='Mean')
        axes[0, 0].fill_between(iterations,
                               [m - s for m, s in zip(tracking_errors_mean, tracking_errors_std)],
                               [m + s for m, s in zip(tracking_errors_mean, tracking_errors_std)],
                               alpha=0.2, label='±1 std')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Tracking Error')
        axes[0, 0].set_title('Deterministic Tracking Error')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 2. Success Rate
        success_rates = [r['success_rate'] * 100 for r in self.all_results]
        axes[0, 1].plot(iterations, success_rates, 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Success Rate (%)')
        axes[0, 1].set_title('Success Rate')
        axes[0, 1].set_ylim([0, 105])
        axes[0, 1].grid(True)

        # 3. Episode Length
        episode_lengths = [r['episode_length_mean'] for r in self.all_results]
        axes[0, 2].plot(iterations, episode_lengths, 'm-', linewidth=2)
        axes[0, 2].set_xlabel('Iteration')
        axes[0, 2].set_ylabel('Episode Length (steps)')
        axes[0, 2].set_title('Episode Length')
        axes[0, 2].grid(True)

        # 4. Energy
        energy_mean = [r['energy_mean'] for r in self.all_results]
        energy_std = [r['energy_std'] for r in self.all_results]
        axes[1, 0].plot(iterations, energy_mean, 'r-', linewidth=2, label='Mean')
        axes[1, 0].fill_between(iterations,
                               [m - s for m, s in zip(energy_mean, energy_std)],
                               [m + s for m, s in zip(energy_mean, energy_std)],
                               alpha=0.2, label='±1 std')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Energy')
        axes[1, 0].set_title('Energy Consumption')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # 5. Fall Rate
        fall_rates = [r['fall_rate'] * 100 for r in self.all_results]
        axes[1, 1].plot(iterations, fall_rates, 'r-', linewidth=2)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Fall Rate (%)')
        axes[1, 1].set_title('Fall Rate')
        axes[1, 1].set_ylim([0, 105])
        axes[1, 1].grid(True)

        # 6. Reward (for comparison)
        rewards = [r['reward_mean'] for r in self.all_results]
        axes[1, 2].plot(iterations, rewards, 'orange', linewidth=2)
        axes[1, 2].set_xlabel('Iteration')
        axes[1, 2].set_ylabel('Reward')
        axes[1, 2].set_title('Reward (Stochastic during training)')
        axes[1, 2].grid(True)

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(output_dir, "evaluation_plots.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        cprint(f"Plot saved to: {plot_path}", "green")
        plt.close()

    def analyze_results(self):
        """Analyze evaluation results and provide D0 conclusions."""
        if len(self.all_results) < 2:
            cprint("Not enough checkpoints for analysis", "yellow")
            return

        cprint("\n" + "="*70, "cyan")
        cprint("D0 BASELINE ANALYSIS", "cyan")
        cprint("="*70, "cyan")

        # Early vs Late performance comparison
        early_results = self.all_results[0]
        late_results = self.all_results[-1]

        cprint("\nEarly vs Late Performance Comparison:", "yellow")
        cprint(f"  Early (iter {early_results['iteration']}):  "
               f"Tracking Error = {early_results['tracking_error_mean']:.4f}, "
               f"Success Rate = {early_results['success_rate']*100:.1f}%", "white")
        cprint(f"  Late  (iter {late_results['iteration']}):  "
               f"Tracking Error = {late_results['tracking_error_mean']:.4f}, "
               f"Success Rate = {late_results['success_rate']*100:.1f}%", "white")

        # Trend analysis
        tracking_errors = [r['tracking_error_mean'] for r in self.all_results]
        success_rates = [r['success_rate'] for r in self.all_results]
        rewards = [r['reward_mean'] for r in self.all_results]

        # Check for degradation
        early_error = tracking_errors[:5]
        late_error = tracking_errors[-5:]
        avg_early_error = np.mean(early_error)
        avg_late_error = np.mean(late_error)

        early_success = np.mean(success_rates[:5])
        late_success = np.mean(success_rates[-5])

        # D0 Conclusion
        cprint("\n" + "="*70, "cyan")
        cprint("D0 CONCLUSION", "cyan")
        cprint("="*70, "cyan")

        error_improvement = (avg_early_error - avg_late_error) / avg_early_error * 100
        success_change = (late_success - early_success) / early_success * 100

        if avg_late_error < avg_early_error * 1.1:  # Less than 10% degradation
            cprint("✅ CONCLUSION A: No Degradation", "green")
            cprint("    → Deterministic tracking performance does NOT degrade over training", "green")
            cprint("    → Any observed reward decrease is likely due to reduced exploration/entropy", "green")
            cprint(f"    → Tracking error: {avg_early_error:.4f} → {avg_late_error:.4f} ({error_improvement:+.1f}%)", "green")
        elif avg_late_error < avg_early_error * 1.3 and late_success > early_success * 0.9:
            cprint("⚠️  CONCLUSION B: Reward Degradation but No Control Degradation", "yellow")
            cprint("    → Reward decreases but deterministic control quality is maintained", "yellow")
            cprint("    → This suggests reward shaping/entropy effects, not true learning degradation", "yellow")
            cprint(f"    → Tracking error: {avg_early_error:.4f} → {avg_late_error:.4f} ({error_improvement:+.1f}%)", "yellow")
            cprint(f"    → Success rate: {early_success*100:.1f}% → {late_success*100:.1f}% ({success_change:+.1f}%)", "yellow")
        else:
            cprint("❌ CONCLUSION C: Deterministic Control Degradation", "red")
            cprint("    → TRUE learning degradation observed in deterministic evaluation", "red")
            cprint("    → Need to investigate training dynamics / reward function", "red")
            cprint(f"    → Tracking error: {avg_early_error:.4f} → {avg_late_error:.4f} ({error_improvement:+.1f}%)", "red")
            cprint(f"    → Success rate: {early_success*100:.1f}% → {late_success*100:.1f}% ({success_change:+.1f}%)", "red")

        cprint("\nNote: This is D0 baseline analysis. True degradation requires", "yellow")
        cprint("      deterministic control degradation (Conclusion C).", "yellow")
        cprint("="*70 + "\n", "cyan")


def main():
    parser = argparse.ArgumentParser(description='D0 Baseline Reproduction - Offline Evaluation')
    parser.add_argument('--task', type=str, default='g1_stu_future_cjm', help='Task name')
    parser.add_argument('--proj_name', type=str, default='g1_stu_future_cjm', help='Project name')
    parser.add_argument('--exptid', type=str, required=True, help='Experiment ID')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--num_rollouts', type=int, default=10, help='Number of rollouts per checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for results')
    parser.add_argument('--checkpoint', type=int, default=-1, help='Evaluate specific checkpoint (-1 for all)')

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        output_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.proj_name, args.exptid, "D0_evaluation")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Create evaluator
    evaluator = OfflineEvaluator(
        task_name=args.task,
        proj_name=args.proj_name,
        exptid=args.exptid,
        device=args.device,
        num_rollouts=args.num_rollouts,
        seed=args.seed
    )

    # Run evaluation
    if args.checkpoint == -1:
        cprint("\n" + "="*70, "cyan")
        cprint("Starting D0 Baseline Evaluation (All Checkpoints)", "cyan")
        cprint("="*70 + "\n", "cyan")
        results = evaluator.evaluate_all_checkpoints()
    else:
        cprint(f"\nEvaluating single checkpoint: {args.checkpoint}", "cyan")
        checkpoint_path = get_policy_path(args.proj_name, args.exptid, args.checkpoint)
        results = [evaluator.evaluate_checkpoint(checkpoint_path)]
        evaluator.all_results = results

    # Save results
    results_path = os.path.join(output_dir, "evaluation_results.json")
    evaluator.save_results(results_path)

    # Generate plots
    evaluator.plot_results(output_dir)

    # Analyze results
    evaluator.analyze_results()

    cprint("\n✓ D0 Baseline Evaluation Complete!", "green")
    cprint(f"Results saved to: {output_dir}", "green")


if __name__ == '__main__':
    main()
