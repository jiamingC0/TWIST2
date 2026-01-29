"""
Single Checkpoint Evaluation for D0 Baseline

Usage:
    python eval_checkpoint.py --exptid <exp_id> --checkpoint <num>

Example:
    python eval_checkpoint.py --exptid 01029_test --checkpoint 20000
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from termcolor import cprint

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from legged_gym.envs import *
from legged_gym.gym_utils import task_registry
from legged_gym import LEGGED_GYM_ROOT_DIR
from rsl_rl.runners import OnPolicyDaggerRunnerCJM


def get_policy_path(proj_name, exptid, checkpoint):
    policy_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", proj_name, exptid)
    return os.path.join(policy_dir, f"model_{checkpoint}.pt")


def set_eval_cfg(env_cfg):
    """Set evaluation configuration for D0 baseline."""
    env_cfg.env.num_envs = 50
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.action_delay = False

    if hasattr(env_cfg, "motion"):
        env_cfg.motion.motion_curriculum = False
        env_cfg.motion.motion_dr_enabled = False

    env_cfg.terrain.curriculum = False

    if hasattr(env_cfg.env, "enable_force_curriculum"):
        env_cfg.env.enable_force_curriculum = False

    env_cfg.env.randomize_start_pos = False
    env_cfg.env.randomize_start_yaw = False
    env_cfg.env.rand_reset = False

    if hasattr(env_cfg.env, 'evaluation_mode'):
        env_cfg.env.evaluation_mode = True
    if hasattr(env_cfg.env, 'force_full_masking'):
        env_cfg.env.force_full_masking = True

    if hasattr(env_cfg.noise, 'noise_increasing_steps'):
        env_cfg.noise.noise_increasing_steps = 0

    env_cfg.env.episode_length_s = 10
    env_cfg.env.debug_viz = False
    env_cfg.env.record_video = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='g1_stu_future_cjm')
    parser.add_argument('--proj_name', type=str, default='g1_stu_future_cjm')
    parser.add_argument('--exptid', type=str, required=True)
    parser.add_argument('--checkpoint', type=int, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_rollouts', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cprint(f"\nEvaluating checkpoint {args.checkpoint} for {args.exptid}", "cyan")

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    set_eval_cfg(env_cfg)

    env = task_registry.make_env(name=args.task, args=None, env_cfg=env_cfg)
    runner = OnPolicyDaggerRunnerCJM(env=env, train_cfg=train_cfg, device=args.device)

    # Load checkpoint
    ckpt_path = get_policy_path(args.proj_name, args.exptid, args.checkpoint)
    state_dict = torch.load(ckpt_path, map_location=args.device)
    runner.alg.actor_critic.load_state_dict(state_dict['model_state_dict'])

    if env_cfg.env.normalize_obs:
        runner.normalizer = state_dict['normalizer']
        if 'critic_normalizer' in state_dict:
            runner.critic_normalizer = state_dict['critic_normalizer']

    runner.alg.actor_critic.test()

    # Evaluate
    env.reset_idx(torch.arange(env.num_envs))
    obs = env.get_observations()
    priv_obs = env.get_privileged_observations()
    critic_obs = priv_obs if priv_obs is not None else obs
    obs, critic_obs = obs.to(args.device), critic_obs.to(args.device)

    if env_cfg.env.normalize_obs:
        obs = runner.normalizer.normalize(obs)
        critic_obs = runner.teacher_normalizer.normalize(critic_obs)

    completed_episodes = 0
    episode_metrics = []
    current_tracking_error = torch.zeros(env.num_envs, device=args.device)
    current_episode_length = torch.zeros(env.num_envs, device=args.device)
    current_energy = torch.zeros(env.num_envs, device=args.device)

    with torch.inference_mode():
        while completed_episodes < args.num_rollouts:
            actions = runner.alg.actor_critic.act_inference(obs)
            next_obs, next_priv_obs, rewards, dones, infos = env.step(actions)
            critic_obs = next_priv_obs if next_priv_obs is not None else next_obs
            next_obs, critic_obs = next_obs.to(args.device), critic_obs.to(args.device)

            if env_cfg.env.normalize_obs:
                next_obs = runner.normalizer.normalize(next_obs)
                critic_obs = runner.teacher_normalizer.normalize(critic_obs)

            tracking_error = env._error_tracking_joint_dof().mean().item()
            energy = (torch.abs(env.torques) * torch.abs(actions)).sum(dim=-1).mean().item()

            for env_id in range(env.num_envs):
                current_tracking_error[env_id] += tracking_error
                current_episode_length[env_id] += 1
                current_energy[env_id] += energy

            for env_id in range(env.num_envs):
                if dones[env_id] and current_episode_length[env_id] > 0:
                    episode_len = current_episode_length[env_id].item()
                    avg_tracking_error = (current_tracking_error[env_id] / episode_len).item()
                    avg_energy = (current_energy[env_id] / episode_len).item()

                    is_fall = True
                    if 'time_outs' in infos and infos['time_outs'][env_id]:
                        is_fall = False

                    episode_metrics.append({
                        'tracking_error': avg_tracking_error,
                        'episode_length': episode_len,
                        'energy': avg_energy,
                        'is_fall': is_fall,
                        'reward': rewards[env_id].item()
                    })

                    current_tracking_error[env_id] = 0
                    current_episode_length[env_id] = 0
                    current_energy[env_id] = 0
                    completed_episodes += 1

                    if completed_episodes >= args.num_rollouts:
                        break

            obs = next_obs
            if completed_episodes >= args.num_rollouts:
                break

    # Print results
    tracking_errors = [m['tracking_error'] for m in episode_metrics]
    episode_lengths = [m['episode_length'] for m in episode_metrics]
    energies = [m['energy'] for m in episode_metrics]
    falls = sum([m['is_fall'] for m in episode_metrics])
    rewards = [m['reward'] for m in episode_metrics]

    cprint(f"\n{'='*60}", "cyan")
    cprint(f"Checkpoint: model_{args.checkpoint:06d}.pt", "cyan")
    cprint(f"{'='*60}", "cyan")
    cprint(f"Tracking Error:  {np.mean(tracking_errors):.4f} ± {np.std(tracking_errors):.4f}", "green")
    cprint(f"Episode Length:  {np.mean(episode_lengths):.2f} steps", "green")
    cprint(f"Energy:         {np.mean(energies):.4f} ± {np.std(energies):.4f}", "green")
    cprint(f"Success Rate:   {(1 - falls/len(episode_metrics))*100:.1f}%", "green")
    cprint(f"Fall Rate:      {(falls/len(episode_metrics))*100:.1f}%", "red" if falls/len(episode_metrics) > 0.5 else "green")
    cprint(f"Reward:         {np.mean(rewards):.4f}", "yellow")
    cprint(f"{'='*60}\n", "cyan")


if __name__ == '__main__':
    main()
