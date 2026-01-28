# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
import statistics
from rich import print

# from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import wandb
# import ml_runlog
import datetime

import numpy as np
from rsl_rl.algorithms import *
from rsl_rl.modules import *
from rsl_rl.storage.replay_buffer import ReplayBuffer
from rsl_rl.env import VecEnv
import sys
from copy import copy, deepcopy
import warnings
# from rsl_rl.utils.running_mean_std import RunningMeanStd
from rsl_rl.utils.normalizer import Normalizer

from legged_gym import LEGGED_GYM_ROOT_DIR


def get_policy_path(proj_name, exptid, checkpoint=-1):
    policy_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", proj_name, exptid)
    if checkpoint == -1:
        models = [file for file in os.listdir(policy_dir) if "model" in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)
    
    return os.path.join(policy_dir, model)


class OnPolicyDaggerRunnerCJM:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu', **kwargs):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self.normalize_obs = env.cfg.env.normalize_obs

        # Teacher configuration
        self.teacher_cfg = train_cfg["teachercfg"]["runner"]
        self.teacher_policy_cfg = train_cfg["teachercfg"]["policy"]
        self.teacher_alg_cfg = train_cfg["teachercfg"]["algorithm"]
        self.warm_iters = self.cfg["warm_iters"]
        self.eval_student = self.cfg["eval_student"]

        # Initialize teacher policy
        teacher_policy_class = eval(self.teacher_cfg["policy_class_name"])
        self.teacher_actor_critic = teacher_policy_class(num_observations=self.env.num_privileged_obs,
                                    num_critic_observations=self.env.num_privileged_obs,
                                    num_motion_observations=self.env.cfg.env.n_priv_mimic_obs,
                                    num_motion_steps=len(self.env.cfg.env.tar_motion_steps_priv),
                                    num_actions=self.env.num_actions,
                                    **self.teacher_policy_cfg).to(self.device)

        if self.normalize_obs:
            self.teacher_normalizer = Normalizer(shape=self.env.num_privileged_obs, device=self.device, dtype=env.obs_buf.dtype)
        else:
            self.teacher_normalizer = None
        self.teacher_actor = self.teacher_actor_critic.act_inference
        
        # Initialize teacher loaded flag
        self.teacher_loaded = False
        
        if not self.eval_student and self.cfg["teacher_experiment_name"] not in ["None", "dummy", None]:
            teacher_policy_pth = get_policy_path(self.cfg["teacher_proj_name"], exptid=self.cfg["teacher_experiment_name"], checkpoint=self.cfg["teacher_checkpoint"])
            self.load_teacher(teacher_policy_pth)
            self.teacher_loaded = True
            print(f"Teacher policy loaded: {teacher_policy_pth}")
        else:
            print("Evaluating student policy only, not loading teacher policy. KL loss will be disabled.")
        
        policy_class = eval(self.cfg["policy_class_name"])
        if "Teleop" in self.cfg["policy_class_name"] or "Tracking" in self.cfg["policy_class_name"]:
            actor_critic = policy_class(num_observations=self.env.num_obs,
                                        num_critic_observations=self.env.num_privileged_obs,
                                        num_motion_observations=self.env.cfg.env.n_mimic_obs,
                                        num_motion_steps=len(self.env.cfg.env.tar_motion_steps),
                                        num_priop_observations=self.env.cfg.env.n_proprio,
                                        num_history_steps=self.env.cfg.env.history_len,
                                        num_actions=self.env.num_actions,
                                        **self.policy_cfg).to(self.device)
        elif "Future" in self.cfg["policy_class_name"]:
            actor_critic = policy_class(num_observations=self.env.num_obs,
                                        num_critic_observations=self.env.num_privileged_obs,
                                        num_motion_observations=self.env.cfg.env.n_mimic_obs,
                                        num_motion_steps=len(self.env.cfg.env.tar_motion_steps),
                                        num_priop_observations=self.env.cfg.env.n_proprio,
                                        num_history_steps=self.env.cfg.env.history_len,
                                        num_actions=self.env.num_actions,
                                        **self.policy_cfg).to(self.device)
        else:
            actor_critic = policy_class(num_observations=self.env.num_obs,
                                        num_critic_observations=self.env.num_privileged_obs,
                                        num_motion_observations=self.env.cfg.env.n_mimic_obs,
                                        num_motion_steps=len(self.env.cfg.env.tar_motion_steps),
                                        num_actions=self.env.num_actions,
                                        **self.policy_cfg).to(self.device)
                
        share_normalizer = (self.env.num_obs == self.env.num_privileged_obs) or self.env.num_privileged_obs is None
            
        if self.normalize_obs:
            # DAgger Runner: Initializing normalizer
            if share_normalizer:
                self.normalizer = Normalizer(shape=self.env.num_obs, device=self.device, dtype=env.obs_buf.dtype)
                self.critic_normalizer = None
            else:
                self.normalizer = Normalizer(shape=self.env.num_obs, device=self.device, dtype=env.obs_buf.dtype)
                self.critic_normalizer = Normalizer(shape=self.env.num_privileged_obs, device=self.device, dtype=env.obs_buf.dtype)
        else:
            self.normalizer = None
            self.critic_normalizer = None
        
        alg_class = eval(self.cfg["algorithm_class_name"]) # DaggerPPO
        self.alg = alg_class(self.env, 
                                  actor_critic,
                                  self.teacher_actor_critic,
                                  teacher_loaded=self.teacher_loaded,
                                  device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.dagger_update_freq = self.alg_cfg["dagger_update_freq"]

        # Evaluation settings
        self.eval_interval = self.cfg.get("eval_interval", 200)
        self.eval_num_episodes = self.cfg.get("eval_num_episodes", 10)
        self.eval_save_metrics = self.cfg.get("eval_save_metrics", True)

        if "Transformer" in self.cfg["policy_class_name"]:
            self.alg.init_storage(
                self.env.num_envs,
                self.num_steps_per_env,
                [self.policy_cfg["obs_context_len"], self.env.num_obs],
                [self.policy_cfg["obs_context_len"], self.env.num_privileged_obs],
                [self.env.num_actions],
            )
        else:
            self.alg.init_storage(
                self.env.num_envs, 
                self.num_steps_per_env, 
                [self.env.num_obs], 
                [self.env.num_privileged_obs], 
                [self.env.num_actions],
            )

        self.learn = self.learn_RL
            
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        print("*************OnPolicyDaggerRunnerCJM init finish*************")
        

    def learn_RL(self, num_learning_iterations, init_at_random_ep_len=False):
        mean_value_loss = 0.
        mean_surrogate_loss = 0.
        mean_disc_loss = 0.
        mean_disc_acc = 0.
        mean_hist_latent_loss = 0.
        mean_priv_reg_loss = 0. 
        priv_reg_coef = 0.
        entropy_coef = 0.
        grad_penalty_coef = 0.

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        if self.normalize_obs:
            obs = self.normalizer.normalize(obs)
            critic_obs = self.teacher_normalizer.normalize(critic_obs)
        infos = {}
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        rew_explr_buffer = deque(maxlen=100)
        rew_entropy_buffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_explr_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_entropy_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        task_rew_buf = deque(maxlen=100)
        cur_task_rew_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        self.start_learning_iteration = copy(self.current_learning_iteration)

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            hist_encoding = it % self.dagger_update_freq == 0
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    # if it < self.warm_iters:
                    #     actions = self.teacher_actor(critic_obs)
                    # else:
                    actions = self.alg.act(obs, critic_obs, infos, hist_encoding)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)  # obs has changed to next_obs !! if done obs has been reset
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    
                    if self.normalize_obs:
                        before_norm_obs = obs.clone()
                        before_norm_critic_obs = critic_obs.clone()
                        # DAgger Runner: Normalizing obs
                        obs = self.normalizer.normalize(obs)
                        critic_obs = self.teacher_normalizer.normalize(critic_obs)
                        if self._need_normalizer_update(it, self.alg_cfg["normalizer_update_iterations"]):
                            self.normalizer.record(before_norm_obs)
                            if self.critic_normalizer is not None:
                                self.critic_normalizer.record(before_norm_critic_obs)
                    
                    total_rew = self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += total_rew
                        cur_reward_explr_sum += 0
                        cur_reward_entropy_sum += 0
                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        rew_explr_buffer.extend(cur_reward_explr_sum[new_ids][:, 0].cpu().numpy().tolist())
                        rew_entropy_buffer.extend(cur_reward_entropy_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        
                        cur_reward_sum[new_ids] = 0
                        cur_reward_explr_sum[new_ids] = 0
                        cur_reward_entropy_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                stop = time.time()
                collection_time = stop - start
                if self.normalize_obs:
                    if self._need_normalizer_update(it, self.alg_cfg["normalizer_update_iterations"]):
                        self.normalizer.update()
                        if self.critic_normalizer is not None:
                            self.critic_normalizer.update()

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            regularization_scale = self.env.cfg.rewards.regularization_scale if hasattr(self.env.cfg.rewards, "regularization_scale") else 1
            average_episode_length = torch.mean(self.env.episode_length.float()).item() if hasattr(self.env, "episode_length") else 0
            mean_motion_difficulty = self.env.mean_motion_difficulty if hasattr(self.env, "mean_motion_difficulty") else 0
            
            self.alg.update_param(it, tot_iter)
            mean_value_loss, mean_surrogate_loss, mean_priv_reg_loss, priv_reg_coef, mean_grad_penalty_loss, grad_penalty_coef, kl_teacher_student_loss = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.eval_interval == 0 and it > 0:
                eval_metrics = self.evaluate_policy(it)
                # Update locals with eval_metrics for logging
                locs['eval_metrics'] = eval_metrics
                self.log(locals)
            else:
                locs['eval_metrics'] = None
            if it < 2500:
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            elif it <= 10000:
                if it % (2*self.save_interval) == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            else:
                if it % (5*self.save_interval) == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        # self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
    
    def _need_normalizer_update(self, iterations, update_iterations):
        return iterations < update_iterations

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        wandb_dict = {}
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # wandb_dict['Episode_rew/' + key] = value
                if "metric" in key:
                    wandb_dict['Episode_rew_metrics/' + key] = value
                else:
                    if "tracking" in key:
                        wandb_dict['Episode_rew_tracking/' + key] = value
                    elif "curriculum" in key:
                        wandb_dict['Episode_curriculum/' + key] = value
                    else:
                        wandb_dict['Episode_rew_regularization/' + key] = value
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n""" # dont print metrics
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        wandb_dict['Loss/value_func'] = locs['mean_value_loss']
        wandb_dict['Loss/surrogate'] = locs['mean_surrogate_loss']
        wandb_dict['Loss/entropy_coef'] = locs['entropy_coef']
        wandb_dict['Loss/learning_rate'] = self.alg.learning_rate
        wandb_dict['Loss/kl_teacher_student'] = locs['kl_teacher_student_loss']
        wandb_dict['Adaptation/hist_latent_loss'] = locs['mean_hist_latent_loss']
        wandb_dict['Adaptation/priv_reg_loss'] = locs['mean_priv_reg_loss']
        wandb_dict['Adaptation/priv_ref_lambda'] = locs['priv_reg_coef']

        wandb_dict['Scale/regularization_scale'] = locs["regularization_scale"]
        if locs['grad_penalty_coef'] != 0:
            wandb_dict['Loss/grad_penalty_loss'] = locs['mean_grad_penalty_loss']
            wandb_dict['Scale/grad_penalty_coef'] = locs["grad_penalty_coef"]
        
        if locs['mean_motion_difficulty'] != 0:
            wandb_dict['Scale/motion_difficulty'] = locs["mean_motion_difficulty"]

        wandb_dict['Policy/mean_noise_std'] = mean_std.item()
        wandb_dict['Perf/total_fps'] = fps
        wandb_dict['Perf/collection time'] = locs['collection_time']
        wandb_dict['Perf/learning_time'] = locs['learn_time']
        if len(locs['rewbuffer']) > 0:
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])

        # Add evaluation metrics if available
        if 'eval_metrics' in locs and locs['eval_metrics'] is not None:
            eval_m = locs['eval_metrics']
            wandb_dict['Eval/tracking_error_mean'] = eval_m['mean_tracking_error']
            wandb_dict['Eval/tracking_error_std'] = eval_m['std_tracking_error']
            wandb_dict['Eval/success_rate'] = eval_m['success_rate']
            wandb_dict['Eval/fall_rate'] = eval_m['fall_rate']
            wandb_dict['Eval/entropy_mean'] = eval_m['mean_entropy']
            # wandb_dict['Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            # wandb_dict['Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        wandb.log(wandb_dict, step=locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        scale_str = f"""{'Regularization_scale:':>{pad}} {locs['regularization_scale']:.4f}\n"""
        average_episode_length = f"""{'Average_episode_length:':>{pad}} {locs['average_episode_length']:.4f}\n"""
        gp_scale_str = f"""{'Grad_penalty_coef:':>{pad}} {locs['grad_penalty_coef']:.4f}\n"""
        motion_difficulty_str = f"""{'Mean_motion_difficulty:':>{pad}} {locs['mean_motion_difficulty']:.4f}\n"""
        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Experiment Name:':>{pad}} {os.path.basename(self.log_dir)}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward (total):':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")

        log_string += f"""{'-' * width}\n"""
        log_string += ep_string
        log_string += f"""{'-' * width}\n"""
        log_string += scale_str
        log_string += average_episode_length
        log_string += gp_scale_str
        log_string += motion_difficulty_str
        curr_it = locs['it'] - self.start_learning_iteration
        eta = self.tot_time / (curr_it + 1) * (locs['num_learning_iterations'] - curr_it)
        mins = eta // 60
        secs = eta % 60
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n""")
        print(log_string)

    def save(self, path, infos=None):
        if self.normalize_obs:
            state_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'normalizer': self.normalizer,
            'critic_normalizer': self.critic_normalizer,
            'infos': infos,
            }
        else:
            state_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }
        torch.save(state_dict, path)
        
        # Save to wandb only if enabled in config
        if getattr(self.cfg, 'save_to_wandb', True):  # Default to True for backward compatibility
            wandb.save(path, base_path=os.path.dirname(path))
            print(f"Saved model to {path} as well as to wandb")
        else:
            print(f"Saved model to {path} (wandb saving disabled)")

    def load(self, path, load_optimizer=True):
        print("*" * 80)
        print("Loading model from {}...".format(path))
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if self.normalize_obs:
            self.normalizer = loaded_dict['normalizer']
            self.critic_normalizer = loaded_dict['critic_normalizer']
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        # self.current_learning_iteration = loaded_dict['iter']
        self.current_learning_iteration = int(os.path.basename(path).split("_")[1].split(".")[0])
        self.env.global_counter = self.current_learning_iteration * 24
        self.env.total_env_steps_counter = self.current_learning_iteration * 24
        print("*" * 80)
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
    
    def get_actor_critic(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic
    
    def get_normalizer(self, device=None):
        if device is not None:
            self.normalizer.to(device)
        return self.normalizer
    
    def load_teacher(self, path):
        print("*" * 80)
        print("Loading teacher policy from {}...".format(path))
        loaded_dict = torch.load(path, map_location=self.device)
        self.teacher_actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if self.normalize_obs:
            self.teacher_normalizer = loaded_dict['normalizer']
        print("*" * 80)

    def evaluate_policy(self, it):
        """Evaluate current policy in deterministic mode without exploration."""
        from termcolor import cprint
        import numpy as np

        # Switch to eval mode (no dropout, deterministic actions)
        self.alg.actor_critic.test()

        # Temporarily disable noise if enabled
        original_add_noise = self.env.cfg.noise.add_noise
        self.env.cfg.noise.add_noise = False

        # Create eval metrics storage
        eval_metrics = {
            'iteration': it,
            'tracking_errors': [],
            'fall_rates': 0,
            'success_rates': 0,
            'policy_entropy': [],
            'episode_lengths': []
        }

        num_eval_envs = min(self.env.num_envs, 100)  # Use up to 100 envs for eval

        # Reset environments
        eval_obs = self.env.reset()
        if self.env.privileged_obs_buf is not None:
            eval_priv_obs = self.env.get_privileged_observations()
        else:
            eval_priv_obs = eval_obs

        eval_obs, eval_priv_obs = eval_obs.to(self.device), eval_priv_obs.to(self.device)

        if self.normalize_obs:
            eval_obs = self.normalizer.normalize(eval_obs)
            eval_priv_obs = self.teacher_normalizer.normalize(eval_priv_obs)

        completed_episodes = 0
        episode_tracking_errors = []
        episode_entropies = []
        episode_falls = 0

        current_episode_tracking_errors = torch.zeros(num_eval_envs, device=self.device)
        current_episode_steps = torch.zeros(num_eval_envs, device=self.device)

        with torch.inference_mode():
            while completed_episodes < self.eval_num_episodes:
                # Get actions without exploration (deterministic)
                actions = self.alg.actor_critic.act_inference(eval_obs)

                # Get entropy from distribution (before deterministic action)
                self.alg.actor_critic.update_distribution(eval_obs)
                entropy = self.alg.actor_critic.entropy.mean().item()
                episode_entropies.append(entropy)

                # Step environment
                next_obs, next_priv_obs, rewards, dones, infos = self.env.step(actions)

                eval_priv_obs = next_priv_obs if next_priv_obs is not None else next_obs
                next_obs, eval_priv_obs = next_obs.to(self.device), eval_priv_obs.to(self.device)

                if self.normalize_obs:
                    next_obs = self.normalizer.normalize(next_obs)
                    eval_priv_obs = self.teacher_normalizer.normalize(eval_priv_obs)

                # Compute tracking error for all environments
                joint_dof_error = self.env._error_tracking_joint_dof().mean().item()

                # Accumulate metrics
                for env_id in range(num_eval_envs):
                    current_episode_tracking_errors[env_id] += joint_dof_error
                    current_episode_steps[env_id] += 1

                # Check for completed episodes
                for env_id in range(num_eval_envs):
                    if dones[env_id] and current_episode_steps[env_id] > 0:
                        # Normalize tracking error by steps
                        avg_tracking_error = (current_episode_tracking_errors[env_id] / current_episode_steps[env_id]).item()

                        # Check if episode ended due to fall (not timeout)
                        if 'time_outs' in infos and infos['time_outs'][env_id]:
                            # Timeout - count as success
                            episode_falls += 0
                        else:
                            # Fell or terminated early - count as fall
                            episode_falls += 1

                        episode_tracking_errors.append(avg_tracking_error)

                        current_episode_tracking_errors[env_id] = 0
                        current_episode_steps[env_id] = 0
                        completed_episodes += 1

                        if completed_episodes >= self.eval_num_episodes:
                            break

                eval_obs = next_obs

                if completed_episodes >= self.eval_num_episodes:
                    break

        # Compute aggregate metrics
        eval_metrics['tracking_errors'] = episode_tracking_errors
        eval_metrics['policy_entropy'] = episode_entropies
        eval_metrics['fall_rate'] = episode_falls / self.eval_num_episodes
        eval_metrics['success_rate'] = 1.0 - eval_metrics['fall_rate']
        eval_metrics['mean_tracking_error'] = np.mean(episode_tracking_errors)
        eval_metrics['std_tracking_error'] = np.std(episode_tracking_errors)
        eval_metrics['mean_entropy'] = np.mean(episode_entropies)

        # Restore original noise setting
        self.env.cfg.noise.add_noise = original_add_noise

        # Print evaluation results
        cprint(f"\n{'='*60}", "cyan")
        cprint(f"Evaluation at iteration {it}", "cyan")
        cprint(f"{'='*60}", "cyan")
        cprint(f"Mean Tracking Error: {eval_metrics['mean_tracking_error']:.4f} ± {eval_metrics['std_tracking_error']:.4f}", "green")
        cprint(f"Success Rate: {eval_metrics['success_rate']*100:.1f}%", "green")
        cprint(f"Fall Rate: {eval_metrics['fall_rate']*100:.1f}%", "red" if eval_metrics['fall_rate'] > 0.5 else "green")
        cprint(f"Mean Policy Entropy: {eval_metrics['mean_entropy']:.4f}", "yellow")
        cprint(f"{'='*60}\n", "cyan")

        # Switch back to train mode
        self.alg.actor_critic.train()

        return eval_metrics
        if self.normalize_obs:
            self.teacher_normalizer = loaded_dict['normalizer']
        print("*" * 80)

       
    def get_teacher_inference_policy(self, device=None):
        self.teacher_actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.teacher_actor_critic.to(device)
        return self.teacher_actor_critic.act_inference