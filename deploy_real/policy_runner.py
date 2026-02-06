"""
Policy runner abstraction for TWIST2 sim2sim.
Handles inference, action scaling and clipping.
"""

from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class PolicyConfig:
    action_scale: np.ndarray
    default_dof_pos: np.ndarray
    action_clip: float = 10.0


class PolicyRunner:
    def __init__(self, policy, device: str, config: PolicyConfig):
        self.policy = policy
        self.device = device
        self.config = config

    def infer_action(self, obs_buf: np.ndarray) -> np.ndarray:
        obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            raw_action = self.policy(obs_tensor).cpu().numpy().squeeze()
        return raw_action

    def scale_action(self, raw_action: np.ndarray) -> np.ndarray:
        clipped = np.clip(raw_action, -self.config.action_clip, self.config.action_clip)
        scaled = clipped * self.config.action_scale
        return scaled

    def to_pd_target(self, scaled_action: np.ndarray) -> np.ndarray:
        return scaled_action + self.config.default_dof_pos
