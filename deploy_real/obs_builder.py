"""
Observation builder for TWIST2 sim2sim.
Encapsulates proprio construction and history/future stacking.
"""

from dataclasses import dataclass
from typing import Deque, Tuple
import numpy as np


@dataclass(frozen=True)
class ObsConfig:
    """Observation config.
    输入/Input: obs 维度与历史长度。
    输出/Output: 配置对象。
    功能/Function: 统一观测维度参数。
    """
    n_mimic_obs: int
    n_proprio: int
    n_obs_single: int
    history_len: int
    total_obs_size: int


class ObservationBuilder:
    """Observation builder.
    输入/Input: 传感状态 + action_mimic + history。
    输出/Output: obs_full / obs_hist / obs_buf。
    功能/Function: 构建策略输入观测并做维度校验。
    """
    def __init__(self, config: ObsConfig, default_dof_pos: np.ndarray, ankle_idx):
        self.config = config
        self.default_dof_pos = default_dof_pos
        self.ankle_idx = ankle_idx

    def build_proprio(self, ang_vel: np.ndarray, rpy: np.ndarray,
                      dof_pos: np.ndarray, dof_vel: np.ndarray,
                      last_action: np.ndarray) -> np.ndarray:
        obs_body_dof_vel = dof_vel.copy()
        obs_body_dof_vel[self.ankle_idx] = 0.0
        obs_proprio = np.concatenate([
            ang_vel * 0.25,
            rpy[:2],
            (dof_pos - self.default_dof_pos),
            obs_body_dof_vel * 0.05,
            last_action
        ])
        return obs_proprio

    def build_state_body(self, ang_vel: np.ndarray, rpy: np.ndarray, dof_pos: np.ndarray) -> np.ndarray:
        return np.concatenate([ang_vel, rpy[:2], dof_pos])

    def build_full_obs(self, action_mimic: np.ndarray, obs_proprio: np.ndarray,
                       history_buf: Deque[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs_full = np.concatenate([action_mimic, obs_proprio])
        obs_hist = np.array(history_buf).flatten()
        future_obs = action_mimic.copy()
        obs_buf = np.concatenate([obs_full, obs_hist, future_obs])
        if obs_buf.shape[0] != self.config.total_obs_size:
            raise ValueError(f"Expected {self.config.total_obs_size} obs, got {obs_buf.shape[0]}")
        return obs_full, obs_hist, obs_buf
