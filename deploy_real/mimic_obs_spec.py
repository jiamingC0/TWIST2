"""
Mimic observation spec and parsing helpers.
Encapsulates magic indices for action_mimic.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class MimicObsSpec:
    """Mimic obs spec.
    输入/Input: action_mimic。
    输出/Output: 解析后的目标字段。
    功能/Function: 统一 action_mimic 布局解析，消除魔法索引。
    """
    # Layout:
    # [0:2] root_vel_xy
    # [2]   root_pos_z
    # [3]   roll
    # [4]   pitch
    # [5]   yaw_ang_vel
    # [6:6+num_dof] dof_pos
    # [..]  task_id (and optional mask at the end)
    num_dof: int = 29

    @property
    def dof_start(self) -> int:
        return 6

    @property
    def dof_end(self) -> int:
        return 6 + self.num_dof

    def is_valid(self, action_mimic: np.ndarray) -> bool:
        return action_mimic is not None and len(action_mimic) >= self.dof_end

    def parse(self, action_mimic: np.ndarray) -> Optional[dict]:
        if not self.is_valid(action_mimic):
            return None
        return {
            "root_vel_xy": np.asarray(action_mimic[0:2], dtype=np.float32),
            "root_pos_z": float(action_mimic[2]),
            "roll": float(action_mimic[3]),
            "pitch": float(action_mimic[4]),
            "yaw_ang_vel": float(action_mimic[5]),
            "dof_pos": np.asarray(action_mimic[self.dof_start:self.dof_end], dtype=np.float32),
        }
