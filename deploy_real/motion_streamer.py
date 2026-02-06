"""
Motion streaming utilities.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from pose.utils.motion_lib_pkl import MotionLib
from deploy_real.mimic_obs_builder import build_mimic_obs


@dataclass
class MotionFrame:
    mimic_obs: np.ndarray
    root_pos: np.ndarray
    root_rot: np.ndarray
    dof_pos: np.ndarray


@dataclass
class MotionStreamer:
    motion_lib: MotionLib
    tar_motion_steps_tensor: object
    control_dt: float
    robot: str
    device: object

    def build_frame(self, t_step: int, task_id: int) -> MotionFrame:
        mimic_obs, root_pos, root_rot, dof_pos, _, _ = build_mimic_obs(
            motion_lib=self.motion_lib,
            t_step=t_step,
            control_dt=self.control_dt,
            tar_motion_steps=self.tar_motion_steps_tensor,
            device=self.device,
            robot_type=self.robot,
            task_id=task_id,
        )
        return MotionFrame(
            mimic_obs=mimic_obs,
            root_pos=root_pos,
            root_rot=root_rot,
            dof_pos=dof_pos,
        )
