"""
Stand manager for pre-stand and cleanup phases.
"""

from dataclasses import dataclass
import numpy as np
import time
import mujoco

from deploy_real.redis_protocol import MotionPhase
from deploy_real.redis_io import RedisIO
from deploy_real.motion_streamer import MotionStreamer


@dataclass
class StandManager:
    """Stand manager.
    输入/Input: stand 时长、task_id、mimic_obs。
    输出/Output: 站立阶段的 Redis 指令流。
    功能/Function: 统一 pre-stand 与 cleanup 时序逻辑。
    """
    redis_io: RedisIO
    motion_streamer: MotionStreamer
    robot: str
    show_viewer: bool
    sim_model: mujoco.MjModel
    sim_data: mujoco.MjData
    viewer: object
    control_dt: float
    robot_base: str

    def pre_stand(self, steps: int, task_id: int = 0, next_tag_fn=None) -> None:
        if steps <= 0:
            return
        for t_step in range(steps):
            t0 = time.time()
            frame = self.motion_streamer.build_frame(t_step, task_id=task_id)
            tag = next_tag_fn() if next_tag_fn is not None else None
            self.redis_io.set_action_body(self.robot, frame.mimic_obs, tag=tag)
            self.redis_io.set_action_hands_neck(self.robot)
            self.redis_io.client.set(self.redis_io.keys.T_STATE, int(time.time() * 1000))
            self.redis_io.set_motion_phase(MotionPhase.PRE_STAND.value)
            if self.show_viewer:
                self.sim_data.qpos[:3] = frame.root_pos
                root_rot = frame.root_rot[[3, 0, 1, 2]]
                self.sim_data.qpos[3:7] = root_rot
                self.sim_data.qpos[7:] = frame.dof_pos
                mujoco.mj_forward(self.sim_model, self.sim_data)
                robot_base_pos = self.sim_data.xpos[self.sim_model.body(self.robot_base).id]
                self.viewer.cam.lookat = robot_base_pos
                self.viewer.cam.distance = 2.0
                self.viewer.sync()
            elapsed = time.time() - t0
            if elapsed < self.control_dt:
                time.sleep(self.control_dt - elapsed)

    def cleanup(self, last_mimic_obs: np.ndarray, target_mimic_obs: np.ndarray, seconds: float = 5.0, next_tag_fn=None) -> None:
        print("[MotionServer] Cleaning up... Interpolating to default mimic_obs...")
        steps = int(seconds / self.control_dt)
        for i in range(steps):
            interp = last_mimic_obs + (target_mimic_obs - last_mimic_obs) * (i / steps)
            tag = next_tag_fn() if next_tag_fn is not None else None
            self.redis_io.set_action_body(self.robot, interp, tag=tag)
            self.redis_io.set_motion_phase(MotionPhase.CLEANUP.value)
            time.sleep(self.control_dt)
        tag = next_tag_fn() if next_tag_fn is not None else None
        self.redis_io.set_action_body(self.robot, target_mimic_obs, tag=tag)
