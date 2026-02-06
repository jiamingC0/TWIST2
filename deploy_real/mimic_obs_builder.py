"""
Build mimic observations from MotionLib.
Extracted to avoid circular imports.
"""

import torch
from pose.utils.motion_lib_pkl import MotionLib
from data_utils.rot_utils import euler_from_quaternion_torch, quat_rotate_inverse_torch


def build_mimic_obs(
    motion_lib: MotionLib,
    t_step: int,
    control_dt: float,
    tar_motion_steps,
    device=None,
    robot_type: str = "g1",
    mask_indicator: bool = False,
    task_id: int = 0
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    motion_times = torch.tensor([t_step * control_dt], device=device).unsqueeze(-1)
    obs_motion_times = tar_motion_steps * control_dt + motion_times
    obs_motion_times = obs_motion_times.flatten()

    motion_ids = torch.zeros(len(tar_motion_steps), dtype=torch.int, device=device)
    motion_task_id = torch.tensor([[task_id]], device=device).reshape(1, -1, 1)

    root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, local_key_body_pos, root_pos_delta_local, root_rot_delta_local = motion_lib.calc_motion_frame(motion_ids, obs_motion_times)

    roll, pitch, yaw = euler_from_quaternion_torch(root_rot, scalar_first=False)
    roll = roll.reshape(1, -1, 1)
    pitch = pitch.reshape(1, -1, 1)
    yaw = yaw.reshape(1, -1, 1)

    root_vel_local = quat_rotate_inverse_torch(root_rot, root_vel, scalar_first=False).reshape(1, -1, 3)
    root_ang_vel_local = quat_rotate_inverse_torch(root_rot, root_ang_vel, scalar_first=False).reshape(1, -1, 3)
    root_pos = root_pos.reshape(1, -1, 3)
    dof_pos = dof_pos.reshape(1, -1, dof_pos.shape[-1])

    if mask_indicator:
        mimic_obs_buf = torch.cat((
                    root_vel_local[..., :2],
                    root_pos[..., 2:3],
                    roll, pitch,
                    root_ang_vel_local[..., 2:3],
                    dof_pos,
                    motion_task_id,
                ), dim=-1)[:, :]
        mask_indicator = torch.ones(1, mimic_obs_buf.shape[1], 1).to(device)
        mimic_obs_buf = torch.cat((mimic_obs_buf, mask_indicator), dim=-1)
    else:
        mimic_obs_buf = torch.cat((
                    root_vel_local[..., :2],
                    root_pos[..., 2:3],
                    roll, pitch,
                    root_ang_vel_local[..., 2:3],
                    dof_pos,
                    motion_task_id,
                ), dim=-1)[:, :]

    mimic_obs_buf = mimic_obs_buf.reshape(1, -1)
    return (
        mimic_obs_buf.detach().cpu().numpy().squeeze(),
        root_pos.detach().cpu().numpy().squeeze(),
        root_rot.detach().cpu().numpy().squeeze(),
        dof_pos.detach().cpu().numpy().squeeze(),
        root_vel.detach().cpu().numpy().squeeze(),
        root_ang_vel.detach().cpu().numpy().squeeze(),
    )
