"""
Metrics recorder for sim2sim evaluation.
Collects per-step metrics and splits them into total/stand/motion buckets.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np


@dataclass
class _Bucket:
    steps: int = 0
    tracking_dof_abs_sum: float = 0.0
    tracking_dof_sq_sum: float = 0.0
    root_pos_z_abs_sum: float = 0.0
    root_vel_xy_l2_sum: float = 0.0
    roll_pitch_abs_sum: float = 0.0
    yaw_ang_vel_abs_sum: float = 0.0
    action_delta_abs_sum: float = 0.0
    torque_abs_sum: float = 0.0
    power_abs_sum: float = 0.0
    root_height_sum: float = 0.0
    root_height_sq_sum: float = 0.0
    roll_sum: float = 0.0
    roll_sq_sum: float = 0.0
    pitch_sum: float = 0.0
    pitch_sq_sum: float = 0.0
    mimic_obs_mismatch_count: int = 0

    def as_metrics(self) -> Dict:
        steps = self.steps
        if steps <= 0:
            return {}
        return {
            "tracking_dof_abs_mean": self.tracking_dof_abs_sum / steps,
            "tracking_dof_abs_std": float(np.sqrt(max(
                0.0,
                self.tracking_dof_sq_sum / steps - (self.tracking_dof_abs_sum / steps) ** 2
            ))),
            "root_pos_z_abs_mean": self.root_pos_z_abs_sum / steps,
            "root_vel_xy_l2_mean": self.root_vel_xy_l2_sum / steps,
            "roll_pitch_abs_mean": self.roll_pitch_abs_sum / steps,
            "yaw_ang_vel_abs_mean": self.yaw_ang_vel_abs_sum / steps,
            "action_delta_abs_mean": self.action_delta_abs_sum / steps,
            "torque_abs_mean": self.torque_abs_sum / steps,
            "power_abs_mean": self.power_abs_sum / steps,
            "root_height_mean": self.root_height_sum / steps,
            "root_height_std": float(np.sqrt(max(
                0.0,
                self.root_height_sq_sum / steps - (self.root_height_sum / steps) ** 2
            ))),
            "roll_mean": self.roll_sum / steps,
            "roll_std": float(np.sqrt(max(
                0.0,
                self.roll_sq_sum / steps - (self.roll_sum / steps) ** 2
            ))),
            "pitch_mean": self.pitch_sum / steps,
            "pitch_std": float(np.sqrt(max(
                0.0,
                self.pitch_sq_sum / steps - (self.pitch_sum / steps) ** 2
            ))),
            "mimic_obs_mismatch_count": self.mimic_obs_mismatch_count,
            "steps": steps,
        }


@dataclass
class MetricsRecorder:
    total: _Bucket = field(default_factory=_Bucket)
    stand: _Bucket = field(default_factory=_Bucket)
    motion: _Bucket = field(default_factory=_Bucket)
    last_action: Optional[np.ndarray] = None

    def update(self,
               *,
               action_mimic: Optional[np.ndarray],
               dof_pos: np.ndarray,
               rpy: np.ndarray,
               ang_vel: np.ndarray,
               root_pos: np.ndarray,
               root_vel_local: np.ndarray,
               raw_action: np.ndarray,
               sim_torque: np.ndarray,
               dof_vel: np.ndarray,
               phase_bucket: Optional[str]) -> None:
        # Select bucket
        if phase_bucket == "stand":
            bucket = self.stand
        elif phase_bucket == "motion":
            bucket = self.motion
        else:
            bucket = None

        # Always update total
        self.total.steps += 1
        if bucket is not None:
            bucket.steps += 1

        if action_mimic is not None and len(action_mimic) >= 6 + 29:
            target_dof = np.asarray(action_mimic[6:6 + 29], dtype=np.float32)
            dof_err = np.abs(dof_pos - target_dof).mean()
            self.total.tracking_dof_abs_sum += float(dof_err)
            self.total.tracking_dof_sq_sum += float(dof_err ** 2)
            if bucket is not None:
                bucket.tracking_dof_abs_sum += float(dof_err)
                bucket.tracking_dof_sq_sum += float(dof_err ** 2)

            target_root_pos_z = float(action_mimic[2])
            target_roll = float(action_mimic[3])
            target_pitch = float(action_mimic[4])
            target_yaw_ang_vel = float(action_mimic[5])
            self.total.root_pos_z_abs_sum += float(abs(root_pos[2] - target_root_pos_z))
            self.total.roll_pitch_abs_sum += float(
                0.5 * (abs(rpy[0] - target_roll) + abs(rpy[1] - target_pitch))
            )
            self.total.yaw_ang_vel_abs_sum += float(abs(ang_vel[2] - target_yaw_ang_vel))
            if bucket is not None:
                bucket.root_pos_z_abs_sum += float(abs(root_pos[2] - target_root_pos_z))
                bucket.roll_pitch_abs_sum += float(
                    0.5 * (abs(rpy[0] - target_roll) + abs(rpy[1] - target_pitch))
                )
                bucket.yaw_ang_vel_abs_sum += float(abs(ang_vel[2] - target_yaw_ang_vel))

            root_vel_xy_err = np.linalg.norm(root_vel_local[:2] - np.asarray(action_mimic[0:2], dtype=np.float32))
            self.total.root_vel_xy_l2_sum += float(root_vel_xy_err)
            if bucket is not None:
                bucket.root_vel_xy_l2_sum += float(root_vel_xy_err)
        else:
            self.total.mimic_obs_mismatch_count += 1
            if bucket is not None:
                bucket.mimic_obs_mismatch_count += 1

        if self.last_action is not None:
            delta = float(np.abs(raw_action - self.last_action).mean())
            self.total.action_delta_abs_sum += delta
            if bucket is not None:
                bucket.action_delta_abs_sum += delta
        self.last_action = raw_action.copy()

        torque_abs_mean = float(np.abs(sim_torque).mean())
        power_abs_mean = float(np.abs(sim_torque * dof_vel).mean())
        self.total.torque_abs_sum += torque_abs_mean
        self.total.power_abs_sum += power_abs_mean
        self.total.root_height_sum += float(root_pos[2])
        self.total.root_height_sq_sum += float(root_pos[2] ** 2)
        self.total.roll_sum += float(rpy[0])
        self.total.roll_sq_sum += float(rpy[0] ** 2)
        self.total.pitch_sum += float(rpy[1])
        self.total.pitch_sq_sum += float(rpy[1] ** 2)
        if bucket is not None:
            bucket.torque_abs_sum += torque_abs_mean
            bucket.power_abs_sum += power_abs_mean
            bucket.root_height_sum += float(root_pos[2])
            bucket.root_height_sq_sum += float(root_pos[2] ** 2)
            bucket.roll_sum += float(rpy[0])
            bucket.roll_sq_sum += float(rpy[0] ** 2)
            bucket.pitch_sum += float(rpy[1])
            bucket.pitch_sq_sum += float(rpy[1] ** 2)

    def as_dict(self) -> Dict[str, Dict]:
        return {
            "total": self.total.as_metrics(),
            "stand": self.stand.as_metrics(),
            "motion": self.motion.as_metrics(),
        }
