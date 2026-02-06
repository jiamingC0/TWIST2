"""
Configuration dataclasses for deploy_real components.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class RealTimePolicyControllerConfig:
    """Policy controller config.
    输入/Input: 仿真、策略、性能相关参数。
    输出/Output: 配置对象。
    功能/Function: 统一实时控制器参数。
    """
    xml_file: str = "assets/g1/g1_sim2sim_29dof.xml"
    device: str = "cuda"
    record_video: bool = False
    record_proprio: bool = False
    measure_fps: bool = False
    limit_fps: bool = True
    policy_frequency: int = 50
    show_viewer: bool = True

    # Simulation timing
    sim_duration: float = 100000.0
    sim_dt: float = 0.001

    # Fall detection
    fall_detection_delay: float = 2.0
    motion_grace_seconds: float = 10.0


@dataclass
class MotionServerConfig:
    """Motion server config.
    输入/Input: motion 播放、站立、cleanup 参数。
    输出/Output: 配置对象。
    功能/Function: 统一 MotionServer 参数。
    """
    robot: str = "unitree_g1_with_hands"
    redis_ip: str = "localhost"
    steps: str = "1"
    use_remote_control: bool = False
    send_start_frame_as_end_frame: bool = False
    show_viewer: bool = False
    play_standing_after_motion: bool = True
    pre_standing_seconds: float = 5.0
    cleanup_seconds: float = 5.0
    control_dt: float = 0.02
