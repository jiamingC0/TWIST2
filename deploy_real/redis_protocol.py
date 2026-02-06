"""
Redis protocol definitions for TWIST2 sim2sim evaluation.

Centralizes keys and phase/flag semantics to avoid string drift.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class MotionPhase(str, Enum):
    PRE_STAND = "pre_stand"
    MOTION = "motion"
    CLEANUP = "cleanup"
    DONE = "done"


@dataclass(frozen=True)
class RedisKeys:
    # Motion status keys
    T_STATE: str = "t_state"
    MOTION_PHASE: str = "motion_phase"
    MOTION_DONE: str = "motion_done"
    POLICY_STOP: str = "policy_stop"

    # Remote control keys
    MOTION_START_SIGNAL: str = "motion_start_signal"
    MOTION_EXIT_SIGNAL: str = "motion_exit_signal"

    # State keys (sim -> redis)
    STATE_BODY: str = "state_body_{robot}"
    STATE_HAND_LEFT: str = "state_hand_left_{robot}"
    STATE_HAND_RIGHT: str = "state_hand_right_{robot}"
    STATE_NECK: str = "state_neck_{robot}"
    ROOT_POS: str = "root_pos_{robot}"

    # Action keys (motion -> redis)
    ACTION_BODY: str = "action_body_{robot}"
    ACTION_HAND_LEFT: str = "action_hand_left_{robot}"
    ACTION_HAND_RIGHT: str = "action_hand_right_{robot}"
    ACTION_NECK: str = "action_neck_{robot}"

    def format(self, key_template: str, robot: str) -> str:
        return key_template.format(robot=robot)


def is_truthy(value) -> bool:
    if value is None:
        return False
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value).lower() in ("1", "true", "yes", "y", "on")


def motion_phase_is_stand(phase: str) -> bool:
    return phase in (MotionPhase.PRE_STAND.value, MotionPhase.CLEANUP.value)


def normalize_motion_phase(phase) -> str:
    if phase is None:
        return ""
    if isinstance(phase, bytes):
        phase = phase.decode("utf-8")
    return str(phase)
