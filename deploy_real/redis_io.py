"""
Redis IO helpers for sim2sim.
Wraps common read/write operations and key formatting.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import json
import numpy as np
import redis

from deploy_real.redis_protocol import RedisKeys


@dataclass
class RedisIO:
    """Redis IO wrapper.
    输入/Input: redis client/pipeline/keys.
    输出/Output: 结构化读写接口。
    功能/Function: 封装 Redis 读写与协议字段。
    """
    client: redis.Redis
    pipeline: redis.client.Pipeline
    keys: RedisKeys

    @classmethod
    def connect(cls, host: str = "localhost", port: int = 6379, db: int = 0) -> "RedisIO":
        client = redis.Redis(host=host, port=port, db=db)
        pipeline = client.pipeline()
        keys = RedisKeys()
        return cls(client=client, pipeline=pipeline, keys=keys)

    def set_state_body(self, robot: str, state_body: np.ndarray) -> None:
        self.pipeline.set(self.keys.format(self.keys.STATE_BODY, robot), json.dumps(state_body.tolist()))

    def set_state_hands_neck(self, robot: str) -> None:
        self.pipeline.set(self.keys.format(self.keys.STATE_HAND_LEFT, robot), json.dumps(np.zeros(7).tolist()))
        self.pipeline.set(self.keys.format(self.keys.STATE_HAND_RIGHT, robot), json.dumps(np.zeros(7).tolist()))
        self.pipeline.set(self.keys.format(self.keys.STATE_NECK, robot), json.dumps(np.zeros(2).tolist()))

    def set_root_pos(self, robot: str, root_pos: np.ndarray) -> None:
        self.pipeline.set(self.keys.format(self.keys.ROOT_POS, robot), json.dumps(root_pos.tolist()))

    def set_t_state(self) -> None:
        import time
        self.pipeline.set(self.keys.T_STATE, int(time.time() * 1000))

    def set_motion_phase(self, phase: str) -> None:
        self.client.set(self.keys.MOTION_PHASE, phase)

    def set_motion_done(self, done: bool) -> None:
        self.client.set(self.keys.MOTION_DONE, int(bool(done)))

    def set_policy_stop(self, stop: bool) -> None:
        self.client.set(self.keys.POLICY_STOP, int(bool(stop)))

    def clear_flags(self) -> None:
        self.client.delete(self.keys.T_STATE)
        self.client.delete(self.keys.MOTION_PHASE)
        self.client.delete(self.keys.MOTION_DONE)
        self.client.delete(self.keys.POLICY_STOP)

    def get_motion_phase(self) -> Optional[bytes]:
        return self.client.get(self.keys.MOTION_PHASE)

    def get_motion_done(self) -> Optional[bytes]:
        return self.client.get(self.keys.MOTION_DONE)

    def get_policy_stop(self) -> Optional[bytes]:
        return self.client.get(self.keys.POLICY_STOP)

    def get_value(self, key: str) -> Optional[bytes]:
        return self.client.get(key)

    def set_value(self, key: str, value) -> None:
        self.client.set(key, value)

    def flush(self) -> None:
        self.pipeline.execute()

    def get_actions(self, robot: str) -> Tuple[Optional[list], Optional[list], Optional[list], Optional[list]]:
        keys = [
            self.keys.format(self.keys.ACTION_BODY, robot),
            self.keys.format(self.keys.ACTION_HAND_LEFT, robot),
            self.keys.format(self.keys.ACTION_HAND_RIGHT, robot),
            self.keys.format(self.keys.ACTION_NECK, robot),
        ]
        for key in keys:
            self.pipeline.get(key)
        results = self.pipeline.execute()
        if not results or len(results) < 4:
            return None, None, None, None
        return (
            json.loads(results[0]),
            json.loads(results[1]),
            json.loads(results[2]),
            json.loads(results[3]),
        )

    def get_state_body(self, robot: str) -> Optional[list]:
        data = self.client.get(self.keys.format(self.keys.STATE_BODY, robot))
        if not data:
            return None
        return json.loads(data)

    def get_root_pos(self, robot: str) -> Optional[list]:
        data = self.client.get(self.keys.format(self.keys.ROOT_POS, robot))
        if not data:
            return None
        return json.loads(data)

    def get_t_state(self) -> Optional[bytes]:
        return self.client.get(self.keys.T_STATE)

    def set_action_body(self, robot: str, action_body: np.ndarray) -> None:
        self.client.set(self.keys.format(self.keys.ACTION_BODY, robot), json.dumps(action_body.tolist()))

    def set_action_hands_neck(self, robot: str) -> None:
        self.client.set(self.keys.format(self.keys.ACTION_HAND_LEFT, robot), json.dumps(np.zeros(7).tolist()))
        self.client.set(self.keys.format(self.keys.ACTION_HAND_RIGHT, robot), json.dumps(np.zeros(7).tolist()))
        self.client.set(self.keys.format(self.keys.ACTION_NECK, robot), json.dumps(np.zeros(2).tolist()))
