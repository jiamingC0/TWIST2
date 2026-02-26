#!/usr/bin/env python3
import pickle
import numpy as np
from types import ModuleType
import sys


class FakeModule(ModuleType):
    def __init__(self, name, real=None):
        super().__init__(name)
        if real:
            self.__dict__.update(real.__dict__)


# Patch numpy for pickle compatibility
np = __import__('numpy')
sys.modules['numpy._core'] = FakeModule('numpy._core', np.core if hasattr(np, 'core') else np)
sys.modules['numpy._core.multiarray'] = FakeModule('numpy._core.multiarray', getattr(np.core, 'multiarray', None))


def read_foot_contact(pkl_path: str, interval: int = 50, max_frames: int = 10):
    """
    读取 pkl 文件的 foot_contact 数据，按间隔输出

    Args:
        pkl_path: pkl 文件路径
        interval: 帧间隔
        max_frames: 最多输出多少帧
    """
    with open(pkl_path, 'rb') as f:
        motion_data = pickle.load(f)

    if 'foot_contact' not in motion_data:
        print(f"错误: {pkl_path} 中没有 foot_contact 数据")
        return

    foot_contact = motion_data['foot_contact']
    total_frames = len(foot_contact)
    fps = motion_data.get('fps', 30.0)

    print(f"文件: {pkl_path}")
    print(f"总帧数: {total_frames}")
    print(f"FPS: {fps}")
    print(f"\n每隔 {interval} 帧输出一次 (最多 {max_frames} 帧):")
    print(f"{'帧号':<6} {'时间(s)':<10} {'左脚':<8} {'右脚':<8}")
    print("-" * 34)

    count = 0
    for i in range(0, total_frames, interval):
        if count >= max_frames:
            break
        left_foot = foot_contact[i, 0]
        right_foot = foot_contact[i, 1]
        left_str = "接触" if left_foot else "离地"
        right_str = "接触" if right_foot else "离地"
        time_s = i / fps
        print(f"{i:<6} {time_s:<10.2f} {left_str:<8} {right_str:<8}")
        count += 1


if __name__ == "__main__":
    pkl_path = "/home/galbot/WorkSpace/TWIST2/assets/example_motions/251215-083327.pkl"
    read_foot_contact(pkl_path, interval=50, max_frames=10)
