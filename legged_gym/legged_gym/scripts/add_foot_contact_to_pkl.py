#!/usr/bin/env python3
import argparse
import os
import pickle
import sys
from types import ModuleType
from typing import List, Optional, Tuple, Any


LEFT_CANDIDATES = [
    "left_ankle_roll_link",
    "left_ankle_link",
    "left_foot",
    "left_toe",
    "l_ankle",
    "l_foot",
]
RIGHT_CANDIDATES = [
    "right_ankle_roll_link",
    "right_ankle_link",
    "right_foot",
    "right_toe",
    "r_ankle",
    "r_foot",
]


class _FakeModule(ModuleType):
    def __init__(self, name: str, real: Optional[ModuleType] = None):
        super().__init__(name)
        if real is not None:
            self.__dict__.update(real.__dict__)


def _patch_numpy_pickle_compat() -> None:
    import numpy as np

    core_mod = np.core if hasattr(np, "core") else np
    multiarray_mod = getattr(core_mod, "multiarray", None)
    sys.modules["numpy._core"] = _FakeModule("numpy._core", core_mod)
    sys.modules["numpy._core.multiarray"] = _FakeModule("numpy._core.multiarray", multiarray_mod)


def _safe_pickle_load(path: str) -> Any:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as e:
        # Old pickles may reference numpy._core on newer numpy installs.
        if "numpy._core" not in str(e):
            raise
        _patch_numpy_pickle_compat()
        with open(path, "rb") as f:
            return pickle.load(f)


def _find_link_idx(link_names: List[str], candidates: List[str]) -> Optional[int]:
    for name in candidates:
        if name in link_names:
            return link_names.index(name)
    lower_names = [n.lower() for n in link_names]
    for cand in candidates:
        cand = cand.lower()
        for i, lname in enumerate(lower_names):
            if cand in lname:
                return i
    return None


def _resolve_input_paths(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        out = []
        for root, _, files in os.walk(input_path):
            for f in files:
                if f.endswith(".pkl"):
                    out.append(os.path.join(root, f))
        return sorted(out)

    if input_path.endswith(".yaml"):
        try:
            import yaml  # type: ignore
            with open(input_path, "r") as f:
                cfg = yaml.safe_load(f)
            root_path = cfg["root_path"]
            motions = cfg["motions"]
            return [os.path.join(root_path, m["file"]) for m in motions]
        except ModuleNotFoundError:
            return _parse_motion_yaml_no_deps(input_path)

    if input_path.endswith(".pkl"):
        return [input_path]

    raise ValueError(f"Unsupported input path: {input_path}")


def _parse_motion_yaml_no_deps(yaml_path: str) -> List[str]:
    root_path = None
    rel_files: List[str] = []
    with open(yaml_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("root_path:"):
                root_path = line.split(":", 1)[1].strip().strip("'\"")
                continue
            if line.startswith("- file:"):
                rel = line.split(":", 1)[1].strip().strip("'\"")
                rel_files.append(rel)

    if root_path is None:
        raise ValueError(f"Cannot parse `root_path` from yaml: {yaml_path}")

    return [os.path.join(root_path, rel) for rel in rel_files]


def _compute_foot_contact(
    root_pos: Any,
    local_body_pos: Any,
    link_body_list: List[str],
    height_eps: float,
    floor_quantile: float,
) -> Any:
    import numpy as np

    left_idx = _find_link_idx(link_body_list, LEFT_CANDIDATES)
    right_idx = _find_link_idx(link_body_list, RIGHT_CANDIDATES)
    if left_idx is None or right_idx is None:
        raise ValueError("Cannot find left/right foot links from link_body_list")

    feet_z = root_pos[:, 2:3] + local_body_pos[:, [left_idx, right_idx], 2]
    floor_z = np.quantile(feet_z.reshape(-1), floor_quantile)
    foot_contact = feet_z <= (floor_z + height_eps)
    return foot_contact.astype(np.bool_)


def _process_one_file(
    pkl_path: str,
    height_eps: float,
    floor_quantile: float,
    overwrite: bool,
) -> Tuple[bool, str]:
    try:
        import numpy as np
    except Exception:
        return False, "[skip] numpy is required to process pkl motion files"

    if not os.path.exists(pkl_path):
        return False, f"[missing] {pkl_path}"

    data = _safe_pickle_load(pkl_path)

    if not isinstance(data, dict):
        return False, f"[skip] not dict: {pkl_path}"

    if "foot_contact" in data and not overwrite:
        return True, f"[skip-exists] {pkl_path}"

    required_keys = ["root_pos", "local_body_pos", "link_body_list"]
    for k in required_keys:
        if k not in data:
            return False, f"[skip] missing key `{k}`: {pkl_path}"

    root_pos = np.asarray(data["root_pos"])
    local_body_pos = np.asarray(data["local_body_pos"])
    link_body_list = list(data["link_body_list"])

    try:
        foot_contact = _compute_foot_contact(
            root_pos=root_pos,
            local_body_pos=local_body_pos,
            link_body_list=link_body_list,
            height_eps=height_eps,
            floor_quantile=floor_quantile,
        )
    except Exception as e:
        return False, f"[skip] {pkl_path}: {e}"

    if foot_contact.shape[0] != root_pos.shape[0]:
        return False, f"[skip] frame mismatch in {pkl_path}"

    data["foot_contact"] = foot_contact
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)
    return True, f"[ok] {pkl_path}"


def main():
    parser = argparse.ArgumentParser(
        description="Add `foot_contact` key to motion PKL files."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input path: a .pkl file, a .yaml motion list, or a directory.",
    )
    parser.add_argument(
        "--height-eps",
        type=float,
        default=0.03,
        help="Contact threshold above estimated floor height (meters).",
    )
    parser.add_argument(
        "--floor-quantile",
        type=float,
        default=0.02,
        help="Quantile used to estimate floor height from foot z values.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing `foot_contact` key.",
    )
    args = parser.parse_args()

    pkl_paths = _resolve_input_paths(args.input)
    if len(pkl_paths) == 0:
        print("No pkl files found.")
        return

    ok_cnt = 0
    fail_cnt = 0
    for p in pkl_paths:
        ok, msg = _process_one_file(
            p,
            height_eps=args.height_eps,
            floor_quantile=args.floor_quantile,
            overwrite=args.overwrite,
        )
        print(msg)
        if ok:
            ok_cnt += 1
        else:
            fail_cnt += 1

    print(f"Done. success={ok_cnt}, failed={fail_cnt}, total={len(pkl_paths)}")


if __name__ == "__main__":
    main()
