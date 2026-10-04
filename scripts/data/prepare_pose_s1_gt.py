
# BEGIN D-BRAN PROJECT BOOTSTRAP
import sys as _dbran_sys
from pathlib import Path as _DbranPath

_dbran_current_file = _DbranPath(__file__).resolve()

for _dbran_candidate in (
    _dbran_current_file.parent,
    *_dbran_current_file.parents,
):
    if (_dbran_candidate / "main_path.py").is_file():
        _dbran_root_string = str(_dbran_candidate)

        if _dbran_root_string not in _dbran_sys.path:
            _dbran_sys.path.insert(0, _dbran_root_string)

        break
else:
    raise RuntimeError(
        "Could not locate the D-BRAN project root from "
        f"{_dbran_current_file}"
    )

from main_path import PROJECT_ROOT
# END D-BRAN PROJECT BOOTSTRAP

import os
import hashlib
import argparse
import torch
from tqdm import tqdm


# ------------------------------------------------------------
# Default joint indices
# ------------------------------------------------------------
ROOT_JOINT_IDX = 0

LEFT_LEG_LEAF_JOINT_IDX = 7
RIGHT_LEG_LEAF_JOINT_IDX = 8
HEAD_LEAF_JOINT_IDX = 12
LEFT_ARM_LEAF_JOINT_IDX = 20
RIGHT_ARM_LEAF_JOINT_IDX = 21

LEAF_ORDER = [
    ("left_leg", LEFT_LEG_LEAF_JOINT_IDX),
    ("right_leg", RIGHT_LEG_LEAF_JOINT_IDX),
    ("head", HEAD_LEAF_JOINT_IDX),
    ("left_arm", LEFT_ARM_LEAF_JOINT_IDX),
    ("right_arm", RIGHT_ARM_LEAF_JOINT_IDX),
]


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def load_list(list_file: str):
    with open(list_file, "r") as f:
        files = [line.strip() for line in f if line.strip()]
    files = [p for p in files if os.path.exists(p)]
    return files


def make_safe_name(src_path: str) -> str:
    stem = os.path.splitext(os.path.basename(src_path))[0]
    digest = hashlib.md5(src_path.encode("utf-8")).hexdigest()[:10]
    return f"{stem}_{digest}"


def ensure_joints_24x3(joints: torch.Tensor) -> torch.Tensor:
    if joints.dim() != 3 or joints.shape[1:] != (24, 3):
        raise ValueError(f"Expected joints with shape (T, 24, 3), got {tuple(joints.shape)}")
    return joints.float()


# ------------------------------------------------------------
# Main processing
# ------------------------------------------------------------
def process_one_file(src_path: str, output_dir: str):
    raw = torch.load(src_path, weights_only=False)

    if "joints" not in raw:
        raise KeyError(f"'joints' missing in raw file: {src_path}")

    joints = ensure_joints_24x3(raw["joints"])             # (T, 24, 3)
    root_pos = joints[:, ROOT_JOINT_IDX, :]                # (T, 3)

    leaf_targets = {}
    leaf_concat = []

    for leaf_name, leaf_idx in LEAF_ORDER:
        p_leaf_root_relative = joints[:, leaf_idx, :] - root_pos
        leaf_targets[leaf_name] = p_leaf_root_relative
        leaf_concat.append(p_leaf_root_relative)

    p_leaf_gt = torch.cat(leaf_concat, dim=1)              # (T, 15)

    p_all_root_relative = joints[:, 1:, :] - root_pos.unsqueeze(1)   # (T, 23, 3)
    p_all_gt_flat = p_all_root_relative.reshape(joints.shape[0], -1) # (T, 69)

    out = {
        "source_file": src_path,
        "num_frames": joints.shape[0],
        "root_joint_idx": ROOT_JOINT_IDX,
        "leaf_order": [name for name, _ in LEAF_ORDER],
        "leaf_joint_indices": [idx for _, idx in LEAF_ORDER],

        "p_leaf_gt": p_leaf_gt,                       # (T, 15)
        "p_all_root_relative": p_all_root_relative,  # (T, 23, 3)
        "p_all_gt_flat": p_all_gt_flat,              # (T, 69)

        "left_leg_p_gt": leaf_targets["left_leg"],    # (T, 3)
        "right_leg_p_gt": leaf_targets["right_leg"],  # (T, 3)
        "head_p_gt": leaf_targets["head"],            # (T, 3)
        "left_arm_p_gt": leaf_targets["left_arm"],    # (T, 3)
        "right_arm_p_gt": leaf_targets["right_arm"],  # (T, 3)
    }

    safe_name = make_safe_name(src_path)
    dst_path = os.path.join(output_dir, f"{safe_name}_pose_s1_gt.pt")
    torch.save(out, dst_path)

    return dst_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_list_file",
        type=str,
        required=True,
        help="Path to the raw source list file, e.g. train_pose.txt or test.txt"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where Pose-S1 GT files will be saved"
    )
    parser.add_argument(
        "--output_list_file",
        type=str,
        required=True,
        help="Path to the output manifest txt"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    raw_files = load_list(args.raw_list_file)
    if len(raw_files) == 0:
        raise RuntimeError(f"No existing files found in {args.raw_list_file}")

    print(f"[info] Found {len(raw_files)} raw files")
    print(f"[info] Output directory: {args.output_dir}")

    saved_paths = []
    for src_path in tqdm(raw_files, desc="Preprocessing Pose-S1 GT", ncols=120):
        dst_path = process_one_file(
            src_path=src_path,
            output_dir=args.output_dir
        )
        saved_paths.append(dst_path)

    with open(args.output_list_file, "w") as f:
        for p in saved_paths:
            f.write(p + "\n")

    print(f"[done] Saved {len(saved_paths)} Pose-S1 GT files")
    print(f"[done] Output list file: {args.output_list_file}")


if __name__ == "__main__":
    main()