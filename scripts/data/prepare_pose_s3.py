"""
Prepare Pose-S3 data for the five-branch PoseS3 + learned fusion design.

Expected input:
    - Raw AMASS/DIP/TotalCapture style files from train_pose.txt/test.txt.
    - PoseS2 full-position prediction files with [T, 69] predictions.

PoseS3 branch design:
    left_leg    input: root+left_leg IMUs + PoseS2 joints [1,4,7,10]      -> 36 dims
                target rotations: reduced joints [1,4]                   -> 12 dims

    right_leg   input: root+right_leg IMUs + PoseS2 joints [2,5,8,11]     -> 36 dims
                target rotations: reduced joints [2,5]                   -> 12 dims

    trunk_head  input: root+head IMUs + PoseS2 joints [3,6,9,12,15]       -> 39 dims
                target rotations: reduced joints [3,6,9,12,15]           -> 30 dims

    left_arm    input: root+left_arm IMUs + PoseS2 joints [13,16,18,20,22]-> 39 dims
                target rotations: reduced joints [13,16,18]              -> 18 dims

    right_arm   input: root+right_arm IMUs + PoseS2 joints [14,17,19,21,23]-> 39 dims
                target rotations: reduced joints [14,17,19]              -> 18 dims

The five branch targets cover joint_set.reduced exactly once.
"""

from __future__ import annotations

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


import argparse
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import torch
from tqdm import tqdm



import articulate as art
from config import paths, joint_set


ROOT_SENSOR_IDX = 5
LEFT_ARM_SENSOR_IDX = 0
RIGHT_ARM_SENSOR_IDX = 1
LEFT_LEG_SENSOR_IDX = 2
RIGHT_LEG_SENSOR_IDX = 3
HEAD_SENSOR_IDX = 4

REDUCED_JOINTS = list(joint_set.reduced)

BRANCH_CONFIG: Dict[str, Dict[str, object]] = {
    "left_leg": {
        "sensor_indices": [ROOT_SENSOR_IDX, LEFT_LEG_SENSOR_IDX],
        "position_joints": [1, 4, 7, 10],
        "reduced_joints": [1, 4],
        "input_dim": 36,
        "output_dim": 12,
    },
    "right_leg": {
        "sensor_indices": [ROOT_SENSOR_IDX, RIGHT_LEG_SENSOR_IDX],
        "position_joints": [2, 5, 8, 11],
        "reduced_joints": [2, 5],
        "input_dim": 36,
        "output_dim": 12,
    },
    "trunk_head": {
        "sensor_indices": [ROOT_SENSOR_IDX, HEAD_SENSOR_IDX],
        "position_joints": [3, 6, 9, 12, 15],
        "reduced_joints": [3, 6, 9, 12, 15],
        "input_dim": 39,
        "output_dim": 30,
    },
    "left_arm": {
        "sensor_indices": [ROOT_SENSOR_IDX, LEFT_ARM_SENSOR_IDX],
        "position_joints": [13, 16, 18, 20, 22],
        "reduced_joints": [13, 16, 18],
        "input_dim": 39,
        "output_dim": 18,
    },
    "right_arm": {
        "sensor_indices": [ROOT_SENSOR_IDX, RIGHT_ARM_SENSOR_IDX],
        "position_joints": [14, 17, 19, 21, 23],
        "reduced_joints": [14, 17, 19],
        "input_dim": 39,
        "output_dim": 18,
    },
}

BRANCH_ORDER = [
    "left_leg",
    "right_leg",
    "trunk_head",
    "left_arm",
    "right_arm",
]

POSE_S2_POSITION_KEYS = [
    "full_joint_position_pred",
    "full_joint_position_assembled",
    "pose_s2_full_position",
    "p_full_pred",
    "pred",
]


def load_list(list_file: str) -> List[str]:
    path = Path(list_file)

    if not path.exists():
        raise FileNotFoundError(f"List file not found: {list_file}")

    files = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    existing = [p for p in files if os.path.exists(p)]
    missing = len(files) - len(existing)

    if missing:
        print(f"[warning] Ignored {missing} missing paths from {list_file}")

    return existing


def normalize_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(path)))


def build_source_map(files: Sequence[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}

    for path in tqdm(files, desc="Indexing PoseS2 predictions", ncols=120):
        data = torch.load(path, map_location="cpu", weights_only=False)

        if "source_file" not in data:
            raise KeyError(f"'source_file' missing in {path}")

        source = normalize_path(str(data["source_file"]))

        if source in mapping:
            raise RuntimeError(
                f"Duplicate source_file in PoseS2 prediction list: {source}"
            )

        mapping[source] = path

    return mapping


def ensure_acc_6x3(acceleration: torch.Tensor) -> torch.Tensor:
    if acceleration.dim() == 3 and tuple(acceleration.shape[1:]) == (6, 3):
        return acceleration.float().contiguous()

    if acceleration.dim() == 2 and acceleration.shape[1] == 18:
        return acceleration.reshape(-1, 6, 3).float().contiguous()

    raise ValueError(
        f"Unsupported acceleration shape: {tuple(acceleration.shape)}"
    )


def ensure_ori_6x3x3(orientation: torch.Tensor) -> torch.Tensor:
    if orientation.dim() == 4 and tuple(orientation.shape[1:]) == (6, 3, 3):
        return orientation.float().contiguous()

    if orientation.dim() == 3 and tuple(orientation.shape[1:]) == (6, 9):
        return orientation.reshape(-1, 6, 3, 3).float().contiguous()

    if orientation.dim() == 2 and orientation.shape[1] == 54:
        return orientation.reshape(-1, 6, 3, 3).float().contiguous()

    raise ValueError(
        f"Unsupported orientation shape: {tuple(orientation.shape)}"
    )


def get_pose_s2_full_position(pred_data: Mapping[str, object]) -> torch.Tensor:
    for key in POSE_S2_POSITION_KEYS:
        value = pred_data.get(key)

        if isinstance(value, torch.Tensor):
            tensor = value.float()

            if tensor.dim() == 3 and tuple(tensor.shape[1:]) == (23, 3):
                return tensor.reshape(tensor.shape[0], 69).contiguous()

            if tensor.dim() == 2 and tensor.shape[1] == 69:
                return tensor.contiguous()

    available = {
        key: tuple(value.shape)
        for key, value in pred_data.items()
        if isinstance(value, torch.Tensor)
    }
    raise KeyError(
        "Could not find PoseS2 full position prediction [T,69]. "
        f"Available tensor keys: {available}"
    )


def build_multi_imu_input(
    acceleration: torch.Tensor,
    orientation: torch.Tensor,
    sensor_indices: Sequence[int],
) -> torch.Tensor:
    features = []

    for sensor_idx in sensor_indices:
        features.append(acceleration[:, sensor_idx, :])
        features.append(orientation[:, sensor_idx].reshape(orientation.shape[0], 9))

    return torch.cat(features, dim=1).float().contiguous()


def full_position_subset(
    full_position: torch.Tensor,
    joint_indices: Sequence[int],
) -> torch.Tensor:
    parts = []

    for joint_idx in joint_indices:
        if joint_idx == 0:
            raise ValueError("PoseS2 full position input does not include root joint 0.")

        start = (joint_idx - 1) * 3
        parts.append(full_position[:, start:start + 3])

    return torch.cat(parts, dim=1).float().contiguous()


def rotmat_to_6d(rotation_matrix: torch.Tensor) -> torch.Tensor:
    first_two_cols = rotation_matrix[..., :, :2]
    return (
        first_two_cols.transpose(-1, -2)
        .reshape(rotation_matrix.shape[:-2] + (6,))
        .float()
    )


def compute_reduced_root_relative_global_6d(
    raw_data: Mapping[str, object],
    body_model,
) -> torch.Tensor:
    pose = raw_data["pose"].float()

    if pose.dim() == 2 and pose.shape[1] == 72:
        pose = pose.reshape(pose.shape[0], 24, 3)

    if pose.dim() != 3 or tuple(pose.shape[1:]) != (24, 3):
        raise ValueError(
            f"Expected pose shape [T,24,3] or [T,72], got {tuple(pose.shape)}"
        )

    pose_rotmat = art.math.axis_angle_to_rotation_matrix(pose).reshape(-1, 24, 3, 3)
    global_rotmat, _ = body_model.forward_kinematics(
        pose_rotmat,
        calc_mesh=False,
    )

    root_rotmat = global_rotmat[:, 0].unsqueeze(1)
    root_inverse = root_rotmat.transpose(-1, -2)
    root_relative_global = torch.matmul(root_inverse, global_rotmat)

    reduced = root_relative_global[:, REDUCED_JOINTS]
    return rotmat_to_6d(reduced).reshape(
        reduced.shape[0],
        len(REDUCED_JOINTS) * 6,
    ).float().contiguous()


def target_branch_from_reduced(
    target_reduced_6d: torch.Tensor,
    reduced_joints: Sequence[int],
) -> torch.Tensor:
    target = target_reduced_6d.reshape(
        target_reduced_6d.shape[0],
        len(REDUCED_JOINTS),
        6,
    )
    parts = []

    for joint_idx in reduced_joints:
        reduced_index = REDUCED_JOINTS.index(joint_idx)
        parts.append(target[:, reduced_index, :])

    return torch.cat(parts, dim=1).float().contiguous()


def validate_branch_partition() -> None:
    predicted_joints = []

    for branch_name in BRANCH_ORDER:
        predicted_joints.extend(BRANCH_CONFIG[branch_name]["reduced_joints"])

    if sorted(predicted_joints) != sorted(REDUCED_JOINTS):
        raise RuntimeError(
            "PoseS3 five-branch reduced joints must cover joint_set.reduced exactly once. "
            f"Expected: {sorted(REDUCED_JOINTS)} | Got: {sorted(predicted_joints)}"
        )


def make_output_name(source_file: str) -> str:
    normalized = normalize_path(source_file)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    stem = Path(source_file).stem
    safe_stem = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in stem
    )
    return f"{safe_stem}_{digest}_pose_s3_five_branch.pt"


def process_one(
    raw_path: str,
    pose_s2_path: str,
    body_model,
    output_dir: str,
) -> str:
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    pose_s2_data = torch.load(pose_s2_path, map_location="cpu", weights_only=False)

    acceleration = ensure_acc_6x3(raw["acc"])
    orientation = ensure_ori_6x3x3(raw["ori"])
    pose_s2_full = get_pose_s2_full_position(pose_s2_data)
    target_reduced_6d = compute_reduced_root_relative_global_6d(
        raw,
        body_model,
    )

    sequence_length = min(
        acceleration.shape[0],
        orientation.shape[0],
        pose_s2_full.shape[0],
        target_reduced_6d.shape[0],
    )

    acceleration = acceleration[:sequence_length]
    orientation = orientation[:sequence_length]
    pose_s2_full = pose_s2_full[:sequence_length]
    target_reduced_6d = target_reduced_6d[:sequence_length]

    item: Dict[str, object] = {
        "source_file": raw_path,
        "pose_s2_source_file": pose_s2_path,
        "num_frames": sequence_length,
        "pose_s2_full_position_input": pose_s2_full,
        "pose3_target_6d_reduced": target_reduced_6d,
        "reduced_joints": REDUCED_JOINTS,
        "branch_order": BRANCH_ORDER,
        "branch_config": BRANCH_CONFIG,
        "rotation_target_format": "root_relative_global_6d_reduced",
    }

    for branch_name in BRANCH_ORDER:
        config = BRANCH_CONFIG[branch_name]

        imu_features = build_multi_imu_input(
            acceleration,
            orientation,
            config["sensor_indices"],
        )
        position_features = full_position_subset(
            pose_s2_full,
            config["position_joints"],
        )
        branch_input = torch.cat(
            [imu_features, position_features],
            dim=1,
        ).float().contiguous()
        branch_target = target_branch_from_reduced(
            target_reduced_6d,
            config["reduced_joints"],
        )

        expected_input_dim = int(config["input_dim"])
        expected_output_dim = int(config["output_dim"])

        if branch_input.shape[1] != expected_input_dim:
            raise RuntimeError(
                f"{branch_name}_input expected dim {expected_input_dim}, "
                f"got {branch_input.shape[1]}"
            )
        if branch_target.shape[1] != expected_output_dim:
            raise RuntimeError(
                f"{branch_name}_target expected dim {expected_output_dim}, "
                f"got {branch_target.shape[1]}"
            )

        item[f"{branch_name}_imu_input"] = imu_features
        item[f"{branch_name}_position_input"] = position_features
        item[f"{branch_name}_input"] = branch_input
        item[f"{branch_name}_target_6d_reduced"] = branch_target
        item[f"{branch_name}_position_joints"] = config["position_joints"]
        item[f"{branch_name}_reduced_joints"] = config["reduced_joints"]

    output_path = os.path.join(
        output_dir,
        make_output_name(raw_path),
    )
    torch.save(item, output_path)

    return output_path


def print_sample(path: str) -> None:
    sample = torch.load(path, map_location="cpu", weights_only=False)

    print("\nSample shapes:")
    print("  source_file:", sample["source_file"])
    print("  pose_s2_full_position_input:", tuple(sample["pose_s2_full_position_input"].shape))
    print("  pose3_target_6d_reduced:", tuple(sample["pose3_target_6d_reduced"].shape))

    for branch_name in BRANCH_ORDER:
        print(f"  {branch_name}_input:", tuple(sample[f"{branch_name}_input"].shape))
        print(
            f"  {branch_name}_target_6d_reduced:",
            tuple(sample[f"{branch_name}_target_6d_reduced"].shape),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_list_file", required=True)
    parser.add_argument("--pose_s2_list_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_list_file", required=True)
    parser.add_argument("--skip_errors", action="store_true")

    args = parser.parse_args()

    validate_branch_partition()
    os.makedirs(args.output_dir, exist_ok=True)

    raw_files = load_list(args.raw_list_file)
    pose_s2_files = load_list(args.pose_s2_list_file)
    pose_s2_map = build_source_map(pose_s2_files)

    body_model = art.ParametricModel(
        paths.smpl_file,
        device=torch.device("cpu"),
    )

    saved_paths: List[str] = []
    skipped = []

    for raw_path in tqdm(
        raw_files,
        desc="Preparing PoseS3 five-branch data",
        ncols=120,
    ):
        normalized_raw = normalize_path(raw_path)

        if normalized_raw not in pose_s2_map:
            skipped.append((raw_path, "missing PoseS2 prediction"))
            continue

        try:
            output_path = process_one(
                raw_path=raw_path,
                pose_s2_path=pose_s2_map[normalized_raw],
                body_model=body_model,
                output_dir=args.output_dir,
            )
            saved_paths.append(output_path)
        except Exception as error:
            if not args.skip_errors:
                raise
            skipped.append((raw_path, repr(error)))

    output_parent = os.path.dirname(args.output_list_file)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    with open(args.output_list_file, "w", encoding="utf-8") as file:
        for path in saved_paths:
            file.write(path + "\n")

    print("\nDone.")
    print("  Raw files:", len(raw_files))
    print("  Saved:", len(saved_paths))
    print("  Skipped:", len(skipped))
    print("  Output list:", args.output_list_file)

    if skipped:
        print("\nSkipped examples:")
        for source, reason in skipped[:20]:
            print(" ", source)
            print("   ", reason)

    if saved_paths:
        print_sample(saved_paths[0])


if __name__ == "__main__":
    main()
