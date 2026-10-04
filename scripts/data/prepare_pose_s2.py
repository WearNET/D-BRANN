
"""
Prepare Pose-S2 data for five fully distributed branches.

Each branch input:
    root IMU (12) + local IMU (12) + local Pose-S1 leaf (3) = 27

Branches and outputs:
    left_leg    -> joints [1, 4, 7, 10]        -> 12
    right_leg   -> joints [2, 5, 8, 11]        -> 12
    trunk_head  -> joints [3, 6, 9, 12, 15]    -> 15
    left_arm    -> joints [13, 16, 18, 20, 22] -> 15
    right_arm   -> joints [14, 17, 19, 21, 23] -> 15
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
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
from tqdm import tqdm



LEFT_ARM_SENSOR_IDX = 0
RIGHT_ARM_SENSOR_IDX = 1
LEFT_LEG_SENSOR_IDX = 2
RIGHT_LEG_SENSOR_IDX = 3
HEAD_SENSOR_IDX = 4
ROOT_SENSOR_IDX = 5

TARGET_CONFIG: Dict[str, Dict[str, object]] = {
    "left_leg": {
        "local_sensor_idx": LEFT_LEG_SENSOR_IDX,
        "leaf_idx": 0,
        "joints": [1, 4, 7, 10],
        "output_dim": 12,
    },
    "right_leg": {
        "local_sensor_idx": RIGHT_LEG_SENSOR_IDX,
        "leaf_idx": 1,
        "joints": [2, 5, 8, 11],
        "output_dim": 12,
    },
    "trunk_head": {
        "local_sensor_idx": HEAD_SENSOR_IDX,
        "leaf_idx": 2,
        "joints": [3, 6, 9, 12, 15],
        "output_dim": 15,
    },
    "left_arm": {
        "local_sensor_idx": LEFT_ARM_SENSOR_IDX,
        "leaf_idx": 3,
        "joints": [13, 16, 18, 20, 22],
        "output_dim": 15,
    },
    "right_arm": {
        "local_sensor_idx": RIGHT_ARM_SENSOR_IDX,
        "leaf_idx": 4,
        "joints": [14, 17, 19, 21, 23],
        "output_dim": 15,
    },
}

TARGET_ORDER = [
    "left_leg",
    "right_leg",
    "trunk_head",
    "left_arm",
    "right_arm",
]

S1_PREDICTION_KEYS = [
    "p_leaf_pred",
    "p_leaf_prediction",
    "pose_s1_pred",
    "leaf_prediction",
    "prediction",
    "pred",
]

POSE_S2_GT_KEYS = [
    "full_joint_position_gt",
    "p_all_gt_flat",
    "p_all_root_relative_flat",
]


def load_list(list_file: str) -> List[str]:
    path = Path(list_file)
    if not path.exists():
        raise FileNotFoundError(f"List file not found: {list_file}")

    entries = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    existing = [entry for entry in entries if os.path.exists(entry)]
    missing = len(entries) - len(existing)

    if missing:
        print(f"[warning] Ignored {missing} missing paths from {list_file}")

    return existing


def normalize_source_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(path)))


def build_source_map(files: Sequence[str], label: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}

    for path in tqdm(files, desc=f"Indexing {label}", ncols=120):
        data = torch.load(path, map_location="cpu", weights_only=False)

        if not isinstance(data, Mapping):
            raise TypeError(f"Expected a dict-like object in {path}")
        if "source_file" not in data:
            raise KeyError(f"'source_file' missing in {path}")

        source = normalize_source_path(str(data["source_file"]))

        if source in mapping:
            raise RuntimeError(
                f"Duplicate source_file in {label}: {source}"
            )

        mapping[source] = path

    return mapping


def pair_files(
    raw_files: Sequence[str],
    s1_prediction_files: Sequence[str],
    pose_s2_gt_files: Sequence[str],
) -> List[Dict[str, str]]:
    s1_map = build_source_map(s1_prediction_files, "Pose-S1 predictions")
    gt_map = build_source_map(pose_s2_gt_files, "Pose-S2 GT")

    pairs: List[Dict[str, str]] = []

    for raw_path in raw_files:
        source = normalize_source_path(raw_path)

        if source in s1_map and source in gt_map:
            pairs.append(
                {
                    "source_file": raw_path,
                    "s1_prediction_file": s1_map[source],
                    "pose_s2_gt_file": gt_map[source],
                }
            )

    if not pairs:
        raise RuntimeError(
            "No matching source_file values were found among the raw, "
            "Pose-S1 prediction, and Pose-S2 GT lists."
        )

    print(
        f"[pairing] raw={len(raw_files)} | "
        f"s1={len(s1_prediction_files)} | "
        f"gt={len(pose_s2_gt_files)} | "
        f"matched={len(pairs)}"
    )
    return pairs


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


def find_tensor(
    data: Mapping[str, object],
    candidate_keys: Sequence[str],
    expected_last_dim: int,
    description: str,
) -> torch.Tensor:
    for key in candidate_keys:
        value = data.get(key)

        if isinstance(value, torch.Tensor):
            tensor = value.float()

            if (
                tensor.dim() == 3
                and tensor.shape[-1] == 3
                and tensor.shape[-2] * 3 == expected_last_dim
            ):
                tensor = tensor.reshape(tensor.shape[0], expected_last_dim)

            if tensor.dim() == 2 and tensor.shape[1] == expected_last_dim:
                return tensor.contiguous()

    candidates = []

    for key, value in data.items():
        if not isinstance(value, torch.Tensor):
            continue

        if value.dim() == 2 and value.shape[1] == expected_last_dim:
            candidates.append((key, value))
        elif (
            value.dim() == 3
            and value.shape[-1] == 3
            and value.shape[-2] * 3 == expected_last_dim
        ):
            candidates.append((key, value.reshape(value.shape[0], -1)))

    if len(candidates) == 1:
        key, tensor = candidates[0]
        print(f"[info] Using inferred {description} key: {key}")
        return tensor.float().contiguous()

    available = {
        key: tuple(value.shape)
        for key, value in data.items()
        if isinstance(value, torch.Tensor)
    }

    raise KeyError(
        f"Could not identify {description} with dimension "
        f"{expected_last_dim}. Tensor keys: {available}"
    )


def build_two_imu_features(
    acceleration: torch.Tensor,
    orientation: torch.Tensor,
    local_sensor_idx: int,
) -> torch.Tensor:
    root_acc = acceleration[:, ROOT_SENSOR_IDX, :]
    root_ori = orientation[:, ROOT_SENSOR_IDX].reshape(-1, 9)
    local_acc = acceleration[:, local_sensor_idx, :]
    local_ori = orientation[:, local_sensor_idx].reshape(-1, 9)

    return torch.cat(
        [root_acc, root_ori, local_acc, local_ori],
        dim=1,
    ).contiguous()


def extract_joint_target(
    full_joint_position_gt: torch.Tensor,
    joints: Sequence[int],
) -> torch.Tensor:
    columns: List[int] = []

    for joint in joints:
        if not 1 <= joint <= 23:
            raise ValueError(
                f"Pose-S2 target joint must be in [1, 23], got {joint}"
            )

        start = (joint - 1) * 3
        columns.extend([start, start + 1, start + 2])

    return full_joint_position_gt[:, columns].contiguous()


def validate_joint_partition() -> None:
    joints: List[int] = []

    for target in TARGET_ORDER:
        joints.extend(TARGET_CONFIG[target]["joints"])

    if sorted(joints) != list(range(1, 24)):
        raise RuntimeError(
            "The five branches must cover joints 1..23 exactly once. "
            f"Current joints: {sorted(joints)}"
        )


def make_output_name(source_file: str) -> str:
    normalized = normalize_source_path(source_file)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    safe_stem = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in Path(source_file).stem
    )
    return f"{safe_stem}_{digest}_pose_s2_full_distributed.pt"


def validate_finite(name: str, tensor: torch.Tensor, source_file: str) -> None:
    if not torch.isfinite(tensor).all():
        bad_count = int((~torch.isfinite(tensor)).sum().item())
        raise ValueError(
            f"{name} contains {bad_count} NaN/Inf values for {source_file}"
        )


def process_one(pair: Mapping[str, str], output_dir: str) -> str:
    raw = torch.load(
        pair["source_file"],
        map_location="cpu",
        weights_only=False,
    )
    s1_data = torch.load(
        pair["s1_prediction_file"],
        map_location="cpu",
        weights_only=False,
    )
    gt_data = torch.load(
        pair["pose_s2_gt_file"],
        map_location="cpu",
        weights_only=False,
    )

    if "acc" not in raw or "ori" not in raw:
        raise KeyError(
            f"Raw file must contain 'acc' and 'ori': {pair['source_file']}"
        )

    acceleration = ensure_acc_6x3(raw["acc"])
    orientation = ensure_ori_6x3x3(raw["ori"])

    p_leaf_pred = find_tensor(
        s1_data,
        S1_PREDICTION_KEYS,
        expected_last_dim=15,
        description="Pose-S1 prediction",
    ).reshape(-1, 5, 3)

    full_gt = find_tensor(
        gt_data,
        POSE_S2_GT_KEYS,
        expected_last_dim=69,
        description="Pose-S2 full joint-position GT",
    )

    sequence_length = min(
        acceleration.shape[0],
        orientation.shape[0],
        p_leaf_pred.shape[0],
        full_gt.shape[0],
    )

    if sequence_length < 1:
        raise ValueError(f"Empty sequence: {pair['source_file']}")

    acceleration = acceleration[:sequence_length]
    orientation = orientation[:sequence_length]
    p_leaf_pred = p_leaf_pred[:sequence_length]
    full_gt = full_gt[:sequence_length]

    output: Dict[str, object] = {
        "source_file": pair["source_file"],
        "s1_prediction_file": pair["s1_prediction_file"],
        "pose_s2_gt_file": pair["pose_s2_gt_file"],
        "sequence_length": sequence_length,
        "input_dim": 27,
        "target_order": TARGET_ORDER,
        "target_config": TARGET_CONFIG,
        "full_joint_position_gt": full_gt,
    }

    for target in TARGET_ORDER:
        config = TARGET_CONFIG[target]

        imu_features = build_two_imu_features(
            acceleration,
            orientation,
            int(config["local_sensor_idx"]),
        )
        local_leaf = p_leaf_pred[:, int(config["leaf_idx"]), :]

        branch_input = torch.cat(
            [imu_features, local_leaf],
            dim=1,
        ).contiguous()
        branch_target = extract_joint_target(
            full_gt,
            config["joints"],
        )

        expected_output_dim = int(config["output_dim"])

        if tuple(branch_input.shape) != (sequence_length, 27):
            raise RuntimeError(
                f"Unexpected {target}_input shape: "
                f"{tuple(branch_input.shape)}"
            )
        if tuple(branch_target.shape) != (
            sequence_length,
            expected_output_dim,
        ):
            raise RuntimeError(
                f"Unexpected {target}_target shape: "
                f"{tuple(branch_target.shape)}"
            )

        validate_finite(
            f"{target}_input",
            branch_input,
            pair["source_file"],
        )
        validate_finite(
            f"{target}_target",
            branch_target,
            pair["source_file"],
        )

        output[f"{target}_input"] = branch_input
        output[f"{target}_target"] = branch_target

    destination = os.path.join(
        output_dir,
        make_output_name(pair["source_file"]),
    )
    torch.save(output, destination)

    return destination


def print_sample(path: str) -> None:
    sample = torch.load(path, map_location="cpu", weights_only=False)

    print("\nSample prepared file")
    print("  source_file:", sample["source_file"])
    print("  sequence_length:", sample["sequence_length"])

    for target in TARGET_ORDER:
        print(
            f"  {target:>10}_input:  "
            f"{tuple(sample[f'{target}_input'].shape)}"
        )
        print(
            f"  {target:>10}_target: "
            f"{tuple(sample[f'{target}_target'].shape)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--raw_list_file", required=True)
    parser.add_argument("--s1_pred_list_file", required=True)
    parser.add_argument("--gt_list_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_list_file", required=True)
    parser.add_argument(
        "--skip_errors",
        action="store_true",
        help="Skip invalid sequences instead of stopping.",
    )

    args = parser.parse_args()

    validate_joint_partition()
    os.makedirs(args.output_dir, exist_ok=True)

    raw_files = load_list(args.raw_list_file)
    s1_files = load_list(args.s1_pred_list_file)
    gt_files = load_list(args.gt_list_file)

    pairs = pair_files(raw_files, s1_files, gt_files)

    saved_paths: List[str] = []
    skipped: List[Tuple[str, str]] = []

    for pair in tqdm(
        pairs,
        desc="Preparing five-branch Pose-S2 data",
        ncols=120,
    ):
        try:
            saved_paths.append(
                process_one(pair, args.output_dir)
            )
        except Exception as error:
            if not args.skip_errors:
                raise
            skipped.append(
                (pair["source_file"], repr(error))
            )

    output_parent = os.path.dirname(args.output_list_file)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    with open(args.output_list_file, "w", encoding="utf-8") as file:
        for path in saved_paths:
            file.write(path + "\n")

    print("\nPreparation complete")
    print("  Matched:", len(pairs))
    print("  Saved:", len(saved_paths))
    print("  Skipped:", len(skipped))
    print("  Output list:", args.output_list_file)

    if skipped:
        print("\nSkipped sequences")
        for source, reason in skipped[:20]:
            print(" ", source)
            print("   ", reason)

    if not saved_paths:
        raise RuntimeError("No prepared files were generated.")

    print_sample(saved_paths[0])


if __name__ == "__main__":
    main()
