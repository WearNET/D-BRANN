"""
Independent full-pipeline profiler for the current distributed architecture.

Compares:
    Original TransPose full pipeline
    vs
    PoseS1 five-branch distributed
    + PoseS2 five-branch full-distributed
    + PoseS3 five-branch distributed
    + PoseS3 learned residual fusion
    + original Trans-B1/B2

It only imports the project modules that are part of the original repo:
    config.py, net.py, utils.py, articulate
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
from main_path import (
    TRANSPOSE_WEIGHTS_FILE,
    POSE_S1_CHECKPOINTS_DIR,
    POSE_S2_CHECKPOINTS_DIR,
    POSE_S3_CHECKPOINTS_DIR,
    FUSION_CHECKPOINT,
)



import os
import sys
import time
import glob
import json
import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm


# ------------------------------------------------------------
# Project path
# ------------------------------------------------------------



# Optional robustness hook. If it is not available, the profiler still works.
try:
    from robustness_full_pipeline_hook import install_fault_injection

    install_fault_injection()
    print("[info] robustness_full_pipeline_hook installed.")
except Exception as exc:
    print(f"[info] robustness_full_pipeline_hook not installed or skipped: {exc}")


from config import paths, joint_set, vel_scale
from net import TransPoseNet
from utils import normalize_and_concat
import articulate as art


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set from --skip_fusion in main(). When True, the PoseS3 learned residual
# fusion module is bypassed (ablation: "w/o residual fusion").
SKIP_FUSION = False


# ------------------------------------------------------------
# Sensor and branch definitions
# ------------------------------------------------------------
ROOT_SENSOR_IDX = 5
LEFT_ARM_SENSOR_IDX = 0
RIGHT_ARM_SENSOR_IDX = 1
LEFT_LEG_SENSOR_IDX = 2
RIGHT_LEG_SENSOR_IDX = 3
HEAD_SENSOR_IDX = 4

LEAF_TARGETS = [
    ("left_leg", LEFT_LEG_SENSOR_IDX),
    ("right_leg", RIGHT_LEG_SENSOR_IDX),
    ("head", HEAD_SENSOR_IDX),
    ("left_arm", LEFT_ARM_SENSOR_IDX),
    ("right_arm", RIGHT_ARM_SENSOR_IDX),
]

LEAF_ORDER = [name for name, _ in LEAF_TARGETS]

POSE_S2_BRANCH_ORDER = [
    "left_leg",
    "right_leg",
    "trunk_head",
    "left_arm",
    "right_arm",
]

POSE_S2_BRANCH_CONFIG: Dict[str, Dict[str, object]] = {
    "left_leg": {
        "local_sensor_idx": LEFT_LEG_SENSOR_IDX,
        "s1_leaf_name": "left_leg",
        "joints": [1, 4, 7, 10],
        "input_dim": 27,
        "output_dim": 12,
    },
    "right_leg": {
        "local_sensor_idx": RIGHT_LEG_SENSOR_IDX,
        "s1_leaf_name": "right_leg",
        "joints": [2, 5, 8, 11],
        "input_dim": 27,
        "output_dim": 12,
    },
    "trunk_head": {
        "local_sensor_idx": HEAD_SENSOR_IDX,
        "s1_leaf_name": "head",
        "joints": [3, 6, 9, 12, 15],
        "input_dim": 27,
        "output_dim": 15,
    },
    "left_arm": {
        "local_sensor_idx": LEFT_ARM_SENSOR_IDX,
        "s1_leaf_name": "left_arm",
        "joints": [13, 16, 18, 20, 22],
        "input_dim": 27,
        "output_dim": 15,
    },
    "right_arm": {
        "local_sensor_idx": RIGHT_ARM_SENSOR_IDX,
        "s1_leaf_name": "right_arm",
        "joints": [14, 17, 19, 21, 23],
        "input_dim": 27,
        "output_dim": 15,
    },
}

POSE_S3_BRANCH_ORDER = [
    "left_leg",
    "right_leg",
    "trunk_head",
    "left_arm",
    "right_arm",
]

POSE_S3_BRANCH_CONFIG: Dict[str, Dict[str, object]] = {
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

REDUCED_JOINTS = [int(x) for x in list(joint_set.reduced)]


# ------------------------------------------------------------
# Model classes
# ------------------------------------------------------------
class DistributedPoseS1(nn.Module):
    def __init__(
        self,
        input_dim=24,
        output_dim=3,
        proj_dim=32,
        rnn_hidden=32,
        rnn_layers=2,
        dropout=0.2,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc_in = nn.Linear(input_dim, proj_dim)
        self.rnn = nn.LSTM(
            input_size=proj_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=False,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(rnn_hidden * 2, output_dim)


class FiveBranchRNN(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        proj_dim=16,
        rnn_hidden=16,
        rnn_layers=2,
        dropout=0.2,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc_in = nn.Linear(input_dim, proj_dim)
        self.rnn = nn.LSTM(
            input_size=proj_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=False,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(rnn_hidden * 2, output_dim)


class FusionNet(nn.Module):
    def __init__(
        self,
        input_dim=90,
        output_dim=90,
        proj_dim=16,
        rnn_hidden=16,
        rnn_layers=1,
        dropout=0.2,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc_in = nn.Linear(input_dim, proj_dim)
        self.rnn = nn.LSTM(
            input_size=proj_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=False,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(rnn_hidden * 2, output_dim)


# ------------------------------------------------------------
# Generic helpers
# ------------------------------------------------------------
def synchronize():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def bytes_to_mb(value):
    return float(value) / (1024.0 ** 2)


def count_params(module):
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def file_size_mb(path):
    if not path or not os.path.exists(path):
        return float("nan")
    return os.path.getsize(path) / (1024.0 ** 2)


def load_list(list_file):
    with open(list_file, "r", encoding="utf-8") as file:
        files = [line.strip() for line in file if line.strip()]
    existing = [path for path in files if os.path.exists(path)]
    missing = len(files) - len(existing)
    if missing:
        print(f"[warning] Ignored {missing} missing paths from {list_file}")
    return existing


def update_peak(old, new):
    if new is None:
        return old
    if old is None:
        return new
    return max(old, new)


def create_streams(names, enabled):
    if DEVICE.type != "cuda" or not enabled:
        return None
    return {name: torch.cuda.Stream(device=DEVICE) for name in names}


def percentile(values, q):
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(torch.quantile(tensor, q))


def remap_state_dict_if_needed(state_dict):
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("fc1."):
            new_key = key.replace("fc1.", "fc_in.", 1)
        elif key.startswith("fc2."):
            new_key = key.replace("fc2.", "fc_out.", 1)
        elif key.startswith("drop."):
            new_key = key.replace("drop.", "dropout.", 1)
        elif key.startswith("linear1."):
            new_key = key.replace("linear1.", "fc_in.", 1)
        elif key.startswith("linear2."):
            new_key = key.replace("linear2.", "fc_out.", 1)
        remapped[new_key] = value
    return remapped


# ------------------------------------------------------------
# Data shape helpers
# ------------------------------------------------------------
def ensure_acc_6x3(acc):
    if acc.dim() == 3 and acc.shape[1:] == (6, 3):
        return acc.float()
    if acc.dim() == 2 and acc.shape[1] == 18:
        return acc.view(acc.shape[0], 6, 3).float()
    raise ValueError(f"Unsupported acc shape: {tuple(acc.shape)}")


def ensure_rotmat_3x3(ori):
    if ori.dim() == 4 and ori.shape[1:] == (6, 3, 3):
        return ori.float()
    if ori.dim() == 2 and ori.shape[1] == 54:
        return ori.view(ori.shape[0], 6, 3, 3).float()
    raise ValueError(f"Unsupported ori shape: {tuple(ori.shape)}")


def build_two_imu_input(acc, ori, root_idx, local_idx):
    acc = ensure_acc_6x3(acc).to(DEVICE, non_blocking=True)
    ori = ensure_rotmat_3x3(ori).to(DEVICE, non_blocking=True)

    a_root = acc[:, root_idx, :]
    r_root = ori[:, root_idx].reshape(acc.shape[0], 9)
    a_local = acc[:, local_idx, :]
    r_local = ori[:, local_idx].reshape(acc.shape[0], 9)

    return torch.cat([a_root, r_root, a_local, r_local], dim=1).float()


def build_multi_imu_input(acc, ori, sensor_indices):
    acc = ensure_acc_6x3(acc).to(DEVICE, non_blocking=True)
    ori = ensure_rotmat_3x3(ori).to(DEVICE, non_blocking=True)

    parts = []
    for sensor_idx in sensor_indices:
        parts.append(acc[:, sensor_idx, :])
        parts.append(ori[:, sensor_idx].reshape(acc.shape[0], 9))

    return torch.cat(parts, dim=1).float()


def full_position_subset(full_position, joint_indices):
    full_position = full_position.to(DEVICE, non_blocking=True).float()
    parts = []
    for joint_idx in joint_indices:
        if joint_idx == 0:
            raise ValueError("PoseS2 full position vector does not include root joint 0.")
        start = (joint_idx - 1) * 3
        parts.append(full_position[:, start:start + 3])
    return torch.cat(parts, dim=1).float()


# ------------------------------------------------------------
# Checkpoint loading
# ------------------------------------------------------------
def load_transpose_net(original_weights, num_past_frame=20, num_future_frame=5):
    old_weights = paths.weights_file
    paths.weights_file = original_weights

    model = TransPoseNet(
        num_past_frame=num_past_frame,
        num_future_frame=num_future_frame,
    ).to(DEVICE)

    paths.weights_file = old_weights

    checkpoint = torch.load(
        original_weights,
        map_location=DEVICE,
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        raise RuntimeError("Unexpected original weights format")

    model.eval()
    return model


def find_checkpoint(root_or_file, target, candidates):
    if os.path.isfile(root_or_file):
        return root_or_file

    paths_to_try = [
        os.path.join(root_or_file, target, pattern.format(target=target))
        for pattern in candidates
    ]

    for path in paths_to_try:
        if os.path.exists(path):
            return path

    matches = sorted(
        glob.glob(
            os.path.join(root_or_file, "**", f"*{target}*.pth"),
            recursive=True,
        )
    )
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find checkpoint for target={target}. Tried:\n"
        + "\n".join(paths_to_try)
    )


def find_fusion_checkpoint(path_or_root):
    if os.path.isfile(path_or_root):
        return path_or_root

    candidates = [
        os.path.join(path_or_root, "best_fusion.pth"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    matches = sorted(
        glob.glob(os.path.join(path_or_root, "**", "*fusion*.pth"), recursive=True)
    )
    if matches:
        return matches[0]

    raise FileNotFoundError(f"No fusion checkpoint found in {path_or_root}")


def load_distributed_s1_model(weights_root, target):
    checkpoint_path = find_checkpoint(
        weights_root,
        target,
        [
            "best_pose_s1_{target}.pth",
        ],
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model = DistributedPoseS1(
        input_dim=int(checkpoint.get("input_dim", 24)),
        output_dim=int(checkpoint.get("output_dim", 3)),
        proj_dim=int(checkpoint.get("proj_dim", 32)),
        rnn_hidden=int(checkpoint.get("rnn_hidden", 32)),
        rnn_layers=int(checkpoint.get("rnn_layers", 2)),
        dropout=float(checkpoint.get("dropout", 0.2)),
    ).to(DEVICE)

    model.load_state_dict(
        remap_state_dict_if_needed(checkpoint["model_state_dict"])
    )
    model.eval()

    return {
        "model": model,
        "checkpoint_path": checkpoint_path,
        "sensor_idx": dict(LEAF_TARGETS)[target],
    }


def load_all_distributed_s1(weights_root):
    return {
        target: load_distributed_s1_model(weights_root, target)
        for target, _ in LEAF_TARGETS
    }


def load_pose_s2_full_branch(weights_root, target):
    checkpoint_path = find_checkpoint(
        weights_root,
        target,
        [
            "best_pose_s2_{target}.pth",
        ],
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )
    config = POSE_S2_BRANCH_CONFIG[target]

    model = FiveBranchRNN(
        input_dim=int(checkpoint.get("input_dim", config["input_dim"])),
        output_dim=int(checkpoint.get("output_dim", config["output_dim"])),
        proj_dim=int(checkpoint.get("proj_dim", 16)),
        rnn_hidden=int(checkpoint.get("rnn_hidden", 16)),
        rnn_layers=int(checkpoint.get("rnn_layers", 2)),
        dropout=float(checkpoint.get("dropout", 0.2)),
    ).to(DEVICE)

    model.load_state_dict(
        remap_state_dict_if_needed(checkpoint["model_state_dict"])
    )
    model.eval()

    print(
        f"[PoseS2] {target}: {checkpoint_path} | "
        f"input={checkpoint.get('input_dim', config['input_dim'])} | "
        f"output={checkpoint.get('output_dim', config['output_dim'])} | "
        f"hidden={checkpoint.get('rnn_hidden', 'unknown')}"
    )

    return {
        "model": model,
        "checkpoint_path": checkpoint_path,
        "config": config,
    }


def load_all_pose_s2_full(weights_root):
    return {
        target: load_pose_s2_full_branch(weights_root, target)
        for target in POSE_S2_BRANCH_ORDER
    }


def load_pose_s3_five_branch_region(weights_root, target):
    checkpoint_path = find_checkpoint(
        weights_root,
        target,
        [
            "best_pose_s3_{target}.pth",
        ],
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )
    config = POSE_S3_BRANCH_CONFIG[target]

    model = FiveBranchRNN(
        input_dim=int(checkpoint.get("input_dim", config["input_dim"])),
        output_dim=int(checkpoint.get("output_dim", config["output_dim"])),
        proj_dim=int(checkpoint.get("proj_dim", 16)),
        rnn_hidden=int(checkpoint.get("rnn_hidden", 16)),
        rnn_layers=int(checkpoint.get("rnn_layers", 2)),
        dropout=float(checkpoint.get("dropout", 0.2)),
    ).to(DEVICE)

    model.load_state_dict(
        remap_state_dict_if_needed(checkpoint["model_state_dict"])
    )
    model.eval()

    print(
        f"[PoseS3] {target}: {checkpoint_path} | "
        f"input={checkpoint.get('input_dim', config['input_dim'])} | "
        f"output={checkpoint.get('output_dim', config['output_dim'])} | "
        f"hidden={checkpoint.get('rnn_hidden', 'unknown')}"
    )

    return {
        "model": model,
        "checkpoint_path": checkpoint_path,
        "config": config,
    }


def load_all_pose_s3_five_branch_regions(weights_root):
    return {
        target: load_pose_s3_five_branch_region(weights_root, target)
        for target in POSE_S3_BRANCH_ORDER
    }


def load_fusion(path_or_root):
    checkpoint_path = find_fusion_checkpoint(path_or_root)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model = FusionNet(
        input_dim=int(checkpoint.get("input_dim", 90)),
        output_dim=int(checkpoint.get("output_dim", 90)),
        proj_dim=int(checkpoint.get("proj_dim", 16)),
        rnn_hidden=int(checkpoint.get("rnn_hidden", 16)),
        rnn_layers=int(checkpoint.get("rnn_layers", 1)),
        dropout=float(checkpoint.get("dropout", 0.2)),
    ).to(DEVICE)

    model.load_state_dict(
        remap_state_dict_if_needed(checkpoint["model_state_dict"])
    )
    model.eval()

    use_pose_s2_position = bool(checkpoint.get("use_pose_s2_position", False))

    print(
        f"[Fusion] {checkpoint_path} | "
        f"input={checkpoint.get('input_dim', 90)} | "
        f"output={checkpoint.get('output_dim', 90)} | "
        f"hidden={checkpoint.get('rnn_hidden', 'unknown')} | "
        f"use_pose_s2_position={use_pose_s2_position}"
    )

    return {
        "model": model,
        "checkpoint_path": checkpoint_path,
        "use_pose_s2_position": use_pose_s2_position,
    }


# ------------------------------------------------------------
# Dense sequence inference
# ------------------------------------------------------------
@torch.inference_mode()
def dense_stage_forward(model, sequence):
    """
    Run Linear -> LSTM -> Linear without PackedSequence.

    sequence: [T, input_dim]
    output:   [T, output_dim]
    """
    if hasattr(model, "stage"):
        stage = model.stage
        hidden = stage.dropout(sequence)
        hidden = torch.relu(stage.linear1(hidden))
        hidden, _ = stage.rnn(hidden.unsqueeze(1))
        return stage.linear2(hidden.squeeze(1))

    hidden = model.dropout(sequence)
    hidden = torch.relu(model.fc_in(hidden))
    hidden, _ = model.rnn(hidden.unsqueeze(1))
    return model.fc_out(hidden.squeeze(1))


@torch.inference_mode()
def run_parallel_branches(models, inputs, order, streams):
    if streams is None:
        return {
            name: dense_stage_forward(models[name]["model"], inputs[name])
            for name in order
        }

    main_stream = torch.cuda.current_stream()
    outputs = {}

    for name in order:
        stream = streams[name]
        stream.wait_stream(main_stream)

        with torch.cuda.stream(stream):
            output = dense_stage_forward(models[name]["model"], inputs[name])
            output.record_stream(stream)
            outputs[name] = output

    for name in order:
        main_stream.wait_stream(streams[name])

    return outputs


# ------------------------------------------------------------
# Distributed pose builders
# ------------------------------------------------------------
def build_pose_s1_inputs(raw):
    inputs = {}
    for target, sensor_idx in LEAF_TARGETS:
        inputs[target] = build_two_imu_input(
            raw["acc"],
            raw["ori"],
            ROOT_SENSOR_IDX,
            sensor_idx,
        )
    return inputs


def build_pose_s2_full_inputs(raw, s1_outputs):
    inputs = {}
    for target in POSE_S2_BRANCH_ORDER:
        config = POSE_S2_BRANCH_CONFIG[target]
        imu_features = build_two_imu_input(
            raw["acc"],
            raw["ori"],
            ROOT_SENSOR_IDX,
            int(config["local_sensor_idx"]),
        )
        leaf_name = str(config["s1_leaf_name"])
        local_leaf = s1_outputs[leaf_name].to(DEVICE, non_blocking=True).float()

        length = min(imu_features.shape[0], local_leaf.shape[0])
        inputs[target] = torch.cat(
            [imu_features[:length], local_leaf[:length]],
            dim=1,
        ).float()
    return inputs


def assemble_pose_s2_full(s2_outputs):
    length = min(output.shape[0] for output in s2_outputs.values())
    full = torch.zeros(length, 69, dtype=torch.float32, device=DEVICE)
    full_view = full.view(length, 23, 3)

    for target in POSE_S2_BRANCH_ORDER:
        config = POSE_S2_BRANCH_CONFIG[target]
        joints = list(config["joints"])
        pred = s2_outputs[target][:length].view(length, len(joints), 3)
        for local_idx, joint_idx in enumerate(joints):
            full_view[:, joint_idx - 1, :] = pred[:, local_idx, :]

    return full


def build_pose_s3_five_branch_inputs(raw, p_full):
    inputs = {}
    for target in POSE_S3_BRANCH_ORDER:
        config = POSE_S3_BRANCH_CONFIG[target]
        imu_features = build_multi_imu_input(
            raw["acc"],
            raw["ori"],
            config["sensor_indices"],
        )
        position_features = full_position_subset(
            p_full,
            config["position_joints"],
        )
        length = min(imu_features.shape[0], position_features.shape[0])
        inputs[target] = torch.cat(
            [imu_features[:length], position_features[:length]],
            dim=1,
        ).float()
    return inputs


def assemble_pose_s3_reduced(s3_outputs):
    length = min(output.shape[0] for output in s3_outputs.values())
    reduced = torch.zeros(
        length,
        len(REDUCED_JOINTS) * 6,
        dtype=torch.float32,
        device=DEVICE,
    )
    reduced_view = reduced.view(length, len(REDUCED_JOINTS), 6)

    for target in POSE_S3_BRANCH_ORDER:
        config = POSE_S3_BRANCH_CONFIG[target]
        reduced_joints = list(config["reduced_joints"])
        pred = s3_outputs[target][:length].view(length, len(reduced_joints), 6)
        for local_idx, joint_idx in enumerate(reduced_joints):
            reduced_idx = REDUCED_JOINTS.index(joint_idx)
            reduced_view[:, reduced_idx, :] = pred[:, local_idx, :]

    return reduced


def apply_fusion(fusion, assembled_reduced, p_full):
    if SKIP_FUSION:
        return assembled_reduced

    if fusion["use_pose_s2_position"]:
        length = min(assembled_reduced.shape[0], p_full.shape[0])
        fusion_input = torch.cat(
            [assembled_reduced[:length], p_full[:length]],
            dim=1,
        ).float()
        assembled_reduced = assembled_reduced[:length]
    else:
        fusion_input = assembled_reduced.float()

    delta = dense_stage_forward(fusion["model"], fusion_input)
    length = min(assembled_reduced.shape[0], delta.shape[0])
    return assembled_reduced[:length] + delta[:length]


@torch.inference_mode()
def distributed_pose_forward(
    raw,
    s1_models,
    s2_models,
    s3_models,
    fusion,
    s1_streams=None,
    s2_streams=None,
    s3_streams=None,
):
    # PoseS1.
    s1_inputs = build_pose_s1_inputs(raw)
    s1_outputs = run_parallel_branches(
        s1_models,
        s1_inputs,
        LEAF_ORDER,
        s1_streams,
    )

    length = min(output.shape[0] for output in s1_outputs.values())
    s1_outputs = {name: output[:length] for name, output in s1_outputs.items()}
    p_leaf = torch.cat([s1_outputs[name] for name in LEAF_ORDER], dim=1)

    # PoseS2 five-branch.
    s2_inputs = build_pose_s2_full_inputs(raw, s1_outputs)
    s2_outputs = run_parallel_branches(
        s2_models,
        s2_inputs,
        POSE_S2_BRANCH_ORDER,
        s2_streams,
    )
    p_full = assemble_pose_s2_full(s2_outputs)

    # PoseS3 five-branch + learned fusion.
    s3_inputs = build_pose_s3_five_branch_inputs(raw, p_full)
    s3_outputs = run_parallel_branches(
        s3_models,
        s3_inputs,
        POSE_S3_BRANCH_ORDER,
        s3_streams,
    )
    assembled_reduced = assemble_pose_s3_reduced(s3_outputs)
    reduced_pose = apply_fusion(fusion, assembled_reduced, p_full)

    imu = normalize_and_concat(raw["acc"], raw["ori"]).float().to(DEVICE)

    length = min(
        imu.shape[0],
        p_leaf.shape[0],
        p_full.shape[0],
        reduced_pose.shape[0],
    )

    return (
        imu[:length],
        p_leaf[:length],
        p_full[:length],
        reduced_pose[:length],
    )


# ------------------------------------------------------------
# Ring buffer for online input
# ------------------------------------------------------------
class IMURingBuffer:
    def __init__(self, window_size):
        self.window_size = window_size
        self.acc = torch.empty(
            window_size,
            6,
            3,
            dtype=torch.float32,
            device=DEVICE,
        )
        self.ori = torch.empty(
            window_size,
            6,
            3,
            3,
            dtype=torch.float32,
            device=DEVICE,
        )
        self.write_pos = 0
        self.initialized = False
        self.indices = torch.arange(window_size, device=DEVICE, dtype=torch.long)

    def reset(self):
        self.write_pos = 0
        self.initialized = False

    def append(self, acc_frame, ori_frame):
        if not self.initialized:
            self.acc.copy_(acc_frame.unsqueeze(0).expand_as(self.acc))
            self.ori.copy_(ori_frame.unsqueeze(0).expand_as(self.ori))
            self.write_pos = 0
            self.initialized = True
        else:
            self.acc[self.write_pos].copy_(acc_frame)
            self.ori[self.write_pos].copy_(ori_frame)
            self.write_pos = (self.write_pos + 1) % self.window_size

        ordered_indices = torch.remainder(self.indices + self.write_pos, self.window_size)
        return {
            "acc": self.acc.index_select(0, ordered_indices),
            "ori": self.ori.index_select(0, ordered_indices),
        }


# ------------------------------------------------------------
# Original pipeline inference
# ------------------------------------------------------------
@torch.inference_mode()
def original_offline(net, raw):
    imu = normalize_and_concat(raw["acc"], raw["ori"]).float().to(DEVICE)
    return net.forward_offline(imu)


@torch.inference_mode()
def original_online_profiled(net, raw, num_future_frame, latency_warmup_frames):
    imu = normalize_and_concat(raw["acc"], raw["ori"]).float().to(DEVICE)
    sequence = torch.cat((imu, imu[-1:].repeat(num_future_frame, 1)), dim=0)

    net.reset()
    outputs = []
    latency_ms = []

    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for index, frame in enumerate(sequence):
        synchronize()
        start = time.perf_counter()
        output = net.forward_online(frame)
        synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        outputs.append(output)

        if index >= latency_warmup_frames and index < imu.shape[0]:
            latency_ms.append(elapsed_ms)

    pose_online, tran_online = [
        torch.stack(items)[num_future_frame:]
        for items in zip(*outputs)
    ]

    peak_alloc = None
    peak_reserved = None
    if DEVICE.type == "cuda":
        peak_alloc = bytes_to_mb(torch.cuda.max_memory_allocated())
        peak_reserved = bytes_to_mb(torch.cuda.max_memory_reserved())

    return pose_online, tran_online, latency_ms, peak_alloc, peak_reserved


# ------------------------------------------------------------
# Distributed pipeline + original Trans-B1/B2
# ------------------------------------------------------------
@torch.inference_mode()
def translation_offline(trans_net, imu, p_leaf, p_full, pose):
    length = min(imu.shape[0], p_leaf.shape[0], p_full.shape[0], pose.shape[0])

    imu = imu[:length]
    p_leaf = p_leaf[:length]
    p_full = p_full[:length]
    pose = pose[:length]

    contact_probability, _ = trans_net.tran_b1.forward(torch.cat((p_leaf, imu), dim=1))
    velocity_pred, _ = trans_net.tran_b2.forward(torch.cat((p_full, imu), dim=1), None)

    root_rotation = imu[:, -9:].view(-1, 3, 3)

    joints = art.math.forward_kinematics(
        pose[:, joint_set.lower_body],
        trans_net.lower_body_bone.expand(pose.shape[0], -1, -1),
        joint_set.lower_body_parent,
    )[1]

    tran_b1_vel = trans_net.gravity_velocity + art.math.lerp(
        torch.cat((torch.zeros(1, 3, device=joints.device), joints[:-1, 7] - joints[1:, 7])),
        torch.cat((torch.zeros(1, 3, device=joints.device), joints[:-1, 8] - joints[1:, 8])),
        contact_probability.max(dim=1).indices.view(-1, 1).cpu(),
    )

    tran_b2_vel = (
        root_rotation.bmm(velocity_pred.unsqueeze(-1)).squeeze(-1).cpu()
        * vel_scale
        / 60
    )

    weight = trans_net._prob_to_weight(
        contact_probability.cpu().max(dim=1).values.sigmoid()
    ).view(-1, 1)

    velocity = art.math.lerp(tran_b2_vel, tran_b1_vel, weight)

    current_root_y = 0.0
    for index in range(velocity.shape[0]):
        current_foot_y = current_root_y + joints[index, 7:9, 1].min().item()
        if current_foot_y + velocity[index, 1].item() <= trans_net.floor_y:
            velocity[index, 1] = trans_net.floor_y - current_foot_y
        current_root_y += velocity[index, 1].item()

    return trans_net.velocity_to_root_position(velocity)


@torch.inference_mode()
def distributed_offline(
    trans_net,
    raw,
    s1_models,
    s2_models,
    s3_models,
    fusion,
):
    imu, p_leaf, p_full, reduced_pose = distributed_pose_forward(
        raw,
        s1_models,
        s2_models,
        s3_models,
        fusion,
        None,
        None,
        None,
    )

    root_rotation = imu[:, -9:].view(-1, 3, 3)
    pose = trans_net._reduced_glb_6d_to_full_local_mat(
        root_rotation.cpu(),
        reduced_pose.cpu(),
    )

    tran = translation_offline(trans_net, imu, p_leaf, p_full, pose)
    return pose, tran


class DistributedOnlineState:
    def __init__(self, trans_net, window_size):
        self.trans_net = trans_net
        self.buffer = IMURingBuffer(window_size)
        self.reset()

    def reset(self):
        self.buffer.reset()
        self.rnn_state = None
        self.current_root_y = 0.0
        self.last_lfoot_pos = self.trans_net.feet_pos[0].clone()
        self.last_rfoot_pos = self.trans_net.feet_pos[1].clone()
        self.last_root_pos = torch.zeros(3)


@torch.inference_mode()
def distributed_online_profiled(
    trans_net,
    raw,
    s1_models,
    s2_models,
    s3_models,
    fusion,
    s1_streams,
    s2_streams,
    s3_streams,
    profile_args,
    latency_warmup_frames,
):
    acc = ensure_acc_6x3(raw["acc"]).float().to(DEVICE)
    ori = ensure_rotmat_3x3(raw["ori"]).float().to(DEVICE)

    acc_sequence = torch.cat(
        (acc, acc[-1:].repeat(profile_args.num_future_frame, 1, 1)),
        dim=0,
    )
    ori_sequence = torch.cat(
        (ori, ori[-1:].repeat(profile_args.num_future_frame, 1, 1, 1)),
        dim=0,
    )

    window_size = profile_args.num_past_frame + profile_args.num_future_frame + 1
    center = profile_args.num_past_frame

    state = DistributedOnlineState(trans_net, window_size)

    poses = []
    translations = []
    latency_ms = []

    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for index, (acc_frame, ori_frame) in enumerate(zip(acc_sequence, ori_sequence)):
        synchronize()
        start = time.perf_counter()

        raw_window = state.buffer.append(acc_frame, ori_frame)

        imu, p_leaf, p_full, reduced_pose = distributed_pose_forward(
            raw_window,
            s1_models,
            s2_models,
            s3_models,
            fusion,
            s1_streams,
            s2_streams,
            s3_streams,
        )

        contact_probability, _ = trans_net.tran_b1.forward(torch.cat((p_leaf, imu), dim=1))
        velocity_pred, state.rnn_state = trans_net.tran_b2.forward(
            torch.cat((p_full, imu), dim=1),
            state.rnn_state,
        )

        contact_probability = contact_probability[center].sigmoid().view(-1).cpu()
        root_rotation = imu[center, -9:].view(3, 3).cpu()

        pose = trans_net._reduced_glb_6d_to_full_local_mat(
            root_rotation,
            reduced_pose[center].cpu(),
        ).squeeze(0)

        lfoot_pos, rfoot_pos = art.math.forward_kinematics(
            pose[joint_set.lower_body].unsqueeze(0),
            trans_net.lower_body_bone.unsqueeze(0),
            joint_set.lower_body_parent,
        )[1][0, 7:9]

        if contact_probability[0] > contact_probability[1]:
            tran_b1_vel = state.last_lfoot_pos - lfoot_pos + trans_net.gravity_velocity
        else:
            tran_b1_vel = state.last_rfoot_pos - rfoot_pos + trans_net.gravity_velocity

        tran_b2_vel = (
            root_rotation.mm(velocity_pred[center].cpu().view(3, 1)).view(3)
            / 60
            * vel_scale
        )

        weight = trans_net._prob_to_weight(contact_probability.max())
        velocity = art.math.lerp(tran_b2_vel, tran_b1_vel, weight)

        current_foot_y = state.current_root_y + min(lfoot_pos[1].item(), rfoot_pos[1].item())
        if current_foot_y + velocity[1].item() <= trans_net.floor_y:
            velocity[1] = trans_net.floor_y - current_foot_y

        state.current_root_y += velocity[1].item()
        state.last_lfoot_pos = lfoot_pos
        state.last_rfoot_pos = rfoot_pos
        state.last_root_pos += velocity

        poses.append(pose)
        translations.append(state.last_root_pos.clone())

        synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if index >= latency_warmup_frames and index < acc.shape[0]:
            latency_ms.append(elapsed_ms)

    pose_online = torch.stack(poses)[profile_args.num_future_frame:]
    tran_online = torch.stack(translations)[profile_args.num_future_frame:]

    peak_alloc = None
    peak_reserved = None
    if DEVICE.type == "cuda":
        peak_alloc = bytes_to_mb(torch.cuda.max_memory_allocated())
        peak_reserved = bytes_to_mb(torch.cuda.max_memory_reserved())

    return pose_online, tran_online, latency_ms, peak_alloc, peak_reserved


# ------------------------------------------------------------
# Evaluation and reporting
# ------------------------------------------------------------
class PoseEvaluator:
    def __init__(self):
        self._eval_fn = art.FullMotionEvaluator(
            paths.smpl_file,
            joint_mask=torch.tensor([1, 2, 16, 17]),
        )

    def eval(self, pose_p, pose_t):
        length = min(pose_p.shape[0], pose_t.shape[0])
        pose_p = pose_p[:length].clone().view(-1, 24, 3, 3)
        pose_t = pose_t[:length].clone().view(-1, 24, 3, 3)
        pose_p[:, joint_set.ignored] = torch.eye(3, device=pose_p.device)
        pose_t[:, joint_set.ignored] = torch.eye(3, device=pose_t.device)
        errs = self._eval_fn(pose_p, pose_t)
        return torch.stack([
            errs[9],
            errs[3],
            errs[0] * 100,
            errs[1] * 100,
            errs[4] / 100,
        ])

    @staticmethod
    def print(errors, title):
        print(f"\n========== {title} ==========")
        names = [
            "SIP Error (deg)",
            "Angular Error (deg)",
            "Positional Error (cm)",
            "Mesh Error (cm)",
            "Jitter Error (100m/s^3)",
        ]
        for index, name in enumerate(names):
            print("%s: %.2f (+/- %.2f)" % (name, float(errors[index, 0]), float(errors[index, 1])))


class BranchRestrictedEvaluator:
    """
    Same masked global angle error formula as PoseEvaluator's SIP metric
    (FullMotionEvaluator errs[9]), but with joint_mask restricted to an
    arbitrary joint subset (e.g. one anatomical branch) instead of the
    fixed hip/shoulder SIP joints. Used for the sensor-robustness
    experiment's "restricted to the joints of the affected branch" metric.
    """

    def __init__(self, joint_indices):
        self.joint_indices = list(joint_indices)
        self._eval_fn = art.FullMotionEvaluator(
            paths.smpl_file,
            joint_mask=torch.tensor(self.joint_indices),
        )

    def eval_sip(self, pose_p, pose_t):
        length = min(pose_p.shape[0], pose_t.shape[0])
        pose_p = pose_p[:length].clone().view(-1, 24, 3, 3)
        pose_t = pose_t[:length].clone().view(-1, 24, 3, 3)
        pose_p[:, joint_set.ignored] = torch.eye(3, device=pose_p.device)
        pose_t[:, joint_set.ignored] = torch.eye(3, device=pose_t.device)
        errs = self._eval_fn(pose_p, pose_t)
        return float(errs[9, 0])


def summarize_eval(errs, title):
    all_errs = torch.stack(errs, dim=0)
    if all_errs.dim() == 3:
        errors = all_errs.mean(dim=0)
    elif all_errs.dim() == 2:
        mean = all_errs.mean(dim=0)
        std = all_errs.std(dim=0)
        errors = torch.stack([mean, std], dim=1)
    else:
        raise RuntimeError(f"Unexpected evaluator error shape: {tuple(all_errs.shape)}")
    PoseEvaluator.print(errors, title)


def summarize_offline_timing(times, frames, peak_alloc, peak_reserved, title):
    times = torch.tensor(times, dtype=torch.float64)
    frames = torch.tensor(frames, dtype=torch.float64)

    ms_per_sequence = times * 1000.0
    throughput_ms_per_frame = (times / frames) * 1000.0
    throughput_fps = frames / times

    print(f"\n========== {title} ==========")
    print(f"Time per sequence (ms):     {ms_per_sequence.mean():.3f} +/- {ms_per_sequence.std(unbiased=False):.3f}")
    print(f"Throughput time/frame (ms): {throughput_ms_per_frame.mean():.6f} +/- {throughput_ms_per_frame.std(unbiased=False):.6f}")
    print(f"Throughput FPS:             {throughput_fps.mean():.3f} +/- {throughput_fps.std(unbiased=False):.3f}")

    if peak_alloc is not None:
        print(f"Peak CUDA alloc (MB):       {peak_alloc:.3f}")
        print(f"Peak CUDA reserved (MB):    {peak_reserved:.3f}")


def summarize_latency(latency_ms, peak_alloc, peak_reserved, title):
    if len(latency_ms) == 0:
        print(f"\n========== {title} ==========")
        print("No latency samples were recorded.")
        return

    values = torch.tensor(latency_ms, dtype=torch.float64)
    mean = float(values.mean())
    median = float(values.median())
    std = float(values.std(unbiased=False))
    minimum = float(values.min())
    maximum = float(values.max())
    p95 = percentile(latency_ms, 0.95)
    p99 = percentile(latency_ms, 0.99)

    print(f"\n========== {title} ==========")
    print(f"Samples:                    {len(latency_ms)}")
    print(f"Mean latency/frame (ms):    {mean:.6f}")
    print(f"Median latency/frame (ms):  {median:.6f}")
    print(f"Std latency/frame (ms):     {std:.6f}")
    print(f"P95 latency/frame (ms):     {p95:.6f}")
    print(f"P99 latency/frame (ms):     {p99:.6f}")
    print(f"Min latency/frame (ms):     {minimum:.6f}")
    print(f"Max latency/frame (ms):     {maximum:.6f}")
    print(f"Mean effective FPS:         {1000.0 / mean:.3f}")

    if peak_alloc is not None:
        print(f"Peak CUDA alloc (MB):       {peak_alloc:.3f}")
        print(f"Peak CUDA reserved (MB):    {peak_reserved:.3f}")


def print_size(original_net, s1_models, s2_models, s3_models, fusion, args):
    orig_pose_params = (
        count_params(original_net.pose_s1)[0]
        + count_params(original_net.pose_s2)[0]
        + count_params(original_net.pose_s3)[0]
    )
    orig_trans_params = count_params(original_net.tran_b1)[0] + count_params(original_net.tran_b2)[0]
    orig_total_params = orig_pose_params + orig_trans_params

    dist_s1_params = sum(count_params(s1_models[name]["model"])[0] for name in LEAF_ORDER)
    dist_s2_params = sum(count_params(s2_models[name]["model"])[0] for name in POSE_S2_BRANCH_ORDER)
    dist_s3_params = sum(count_params(s3_models[name]["model"])[0] for name in POSE_S3_BRANCH_ORDER)
    dist_fusion_params = count_params(fusion["model"])[0]

    dist_total_params = dist_s1_params + dist_s2_params + dist_s3_params + dist_fusion_params + orig_trans_params

    print("\n==================== SIZE STATS ====================")
    print("Original full pipeline")
    print(f"  Pose params total:         {orig_pose_params}")
    print(f"  Trans-B1+B2 params:        {orig_trans_params}")
    print(f"  Full pipeline params:      {orig_total_params}")
    print(f"  Full pipeline fp32 (MB):   {orig_total_params * 4 / (1024 ** 2):.3f}")
    print(f"  Original weights file MB:  {file_size_mb(args.original_weights):.3f}")

    print("\nDistributed five-branch pose + original Trans-B1/B2")
    print(f"  Distributed S1 params:     {dist_s1_params}")
    print(f"  PoseS2 five-branch params: {dist_s2_params}")
    print(f"  PoseS3 five-branch params: {dist_s3_params}")
    print(f"  Fusion params:      {dist_fusion_params}")
    print(f"  Original Trans-B1+B2:      {orig_trans_params}")
    print(f"  Full pipeline params:      {dist_total_params}")
    print(f"  Full pipeline fp32 (MB):   {dist_total_params * 4 / (1024 ** 2):.3f}")

    print("\nDistributed / Original full ratio")
    print(f"  Param ratio:               {dist_total_params / float(orig_total_params):.4f}")


# ------------------------------------------------------------
# Warmup
# ------------------------------------------------------------
@torch.inference_mode()
def warmup_online_models(
    original_net,
    trans_net_for_distributed,
    raw,
    s1_models,
    s2_models,
    s3_models,
    fusion,
    s1_streams,
    s2_streams,
    s3_streams,
    profile_args,
    frames,
):
    acc = ensure_acc_6x3(raw["acc"]).float().to(DEVICE)
    ori = ensure_rotmat_3x3(raw["ori"]).float().to(DEVICE)
    imu = normalize_and_concat(acc, ori).float().to(DEVICE)

    count = min(frames, imu.shape[0])

    original_net.reset()
    for index in range(count):
        _ = original_net.forward_online(imu[index])

    buffer = IMURingBuffer(profile_args.num_past_frame + profile_args.num_future_frame + 1)
    for index in range(count):
        raw_window = buffer.append(acc[index], ori[index])
        _ = distributed_pose_forward(
            raw_window,
            s1_models,
            s2_models,
            s3_models,
            fusion,
            s1_streams,
            s2_streams,
            s3_streams,
        )

    synchronize()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--raw_list_file", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default=None)

    parser.add_argument("--original_weights", type=str, default=str(TRANSPOSE_WEIGHTS_FILE))
    parser.add_argument("--distributed_s1_root", type=str, default=str(POSE_S1_CHECKPOINTS_DIR))
    parser.add_argument("--pose_s2_full_root", type=str, default=str(POSE_S2_CHECKPOINTS_DIR))
    parser.add_argument("--pose_s3_five_branch_root", type=str, default=str(POSE_S3_CHECKPOINTS_DIR))
    parser.add_argument("--fusion_weights", type=str, default=str(FUSION_CHECKPOINT))

    parser.add_argument("--evaluate_online", action="store_true")
    parser.add_argument("--num_past_frame", type=int, default=20)
    parser.add_argument("--num_future_frame", type=int, default=5)
    parser.add_argument("--profile_repeat", type=int, default=1)
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--latency_warmup_frames", type=int, default=30)
    parser.add_argument("--use_cuda_streams", action="store_true")

    parser.add_argument("--skip_fusion", action="store_true",
                         help="Ablation: bypass the PoseS3 learned residual fusion module.")
    parser.add_argument("--branch_restricted_joints", type=str, default=None,
                         help="Comma-separated SMPL joint indices. When set, also reports the "
                              "SIP-style masked global angle error restricted to this joint subset "
                              "(e.g. one anatomical branch), for both pipelines.")
    parser.add_argument("--branch_restricted_label", type=str, default="branch",
                         help="Label used when printing the branch-restricted metric.")
    parser.add_argument("--export_per_sequence", type=str, default=None,
                         help="Path to write per-sequence offline errors (JSON) for both pipelines.")
    parser.add_argument("--export_per_sequence_online", type=str, default=None,
                         help="Path to write per-sequence online errors (JSON) for both pipelines. "
                              "Requires --evaluate_online.")

    args = parser.parse_args()

    global SKIP_FUSION
    SKIP_FUSION = bool(args.skip_fusion)
    print("Skip PoseS3 residual fusion:", SKIP_FUSION)

    branch_evaluator = None
    branch_joint_indices = None
    if args.branch_restricted_joints is not None:
        branch_joint_indices = [int(x) for x in args.branch_restricted_joints.split(",") if x.strip() != ""]
        branch_evaluator = BranchRestrictedEvaluator(branch_joint_indices)
        print(f"Branch-restricted eval ('{args.branch_restricted_label}') joints: {branch_joint_indices}")

    print("DEVICE:", DEVICE)
    print("Use CUDA streams:", bool(args.use_cuda_streams and DEVICE.type == "cuda"))
    print("Online latency mode: frame-by-frame, batch size 1, fixed window")
    print("Latency warmup frames:", args.latency_warmup_frames)
    print("Note: Trans-B1 and Trans-B2 are original modules in both pipelines.")
    print("Distributed pose: S1 5-branch + S2 5-branch + S3 5-branch + learned fusion.")

    profile_args = SimpleNamespace(
        num_past_frame=args.num_past_frame,
        num_future_frame=args.num_future_frame,
    )

    evaluator = PoseEvaluator()

    original_net = load_transpose_net(args.original_weights, args.num_past_frame, args.num_future_frame)
    trans_net_for_distributed = load_transpose_net(args.original_weights, args.num_past_frame, args.num_future_frame)

    s1_models = load_all_distributed_s1(args.distributed_s1_root)
    s2_models = load_all_pose_s2_full(args.pose_s2_full_root)
    s3_models = load_all_pose_s3_five_branch_regions(args.pose_s3_five_branch_root)
    fusion = load_fusion(args.fusion_weights)

    s1_streams = create_streams(LEAF_ORDER, args.use_cuda_streams)
    s2_streams = create_streams(POSE_S2_BRANCH_ORDER, args.use_cuda_streams)
    s3_streams = create_streams(POSE_S3_BRANCH_ORDER, args.use_cuda_streams)

    print_size(original_net, s1_models, s2_models, s3_models, fusion, args)

    raw_paths = None
    if args.dataset_dir is not None:
        data = torch.load(os.path.join(args.dataset_dir, "test.pt"), map_location="cpu", weights_only=False)
        raw_items = [
            {"acc": acc, "ori": ori, "pose": pose, "tran": tran}
            for acc, ori, pose, tran in zip(data["acc"], data["ori"], data["pose"], data["tran"])
        ]
    elif args.raw_list_file is not None:
        raw_paths = load_list(args.raw_list_file)
        raw_items = [
            torch.load(path, map_location="cpu", weights_only=False)
            for path in raw_paths
        ]
    else:
        raise RuntimeError("Provide either --raw_list_file or --dataset_dir")

    if args.max_sequences is not None:
        raw_items = raw_items[:args.max_sequences]
        if raw_paths is not None:
            raw_paths = raw_paths[:args.max_sequences]

    if len(raw_items) == 0:
        raise RuntimeError("No sequences found.")

    if raw_paths is None:
        raw_paths = [None] * len(raw_items)

    if args.warmup:
        warmup_online_models(
            original_net,
            trans_net_for_distributed,
            raw_items[0],
            s1_models,
            s2_models,
            s3_models,
            fusion,
            s1_streams,
            s2_streams,
            s3_streams,
            profile_args,
            max(args.latency_warmup_frames, 1),
        )

    offline_orig_errs = []
    offline_dist_errs = []
    online_orig_errs = []
    online_dist_errs = []

    branch_orig_errs = []
    branch_dist_errs = []
    per_sequence_records = []
    per_sequence_records_online = []

    offline_orig_times = []
    offline_dist_times = []
    offline_orig_frames = []
    offline_dist_frames = []

    original_latency_ms = []
    distributed_latency_ms = []

    offline_orig_peak_alloc = None
    offline_orig_peak_reserved = None
    offline_dist_peak_alloc = None
    offline_dist_peak_reserved = None

    online_orig_peak_alloc = None
    online_orig_peak_reserved = None
    online_dist_peak_alloc = None
    online_dist_peak_reserved = None

    iterator = tqdm(list(zip(raw_items, raw_paths)), desc="Evaluating full pipeline with Trans-B1/B2", ncols=120)

    for raw, raw_path in iterator:
        pose_t = art.math.axis_angle_to_rotation_matrix(raw["pose"]).view(-1, 24, 3, 3)
        frame_count = float(pose_t.shape[0])

        if DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        synchronize()
        start = time.perf_counter()
        pose_orig_off, _ = original_offline(original_net, raw)
        synchronize()
        elapsed = time.perf_counter() - start

        offline_orig_errs.append(evaluator.eval(pose_orig_off, pose_t))
        offline_orig_times.append(elapsed)
        offline_orig_frames.append(frame_count)

        if DEVICE.type == "cuda":
            offline_orig_peak_alloc = update_peak(offline_orig_peak_alloc, bytes_to_mb(torch.cuda.max_memory_allocated()))
            offline_orig_peak_reserved = update_peak(offline_orig_peak_reserved, bytes_to_mb(torch.cuda.max_memory_reserved()))

        if DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        synchronize()
        start = time.perf_counter()
        pose_dist_off, _ = distributed_offline(
            trans_net_for_distributed,
            raw,
            s1_models,
            s2_models,
            s3_models,
            fusion,
        )
        synchronize()
        elapsed = time.perf_counter() - start

        offline_dist_errs.append(evaluator.eval(pose_dist_off, pose_t))
        offline_dist_times.append(elapsed)
        offline_dist_frames.append(frame_count)

        if DEVICE.type == "cuda":
            offline_dist_peak_alloc = update_peak(offline_dist_peak_alloc, bytes_to_mb(torch.cuda.max_memory_allocated()))
            offline_dist_peak_reserved = update_peak(offline_dist_peak_reserved, bytes_to_mb(torch.cuda.max_memory_reserved()))

        if branch_evaluator is not None:
            branch_orig_errs.append(branch_evaluator.eval_sip(pose_orig_off, pose_t))
            branch_dist_errs.append(branch_evaluator.eval_sip(pose_dist_off, pose_t))

        if args.export_per_sequence is not None:
            orig_vec = offline_orig_errs[-1][:, 0]
            dist_vec = offline_dist_errs[-1][:, 0]
            record = {
                "path": raw_path,
                "source": raw.get("source") if isinstance(raw, dict) else None,
                "index": raw.get("index") if isinstance(raw, dict) else None,
                "frame_count": frame_count,
                "sip_orig": float(orig_vec[0]),
                "sip_dist": float(dist_vec[0]),
                "ang_orig": float(orig_vec[1]),
                "ang_dist": float(dist_vec[1]),
                "pos_orig": float(orig_vec[2]),
                "pos_dist": float(dist_vec[2]),
                "mesh_orig": float(orig_vec[3]),
                "mesh_dist": float(dist_vec[3]),
                "jitter_orig": float(orig_vec[4]),
                "jitter_dist": float(dist_vec[4]),
            }
            if branch_evaluator is not None:
                record["branch_sip_orig"] = branch_orig_errs[-1]
                record["branch_sip_dist"] = branch_dist_errs[-1]
            per_sequence_records.append(record)

        if args.evaluate_online:
            pose_orig_on, _, orig_samples, peak_alloc, peak_reserved = original_online_profiled(
                original_net,
                raw,
                args.num_future_frame,
                args.latency_warmup_frames,
            )

            online_orig_errs.append(evaluator.eval(pose_orig_on, pose_t))
            original_latency_ms.extend(orig_samples)
            online_orig_peak_alloc = update_peak(online_orig_peak_alloc, peak_alloc)
            online_orig_peak_reserved = update_peak(online_orig_peak_reserved, peak_reserved)

            pose_dist_on, _, dist_samples, peak_alloc, peak_reserved = distributed_online_profiled(
                trans_net_for_distributed,
                raw,
                s1_models,
                s2_models,
                s3_models,
                fusion,
                s1_streams,
                s2_streams,
                s3_streams,
                profile_args,
                args.latency_warmup_frames,
            )

            online_dist_errs.append(evaluator.eval(pose_dist_on, pose_t))
            distributed_latency_ms.extend(dist_samples)
            online_dist_peak_alloc = update_peak(online_dist_peak_alloc, peak_alloc)
            online_dist_peak_reserved = update_peak(online_dist_peak_reserved, peak_reserved)

            if args.export_per_sequence_online is not None:
                orig_vec_on = online_orig_errs[-1][:, 0]
                dist_vec_on = online_dist_errs[-1][:, 0]
                record_on = {
                    "path": raw_path,
                    "source": raw.get("source") if isinstance(raw, dict) else None,
                    "index": raw.get("index") if isinstance(raw, dict) else None,
                    "frame_count": frame_count,
                    "sip_orig": float(orig_vec_on[0]),
                    "sip_dist": float(dist_vec_on[0]),
                    "ang_orig": float(orig_vec_on[1]),
                    "ang_dist": float(dist_vec_on[1]),
                    "pos_orig": float(orig_vec_on[2]),
                    "pos_dist": float(dist_vec_on[2]),
                    "mesh_orig": float(orig_vec_on[3]),
                    "mesh_dist": float(dist_vec_on[3]),
                    "jitter_orig": float(orig_vec_on[4]),
                    "jitter_dist": float(dist_vec_on[4]),
                }
                per_sequence_records_online.append(record_on)

    print("\n==================== OFFLINE EVALUATION ====================")
    summarize_eval(offline_orig_errs, "Original full pipeline")
    summarize_eval(offline_dist_errs, "Distributed five-branch pose + original Trans-B1/B2")

    summarize_offline_timing(
        offline_orig_times,
        offline_orig_frames,
        offline_orig_peak_alloc,
        offline_orig_peak_reserved,
        "OFFLINE THROUGHPUT - Original full pipeline",
    )
    summarize_offline_timing(
        offline_dist_times,
        offline_dist_frames,
        offline_dist_peak_alloc,
        offline_dist_peak_reserved,
        "OFFLINE THROUGHPUT - Distributed five-branch pose + original Trans-B1/B2",
    )

    if args.evaluate_online:
        print("\n==================== ONLINE EVALUATION ====================")
        summarize_eval(online_orig_errs, "Original full pipeline")
        summarize_eval(online_dist_errs, "Distributed five-branch pose + original Trans-B1/B2")

        summarize_latency(
            original_latency_ms,
            online_orig_peak_alloc,
            online_orig_peak_reserved,
            "ONLINE LATENCY - Original full pipeline",
        )
        summarize_latency(
            distributed_latency_ms,
            online_dist_peak_alloc,
            online_dist_peak_reserved,
            "ONLINE LATENCY - Distributed five-branch pose + original Trans-B1/B2",
        )

    if branch_evaluator is not None:
        branch_orig = torch.tensor(branch_orig_errs)
        branch_dist = torch.tensor(branch_dist_errs)
        print(f"\n==================== BRANCH-RESTRICTED SIP ('{args.branch_restricted_label}') ====================")
        print(f"Joints: {branch_joint_indices}")
        print("Original full pipeline:                  %.2f (+/- %.2f)" % (
            float(branch_orig.mean()), float(branch_orig.std(unbiased=False))))
        print("Distributed five-branch + fusion:         %.2f (+/- %.2f)" % (
            float(branch_dist.mean()), float(branch_dist.std(unbiased=False))))

    if args.export_per_sequence is not None:
        export_path = Path(args.export_per_sequence)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with export_path.open("w", encoding="utf-8") as f:
            json.dump(per_sequence_records, f, indent=2)
        print(f"\n[info] Wrote per-sequence errors for {len(per_sequence_records)} sequences to {export_path}")

    if args.export_per_sequence_online is not None:
        export_path_online = Path(args.export_per_sequence_online)
        export_path_online.parent.mkdir(parents=True, exist_ok=True)
        with export_path_online.open("w", encoding="utf-8") as f:
            json.dump(per_sequence_records_online, f, indent=2)
        print(f"\n[info] Wrote per-sequence online errors for {len(per_sequence_records_online)} "
              f"sequences to {export_path_online}")


if __name__ == "__main__":
    main()
