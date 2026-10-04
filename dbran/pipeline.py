"""
Reusable D-BRAN inference pipeline.

This module extracts the validated five-branch D-BRAN inference path from
`scripts/profile/profile_full_pipeline.py` so it can be reused by:

- offline evaluation;
- online/real-time inference;
- the future Xsens receiver;
- the future Unity bridge.

The current full pipeline is:

    D-BRAN Pose-S1 (five branches, 32 hidden units)
    -> D-BRAN Pose-S2 (five branches, 16 hidden units)
    -> D-BRAN Pose-S3 (five branches, 16 hidden units)
    -> learned residual rotation fusion (16 hidden units)
    -> original TransPose Trans-B1 and Trans-B2.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

import articulate as art
from config import joint_set, paths, vel_scale
from main_path import (
    FUSION_CHECKPOINT,
    POSE_S1_CHECKPOINTS_DIR,
    POSE_S2_CHECKPOINTS_DIR,
    POSE_S3_CHECKPOINTS_DIR,
    TRANSPOSE_WEIGHTS_FILE,
)
from net import TransPoseNet
from utils import normalize_and_concat


# ---------------------------------------------------------------------------
# Sensor order and anatomical branches
# ---------------------------------------------------------------------------

LEFT_ARM_SENSOR_IDX = 0
RIGHT_ARM_SENSOR_IDX = 1
LEFT_LEG_SENSOR_IDX = 2
RIGHT_LEG_SENSOR_IDX = 3
HEAD_SENSOR_IDX = 4
ROOT_SENSOR_IDX = 5

SENSOR_ORDER = (
    "left_arm",
    "right_arm",
    "left_leg",
    "right_leg",
    "head",
    "root",
)

LEAF_TARGETS = (
    ("left_leg", LEFT_LEG_SENSOR_IDX),
    ("right_leg", RIGHT_LEG_SENSOR_IDX),
    ("head", HEAD_SENSOR_IDX),
    ("left_arm", LEFT_ARM_SENSOR_IDX),
    ("right_arm", RIGHT_ARM_SENSOR_IDX),
)

LEAF_ORDER = tuple(name for name, _ in LEAF_TARGETS)

POSE_S2_BRANCH_ORDER = (
    "left_leg",
    "right_leg",
    "trunk_head",
    "left_arm",
    "right_arm",
)

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

POSE_S3_BRANCH_ORDER = POSE_S2_BRANCH_ORDER

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

REDUCED_JOINTS = tuple(int(index) for index in joint_set.reduced)


# ---------------------------------------------------------------------------
# Public result containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DBranOfflineOutput:
    """Complete output of an offline D-BRAN sequence inference."""

    pose: torch.Tensor
    translation: torch.Tensor
    imu: torch.Tensor
    leaf_positions: torch.Tensor
    full_positions: torch.Tensor
    reduced_pose_6d: torch.Tensor


@dataclass(frozen=True)
class DBranOnlineOutput:
    """Output generated after one new real-time IMU frame is appended."""

    pose: torch.Tensor
    translation: torch.Tensor
    imu: torch.Tensor
    leaf_positions: torch.Tensor
    full_positions: torch.Tensor
    reduced_pose_6d: torch.Tensor
    input_frame_index: int
    output_frame_index: int
    has_future_context: bool
    has_full_window: bool


@dataclass(frozen=True)
class DBranOnlineSequenceOutput:
    """Stacked online outputs produced from a complete recorded sequence."""

    pose: torch.Tensor
    translation: torch.Tensor


# ---------------------------------------------------------------------------
# Network definitions used by the distributed checkpoints
# ---------------------------------------------------------------------------

class DistributedPoseS1(nn.Module):
    def __init__(
        self,
        input_dim: int = 24,
        output_dim: int = 3,
        proj_dim: int = 32,
        rnn_hidden: int = 32,
        rnn_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
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
        input_dim: int,
        output_dim: int,
        proj_dim: int = 16,
        rnn_hidden: int = 16,
        rnn_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
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
        input_dim: int = 90,
        output_dim: int = 90,
        proj_dim: int = 16,
        rnn_hidden: int = 16,
        rnn_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
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


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return resolved


def _require_file(path: Union[str, os.PathLike], description: str) -> str:
    resolved = os.fspath(Path(path).expanduser().resolve())
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def _require_directory(path: Union[str, os.PathLike], description: str) -> str:
    resolved = os.fspath(Path(path).expanduser().resolve())
    if not os.path.isdir(resolved):
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def _remap_state_dict_if_needed(
    state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    remapped: Dict[str, torch.Tensor] = {}
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


def _checkpoint_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise RuntimeError(
            f"Unexpected checkpoint type: {type(checkpoint).__name__}"
        )

    if not isinstance(state_dict, Mapping):
        raise RuntimeError("Checkpoint model_state_dict is not a mapping.")
    return state_dict


def _find_checkpoint(
    root_or_file: str,
    target: str,
    candidates: Sequence[str],
) -> str:
    if os.path.isfile(root_or_file):
        return root_or_file

    paths_to_try = [
        os.path.join(root_or_file, target, pattern.format(target=target))
        for pattern in candidates
    ]
    for candidate_path in paths_to_try:
        if os.path.isfile(candidate_path):
            return candidate_path

    matches = sorted(
        glob.glob(
            os.path.join(root_or_file, "**", f"*{target}*.pth"),
            recursive=True,
        )
    )
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find a checkpoint for target={target}. Tried:\n"
        + "\n".join(paths_to_try)
    )


def _find_fusion_checkpoint(path_or_root: str) -> str:
    if os.path.isfile(path_or_root):
        return path_or_root

    candidates = (
        "best_fusion.pth",
    )
    for filename in candidates:
        candidate_path = os.path.join(path_or_root, filename)
        if os.path.isfile(candidate_path):
            return candidate_path

    matches = sorted(
        glob.glob(os.path.join(path_or_root, "**", "*fusion*.pth"), recursive=True)
    )
    if matches:
        return matches[0]

    raise FileNotFoundError(f"No fusion checkpoint found in {path_or_root}")


@torch.inference_mode()
def _dense_stage_forward(model: nn.Module, sequence: torch.Tensor) -> torch.Tensor:
    """Run Linear -> LSTM -> Linear on a dense [T, D] sequence."""
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


def _ensure_acc_6x3(acc: torch.Tensor) -> torch.Tensor:
    if acc.ndim == 3 and tuple(acc.shape[1:]) == (6, 3):
        return acc.float()
    if acc.ndim == 2 and acc.shape[1] == 18:
        return acc.reshape(acc.shape[0], 6, 3).float()
    raise ValueError(
        "Acceleration must have shape [T, 6, 3] or [T, 18], "
        f"but received {tuple(acc.shape)}."
    )


def _ensure_ori_6x3x3(ori: torch.Tensor) -> torch.Tensor:
    if ori.ndim == 4 and tuple(ori.shape[1:]) == (6, 3, 3):
        return ori.float()
    if ori.ndim == 2 and ori.shape[1] == 54:
        return ori.reshape(ori.shape[0], 6, 3, 3).float()
    raise ValueError(
        "Orientation must have shape [T, 6, 3, 3] or [T, 54], "
        f"but received {tuple(ori.shape)}."
    )


def _ensure_acc_frame(acc_frame: torch.Tensor) -> torch.Tensor:
    if acc_frame.ndim == 3 and acc_frame.shape[0] == 1:
        acc_frame = acc_frame[0]
    if acc_frame.ndim == 1 and acc_frame.numel() == 18:
        acc_frame = acc_frame.reshape(6, 3)
    if acc_frame.ndim != 2 or tuple(acc_frame.shape) != (6, 3):
        raise ValueError(
            "An online acceleration frame must have shape [6, 3] or [18], "
            f"but received {tuple(acc_frame.shape)}."
        )
    return acc_frame.float()


def _ensure_ori_frame(ori_frame: torch.Tensor) -> torch.Tensor:
    if ori_frame.ndim == 4 and ori_frame.shape[0] == 1:
        ori_frame = ori_frame[0]
    if ori_frame.ndim == 1 and ori_frame.numel() == 54:
        ori_frame = ori_frame.reshape(6, 3, 3)
    if ori_frame.ndim != 3 or tuple(ori_frame.shape) != (6, 3, 3):
        raise ValueError(
            "An online orientation frame must have shape [6, 3, 3] or [54], "
            f"but received {tuple(ori_frame.shape)}."
        )
    return ori_frame.float()


def _full_position_subset(
    full_position: torch.Tensor,
    joint_indices: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    full_position = full_position.to(device, non_blocking=True).float()
    parts = []
    for joint_idx in joint_indices:
        if joint_idx == 0:
            raise ValueError("The Pose-S2 full-position vector excludes root joint 0.")
        start = (joint_idx - 1) * 3
        parts.append(full_position[:, start : start + 3])
    return torch.cat(parts, dim=1).float()


# ---------------------------------------------------------------------------
# Online ring buffer
# ---------------------------------------------------------------------------

class _IMURingBuffer:
    def __init__(self, window_size: int, device: torch.device) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive.")

        self.window_size = int(window_size)
        self.device = device
        self.acc = torch.empty(
            self.window_size,
            6,
            3,
            dtype=torch.float32,
            device=device,
        )
        self.ori = torch.empty(
            self.window_size,
            6,
            3,
            3,
            dtype=torch.float32,
            device=device,
        )
        self.indices = torch.arange(
            self.window_size,
            dtype=torch.long,
            device=device,
        )
        self.reset()

    def reset(self) -> None:
        self.write_pos = 0
        self.received_frames = 0
        self.initialized = False

    def append(
        self,
        acc_frame: torch.Tensor,
        ori_frame: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        acc_frame = acc_frame.to(self.device, non_blocking=True)
        ori_frame = ori_frame.to(self.device, non_blocking=True)

        if not self.initialized:
            self.acc.copy_(acc_frame.unsqueeze(0).expand_as(self.acc))
            self.ori.copy_(ori_frame.unsqueeze(0).expand_as(self.ori))
            self.write_pos = 0
            self.initialized = True
        else:
            self.acc[self.write_pos].copy_(acc_frame)
            self.ori[self.write_pos].copy_(ori_frame)
            self.write_pos = (self.write_pos + 1) % self.window_size

        self.received_frames += 1
        ordered_indices = torch.remainder(
            self.indices + self.write_pos,
            self.window_size,
        )
        return {
            "acc": self.acc.index_select(0, ordered_indices),
            "ori": self.ori.index_select(0, ordered_indices),
        }


# ---------------------------------------------------------------------------
# D-BRAN pipeline
# ---------------------------------------------------------------------------

class DBranPipeline:
    """Load and execute the complete validated D-BRAN pipeline."""

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = "auto",
        num_past_frame: int = 20,
        num_future_frame: int = 5,
        use_cuda_streams: bool = False,
        transpose_weights: Union[str, os.PathLike] = TRANSPOSE_WEIGHTS_FILE,
        pose_s1_root: Union[str, os.PathLike] = POSE_S1_CHECKPOINTS_DIR,
        pose_s2_root: Union[str, os.PathLike] = POSE_S2_CHECKPOINTS_DIR,
        pose_s3_root: Union[str, os.PathLike] = POSE_S3_CHECKPOINTS_DIR,
        fusion_checkpoint: Union[str, os.PathLike] = FUSION_CHECKPOINT,
        verbose: bool = True,
    ) -> None:
        self.device = _resolve_device(device)
        self.num_past_frame = int(num_past_frame)
        self.num_future_frame = int(num_future_frame)
        self.window_size = self.num_past_frame + self.num_future_frame + 1
        self.center_index = self.num_past_frame
        self.use_cuda_streams = bool(
            use_cuda_streams and self.device.type == "cuda"
        )
        self.verbose = bool(verbose)

        if self.num_past_frame < 0 or self.num_future_frame < 0:
            raise ValueError("Past and future frame counts cannot be negative.")

        self.transpose_weights = _require_file(
            transpose_weights,
            "Original TransPose checkpoint",
        )
        self.pose_s1_root = _require_directory(
            pose_s1_root,
            "Pose-S1 checkpoint directory",
        )
        self.pose_s2_root = _require_directory(
            pose_s2_root,
            "Pose-S2 checkpoint directory",
        )
        self.pose_s3_root = _require_directory(
            pose_s3_root,
            "Pose-S3 checkpoint directory",
        )

        fusion_path = Path(fusion_checkpoint).expanduser().resolve()
        if fusion_path.is_dir():
            self.fusion_checkpoint = _find_fusion_checkpoint(
                os.fspath(fusion_path)
            )
        else:
            self.fusion_checkpoint = _require_file(
                fusion_path,
                "Fusion checkpoint",
            )

        self.translation_net = self._load_transpose_net()
        self.pose_s1_models = self._load_all_pose_s1_models()
        self.pose_s2_models = self._load_all_pose_s2_models()
        self.pose_s3_models = self._load_all_pose_s3_models()
        self.fusion = self._load_fusion()

        self.pose_s1_streams = self._create_streams(LEAF_ORDER)
        self.pose_s2_streams = self._create_streams(POSE_S2_BRANCH_ORDER)
        self.pose_s3_streams = self._create_streams(POSE_S3_BRANCH_ORDER)

        self._online_buffer = _IMURingBuffer(self.window_size, self.device)
        self.reset_online_state()

        if self.verbose:
            print(f"[D-BRAN] Device: {self.device}")
            print(f"[D-BRAN] Online window: {self.window_size} frames "
                  f"({self.num_past_frame} past + current + "
                  f"{self.num_future_frame} future)")
            print(f"[D-BRAN] CUDA streams: {self.use_cuda_streams}")

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def _load_transpose_net(self) -> TransPoseNet:
        old_weights = paths.weights_file
        try:
            paths.weights_file = self.transpose_weights
            model = TransPoseNet(
                num_past_frame=self.num_past_frame,
                num_future_frame=self.num_future_frame,
            ).to(self.device)
        finally:
            paths.weights_file = old_weights

        checkpoint = torch.load(
            self.transpose_weights,
            map_location=self.device,
            weights_only=False,
        )
        model.load_state_dict(_checkpoint_state_dict(checkpoint))
        model.eval()
        return model

    def _load_pose_s1_model(self, target: str) -> Dict[str, object]:
        checkpoint_path = _find_checkpoint(
            self.pose_s1_root,
            target,
            (
                "best_pose_s1_{target}.pth",
            ),
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f"Unexpected Pose-S1 checkpoint: {checkpoint_path}")

        model = DistributedPoseS1(
            input_dim=int(checkpoint.get("input_dim", 24)),
            output_dim=int(checkpoint.get("output_dim", 3)),
            proj_dim=int(checkpoint.get("proj_dim", 32)),
            rnn_hidden=int(checkpoint.get("rnn_hidden", 32)),
            rnn_layers=int(checkpoint.get("rnn_layers", 2)),
            dropout=float(checkpoint.get("dropout", 0.2)),
        ).to(self.device)
        model.load_state_dict(
            _remap_state_dict_if_needed(_checkpoint_state_dict(checkpoint))
        )
        model.eval()

        if self.verbose:
            print(f"[Pose-S1] {target}: {checkpoint_path}")

        return {
            "model": model,
            "checkpoint_path": checkpoint_path,
            "sensor_idx": dict(LEAF_TARGETS)[target],
        }

    def _load_all_pose_s1_models(self) -> Dict[str, Dict[str, object]]:
        return {
            target: self._load_pose_s1_model(target)
            for target in LEAF_ORDER
        }

    def _load_pose_s2_model(self, target: str) -> Dict[str, object]:
        checkpoint_path = _find_checkpoint(
            self.pose_s2_root,
            target,
            (
                "best_pose_s2_{target}.pth",
            ),
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f"Unexpected Pose-S2 checkpoint: {checkpoint_path}")

        branch_config = POSE_S2_BRANCH_CONFIG[target]
        model = FiveBranchRNN(
            input_dim=int(checkpoint.get("input_dim", branch_config["input_dim"])),
            output_dim=int(checkpoint.get("output_dim", branch_config["output_dim"])),
            proj_dim=int(checkpoint.get("proj_dim", 16)),
            rnn_hidden=int(checkpoint.get("rnn_hidden", 16)),
            rnn_layers=int(checkpoint.get("rnn_layers", 2)),
            dropout=float(checkpoint.get("dropout", 0.2)),
        ).to(self.device)
        model.load_state_dict(
            _remap_state_dict_if_needed(_checkpoint_state_dict(checkpoint))
        )
        model.eval()

        if self.verbose:
            print(f"[Pose-S2] {target}: {checkpoint_path}")

        return {
            "model": model,
            "checkpoint_path": checkpoint_path,
            "config": branch_config,
        }

    def _load_all_pose_s2_models(self) -> Dict[str, Dict[str, object]]:
        return {
            target: self._load_pose_s2_model(target)
            for target in POSE_S2_BRANCH_ORDER
        }

    def _load_pose_s3_model(self, target: str) -> Dict[str, object]:
        checkpoint_path = _find_checkpoint(
            self.pose_s3_root,
            target,
            (
                "best_pose_s3_{target}.pth",
            ),
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f"Unexpected Pose-S3 checkpoint: {checkpoint_path}")

        branch_config = POSE_S3_BRANCH_CONFIG[target]
        model = FiveBranchRNN(
            input_dim=int(checkpoint.get("input_dim", branch_config["input_dim"])),
            output_dim=int(checkpoint.get("output_dim", branch_config["output_dim"])),
            proj_dim=int(checkpoint.get("proj_dim", 16)),
            rnn_hidden=int(checkpoint.get("rnn_hidden", 16)),
            rnn_layers=int(checkpoint.get("rnn_layers", 2)),
            dropout=float(checkpoint.get("dropout", 0.2)),
        ).to(self.device)
        model.load_state_dict(
            _remap_state_dict_if_needed(_checkpoint_state_dict(checkpoint))
        )
        model.eval()

        if self.verbose:
            print(f"[Pose-S3] {target}: {checkpoint_path}")

        return {
            "model": model,
            "checkpoint_path": checkpoint_path,
            "config": branch_config,
        }

    def _load_all_pose_s3_models(self) -> Dict[str, Dict[str, object]]:
        return {
            target: self._load_pose_s3_model(target)
            for target in POSE_S3_BRANCH_ORDER
        }

    def _load_fusion(self) -> Dict[str, object]:
        checkpoint = torch.load(
            self.fusion_checkpoint,
            map_location=self.device,
            weights_only=False,
        )
        if not isinstance(checkpoint, dict):
            raise RuntimeError(
                f"Unexpected fusion checkpoint: {self.fusion_checkpoint}"
            )

        model = FusionNet(
            input_dim=int(checkpoint.get("input_dim", 90)),
            output_dim=int(checkpoint.get("output_dim", 90)),
            proj_dim=int(checkpoint.get("proj_dim", 16)),
            rnn_hidden=int(checkpoint.get("rnn_hidden", 16)),
            rnn_layers=int(checkpoint.get("rnn_layers", 1)),
            dropout=float(checkpoint.get("dropout", 0.2)),
        ).to(self.device)
        model.load_state_dict(
            _remap_state_dict_if_needed(_checkpoint_state_dict(checkpoint))
        )
        model.eval()

        use_pose_s2_position = bool(
            checkpoint.get("use_pose_s2_position", False)
        )

        if self.verbose:
            print(
                f"[Fusion] {self.fusion_checkpoint} | "
                f"use_pose_s2_position={use_pose_s2_position}"
            )

        return {
            "model": model,
            "checkpoint_path": self.fusion_checkpoint,
            "use_pose_s2_position": use_pose_s2_position,
        }

    def _create_streams(
        self,
        names: Sequence[str],
    ) -> Optional[Dict[str, torch.cuda.Stream]]:
        if not self.use_cuda_streams:
            return None
        return {
            name: torch.cuda.Stream(device=self.device)
            for name in names
        }

    # ------------------------------------------------------------------
    # Distributed pose stages
    # ------------------------------------------------------------------

    def _build_two_imu_input(
        self,
        acc: torch.Tensor,
        ori: torch.Tensor,
        root_idx: int,
        local_idx: int,
    ) -> torch.Tensor:
        acc = _ensure_acc_6x3(acc).to(self.device, non_blocking=True)
        ori = _ensure_ori_6x3x3(ori).to(self.device, non_blocking=True)

        return torch.cat(
            (
                acc[:, root_idx, :],
                ori[:, root_idx].reshape(acc.shape[0], 9),
                acc[:, local_idx, :],
                ori[:, local_idx].reshape(acc.shape[0], 9),
            ),
            dim=1,
        ).float()

    def _build_multi_imu_input(
        self,
        acc: torch.Tensor,
        ori: torch.Tensor,
        sensor_indices: Sequence[int],
    ) -> torch.Tensor:
        acc = _ensure_acc_6x3(acc).to(self.device, non_blocking=True)
        ori = _ensure_ori_6x3x3(ori).to(self.device, non_blocking=True)

        parts = []
        for sensor_idx in sensor_indices:
            parts.append(acc[:, sensor_idx, :])
            parts.append(ori[:, sensor_idx].reshape(acc.shape[0], 9))
        return torch.cat(parts, dim=1).float()

    def _run_parallel_branches(
        self,
        models: Mapping[str, Mapping[str, object]],
        inputs: Mapping[str, torch.Tensor],
        order: Sequence[str],
        streams: Optional[Mapping[str, torch.cuda.Stream]],
    ) -> Dict[str, torch.Tensor]:
        if streams is None:
            return {
                name: _dense_stage_forward(
                    models[name]["model"],  # type: ignore[arg-type]
                    inputs[name],
                )
                for name in order
            }

        main_stream = torch.cuda.current_stream(device=self.device)
        outputs: Dict[str, torch.Tensor] = {}

        for name in order:
            stream = streams[name]
            stream.wait_stream(main_stream)
            with torch.cuda.stream(stream):
                output = _dense_stage_forward(
                    models[name]["model"],  # type: ignore[arg-type]
                    inputs[name],
                )
                output.record_stream(stream)
                outputs[name] = output

        for name in order:
            main_stream.wait_stream(streams[name])

        return outputs

    def _build_pose_s1_inputs(
        self,
        raw: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return {
            target: self._build_two_imu_input(
                raw["acc"],
                raw["ori"],
                ROOT_SENSOR_IDX,
                sensor_idx,
            )
            for target, sensor_idx in LEAF_TARGETS
        }

    def _build_pose_s2_inputs(
        self,
        raw: Mapping[str, torch.Tensor],
        pose_s1_outputs: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        inputs: Dict[str, torch.Tensor] = {}
        for target in POSE_S2_BRANCH_ORDER:
            branch_config = POSE_S2_BRANCH_CONFIG[target]
            imu_features = self._build_two_imu_input(
                raw["acc"],
                raw["ori"],
                ROOT_SENSOR_IDX,
                int(branch_config["local_sensor_idx"]),
            )
            leaf_name = str(branch_config["s1_leaf_name"])
            local_leaf = pose_s1_outputs[leaf_name].to(
                self.device,
                non_blocking=True,
            ).float()
            length = min(imu_features.shape[0], local_leaf.shape[0])
            inputs[target] = torch.cat(
                (imu_features[:length], local_leaf[:length]),
                dim=1,
            ).float()
        return inputs

    def _assemble_pose_s2(
        self,
        pose_s2_outputs: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        length = min(output.shape[0] for output in pose_s2_outputs.values())
        full = torch.zeros(
            length,
            69,
            dtype=torch.float32,
            device=self.device,
        )
        full_view = full.reshape(length, 23, 3)

        for target in POSE_S2_BRANCH_ORDER:
            joints = list(POSE_S2_BRANCH_CONFIG[target]["joints"])
            prediction = pose_s2_outputs[target][:length].reshape(
                length,
                len(joints),
                3,
            )
            for local_idx, joint_idx in enumerate(joints):
                full_view[:, joint_idx - 1, :] = prediction[:, local_idx, :]
        return full

    def _build_pose_s3_inputs(
        self,
        raw: Mapping[str, torch.Tensor],
        full_positions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        inputs: Dict[str, torch.Tensor] = {}
        for target in POSE_S3_BRANCH_ORDER:
            branch_config = POSE_S3_BRANCH_CONFIG[target]
            imu_features = self._build_multi_imu_input(
                raw["acc"],
                raw["ori"],
                branch_config["sensor_indices"],  # type: ignore[arg-type]
            )
            position_features = _full_position_subset(
                full_positions,
                branch_config["position_joints"],  # type: ignore[arg-type]
                self.device,
            )
            length = min(imu_features.shape[0], position_features.shape[0])
            inputs[target] = torch.cat(
                (imu_features[:length], position_features[:length]),
                dim=1,
            ).float()
        return inputs

    def _assemble_pose_s3(
        self,
        pose_s3_outputs: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        length = min(output.shape[0] for output in pose_s3_outputs.values())
        reduced = torch.zeros(
            length,
            len(REDUCED_JOINTS) * 6,
            dtype=torch.float32,
            device=self.device,
        )
        reduced_view = reduced.reshape(length, len(REDUCED_JOINTS), 6)

        for target in POSE_S3_BRANCH_ORDER:
            reduced_joints = list(
                POSE_S3_BRANCH_CONFIG[target]["reduced_joints"]
            )
            prediction = pose_s3_outputs[target][:length].reshape(
                length,
                len(reduced_joints),
                6,
            )
            for local_idx, joint_idx in enumerate(reduced_joints):
                reduced_idx = REDUCED_JOINTS.index(joint_idx)
                reduced_view[:, reduced_idx, :] = prediction[:, local_idx, :]
        return reduced

    def _apply_fusion(
        self,
        assembled_reduced: torch.Tensor,
        full_positions: torch.Tensor,
    ) -> torch.Tensor:
        if bool(self.fusion["use_pose_s2_position"]):
            length = min(
                assembled_reduced.shape[0],
                full_positions.shape[0],
            )
            fusion_input = torch.cat(
                (
                    assembled_reduced[:length],
                    full_positions[:length],
                ),
                dim=1,
            ).float()
            assembled_reduced = assembled_reduced[:length]
        else:
            fusion_input = assembled_reduced.float()

        delta = _dense_stage_forward(
            self.fusion["model"],  # type: ignore[arg-type]
            fusion_input,
        )
        length = min(assembled_reduced.shape[0], delta.shape[0])
        return assembled_reduced[:length] + delta[:length]

    @torch.inference_mode()
    def forward_pose(
        self,
        acc: torch.Tensor,
        ori: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run only the distributed pose stages and return GPU intermediates."""
        raw = {
            "acc": _ensure_acc_6x3(acc),
            "ori": _ensure_ori_6x3x3(ori),
        }

        pose_s1_inputs = self._build_pose_s1_inputs(raw)
        pose_s1_outputs = self._run_parallel_branches(
            self.pose_s1_models,
            pose_s1_inputs,
            LEAF_ORDER,
            self.pose_s1_streams,
        )
        length = min(output.shape[0] for output in pose_s1_outputs.values())
        pose_s1_outputs = {
            name: output[:length]
            for name, output in pose_s1_outputs.items()
        }
        leaf_positions = torch.cat(
            [pose_s1_outputs[name] for name in LEAF_ORDER],
            dim=1,
        )

        pose_s2_inputs = self._build_pose_s2_inputs(raw, pose_s1_outputs)
        pose_s2_outputs = self._run_parallel_branches(
            self.pose_s2_models,
            pose_s2_inputs,
            POSE_S2_BRANCH_ORDER,
            self.pose_s2_streams,
        )
        full_positions = self._assemble_pose_s2(pose_s2_outputs)

        pose_s3_inputs = self._build_pose_s3_inputs(raw, full_positions)
        pose_s3_outputs = self._run_parallel_branches(
            self.pose_s3_models,
            pose_s3_inputs,
            POSE_S3_BRANCH_ORDER,
            self.pose_s3_streams,
        )
        assembled_reduced = self._assemble_pose_s3(pose_s3_outputs)
        reduced_pose_6d = self._apply_fusion(
            assembled_reduced,
            full_positions,
        )

        imu = normalize_and_concat(raw["acc"], raw["ori"]).float().to(
            self.device,
            non_blocking=True,
        )
        length = min(
            imu.shape[0],
            leaf_positions.shape[0],
            full_positions.shape[0],
            reduced_pose_6d.shape[0],
        )
        return (
            imu[:length],
            leaf_positions[:length],
            full_positions[:length],
            reduced_pose_6d[:length],
        )

    # ------------------------------------------------------------------
    # Translation and full-pose conversion
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _translation_offline(
        self,
        imu: torch.Tensor,
        leaf_positions: torch.Tensor,
        full_positions: torch.Tensor,
        pose: torch.Tensor,
    ) -> torch.Tensor:
        length = min(
            imu.shape[0],
            leaf_positions.shape[0],
            full_positions.shape[0],
            pose.shape[0],
        )
        imu = imu[:length]
        leaf_positions = leaf_positions[:length]
        full_positions = full_positions[:length]
        pose = pose[:length]

        contact_probability, _ = self.translation_net.tran_b1.forward(
            torch.cat((leaf_positions, imu), dim=1)
        )
        velocity_prediction, _ = self.translation_net.tran_b2.forward(
            torch.cat((full_positions, imu), dim=1),
            None,
        )
        root_rotation = imu[:, -9:].reshape(-1, 3, 3)

        joints = art.math.forward_kinematics(
            pose[:, joint_set.lower_body],
            self.translation_net.lower_body_bone.expand(pose.shape[0], -1, -1),
            joint_set.lower_body_parent,
        )[1]

        tran_b1_velocity = self.translation_net.gravity_velocity + art.math.lerp(
            torch.cat(
                (
                    torch.zeros(1, 3, device=joints.device),
                    joints[:-1, 7] - joints[1:, 7],
                )
            ),
            torch.cat(
                (
                    torch.zeros(1, 3, device=joints.device),
                    joints[:-1, 8] - joints[1:, 8],
                )
            ),
            contact_probability.max(dim=1).indices.reshape(-1, 1).cpu(),
        )
        tran_b2_velocity = (
            root_rotation.bmm(velocity_prediction.unsqueeze(-1))
            .squeeze(-1)
            .cpu()
            * vel_scale
            / 60
        )
        weight = self.translation_net._prob_to_weight(
            contact_probability.cpu().max(dim=1).values.sigmoid()
        ).reshape(-1, 1)
        velocity = art.math.lerp(
            tran_b2_velocity,
            tran_b1_velocity,
            weight,
        )

        current_root_y = 0.0
        for frame_index in range(velocity.shape[0]):
            current_foot_y = (
                current_root_y
                + joints[frame_index, 7:9, 1].min().item()
            )
            if (
                current_foot_y + velocity[frame_index, 1].item()
                <= self.translation_net.floor_y
            ):
                velocity[frame_index, 1] = (
                    self.translation_net.floor_y - current_foot_y
                )
            current_root_y += velocity[frame_index, 1].item()

        return self.translation_net.velocity_to_root_position(velocity)

    @torch.inference_mode()
    def forward_offline(
        self,
        acc: torch.Tensor,
        ori: torch.Tensor,
    ) -> DBranOfflineOutput:
        """Run the complete D-BRAN pipeline on a recorded sequence."""
        imu, leaf_positions, full_positions, reduced_pose_6d = self.forward_pose(
            acc,
            ori,
        )

        root_rotation = imu[:, -9:].reshape(-1, 3, 3)
        pose = self.translation_net._reduced_glb_6d_to_full_local_mat(
            root_rotation.cpu(),
            reduced_pose_6d.cpu(),
        )
        translation = self._translation_offline(
            imu,
            leaf_positions,
            full_positions,
            pose,
        )

        return DBranOfflineOutput(
            pose=pose,
            translation=translation,
            imu=imu.detach().cpu(),
            leaf_positions=leaf_positions.detach().cpu(),
            full_positions=full_positions.detach().cpu(),
            reduced_pose_6d=reduced_pose_6d.detach().cpu(),
        )

    # ------------------------------------------------------------------
    # Real-time inference
    # ------------------------------------------------------------------

    def reset_online_state(self) -> None:
        """Reset the temporal window and translation state."""
        self._online_buffer.reset()
        self._online_rnn_state = None
        self._online_current_root_y = 0.0
        self._online_last_lfoot_pos = self.translation_net.feet_pos[0].clone()
        self._online_last_rfoot_pos = self.translation_net.feet_pos[1].clone()
        self._online_last_root_pos = torch.zeros(3)
        self._online_input_frame_index = -1

    @property
    def online_received_frames(self) -> int:
        return self._online_buffer.received_frames

    @torch.inference_mode()
    def forward_online(
        self,
        acc_frame: torch.Tensor,
        ori_frame: torch.Tensor,
    ) -> DBranOnlineOutput:
        """
        Append one synchronized six-IMU frame and produce one delayed output.

        Once the stream is established, the returned pose corresponds to
        `num_future_frame` frames before the newest input frame.
        """
        acc_frame = _ensure_acc_frame(acc_frame)
        ori_frame = _ensure_ori_frame(ori_frame)
        raw_window = self._online_buffer.append(acc_frame, ori_frame)
        self._online_input_frame_index += 1

        imu, leaf_positions, full_positions, reduced_pose_6d = self.forward_pose(
            raw_window["acc"],
            raw_window["ori"],
        )

        contact_probability, _ = self.translation_net.tran_b1.forward(
            torch.cat((leaf_positions, imu), dim=1)
        )
        velocity_prediction, self._online_rnn_state = (
            self.translation_net.tran_b2.forward(
                torch.cat((full_positions, imu), dim=1),
                self._online_rnn_state,
            )
        )

        center = self.center_index
        contact_probability = (
            contact_probability[center].sigmoid().reshape(-1).cpu()
        )
        root_rotation = imu[center, -9:].reshape(3, 3).cpu()
        center_reduced_pose = reduced_pose_6d[center].cpu()
        pose = self.translation_net._reduced_glb_6d_to_full_local_mat(
            root_rotation,
            center_reduced_pose,
        ).squeeze(0)

        left_foot_position, right_foot_position = art.math.forward_kinematics(
            pose[joint_set.lower_body].unsqueeze(0),
            self.translation_net.lower_body_bone.unsqueeze(0),
            joint_set.lower_body_parent,
        )[1][0, 7:9]

        if contact_probability[0] > contact_probability[1]:
            tran_b1_velocity = (
                self._online_last_lfoot_pos
                - left_foot_position
                + self.translation_net.gravity_velocity
            )
        else:
            tran_b1_velocity = (
                self._online_last_rfoot_pos
                - right_foot_position
                + self.translation_net.gravity_velocity
            )

        tran_b2_velocity = (
            root_rotation.mm(
                velocity_prediction[center].cpu().reshape(3, 1)
            ).reshape(3)
            / 60
            * vel_scale
        )
        weight = self.translation_net._prob_to_weight(
            contact_probability.max()
        )
        velocity = art.math.lerp(
            tran_b2_velocity,
            tran_b1_velocity,
            weight,
        )

        current_foot_y = self._online_current_root_y + min(
            left_foot_position[1].item(),
            right_foot_position[1].item(),
        )
        if current_foot_y + velocity[1].item() <= self.translation_net.floor_y:
            velocity[1] = self.translation_net.floor_y - current_foot_y

        self._online_current_root_y += velocity[1].item()
        self._online_last_lfoot_pos = left_foot_position
        self._online_last_rfoot_pos = right_foot_position
        self._online_last_root_pos += velocity

        output_frame_index = (
            self._online_input_frame_index - self.num_future_frame
        )

        return DBranOnlineOutput(
            pose=pose,
            translation=self._online_last_root_pos.clone(),
            imu=imu[center].detach().cpu(),
            leaf_positions=leaf_positions[center].detach().cpu(),
            full_positions=full_positions[center].detach().cpu(),
            reduced_pose_6d=center_reduced_pose.detach().cpu(),
            input_frame_index=self._online_input_frame_index,
            output_frame_index=output_frame_index,
            has_future_context=output_frame_index >= 0,
            has_full_window=(
                self._online_buffer.received_frames >= self.window_size
            ),
        )

    @torch.inference_mode()
    def forward_online_sequence(
        self,
        acc: torch.Tensor,
        ori: torch.Tensor,
        pad_future: bool = True,
    ) -> DBranOnlineSequenceOutput:
        """
        Replay a recorded sequence through the real-time path.

        With `pad_future=True`, the last input frame is repeated
        `num_future_frame` times and the initial warm-up outputs are removed.
        This matches the current profiler's online evaluation protocol.
        """
        acc = _ensure_acc_6x3(acc)
        ori = _ensure_ori_6x3x3(ori)
        if acc.shape[0] != ori.shape[0]:
            raise ValueError(
                "Acceleration and orientation sequence lengths do not match: "
                f"{acc.shape[0]} vs {ori.shape[0]}."
            )
        if acc.shape[0] == 0:
            raise ValueError("Cannot run an empty sequence.")

        if pad_future and self.num_future_frame > 0:
            acc_sequence = torch.cat(
                (acc, acc[-1:].repeat(self.num_future_frame, 1, 1)),
                dim=0,
            )
            ori_sequence = torch.cat(
                (ori, ori[-1:].repeat(self.num_future_frame, 1, 1, 1)),
                dim=0,
            )
        else:
            acc_sequence = acc
            ori_sequence = ori

        self.reset_online_state()
        poses = []
        translations = []
        for acc_frame, ori_frame in zip(acc_sequence, ori_sequence):
            output = self.forward_online(acc_frame, ori_frame)
            poses.append(output.pose)
            translations.append(output.translation)

        pose = torch.stack(poses)
        translation = torch.stack(translations)

        if pad_future and self.num_future_frame > 0:
            pose = pose[self.num_future_frame :]
            translation = translation[self.num_future_frame :]

        return DBranOnlineSequenceOutput(
            pose=pose,
            translation=translation,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def parameter_counts(self) -> Dict[str, int]:
        """Return parameter counts for each D-BRAN component."""
        pose_s1 = sum(
            sum(parameter.numel() for parameter in entry["model"].parameters())
            for entry in self.pose_s1_models.values()
        )
        pose_s2 = sum(
            sum(parameter.numel() for parameter in entry["model"].parameters())
            for entry in self.pose_s2_models.values()
        )
        pose_s3 = sum(
            sum(parameter.numel() for parameter in entry["model"].parameters())
            for entry in self.pose_s3_models.values()
        )
        fusion = sum(
            parameter.numel()
            for parameter in self.fusion["model"].parameters()
        )
        translation = sum(
            parameter.numel()
            for module in (
                self.translation_net.tran_b1,
                self.translation_net.tran_b2,
            )
            for parameter in module.parameters()
        )
        return {
            "pose_s1": pose_s1,
            "pose_s2": pose_s2,
            "pose_s3": pose_s3,
            "fusion": fusion,
            "translation": translation,
            "total": pose_s1 + pose_s2 + pose_s3 + fusion + translation,
        }
