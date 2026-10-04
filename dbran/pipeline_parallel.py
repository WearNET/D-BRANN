"""
Process-parallel variant of DBranPipeline.

DBranPipeline (pipeline.py) runs the five anatomical pose branches either
fully sequentially, or concurrently via CUDA streams inside one process.
This module instead runs each branch in its own OS process, each with its
own CUDA context, communicating with a central orchestrator process through
multiprocessing queues.

This only changes HOW Stage 1 -> Stage 2 -> Stage 3 per branch is computed.
Translation (original TransPose Trans-B1/Trans-B2) and residual fusion are
untouched and still run in the orchestrator process, reusing DBranPipeline's
own loading and math (_load_transpose_net, _load_fusion, _assemble_pose_s2,
_assemble_pose_s3, _apply_fusion) so that behavior does not drift from the
validated pipeline.

Key simplification made possible by the current branch configuration: for
every branch, POSE_S3_BRANCH_CONFIG[target]["position_joints"] is exactly
POSE_S2_BRANCH_CONFIG[target]["joints"] -- i.e. each branch's Stage 3 only
ever reads back the joints it produced itself in Stage 2. So a branch's
Stage 3 can consume its own Stage 2 output directly, instead of going
through the shared `full_positions` buffer DBranPipeline builds for that
purpose. `full_positions` is still assembled here because Trans-B2
(translation) needs the combined five-branch position vector.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.multiprocessing as mp

from main_path import (
    FUSION_CHECKPOINT,
    POSE_S1_CHECKPOINTS_DIR,
    POSE_S2_CHECKPOINTS_DIR,
    POSE_S3_CHECKPOINTS_DIR,
    TRANSPOSE_WEIGHTS_FILE,
)
from utils import normalize_and_concat

from dbran.pipeline import (
    DBranPipeline,
    DistributedPoseS1,
    FiveBranchRNN,
    LEAF_TARGETS,
    POSE_S2_BRANCH_CONFIG,
    POSE_S2_BRANCH_ORDER,
    POSE_S3_BRANCH_CONFIG,
    ROOT_SENSOR_IDX,
    _checkpoint_state_dict,
    _dense_stage_forward,
    _ensure_acc_6x3,
    _ensure_ori_6x3x3,
    _find_checkpoint,
    _find_fusion_checkpoint,
    _IMURingBuffer,
    _remap_state_dict_if_needed,
    _require_directory,
    _require_file,
    _resolve_device,
)

try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass


def _two_imu_input(acc, ori, root_idx, local_idx, device):
    acc = _ensure_acc_6x3(acc).to(device, non_blocking=True)
    ori = _ensure_ori_6x3x3(ori).to(device, non_blocking=True)
    return torch.cat(
        (
            acc[:, root_idx, :],
            ori[:, root_idx].reshape(acc.shape[0], 9),
            acc[:, local_idx, :],
            ori[:, local_idx].reshape(acc.shape[0], 9),
        ),
        dim=1,
    ).float()


def _multi_imu_input(acc, ori, sensor_indices, device):
    acc = _ensure_acc_6x3(acc).to(device, non_blocking=True)
    ori = _ensure_ori_6x3x3(ori).to(device, non_blocking=True)
    parts = []
    for sensor_idx in sensor_indices:
        parts.append(acc[:, sensor_idx, :])
        parts.append(ori[:, sensor_idx].reshape(acc.shape[0], 9))
    return torch.cat(parts, dim=1).float()


def _load_branch_stage(root, target, prefix, config, device):
    path = _find_checkpoint(root, target, (f"best_{prefix}_{{target}}.pth",))
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_cls = DistributedPoseS1 if prefix == "pose_s1" else FiveBranchRNN
    model = model_cls(
        input_dim=int(checkpoint.get("input_dim", config.get("input_dim", 24))),
        output_dim=int(checkpoint.get("output_dim", config.get("output_dim", 3))),
        proj_dim=int(checkpoint.get("proj_dim", 32 if prefix == "pose_s1" else 16)),
        rnn_hidden=int(checkpoint.get("rnn_hidden", 32 if prefix == "pose_s1" else 16)),
        rnn_layers=int(checkpoint.get("rnn_layers", 2)),
        dropout=float(checkpoint.get("dropout", 0.2)),
    ).to(device)
    model.load_state_dict(_remap_state_dict_if_needed(_checkpoint_state_dict(checkpoint)))
    model.eval()
    return model


def _branch_worker_main(
    branch_name: str,
    s1_leaf_name: str,
    pose_s1_root: str,
    pose_s2_root: str,
    pose_s3_root: str,
    device_str: str,
    in_queue: "mp.Queue",
    out_queue: "mp.Queue",
    ready_queue: "mp.Queue",
) -> None:
    device = torch.device(device_str)

    s1_leaf_sensor_idx = dict(LEAF_TARGETS)[s1_leaf_name]
    s2_config = POSE_S2_BRANCH_CONFIG[branch_name]
    s3_config = POSE_S3_BRANCH_CONFIG[branch_name]

    s1_model = _load_branch_stage(pose_s1_root, s1_leaf_name, "pose_s1", {}, device)
    s2_model = _load_branch_stage(pose_s2_root, branch_name, "pose_s2", s2_config, device)
    s3_model = _load_branch_stage(pose_s3_root, branch_name, "pose_s3", s3_config, device)

    total_params = sum(
        p.numel()
        for model in (s1_model, s2_model, s3_model)
        for p in model.parameters()
    )
    ready_queue.put((branch_name, total_params))

    s2_local_idx = int(s2_config["local_sensor_idx"])
    s3_sensor_indices = s3_config["sensor_indices"]

    while True:
        item = in_queue.get()
        if item is None:
            break
        acc_cpu, ori_cpu = item

        with torch.no_grad():
            s1_in = _two_imu_input(acc_cpu, ori_cpu, ROOT_SENSOR_IDX, s1_leaf_sensor_idx, device)
            leaf_out = _dense_stage_forward(s1_model, s1_in)

            s2_imu = _two_imu_input(acc_cpu, ori_cpu, ROOT_SENSOR_IDX, s2_local_idx, device)
            length = min(s2_imu.shape[0], leaf_out.shape[0])
            s2_out = _dense_stage_forward(
                s2_model, torch.cat((s2_imu[:length], leaf_out[:length]), dim=1)
            )

            s3_imu = _multi_imu_input(acc_cpu, ori_cpu, s3_sensor_indices, device)
            length = min(s3_imu.shape[0], s2_out.shape[0])
            s3_out = _dense_stage_forward(
                s3_model, torch.cat((s3_imu[:length], s2_out[:length]), dim=1)
            )

        out_queue.put((leaf_out.cpu(), s2_out.cpu(), s3_out.cpu()))


class DBranParallelPipeline(DBranPipeline):
    """DBranPipeline with the five pose branches each in their own process."""

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = "auto",
        num_past_frame: int = 20,
        num_future_frame: int = 5,
        transpose_weights: Union[str, os.PathLike] = TRANSPOSE_WEIGHTS_FILE,
        pose_s1_root: Union[str, os.PathLike] = POSE_S1_CHECKPOINTS_DIR,
        pose_s2_root: Union[str, os.PathLike] = POSE_S2_CHECKPOINTS_DIR,
        pose_s3_root: Union[str, os.PathLike] = POSE_S3_CHECKPOINTS_DIR,
        fusion_checkpoint: Union[str, os.PathLike] = FUSION_CHECKPOINT,
        worker_device: Optional[Union[str, torch.device]] = None,
        verbose: bool = True,
    ) -> None:
        # Deliberately not calling DBranPipeline.__init__: it loads all 15
        # branch checkpoints into this process. Only the pieces that stay
        # in the orchestrator (translation + fusion) are set up here.
        self.device = _resolve_device(device)
        self.num_past_frame = int(num_past_frame)
        self.num_future_frame = int(num_future_frame)
        self.window_size = self.num_past_frame + self.num_future_frame + 1
        self.center_index = self.num_past_frame
        self.use_cuda_streams = False
        self.verbose = bool(verbose)

        self.transpose_weights = _require_file(transpose_weights, "Original TransPose checkpoint")
        self.pose_s1_root = _require_directory(pose_s1_root, "Pose-S1 checkpoint directory")
        self.pose_s2_root = _require_directory(pose_s2_root, "Pose-S2 checkpoint directory")
        self.pose_s3_root = _require_directory(pose_s3_root, "Pose-S3 checkpoint directory")

        fusion_path = Path(fusion_checkpoint).expanduser().resolve()
        self.fusion_checkpoint = (
            _find_fusion_checkpoint(os.fspath(fusion_path))
            if fusion_path.is_dir()
            else _require_file(fusion_path, "Fusion checkpoint")
        )

        self.translation_net = self._load_transpose_net()
        self.fusion = self._load_fusion()

        self.pose_s1_models = {}
        self.pose_s2_models = {}
        self.pose_s3_models = {}
        self.pose_s1_streams = None
        self.pose_s2_streams = None
        self.pose_s3_streams = None

        self._online_buffer = _IMURingBuffer(self.window_size, self.device)
        self.reset_online_state()

        worker_device_str = str(worker_device or self.device)
        self._branch_params = 0
        self._in_queues: Dict[str, "mp.Queue"] = {}
        self._out_queues: Dict[str, "mp.Queue"] = {}
        self._workers: Dict[str, "mp.Process"] = {}

        ready_queue: "mp.Queue" = mp.Queue()
        for branch_name in POSE_S2_BRANCH_ORDER:
            s1_leaf_name = POSE_S2_BRANCH_CONFIG[branch_name]["s1_leaf_name"]
            in_q: "mp.Queue" = mp.Queue()
            out_q: "mp.Queue" = mp.Queue()
            proc = mp.Process(
                target=_branch_worker_main,
                args=(
                    branch_name,
                    s1_leaf_name,
                    str(self.pose_s1_root),
                    str(self.pose_s2_root),
                    str(self.pose_s3_root),
                    worker_device_str,
                    in_q,
                    out_q,
                    ready_queue,
                ),
                daemon=True,
            )
            proc.start()
            self._in_queues[branch_name] = in_q
            self._out_queues[branch_name] = out_q
            self._workers[branch_name] = proc

        for _ in POSE_S2_BRANCH_ORDER:
            name, params = ready_queue.get(timeout=180)
            self._branch_params += params
            if self.verbose:
                print(f"[DBranParallelPipeline] worker ready: {name} ({params:,} params)")

        if self.verbose:
            print(f"[DBranParallelPipeline] Orchestrator device: {self.device}")
            print(f"[DBranParallelPipeline] Worker device:       {worker_device_str}")
            print(f"[DBranParallelPipeline] {len(self._workers)} branch processes running")

    def forward_pose(self, acc: torch.Tensor, ori: torch.Tensor):
        raw_acc = _ensure_acc_6x3(acc).detach().cpu()
        raw_ori = _ensure_ori_6x3x3(ori).detach().cpu()

        for branch_name in POSE_S2_BRANCH_ORDER:
            self._in_queues[branch_name].put((raw_acc, raw_ori))

        leaf_outputs, s2_outputs, s3_outputs = {}, {}, {}
        for branch_name in POSE_S2_BRANCH_ORDER:
            leaf_out, s2_out, s3_out = self._out_queues[branch_name].get()
            leaf_outputs[branch_name] = leaf_out.to(self.device, non_blocking=True)
            s2_outputs[branch_name] = s2_out.to(self.device, non_blocking=True)
            s3_outputs[branch_name] = s3_out.to(self.device, non_blocking=True)

        leaf_positions = torch.cat(
            [leaf_outputs[name] for name in POSE_S2_BRANCH_ORDER], dim=1
        )
        full_positions = self._assemble_pose_s2(s2_outputs)
        assembled_reduced = self._assemble_pose_s3(s3_outputs)
        reduced_pose_6d = self._apply_fusion(assembled_reduced, full_positions)

        imu = normalize_and_concat(raw_acc, raw_ori).float().to(
            self.device, non_blocking=True
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

    def parameter_counts(self) -> Dict[str, int]:
        fusion = sum(p.numel() for p in self.fusion["model"].parameters())
        translation = sum(
            p.numel()
            for module in (self.translation_net.tran_b1, self.translation_net.tran_b2)
            for p in module.parameters()
        )
        return {
            "pose_branches_total (5 processes)": self._branch_params,
            "fusion": fusion,
            "translation": translation,
            "total": self._branch_params + fusion + translation,
        }

    def shutdown(self) -> None:
        for branch_name, proc in self._workers.items():
            if proc.is_alive():
                self._in_queues[branch_name].put(None)
        for proc in self._workers.values():
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
