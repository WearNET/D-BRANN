"""
Analytical FLOPs estimator for the proposed online pose pipeline.

This script reports FLOPs per online inference window.

Convention:
    1 MAC = 2 FLOPs
    Bias additions and elementwise activations are ignored.
    LSTM FLOPs include input-to-hidden and hidden-to-hidden matrix products
    for the four LSTM gates.

Default online window:
    20 past frames + current frame + 5 future frames = 26 frames
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
from dataclasses import dataclass
from typing import List


@dataclass
class RNNBlock:
    name: str
    input_dim: int
    proj_dim: int
    hidden_dim: int
    output_dim: int
    rnn_layers: int
    bidirectional: bool = True
    copies: int = 1


def linear_flops_per_step(input_dim: int, output_dim: int) -> int:
    return 2 * input_dim * output_dim


def lstm_flops_per_step(
    input_dim: int,
    hidden_dim: int,
    layers: int,
    bidirectional: bool,
) -> int:
    directions = 2 if bidirectional else 1
    total = 0
    current_input_dim = input_dim

    for _ in range(layers):
        per_direction = 8 * hidden_dim * (current_input_dim + hidden_dim)
        total += directions * per_direction
        current_input_dim = hidden_dim * directions

    return total


def rnn_block_flops_per_window(block: RNNBlock, window_size: int) -> int:
    per_step = 0
    per_step += linear_flops_per_step(block.input_dim, block.proj_dim)
    per_step += lstm_flops_per_step(
        input_dim=block.proj_dim,
        hidden_dim=block.hidden_dim,
        layers=block.rnn_layers,
        bidirectional=block.bidirectional,
    )
    per_step += linear_flops_per_step(
        block.hidden_dim * (2 if block.bidirectional else 1),
        block.output_dim,
    )

    return per_step * window_size * block.copies


def format_flops(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.6f} GFLOPs"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.6f} MFLOPs"
    if value >= 1_000:
        return f"{value / 1_000:.6f} KFLOPs"
    return f"{value} FLOPs"


def build_final_pose_blocks(
    pose_s2_hidden: int,
    pose_s3_hidden: int,
) -> List[RNNBlock]:
    blocks = []

    # PoseS1 final: 5 branches, all identical.
    blocks.append(
        RNNBlock(
            name="PoseS1 distributed 32h",
            input_dim=24,
            proj_dim=32,
            hidden_dim=32,
            output_dim=3,
            rnn_layers=2,
            bidirectional=True,
            copies=5,
        )
    )

    # PoseS2 final candidate: five branches.
    pose_s2_outputs = [12, 12, 15, 15, 15]
    for branch_name, output_dim in zip(
        ["PoseS2 left_leg", "PoseS2 right_leg", "PoseS2 trunk_head", "PoseS2 left_arm", "PoseS2 right_arm"],
        pose_s2_outputs,
    ):
        blocks.append(
            RNNBlock(
                name=f"{branch_name} {pose_s2_hidden}h",
                input_dim=27,
                proj_dim=pose_s2_hidden,
                hidden_dim=pose_s2_hidden,
                output_dim=output_dim,
                rnn_layers=2,
                bidirectional=True,
                copies=1,
            )
        )

    # PoseS3 proposed five-branch regions.
    pose_s3_regions = [
        ("PoseS3 left_leg", 36, 12),
        ("PoseS3 right_leg", 36, 12),
        ("PoseS3 trunk_head", 39, 30),
        ("PoseS3 left_arm", 39, 18),
        ("PoseS3 right_arm", 39, 18),
    ]
    for name, input_dim, output_dim in pose_s3_regions:
        blocks.append(
            RNNBlock(
                name=f"{name} {pose_s3_hidden}h",
                input_dim=input_dim,
                proj_dim=pose_s3_hidden,
                hidden_dim=pose_s3_hidden,
                output_dim=output_dim,
                rnn_layers=2,
                bidirectional=True,
                copies=1,
            )
        )

    # Learned residual fusion.
    blocks.append(
        RNNBlock(
            name=f"Fusion {pose_s3_hidden}h",
            input_dim=90,
            proj_dim=pose_s3_hidden,
            hidden_dim=pose_s3_hidden,
            output_dim=90,
            rnn_layers=1,
            bidirectional=True,
            copies=1,
        )
    )

    return blocks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_size", type=int, default=26)
    parser.add_argument("--pose_s2_hidden", type=int, default=16, choices=[16, 24, 32])
    parser.add_argument("--pose_s3_hidden", type=int, default=16)

    args = parser.parse_args()

    blocks = build_final_pose_blocks(
        pose_s2_hidden=args.pose_s2_hidden,
        pose_s3_hidden=args.pose_s3_hidden,
    )

    stage_totals = {
        "PoseS1": 0,
        "PoseS2": 0,
        "PoseS3": 0,
        "Fusion": 0,
    }

    print("=" * 88)
    print(f"FLOPs per online window: {args.window_size} frames")
    print("Convention: 1 MAC = 2 FLOPs; bias/activation FLOPs ignored.")
    print("=" * 88)

    for block in blocks:
        flops = rnn_block_flops_per_window(block, args.window_size)

        if block.name.startswith("PoseS1"):
            stage_totals["PoseS1"] += flops
        elif block.name.startswith("PoseS2"):
            stage_totals["PoseS2"] += flops
        elif block.name.startswith("PoseS3"):
            stage_totals["PoseS3"] += flops
        elif block.name.startswith("Fusion"):
            stage_totals["Fusion"] += flops

        print(f"{block.name:<36} {flops:>15,d}  {format_flops(flops)}")

    pose_pipeline_total = sum(stage_totals.values())

    print("-" * 88)
    for stage_name in ["PoseS1", "PoseS2", "PoseS3", "Fusion"]:
        value = stage_totals[stage_name]
        print(f"{stage_name:<36} {value:>15,d}  {format_flops(value)}")
    print("-" * 88)
    print(f"{'Pose pipeline total':<36} {pose_pipeline_total:>15,d}  {format_flops(pose_pipeline_total)}")
    print("=" * 88)


if __name__ == "__main__":
    main()
