"""
Precompute PoseS3 five-branch outputs and assemble them into [T, 90].

Input:
    Prepared PoseS3 five-branch files from prepare_pose_s3.py

Output per sequence:
    pose3_assembled_pred_6d_reduced [T, 90]
    pose3_target_6d_reduced         [T, 90]
    pose_s2_full_position_input     [T, 69]

These files are then used to train the learned PoseS3 fusion network.
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
import os
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence
from tqdm import tqdm



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REDUCED_JOINTS = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]

BRANCH_CONFIG: Dict[str, Dict[str, object]] = {
    "left_leg": {
        "input_dim": 36,
        "output_dim": 12,
        "reduced_joints": [1, 4],
    },
    "right_leg": {
        "input_dim": 36,
        "output_dim": 12,
        "reduced_joints": [2, 5],
    },
    "trunk_head": {
        "input_dim": 39,
        "output_dim": 30,
        "reduced_joints": [3, 6, 9, 12, 15],
    },
    "left_arm": {
        "input_dim": 39,
        "output_dim": 18,
        "reduced_joints": [13, 16, 18],
    },
    "right_arm": {
        "input_dim": 39,
        "output_dim": 18,
        "reduced_joints": [14, 17, 19],
    },
}

BRANCH_ORDER = [
    "left_leg",
    "right_leg",
    "trunk_head",
    "left_arm",
    "right_arm",
]


class PoseS3FiveBranchRegionNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        proj_dim: int = 16,
        rnn_hidden: int = 16,
        rnn_layers: int = 2,
        dropout: float = 0.2,
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

    def forward(self, sequence: PackedSequence) -> PackedSequence:
        if not isinstance(sequence, PackedSequence):
            raise RuntimeError("PoseS3FiveBranchRegionNet expects PackedSequence.")

        data = self.dropout(sequence.data)
        data = torch.relu(self.fc_in(data))

        packed = PackedSequence(
            data,
            sequence.batch_sizes,
            sequence.sorted_indices,
            sequence.unsorted_indices,
        )
        output, _ = self.rnn(packed)
        prediction = self.fc_out(output.data)

        return PackedSequence(
            prediction,
            output.batch_sizes,
            output.sorted_indices,
            output.unsorted_indices,
        )


def load_list(list_file: str) -> List[str]:
    path = Path(list_file)

    if not path.exists():
        raise FileNotFoundError(f"List file not found: {list_file}")

    files = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    return [path for path in files if os.path.exists(path)]


def find_checkpoint(weights_root: str, target: str) -> str:
    path = os.path.join(
        weights_root,
        target,
        f"best_pose_s3_{target}.pth",
    )

    if os.path.exists(path):
        return path

    raise FileNotFoundError(
        f"Could not find checkpoint for target={target}. Tried: {path}"
    )


def load_branch_model(weights_root: str, target: str):
    checkpoint_path = find_checkpoint(weights_root, target)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    config = BRANCH_CONFIG[target]
    input_dim = int(checkpoint.get("input_dim", config["input_dim"]))
    output_dim = int(checkpoint.get("output_dim", config["output_dim"]))
    proj_dim = int(checkpoint.get("proj_dim", 16))
    rnn_hidden = int(checkpoint.get("rnn_hidden", 16))
    rnn_layers = int(checkpoint.get("rnn_layers", 2))
    dropout = float(checkpoint.get("dropout", 0.2))

    model = PoseS3FiveBranchRegionNet(
        input_dim=input_dim,
        output_dim=output_dim,
        proj_dim=proj_dim,
        rnn_hidden=rnn_hidden,
        rnn_layers=rnn_layers,
        dropout=dropout,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint_path


@torch.inference_mode()
def infer_one_branch(model, branch_input: torch.Tensor) -> torch.Tensor:
    branch_input = branch_input.float().unsqueeze(0).to(DEVICE)
    lengths = torch.tensor([branch_input.shape[1]], dtype=torch.long)

    packed_input = pack_padded_sequence(
        branch_input,
        lengths.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )
    packed_prediction = model(packed_input)

    return packed_prediction.data.detach().cpu()


def insert_branch_prediction(
    assembled: torch.Tensor,
    branch_prediction: torch.Tensor,
    reduced_joints: Sequence[int],
) -> None:
    branch_prediction = branch_prediction.reshape(
        branch_prediction.shape[0],
        len(reduced_joints),
        6,
    )
    assembled_view = assembled.reshape(
        assembled.shape[0],
        len(REDUCED_JOINTS),
        6,
    )

    for local_index, joint_idx in enumerate(reduced_joints):
        reduced_index = REDUCED_JOINTS.index(joint_idx)
        assembled_view[:, reduced_index, :] = branch_prediction[:, local_index, :]


def process_one_file(
    data_path: str,
    models: Dict[str, nn.Module],
    output_dir: str,
    index: int,
) -> str:
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    sequence_length = int(data["num_frames"])

    assembled = torch.zeros(
        sequence_length,
        len(REDUCED_JOINTS) * 6,
        dtype=torch.float32,
    )
    branch_outputs = {}

    for target in BRANCH_ORDER:
        branch_input = data[f"{target}_input"].float()
        branch_prediction = infer_one_branch(models[target], branch_input)

        branch_outputs[target] = branch_prediction
        insert_branch_prediction(
            assembled,
            branch_prediction,
            BRANCH_CONFIG[target]["reduced_joints"],
        )

    output = {
        "source_file": data["source_file"],
        "pose_s2_source_file": data.get("pose_s2_source_file", ""),
        "prepared_pose_s3_source_file": data_path,
        "num_frames": sequence_length,
        "branch_order": BRANCH_ORDER,
        "branch_config": BRANCH_CONFIG,
        "reduced_joints": REDUCED_JOINTS,
        "pose3_assembled_pred_6d_reduced": assembled,
        "pose3_target_6d_reduced": data["pose3_target_6d_reduced"].float(),
        "pose_s2_full_position_input": data["pose_s2_full_position_input"].float(),
    }

    for target in BRANCH_ORDER:
        output[f"{target}_pred_6d_reduced"] = branch_outputs[target]
        output[f"{target}_target_6d_reduced"] = data[f"{target}_target_6d_reduced"].float()

    output_path = os.path.join(
        output_dir,
        f"pose_s3_pred_{index:06d}.pt",
    )
    torch.save(output, output_path)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_list_file", required=True)
    parser.add_argument("--weights_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_list_file", required=True)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = load_list(args.data_list_file)

    if not files:
        raise RuntimeError(f"No files found in {args.data_list_file}")

    models = {}
    print("Loading branch models:")
    for target in BRANCH_ORDER:
        model, checkpoint_path = load_branch_model(args.weights_root, target)
        models[target] = model
        print(f"  {target}: {checkpoint_path}")

    saved_paths = []

    for index, data_path in enumerate(
        tqdm(files, desc="Precomputing PoseS3 five-branch outputs", ncols=120)
    ):
        output_path = process_one_file(
            data_path=data_path,
            models=models,
            output_dir=args.output_dir,
            index=index,
        )
        saved_paths.append(output_path)

    output_parent = os.path.dirname(args.output_list_file)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    with open(args.output_list_file, "w", encoding="utf-8") as file:
        for path in saved_paths:
            file.write(path + "\n")

    print("\nDone.")
    print("  Saved:", len(saved_paths))
    print("  Output list:", args.output_list_file)

    sample = torch.load(saved_paths[0], map_location="cpu", weights_only=False)
    print("\nSample shapes:")
    print("  pose3_assembled_pred_6d_reduced:", tuple(sample["pose3_assembled_pred_6d_reduced"].shape))
    print("  pose3_target_6d_reduced:", tuple(sample["pose3_target_6d_reduced"].shape))
    print("  pose_s2_full_position_input:", tuple(sample["pose_s2_full_position_input"].shape))


if __name__ == "__main__":
    main()
