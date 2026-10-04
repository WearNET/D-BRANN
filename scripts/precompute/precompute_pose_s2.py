"""
Precompute Pose-S2 full-distributed outputs.

This script loads the five trained Pose-S2 branch models and assembles their
outputs into the standard Pose-S2 full joint-position vector:

    full_joint_position_pred [T, 69]

Expected prepared input files:
    data/dataset_train/train_pose_s2.txt
    data/dataset_train/test_pose_s2.txt

Each referenced .pt file must contain:
    source_file
    left_leg_input      [T, 27]
    right_leg_input     [T, 27]
    trunk_head_input    [T, 27]
    left_arm_input      [T, 27]
    right_arm_input     [T, 27]

Optional:
    full_joint_position_gt [T, 69]

Expected checkpoints:
    weights_root/
    ├── left_leg/best_pose_s2_left_leg.pth
    ├── right_leg/best_pose_s2_right_leg.pth
    ├── trunk_head/best_pose_s2_trunk_head.pth
    ├── left_arm/best_pose_s2_left_arm.pth
    └── right_arm/best_pose_s2_right_arm.pth
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

TARGET_ORDER = [
    "left_leg",
    "right_leg",
    "trunk_head",
    "left_arm",
    "right_arm",
]

TARGET_CONFIG: Dict[str, Dict[str, object]] = {
    "left_leg": {
        "joints": [1, 4, 7, 10],
        "output_dim": 12,
    },
    "right_leg": {
        "joints": [2, 5, 8, 11],
        "output_dim": 12,
    },
    "trunk_head": {
        "joints": [3, 6, 9, 12, 15],
        "output_dim": 15,
    },
    "left_arm": {
        "joints": [13, 16, 18, 20, 22],
        "output_dim": 15,
    },
    "right_arm": {
        "joints": [14, 17, 19, 21, 23],
        "output_dim": 15,
    },
}


class PoseS2FullDistributedBranch(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        proj_dim: int,
        rnn_hidden: int,
        rnn_layers: int,
        dropout: float,
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
            raise RuntimeError(
                "PoseS2FullDistributedBranch expects a PackedSequence."
            )

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
    existing = [file_path for file_path in files if os.path.exists(file_path)]
    missing = len(files) - len(existing)

    if missing:
        print(f"[warning] Ignored {missing} missing paths from {list_file}")

    return existing


def find_checkpoint(weights_root: str, target: str) -> str:
    candidates = [
        os.path.join(
            weights_root,
            target,
            f"best_pose_s2_{target}.pth",
        ),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not find checkpoint for target={target}. Tried:\n"
        + "\n".join(candidates)
    )


def load_branch_model(weights_root: str, target: str) -> PoseS2FullDistributedBranch:
    checkpoint_path = find_checkpoint(weights_root, target)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    output_dim = int(
        checkpoint.get(
            "output_dim",
            TARGET_CONFIG[target]["output_dim"],
        )
    )

    model = PoseS2FullDistributedBranch(
        input_dim=int(checkpoint.get("input_dim", 27)),
        output_dim=output_dim,
        proj_dim=int(checkpoint.get("proj_dim", 16)),
        rnn_hidden=int(checkpoint.get("rnn_hidden", 16)),
        rnn_layers=int(checkpoint.get("rnn_layers", 2)),
        dropout=float(checkpoint.get("dropout", 0.2)),
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        f"[model] {target}: {checkpoint_path} | "
        f"proj={checkpoint.get('proj_dim', 'unknown')} | "
        f"hidden={checkpoint.get('rnn_hidden', 'unknown')} | "
        f"output={output_dim}"
    )

    return model


@torch.inference_mode()
def infer_branch(
    model: PoseS2FullDistributedBranch,
    branch_input: torch.Tensor,
) -> torch.Tensor:
    if branch_input.dim() != 2 or branch_input.shape[1] != 27:
        raise ValueError(
            f"Expected branch input [T,27], got {tuple(branch_input.shape)}"
        )

    branch_input = branch_input.float().unsqueeze(0).to(DEVICE)
    lengths = torch.tensor(
        [branch_input.shape[1]],
        dtype=torch.long,
    )

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
    joints: Sequence[int],
) -> None:
    if branch_prediction.shape[1] != len(joints) * 3:
        raise ValueError(
            f"Prediction dimension {branch_prediction.shape[1]} does not match "
            f"{len(joints)} joints."
        )

    prediction = branch_prediction.reshape(
        branch_prediction.shape[0],
        len(joints),
        3,
    )

    assembled_view = assembled.reshape(
        assembled.shape[0],
        23,
        3,
    )

    for local_index, joint_idx in enumerate(joints):
        if not 1 <= joint_idx <= 23:
            raise ValueError(f"Joint index must be in [1,23], got {joint_idx}")

        assembled_view[:, joint_idx - 1, :] = prediction[:, local_index, :]


def make_output_name(index: int, prepared_path: str, suffix: str) -> str:
    stem = Path(prepared_path).stem
    safe_stem = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in stem
    )

    if suffix:
        return f"{safe_stem}_{suffix}_pose_s2_pred.pt"

    return f"{safe_stem}_pose_s2_pred_{index:06d}.pt"


def process_one_file(
    prepared_path: str,
    models: Dict[str, PoseS2FullDistributedBranch],
    output_dir: str,
    index: int,
    suffix: str,
) -> str:
    data = torch.load(
        prepared_path,
        map_location="cpu",
        weights_only=False,
    )

    if "source_file" not in data:
        raise KeyError(f"'source_file' missing in prepared file: {prepared_path}")

    sequence_lengths = []

    for target in TARGET_ORDER:
        input_key = f"{target}_input"

        if input_key not in data:
            raise KeyError(f"'{input_key}' missing in prepared file: {prepared_path}")

        sequence_lengths.append(data[input_key].shape[0])

    sequence_length = min(sequence_lengths)

    assembled = torch.zeros(
        sequence_length,
        69,
        dtype=torch.float32,
    )
    branch_predictions = {}

    for target in TARGET_ORDER:
        branch_input = data[f"{target}_input"][:sequence_length].float()
        branch_prediction = infer_branch(
            models[target],
            branch_input,
        )

        if branch_prediction.shape[0] != sequence_length:
            raise RuntimeError(
                f"{target} prediction length mismatch: "
                f"{branch_prediction.shape[0]} vs {sequence_length}"
            )

        branch_predictions[target] = branch_prediction

        insert_branch_prediction(
            assembled,
            branch_prediction,
            TARGET_CONFIG[target]["joints"],
        )

    output = {
        "source_file": data["source_file"],
        "prepared_pose_s2_source_file": prepared_path,
        "num_frames": sequence_length,
        "target_order": TARGET_ORDER,
        "target_config": TARGET_CONFIG,
        "full_joint_position_pred": assembled,
    }

    if "full_joint_position_gt" in data:
        output["full_joint_position_gt"] = data["full_joint_position_gt"][:sequence_length].float()

    for target in TARGET_ORDER:
        output[f"{target}_pred"] = branch_predictions[target]

        target_key = f"{target}_target"
        if target_key in data:
            output[target_key] = data[target_key][:sequence_length].float()

    output_path = os.path.join(
        output_dir,
        make_output_name(index, prepared_path, suffix),
    )
    torch.save(output, output_path)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_list_file",
        required=True,
        help="List of prepared Pose-S2 full-distributed files.",
    )
    parser.add_argument(
        "--weights_root",
        required=True,
        help="Root directory containing the five trained branch folders.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where prediction .pt files will be saved.",
    )
    parser.add_argument(
        "--output_list_file",
        required=True,
        help="Manifest file containing the generated prediction files.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Optional filename suffix, for example 16h or 24h.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    prepared_files = load_list(args.data_list_file)

    if not prepared_files:
        raise RuntimeError(f"No prepared files found in {args.data_list_file}")

    print("DEVICE:", DEVICE)
    print("Prepared files:", len(prepared_files))
    print("Weights root:", args.weights_root)

    models = {
        target: load_branch_model(args.weights_root, target)
        for target in TARGET_ORDER
    }

    saved_paths = []

    for index, prepared_path in enumerate(
        tqdm(
            prepared_files,
            desc="Precomputing Pose-S2 full-distributed outputs",
            ncols=120,
        )
    ):
        output_path = process_one_file(
            prepared_path=prepared_path,
            models=models,
            output_dir=args.output_dir,
            index=index,
            suffix=args.suffix,
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

    sample = torch.load(
        saved_paths[0],
        map_location="cpu",
        weights_only=False,
    )
    print("\nSample output:")
    print("  source_file:", sample["source_file"])
    print("  full_joint_position_pred:", tuple(sample["full_joint_position_pred"].shape))

    if "full_joint_position_gt" in sample:
        print("  full_joint_position_gt:", tuple(sample["full_joint_position_gt"].shape))


if __name__ == "__main__":
    main()
