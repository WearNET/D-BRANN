
"""
Train the five fully distributed Pose-S2 branches from prepared files.

No Pose-S2 fusion network is used.
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
    POSE_S2_TRAIN_LIST,
    POSE_S2_TEST_LIST,
    RETRAINED_CHECKPOINTS_DIR,
)



import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    PackedSequence,
    pack_padded_sequence,
    pad_packed_sequence,
    pad_sequence,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm



try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

TARGET_ORDER = [
    "left_leg",
    "right_leg",
    "trunk_head",
    "left_arm",
    "right_arm",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


class PreparedPoseS2BranchDataset(Dataset):
    def __init__(
        self,
        file_paths: Sequence[str],
        target: str,
        training: bool,
        leaf_noise_sigma: float,
    ):
        if target not in TARGET_CONFIG:
            raise ValueError(f"Unknown target: {target}")

        self.file_paths = list(file_paths)
        self.target = target
        self.training = training
        self.leaf_noise_sigma = float(leaf_noise_sigma)
        self.input_key = f"{target}_input"
        self.target_key = f"{target}_target"

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.file_paths[index]
        data = torch.load(path, map_location="cpu", weights_only=False)

        if self.input_key not in data:
            raise KeyError(f"'{self.input_key}' missing in {path}")
        if self.target_key not in data:
            raise KeyError(f"'{self.target_key}' missing in {path}")

        branch_input = data[self.input_key].float().clone()
        branch_target = data[self.target_key].float()

        if branch_input.dim() != 2 or branch_input.shape[1] != 27:
            raise ValueError(
                f"{self.input_key} must be [T, 27], "
                f"got {tuple(branch_input.shape)} in {path}"
            )

        expected_output_dim = int(
            TARGET_CONFIG[self.target]["output_dim"]
        )

        if (
            branch_target.dim() != 2
            or branch_target.shape[1] != expected_output_dim
        ):
            raise ValueError(
                f"{self.target_key} must be [T, {expected_output_dim}], "
                f"got {tuple(branch_target.shape)} in {path}"
            )

        sequence_length = min(
            branch_input.shape[0],
            branch_target.shape[0],
        )
        branch_input = branch_input[:sequence_length]
        branch_target = branch_target[:sequence_length]

        # Only the local Pose-S1 leaf occupies the final three input values.
        if self.training and self.leaf_noise_sigma > 0:
            branch_input[:, -3:] += (
                torch.randn_like(branch_input[:, -3:])
                * self.leaf_noise_sigma
            )

        return branch_input, branch_target


def collate_sequences(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs, targets = zip(*batch)
    lengths = torch.tensor(
        [sequence.shape[0] for sequence in inputs],
        dtype=torch.long,
    )

    return (
        pad_sequence(inputs, batch_first=True),
        pad_sequence(targets, batch_first=True),
        lengths,
    )


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
                "PoseS2FullDistributedBranch expects PackedSequence."
            )

        projected = self.dropout(sequence.data)
        projected = torch.relu(self.fc_in(projected))

        packed_projected = PackedSequence(
            projected,
            sequence.batch_sizes,
            sequence.sorted_indices,
            sequence.unsorted_indices,
        )

        recurrent_output, _ = self.rnn(packed_projected)
        prediction = self.fc_out(recurrent_output.data)

        return PackedSequence(
            prediction,
            recurrent_output.batch_sizes,
            recurrent_output.sorted_indices,
            recurrent_output.unsorted_indices,
        )


def forward_padded(
    model: PoseS2FullDistributedBranch,
    inputs_padded: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    packed = pack_padded_sequence(
        inputs_padded,
        lengths.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )
    prediction_packed = model(packed)
    prediction_padded, _ = pad_packed_sequence(
        prediction_packed,
        batch_first=True,
        total_length=inputs_padded.shape[1],
    )
    return prediction_padded


def build_valid_mask(
    lengths: torch.Tensor,
    maximum_length: int,
    device: torch.device,
) -> torch.Tensor:
    indices = torch.arange(
        maximum_length,
        device=device,
    ).unsqueeze(0)

    return indices < lengths.to(device).unsqueeze(1)


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    expanded_mask = valid_mask.unsqueeze(-1).expand_as(prediction)
    difference = (
        prediction[expanded_mask]
        - target[expanded_mask]
    )
    return torch.mean(difference ** 2)


def position_error_sum_cm(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[float, int]:
    output_dim = prediction.shape[-1]
    joint_count = output_dim // 3

    prediction = prediction.reshape(
        prediction.shape[0],
        prediction.shape[1],
        joint_count,
        3,
    )
    target = target.reshape_as(prediction)

    prediction = prediction[valid_mask]
    target = target[valid_mask]

    distances = torch.norm(
        prediction - target,
        dim=-1,
    )

    return (
        float(distances.sum().item() * 100.0),
        int(distances.numel()),
    )


def run_epoch(
    model: PoseS2FullDistributedBranch,
    loader: DataLoader,
    optimizer,
    scheduler,
    clip_value: float,
    training: bool,
) -> Tuple[float, float]:
    model.train(training)

    total_squared_error = 0.0
    total_scalar_values = 0
    total_position_error_cm = 0.0
    total_joint_samples = 0

    context = torch.enable_grad() if training else torch.inference_mode()

    with context:
        progress = tqdm(
            loader,
            desc="Train" if training else "Eval",
            ncols=120,
        )

        for inputs_padded, targets_padded, lengths in progress:
            inputs_padded = inputs_padded.to(
                DEVICE,
                non_blocking=True,
            )
            targets_padded = targets_padded.to(
                DEVICE,
                non_blocking=True,
            )

            valid_mask = build_valid_mask(
                lengths,
                inputs_padded.shape[1],
                DEVICE,
            )

            if training:
                optimizer.zero_grad(set_to_none=True)

            prediction = forward_padded(
                model,
                inputs_padded,
                lengths,
            )

            loss = masked_mse(
                prediction,
                targets_padded,
                valid_mask,
            )

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    clip_value,
                )
                optimizer.step()
                scheduler.step()

            with torch.no_grad():
                expanded_mask = valid_mask.unsqueeze(-1).expand_as(
                    prediction
                )
                difference = (
                    prediction[expanded_mask]
                    - targets_padded[expanded_mask]
                )

                total_squared_error += float(
                    torch.sum(difference ** 2).item()
                )
                total_scalar_values += int(difference.numel())

                position_sum, position_count = position_error_sum_cm(
                    prediction,
                    targets_padded,
                    valid_mask,
                )
                total_position_error_cm += position_sum
                total_joint_samples += position_count

            running_loss = (
                total_squared_error
                / max(total_scalar_values, 1)
            )
            running_position = (
                total_position_error_cm
                / max(total_joint_samples, 1)
            )

            progress.set_postfix(
                loss=f"{running_loss:.6f}",
                pos_cm=f"{running_position:.4f}",
            )

    return (
        total_squared_error / max(total_scalar_values, 1),
        total_position_error_cm / max(total_joint_samples, 1),
    )


def initialize_wandb(
    args: argparse.Namespace,
    target: str,
    parameter_count: int,
):
    if not args.use_wandb:
        return None

    if not WANDB_AVAILABLE:
        raise RuntimeError(
            "wandb is not installed, but --use_wandb was requested."
        )

    run_name = (
        args.wandb_run_name
        or f"pose_s2_{target}_{args.rnn_hidden}h"
    )

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=run_name,
        group=args.wandb_group or None,
        tags=[
            tag.strip()
            for tag in args.wandb_tags.split(",")
            if tag.strip()
        ],
        config={
            **vars(args),
            "active_target": target,
            "parameter_count": parameter_count,
            "target_config": TARGET_CONFIG[target],
        },
        reinit=True,
    )


def validate_prepared_sample(path: str, target: str) -> None:
    data = torch.load(path, map_location="cpu", weights_only=False)

    branch_input = data[f"{target}_input"]
    branch_target = data[f"{target}_target"]
    expected_output_dim = int(TARGET_CONFIG[target]["output_dim"])

    if branch_input.dim() != 2 or branch_input.shape[1] != 27:
        raise ValueError(
            f"{target}_input must be [T, 27], "
            f"got {tuple(branch_input.shape)}"
        )
    if (
        branch_target.dim() != 2
        or branch_target.shape[1] != expected_output_dim
    ):
        raise ValueError(
            f"{target}_target must be [T, {expected_output_dim}], "
            f"got {tuple(branch_target.shape)}"
        )

    print(
        f"[sanity] {target}: "
        f"input={tuple(branch_input.shape)} | "
        f"target={tuple(branch_target.shape)}"
    )


def train_target(
    args: argparse.Namespace,
    target: str,
    train_files_all: Sequence[str],
    test_files: Sequence[str],
) -> Dict[str, float]:
    set_seed(args.seed)

    shuffled = list(train_files_all)
    random.shuffle(shuffled)

    validation_count = max(
        1,
        int(round(len(shuffled) * args.val_ratio)),
    )
    validation_count = min(
        validation_count,
        len(shuffled) - 1,
    )

    train_files = shuffled[:-validation_count]
    validation_files = shuffled[-validation_count:]

    print("\n" + "=" * 78)
    print(f"TRAINING FULLY DISTRIBUTED POSE-S2 BRANCH: {target}")
    print("=" * 78)
    print("DEVICE:", DEVICE)
    print(
        f"[Data] Train: {len(train_files)} | "
        f"Val: {len(validation_files)} | "
        f"Test: {len(test_files)}"
    )

    validate_prepared_sample(train_files[0], target)

    train_dataset = PreparedPoseS2BranchDataset(
        train_files,
        target,
        training=True,
        leaf_noise_sigma=args.leaf_noise_sigma,
    )
    validation_dataset = PreparedPoseS2BranchDataset(
        validation_files,
        target,
        training=False,
        leaf_noise_sigma=0.0,
    )
    test_dataset = PreparedPoseS2BranchDataset(
        test_files,
        target,
        training=False,
        leaf_noise_sigma=0.0,
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "collate_fn": collate_sequences,
        "num_workers": args.num_workers,
        "pin_memory": DEVICE.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    output_dim = int(TARGET_CONFIG[target]["output_dim"])

    pretrained_state_dict = None
    if args.pretrained_checkpoint_dir:
        pretrained_path = os.path.join(
            args.pretrained_checkpoint_dir,
            target,
            f"best_pose_s2_{target}.pth",
        )
        print(f"[Fine-tune] Loading pretrained checkpoint: {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location=DEVICE, weights_only=False)

        if ckpt.get("target") != target:
            print(
                f"  [warning] checkpoint target='{ckpt.get('target')}' does not "
                f"match target='{target}'"
            )

        for key in ("proj_dim", "rnn_hidden", "rnn_layers", "dropout"):
            if key in ckpt and getattr(args, key) != ckpt[key]:
                print(
                    f"  [Fine-tune] overriding --{key}={getattr(args, key)} with "
                    f"checkpoint value {ckpt[key]} (architecture must match to load weights)"
                )
                setattr(args, key, ckpt[key])

        pretrained_state_dict = ckpt["model_state_dict"]

        if args.learning_rate == 1e-3:
            print(
                "  [Fine-tune] using the from-scratch default learning rate "
                "(1e-3) -- consider passing a lower --learning_rate "
                "(e.g. 1e-4) when fine-tuning on a small dataset."
            )

    model = PoseS2FullDistributedBranch(
        input_dim=27,
        output_dim=output_dim,
        proj_dim=args.proj_dim,
        rnn_hidden=args.rnn_hidden,
        rnn_layers=args.rnn_layers,
        dropout=args.dropout,
    ).to(DEVICE)

    if pretrained_state_dict is not None:
        model.load_state_dict(pretrained_state_dict)
        print("[Fine-tune] Pretrained weights loaded.")

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Input dimension:", 27)
    print("Output dimension:", output_dim)
    print("Joints:", TARGET_CONFIG[target]["joints"])
    print("Parameters:", parameter_count)
    print("Leaf noise sigma:", args.leaf_noise_sigma)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.decay_steps,
        gamma=args.decay_rate,
    )

    target_save_dir = os.path.join(
        args.save_dir,
        target,
    )
    os.makedirs(target_save_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        target_save_dir,
        f"best_pose_s2_{target}.pth",
    )

    run = initialize_wandb(
        args,
        target,
        parameter_count,
    )

    if run is not None:
        wandb.watch(
            model,
            log="gradients",
            log_freq=200,
        )

    best_validation_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_position_error = run_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            args.clip_val,
            training=True,
        )
        validation_loss, validation_position_error = run_epoch(
            model,
            validation_loader,
            None,
            None,
            args.clip_val,
            training=False,
        )

        learning_rate = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d} -> "
            f"TrainLoss: {train_loss:.6f} | "
            f"TrainPos(cm): {train_position_error:.4f} | "
            f"ValLoss: {validation_loss:.6f} | "
            f"ValPos(cm): {validation_position_error:.4f} | "
            f"LR: {learning_rate:.8f}"
        )

        if run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/position_error_cm": train_position_error,
                    "val/loss": validation_loss,
                    "val/position_error_cm": validation_position_error,
                    "learning_rate": learning_rate,
                }
            )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "target": target,
                    "target_config": TARGET_CONFIG[target],
                    "input_dim": 27,
                    "output_dim": output_dim,
                    "proj_dim": args.proj_dim,
                    "rnn_hidden": args.rnn_hidden,
                    "rnn_layers": args.rnn_layers,
                    "dropout": args.dropout,
                    "bidirectional": True,
                    "leaf_noise_sigma": args.leaf_noise_sigma,
                    "parameter_count": parameter_count,
                    "best_validation_loss": best_validation_loss,
                    "epoch": epoch,
                },
                checkpoint_path,
            )

            print(f"  [*] Saved best model to: {checkpoint_path}")
        else:
            patience_counter += 1
            print(
                f"  [!] Patience: "
                f"{patience_counter}/{args.patience}"
            )

            if patience_counter >= args.patience:
                print("Early stopping.")
                break

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    final_validation_loss, final_validation_position = run_epoch(
        model,
        validation_loader,
        None,
        None,
        args.clip_val,
        training=False,
    )
    final_test_loss, final_test_position = run_epoch(
        model,
        test_loader,
        None,
        None,
        args.clip_val,
        training=False,
    )

    print(
        f"\nVAL  -> Loss: {final_validation_loss:.6f} | "
        f"PositionError(cm): {final_validation_position:.4f}"
    )
    print(
        f"TEST -> Loss: {final_test_loss:.6f} | "
        f"PositionError(cm): {final_test_position:.4f}"
    )

    if run is not None:
        wandb.log(
            {
                "final/val_loss": final_validation_loss,
                "final/val_position_error_cm": final_validation_position,
                "final/test_loss": final_test_loss,
                "final/test_position_error_cm": final_test_position,
            }
        )
        wandb.finish()

    return {
        "val_loss": final_validation_loss,
        "val_position_error_cm": final_validation_position,
        "test_loss": final_test_loss,
        "test_position_error_cm": final_test_position,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        required=True,
        choices=[*TARGET_ORDER, "all"],
    )
    parser.add_argument(
        "--train_list_file",
        default=str(POSE_S2_TRAIN_LIST),
    )
    parser.add_argument(
        "--test_list_file",
        default=str(POSE_S2_TEST_LIST),
    )
    parser.add_argument(
        "--save_dir",
        default=str(RETRAINED_CHECKPOINTS_DIR / "pose_s2"),
    )
    parser.add_argument(
        "--pretrained_checkpoint_dir",
        default="",
        help=(
            "Root dir of existing Pose-S2 checkpoints to fine-tune from, e.g. "
            "checkpoints/dbran_pose_s2. The per-target path is "
            "built automatically as <dir>/<target>/best_pose_s2_"
            "<target>.pth -- this avoids accidentally pointing every branch at "
            "the same checkpoint when using --target all. Leave empty to train "
            "from scratch (default behavior, unchanged)."
        ),
    )

    parser.add_argument("--proj_dim", type=int, default=16)
    parser.add_argument("--rnn_hidden", type=int, default=16)
    parser.add_argument("--rnn_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--clip_val", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--decay_rate", type=float, default=0.96)
    parser.add_argument("--decay_steps", type=int, default=2000)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--leaf_noise_sigma", type=float, default=0.04)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument(
        "--wandb_project",
        default="dbran-pose-s2",
    )
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_run_name", default="")
    parser.add_argument(
        "--wandb_group",
        default="pose_s2_16h",
    )
    parser.add_argument(
        "--wandb_tags",
        default="pose_s2,blstm,no_fusion",
    )

    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    set_seed(args.seed)

    train_files = load_list(args.train_list_file)
    test_files = load_list(args.test_list_file)

    if len(train_files) < 2:
        raise RuntimeError(
            "The training list must contain at least two valid files."
        )
    if not test_files:
        raise RuntimeError(
            "The test list contains no valid files."
        )

    targets = TARGET_ORDER if args.target == "all" else [args.target]
    results: Dict[str, Dict[str, float]] = {}

    for target in targets:
        results[target] = train_target(
            args,
            target,
            train_files,
            test_files,
        )

    print("\n" + "=" * 78)
    print("FINAL FIVE-BRANCH POSE-S2 RESULTS")
    print("=" * 78)

    for target in targets:
        metrics = results[target]
        print(
            f"{target:>10}: "
            f"ValPos={metrics['val_position_error_cm']:.4f} cm | "
            f"TestPos={metrics['test_position_error_cm']:.4f} cm"
        )


if __name__ == "__main__":
    main()
