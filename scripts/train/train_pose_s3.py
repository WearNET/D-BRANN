"""
Train PoseS3 five-branch models.

Use:
    --target all
or one of:
    left_leg, right_leg, trunk_head, left_arm, right_arm
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
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import (
    PackedSequence,
    pack_padded_sequence,
    pad_sequence,
)
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BRANCH_CONFIG: Dict[str, Dict[str, object]] = {
    "left_leg": {
        "input_dim": 36,
        "output_dim": 12,
        "imu_dim": 24,
        "position_joints": [1, 4, 7, 10],
        "reduced_joints": [1, 4],
    },
    "right_leg": {
        "input_dim": 36,
        "output_dim": 12,
        "imu_dim": 24,
        "position_joints": [2, 5, 8, 11],
        "reduced_joints": [2, 5],
    },
    "trunk_head": {
        "input_dim": 39,
        "output_dim": 30,
        "imu_dim": 24,
        "position_joints": [3, 6, 9, 12, 15],
        "reduced_joints": [3, 6, 9, 12, 15],
    },
    "left_arm": {
        "input_dim": 39,
        "output_dim": 18,
        "imu_dim": 24,
        "position_joints": [13, 16, 18, 20, 22],
        "reduced_joints": [13, 16, 18],
    },
    "right_arm": {
        "input_dim": 39,
        "output_dim": 18,
        "imu_dim": 24,
        "position_joints": [14, 17, 19, 21, 23],
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
            raise RuntimeError("PoseS3FiveBranchRegionNet expects a PackedSequence.")

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


def sixd_to_rotmat(x6d: torch.Tensor) -> torch.Tensor:
    x = x6d.reshape(*x6d.shape[:-1], 2, 3)
    a1 = x[..., 0, :]
    a2 = x[..., 1, :]

    b1 = F.normalize(a1, dim=-1)
    dot = torch.sum(b1 * a2, dim=-1, keepdim=True)
    b2 = F.normalize(a2 - dot * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)


def angular_error_deg(
    prediction_6d: torch.Tensor,
    target_6d: torch.Tensor,
    num_joints: int,
) -> torch.Tensor:
    prediction = prediction_6d.reshape(-1, num_joints, 6)
    target = target_6d.reshape(-1, num_joints, 6)

    prediction_matrix = sixd_to_rotmat(prediction)
    target_matrix = sixd_to_rotmat(target)

    relative = torch.matmul(
        prediction_matrix.transpose(-1, -2),
        target_matrix,
    )
    trace = (
        relative[..., 0, 0]
        + relative[..., 1, 1]
        + relative[..., 2, 2]
    )
    cosine = torch.clamp(
        (trace - 1.0) / 2.0,
        min=-1.0 + 1e-6,
        max=1.0 - 1e-6,
    )

    return torch.rad2deg(torch.acos(cosine)).mean()


class PoseS3FiveBranchDataset(Dataset):
    def __init__(
        self,
        files: Sequence[str],
        target: str,
        training: bool,
        position_noise_std: float,
    ):
        if target not in BRANCH_CONFIG:
            raise ValueError(f"Unknown target: {target}")

        self.files = list(files)
        self.target = target
        self.training = training
        self.position_noise_std = float(position_noise_std)

        self.input_key = f"{target}_input"
        self.target_key = f"{target}_target_6d_reduced"
        self.imu_dim = int(BRANCH_CONFIG[target]["imu_dim"])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        data = torch.load(path, map_location="cpu", weights_only=False)

        branch_input = data[self.input_key].float().clone()
        branch_target = data[self.target_key].float()

        if self.training and self.position_noise_std > 0:
            branch_input[:, self.imu_dim:] += (
                torch.randn_like(branch_input[:, self.imu_dim:])
                * self.position_noise_std
            )

        sequence_length = min(branch_input.shape[0], branch_target.shape[0])

        return (
            branch_input[:sequence_length],
            branch_target[:sequence_length],
            sequence_length,
        )


def collate_batch(batch):
    inputs, targets, lengths = zip(*batch)

    lengths = torch.tensor(lengths, dtype=torch.long)
    inputs = pad_sequence(inputs, batch_first=True)
    targets = pad_sequence(targets, batch_first=True)

    return inputs, targets, lengths


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    grad_clip,
    num_joints,
    training,
):
    model.train(training)

    total_loss = 0.0
    total_angular = 0.0
    total_batches = 0

    context = torch.enable_grad() if training else torch.inference_mode()

    with context:
        for inputs, targets, lengths in tqdm(
            loader,
            desc="Train" if training else "Eval",
            ncols=120,
        ):
            inputs = inputs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            packed_inputs = pack_padded_sequence(
                inputs,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_targets = pack_padded_sequence(
                targets,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )

            if training:
                optimizer.zero_grad(set_to_none=True)

            prediction = model(packed_inputs)
            loss = criterion(prediction.data, packed_targets.data)
            angular = angular_error_deg(
                prediction.data,
                packed_targets.data,
                num_joints,
            )

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                )
                optimizer.step()

            total_loss += float(loss.item())
            total_angular += float(angular.item())
            total_batches += 1

    return {
        "loss": total_loss / max(total_batches, 1),
        "angular_error_deg": total_angular / max(total_batches, 1),
    }


def train_one_target(args, target, train_files, test_files):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = BRANCH_CONFIG[target]
    input_dim = int(config["input_dim"])
    output_dim = int(config["output_dim"])
    num_joints = output_dim // 6

    full_train = PoseS3FiveBranchDataset(
        train_files,
        target=target,
        training=True,
        position_noise_std=args.position_noise_std,
    )
    test_dataset = PoseS3FiveBranchDataset(
        test_files,
        target=target,
        training=False,
        position_noise_std=0.0,
    )

    n_total = len(full_train)
    n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(
        full_train,
        [n_train, n_val],
        generator=generator,
    )

    # Make sure validation does not inject noise.
    val_dataset.dataset.training = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    pretrained_state_dict = None
    if args.pretrained_checkpoint_dir:
        pretrained_path = os.path.join(
            args.pretrained_checkpoint_dir,
            target,
            f"best_pose_s3_{target}.pth",
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

        if args.lr == 1e-3:
            print(
                "  [Fine-tune] using the from-scratch default learning rate "
                "(1e-3) -- consider passing a lower --lr (e.g. 1e-4) when "
                "fine-tuning on a small dataset."
            )

    model = PoseS3FiveBranchRegionNet(
        input_dim=input_dim,
        output_dim=output_dim,
        proj_dim=args.proj_dim,
        rnn_hidden=args.rnn_hidden,
        rnn_layers=args.rnn_layers,
        dropout=args.dropout,
    ).to(DEVICE)

    if pretrained_state_dict is not None:
        model.load_state_dict(pretrained_state_dict)
        print("[Fine-tune] Pretrained weights loaded.")

    parameter_count = sum(p.numel() for p in model.parameters())

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    save_target_dir = os.path.join(args.save_dir, target)
    os.makedirs(save_target_dir, exist_ok=True)
    best_path = os.path.join(
        save_target_dir,
        f"best_pose_s3_{target}.pth",
    )

    run = None
    if args.use_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"pose_s3_{target}_{args.rnn_hidden}h",
            group=args.wandb_group,
            config={**vars(args), "target": target, "branch_config": config},
            reinit=True,
        )

    print("\n" + "=" * 80)
    print(f"Training PoseS3 five-branch region: {target}")
    print("=" * 80)
    print("Device:", DEVICE)
    print("Input dim:", input_dim)
    print("Output dim:", output_dim)
    print("Num reduced joints:", num_joints)
    print("Parameters:", parameter_count)
    print(f"Train/Val/Test: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")

    best_val_angular = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
            num_joints=num_joints,
            training=True,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            grad_clip=args.grad_clip,
            num_joints=num_joints,
            training=False,
        )

        print(
            f"TrainLoss={train_metrics['loss']:.6f} | "
            f"TrainAng={train_metrics['angular_error_deg']:.4f} deg | "
            f"ValLoss={val_metrics['loss']:.6f} | "
            f"ValAng={val_metrics['angular_error_deg']:.4f} deg"
        )

        if run is not None:
            import wandb

            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_metrics["loss"],
                    "train/angular_error_deg": train_metrics["angular_error_deg"],
                    "val/loss": val_metrics["loss"],
                    "val/angular_error_deg": val_metrics["angular_error_deg"],
                }
            )

        if val_metrics["angular_error_deg"] < best_val_angular:
            best_val_angular = val_metrics["angular_error_deg"]
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "target": target,
                    "branch_config": config,
                    "input_dim": input_dim,
                    "output_dim": output_dim,
                    "proj_dim": args.proj_dim,
                    "rnn_hidden": args.rnn_hidden,
                    "rnn_layers": args.rnn_layers,
                    "dropout": args.dropout,
                    "parameter_count": parameter_count,
                    "best_val_angular_error_deg": best_val_angular,
                    "epoch": epoch,
                },
                best_path,
            )
            print(f"  [*] Saved best model to: {best_path}")
        else:
            patience_counter += 1
            print(f"  [!] Patience: {patience_counter}/{args.patience}")

            if patience_counter >= args.patience:
                print("Early stopping.")
                break

    checkpoint = torch.load(
        best_path,
        map_location=DEVICE,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    val_metrics = run_epoch(
        model=model,
        loader=val_loader,
        criterion=criterion,
        optimizer=None,
        grad_clip=args.grad_clip,
        num_joints=num_joints,
        training=False,
    )
    test_metrics = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        optimizer=None,
        grad_clip=args.grad_clip,
        num_joints=num_joints,
        training=False,
    )

    print(
        f"\nVAL  -> Loss={val_metrics['loss']:.6f} | "
        f"Angular={val_metrics['angular_error_deg']:.4f} deg"
    )
    print(
        f"TEST -> Loss={test_metrics['loss']:.6f} | "
        f"Angular={test_metrics['angular_error_deg']:.4f} deg"
    )

    if run is not None:
        import wandb

        wandb.log(
            {
                "final/val_loss": val_metrics["loss"],
                "final/val_angular_error_deg": val_metrics["angular_error_deg"],
                "final/test_loss": test_metrics["loss"],
                "final/test_angular_error_deg": test_metrics["angular_error_deg"],
            }
        )
        wandb.finish()

    return {
        "val_angular_error_deg": val_metrics["angular_error_deg"],
        "test_angular_error_deg": test_metrics["angular_error_deg"],
        "parameters": parameter_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        required=True,
        choices=[*BRANCH_ORDER, "all"],
    )
    parser.add_argument("--train_list_file", required=True)
    parser.add_argument("--test_list_file", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument(
        "--pretrained_checkpoint_dir",
        default="",
        help=(
            "Root dir of existing Pose-S3 checkpoints to fine-tune from, e.g. "
            "checkpoints/dbran_pose_s3. The per-target path is "
            "built automatically as <dir>/<target>/"
            "best_pose_s3_<target>.pth. Leave empty to "
            "train from scratch (default behavior, unchanged)."
        ),
    )

    parser.add_argument("--proj_dim", type=int, default=16)
    parser.add_argument("--rnn_hidden", type=int, default=16)
    parser.add_argument("--rnn_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--position_noise_std", type=float, default=0.025)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument(
        "--wandb_project",
        default="dbran-pose-s3",
    )
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_group", default=None)

    args = parser.parse_args()

    train_files = load_list(args.train_list_file)
    test_files = load_list(args.test_list_file)

    if not train_files:
        raise RuntimeError(f"No training files found in {args.train_list_file}")
    if not test_files:
        raise RuntimeError(f"No test files found in {args.test_list_file}")

    targets = BRANCH_ORDER if args.target == "all" else [args.target]
    results = {}

    for target in targets:
        results[target] = train_one_target(
            args,
            target,
            train_files,
            test_files,
        )

    print("\n" + "=" * 80)
    print("FINAL POSES3 FIVE-BRANCH REGION RESULTS")
    print("=" * 80)

    for target, metrics in results.items():
        print(
            f"{target:>10}: "
            f"ValAng={metrics['val_angular_error_deg']:.4f} deg | "
            f"TestAng={metrics['test_angular_error_deg']:.4f} deg | "
            f"Params={metrics['parameters']}"
        )


if __name__ == "__main__":
    main()
