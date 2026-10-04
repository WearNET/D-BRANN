"""
Train the learned fusion network that refines the assembled Pose-S3 output.

Input:
    pose3_assembled_pred_6d_reduced [T, 90]

Output:
    delta [T, 90]

Final prediction:
    pose3_assembled_pred_6d_reduced + delta
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
from typing import List, Sequence, Tuple

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


class FusionNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 90,
        proj_dim: int = 16,
        rnn_hidden: int = 16,
        rnn_layers: int = 1,
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
            raise RuntimeError("FusionNet expects PackedSequence.")

        data = self.dropout(sequence.data)
        data = torch.relu(self.fc_in(data))

        packed = PackedSequence(
            data,
            sequence.batch_sizes,
            sequence.sorted_indices,
            sequence.unsorted_indices,
        )
        output, _ = self.rnn(packed)
        delta = self.fc_out(output.data)

        return PackedSequence(
            delta,
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


class FusionDataset(Dataset):
    def __init__(self, files: Sequence[str], use_pose_s2_position: bool = False):
        self.files = list(files)
        self.use_pose_s2_position = bool(use_pose_s2_position)

        if not self.files:
            raise RuntimeError("Fusion dataset is empty.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        data = torch.load(path, map_location="cpu", weights_only=False)

        assembled = data["pose3_assembled_pred_6d_reduced"].float()
        target = data["pose3_target_6d_reduced"].float()

        inputs = [assembled]

        if self.use_pose_s2_position:
            inputs.append(data["pose_s2_full_position_input"].float())

        network_input = torch.cat(inputs, dim=1)

        sequence_length = min(
            network_input.shape[0],
            assembled.shape[0],
            target.shape[0],
        )

        return (
            network_input[:sequence_length],
            assembled[:sequence_length],
            target[:sequence_length],
            sequence_length,
        )


def collate_batch(batch):
    inputs, assembled, targets, lengths = zip(*batch)

    lengths = torch.tensor(lengths, dtype=torch.long)
    inputs = pad_sequence(inputs, batch_first=True)
    assembled = pad_sequence(assembled, batch_first=True)
    targets = pad_sequence(targets, batch_first=True)

    return inputs, assembled, targets, lengths


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
    total_assembled_loss = 0.0
    total_assembled_angular = 0.0
    total_batches = 0

    context = torch.enable_grad() if training else torch.inference_mode()

    with context:
        for network_input, assembled, target, lengths in tqdm(
            loader,
            desc="Train" if training else "Eval",
            ncols=120,
        ):
            network_input = network_input.to(DEVICE, non_blocking=True)
            assembled = assembled.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)

            packed_input = pack_padded_sequence(
                network_input,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_assembled = pack_padded_sequence(
                assembled,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_target = pack_padded_sequence(
                target,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )

            if training:
                optimizer.zero_grad(set_to_none=True)

            delta = model(packed_input)
            prediction_data = packed_assembled.data + delta.data

            loss = criterion(prediction_data, packed_target.data)
            angular = angular_error_deg(
                prediction_data,
                packed_target.data,
                num_joints,
            )

            assembled_loss = criterion(
                packed_assembled.data,
                packed_target.data,
            )
            assembled_angular = angular_error_deg(
                packed_assembled.data,
                packed_target.data,
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
            total_assembled_loss += float(assembled_loss.item())
            total_assembled_angular += float(assembled_angular.item())
            total_batches += 1

    return {
        "loss": total_loss / max(total_batches, 1),
        "angular_error_deg": total_angular / max(total_batches, 1),
        "assembled_loss": total_assembled_loss / max(total_batches, 1),
        "assembled_angular_error_deg": total_assembled_angular / max(total_batches, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pred_list_file", required=True)
    parser.add_argument("--test_pred_list_file", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument(
        "--pretrained_checkpoint",
        default="",
        help=(
            "Path to an existing best_fusion.pth "
            "to fine-tune from. There is only one fusion model (not per-"
            "branch), so a single file path is unambiguous here. Leave empty "
            "to train from scratch (default behavior, unchanged)."
        ),
    )

    parser.add_argument("--use_pose_s2_position", action="store_true")

    parser.add_argument("--proj_dim", type=int, default=16)
    parser.add_argument("--rnn_hidden", type=int, default=16)
    parser.add_argument("--rnn_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument(
        "--wandb_project",
        default="dbran-fusion",
    )
    parser.add_argument(
        "--wandb_run_name",
        default="fusion",
    )
    parser.add_argument("--wandb_group", default=None)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    train_files = load_list(args.train_pred_list_file)
    test_files = load_list(args.test_pred_list_file)

    full_train = FusionDataset(
        train_files,
        use_pose_s2_position=args.use_pose_s2_position,
    )
    test_dataset = FusionDataset(
        test_files,
        use_pose_s2_position=args.use_pose_s2_position,
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

    sample_input, sample_assembled, sample_target, _ = full_train[0]
    input_dim = sample_input.shape[1]
    output_dim = sample_target.shape[1]

    if output_dim != 90:
        raise ValueError(f"Expected PoseS3 fusion target output_dim=90, got {output_dim}")

    num_joints = output_dim // 6

    pretrained_state_dict = None
    if args.pretrained_checkpoint:
        print(f"[Fine-tune] Loading pretrained checkpoint: {args.pretrained_checkpoint}")
        ckpt = torch.load(args.pretrained_checkpoint, map_location=DEVICE, weights_only=False)

        if ckpt.get("input_dim") not in (None, input_dim):
            raise RuntimeError(
                f"Checkpoint input_dim={ckpt.get('input_dim')} does not match "
                f"the current dataset's input_dim={input_dim} -- check "
                f"--use_pose_s2_position matches how the checkpoint was trained."
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

    model = FusionNet(
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

    run = None
    if args.use_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            group=args.wandb_group,
            config={**vars(args), "parameter_count": parameter_count},
        )

    print("DEVICE:", DEVICE)
    print("Input dim:", input_dim)
    print("Output dim:", output_dim)
    print("Num joints:", num_joints)
    print("Parameters:", parameter_count)
    print("Use PoseS2 position input:", args.use_pose_s2_position)
    print(f"Train/Val/Test: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")

    best_val_angular = float("inf")
    patience_counter = 0
    best_path = os.path.join(
        args.save_dir,
        "best_fusion.pth",
    )

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
            f"ValAng={val_metrics['angular_error_deg']:.4f} deg | "
            f"AssembledValAng={val_metrics['assembled_angular_error_deg']:.4f} deg"
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
                    "val/assembled_angular_error_deg": val_metrics["assembled_angular_error_deg"],
                }
            )

        if val_metrics["angular_error_deg"] < best_val_angular:
            best_val_angular = val_metrics["angular_error_deg"]
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "output_dim": output_dim,
                    "proj_dim": args.proj_dim,
                    "rnn_hidden": args.rnn_hidden,
                    "rnn_layers": args.rnn_layers,
                    "dropout": args.dropout,
                    "use_pose_s2_position": args.use_pose_s2_position,
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
        f"Angular={val_metrics['angular_error_deg']:.4f} deg | "
        f"AssembledAngular={val_metrics['assembled_angular_error_deg']:.4f} deg"
    )
    print(
        f"TEST -> Loss={test_metrics['loss']:.6f} | "
        f"Angular={test_metrics['angular_error_deg']:.4f} deg | "
        f"AssembledAngular={test_metrics['assembled_angular_error_deg']:.4f} deg"
    )

    if run is not None:
        import wandb

        wandb.log(
            {
                "final/val_loss": val_metrics["loss"],
                "final/val_angular_error_deg": val_metrics["angular_error_deg"],
                "final/val_assembled_angular_error_deg": val_metrics["assembled_angular_error_deg"],
                "final/test_loss": test_metrics["loss"],
                "final/test_angular_error_deg": test_metrics["angular_error_deg"],
                "final/test_assembled_angular_error_deg": test_metrics["assembled_angular_error_deg"],
            }
        )
        wandb.finish()


if __name__ == "__main__":
    main()
