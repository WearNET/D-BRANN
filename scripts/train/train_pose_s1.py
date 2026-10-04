
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
    POSE_S1_GT_TRAIN_LIST,
    POSE_S1_GT_TEST_LIST,
    RETRAINED_CHECKPOINTS_DIR,
)


import os
import random
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, PackedSequence
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS = 150
CLIP_VAL = 1.0
PATIENCE = 10
DECAY_RATE = 0.96
DECAY_STEPS = 2000
VAL_RATIO = 0.10
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_TRAIN_GT_LIST = str(POSE_S1_GT_TRAIN_LIST)
DEFAULT_TEST_GT_LIST = str(POSE_S1_GT_TEST_LIST)
DEFAULT_SAVE_DIR = str(RETRAINED_CHECKPOINTS_DIR / "pose_s1")

# ------------------------------------------------------------
# IMU order aligned with TransPose live demo convention
# 0 = left lower arm
# 1 = right lower arm
# 2 = left lower leg
# 3 = right lower leg
# 4 = head
# 5 = root
# ------------------------------------------------------------
ROOT_SENSOR_IDX = 5
LEFT_ARM_SENSOR_IDX = 0
RIGHT_ARM_SENSOR_IDX = 1
LEFT_LEG_SENSOR_IDX = 2
RIGHT_LEG_SENSOR_IDX = 3
HEAD_SENSOR_IDX = 4

TARGET_CONFIG = {
    "left_arm": {
        "sensor_idx": LEFT_ARM_SENSOR_IDX,
        "gt_key": "left_arm_p_gt",
    },
    "right_arm": {
        "sensor_idx": RIGHT_ARM_SENSOR_IDX,
        "gt_key": "right_arm_p_gt",
    },
    "left_leg": {
        "sensor_idx": LEFT_LEG_SENSOR_IDX,
        "gt_key": "left_leg_p_gt",
    },
    "right_leg": {
        "sensor_idx": RIGHT_LEG_SENSOR_IDX,
        "gt_key": "right_leg_p_gt",
    },
    "head": {
        "sensor_idx": HEAD_SENSOR_IDX,
        "gt_key": "head_p_gt",
    },
}

TARGET_ORDER = ["left_leg", "right_leg", "head", "left_arm", "right_arm"]


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# Input helpers
# ------------------------------------------------------------
def ensure_rotmat_3x3(ori: torch.Tensor) -> torch.Tensor:
    if ori.dim() == 4 and ori.shape[1:] == (6, 3, 3):
        return ori.float()
    if ori.dim() == 2 and ori.shape[1] == 54:
        T = ori.shape[0]
        return ori.view(T, 6, 3, 3).float()
    raise ValueError(f"Unsupported ori shape: {tuple(ori.shape)}")


def ensure_acc_6x3(acc: torch.Tensor) -> torch.Tensor:
    if acc.dim() == 3 and acc.shape[1:] == (6, 3):
        return acc.float()
    if acc.dim() == 2 and acc.shape[1] == 18:
        T = acc.shape[0]
        return acc.view(T, 6, 3).float()
    raise ValueError(f"Unsupported acc shape: {tuple(acc.shape)}")


def build_two_imu_input(acc: torch.Tensor,
                        ori: torch.Tensor,
                        root_idx: int,
                        local_idx: int) -> torch.Tensor:
    """
    Build per-frame feature vector:
      x_raw(t) = [a_root(3), R_root(9), a_local(3), R_local(9)] -> 24 dims
    """
    acc = ensure_acc_6x3(acc)
    ori = ensure_rotmat_3x3(ori)

    a_root = acc[:, root_idx, :]
    a_local = acc[:, local_idx, :]

    R_root = ori[:, root_idx, :, :].reshape(-1, 9)
    R_local = ori[:, local_idx, :, :].reshape(-1, 9)

    x = torch.cat([a_root, R_root, a_local, R_local], dim=1)
    return x


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
class DistributedPoseS1NoClassifier(nn.Module):
    """
    One distributed Pose-S1 sub-network without classifier input.

    Input per frame:
      [a_root(3), R_root(9), a_local(3), R_local(9)] -> 24 dims

    Output per frame:
      local leaf position relative to root -> 3 dims
    """
    def __init__(self,
                 input_dim=24,
                 proj_dim=64,
                 rnn_hidden=64,
                 rnn_layers=2,
                 dropout=0.2,
                 bidirectional=True,
                 output_dim=3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc_in = nn.Linear(input_dim, proj_dim)
        self.rnn = nn.LSTM(
            input_size=proj_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=False,
            bidirectional=bidirectional,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )
        out_dim = rnn_hidden * (2 if bidirectional else 1)
        self.fc_out = nn.Linear(out_dim, output_dim)

    def forward(self, x: PackedSequence) -> PackedSequence:
        if not isinstance(x, PackedSequence):
            raise RuntimeError("DistributedPoseS1NoClassifier expects a PackedSequence")

        data = x.data
        data = self.dropout(data)
        data = torch.relu(self.fc_in(data))

        packed = PackedSequence(data, x.batch_sizes, x.sorted_indices, x.unsorted_indices)
        out_packed, _ = self.rnn(packed)

        pred = self.fc_out(out_packed.data)
        return PackedSequence(
            pred,
            out_packed.batch_sizes,
            out_packed.sorted_indices,
            out_packed.unsorted_indices
        )


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def load_list(list_file):
    with open(list_file, "r") as f:
        files = [line.strip() for line in f if line.strip()]
    files = [p for p in files if os.path.exists(p)]
    return files


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
class PoseS1DistributedNoClassifierDataset(Dataset):
    """
    Each item:
      - loads one Pose-S1 GT file
      - reads source_file from it
      - loads raw acc and ori from source_file
      - builds input: [root+local raw] -> 24 dims
      - target: local leaf root-relative position -> 3 dims
    """
    def __init__(self, gt_file_list, target_name: str, root_sensor_idx: int = ROOT_SENSOR_IDX):
        if target_name not in TARGET_CONFIG:
            raise ValueError(f"Unknown target_name: {target_name}")

        self.gt_file_paths = [p for p in gt_file_list if os.path.exists(p)]
        self.target_name = target_name
        self.root_sensor_idx = root_sensor_idx

        self.local_sensor_idx = TARGET_CONFIG[target_name]["sensor_idx"]
        self.gt_key = TARGET_CONFIG[target_name]["gt_key"]

    def __len__(self):
        return len(self.gt_file_paths)

    def __getitem__(self, idx):
        gt_path = self.gt_file_paths[idx]
        gt_data = torch.load(gt_path, weights_only=False)

        if "source_file" not in gt_data:
            raise KeyError(f"'source_file' missing in GT file: {gt_path}")

        src_path = gt_data["source_file"]
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Raw source file not found: {src_path}")

        if self.gt_key not in gt_data:
            raise KeyError(f"Missing GT key '{self.gt_key}' in {gt_path}")

        raw = torch.load(src_path, weights_only=False)

        if "acc" not in raw or "ori" not in raw:
            raise KeyError(f"'acc' or 'ori' missing in raw file: {src_path}")

        x = build_two_imu_input(
            acc=raw["acc"],
            ori=raw["ori"],
            root_idx=self.root_sensor_idx,
            local_idx=self.local_sensor_idx,
        )  # (T, 24)

        y = gt_data[self.gt_key].float()  # (T, 3)

        if x.shape[0] != y.shape[0]:
            raise RuntimeError(
                f"Length mismatch for source_file={src_path}: x={x.shape[0]}, y={y.shape[0]}"
            )

        return x, y


def collate_fn(batch):
    batch.sort(key=lambda z: z[0].shape[0], reverse=True)
    xs, ys = zip(*batch)

    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    x_pad = pad_sequence(xs, batch_first=True)
    y_pad = pad_sequence(ys, batch_first=True)

    return x_pad, y_pad, lengths


# ------------------------------------------------------------
# Eval helpers
# ------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_batches = 0

    total_points = 0
    total_pos_error = 0.0

    for x_pad, y_pad, lengths in tqdm(loader, desc="Eval", ncols=120):
        x_pad = x_pad.to(DEVICE)
        y_pad = y_pad.to(DEVICE)

        packed_x = pack_padded_sequence(
            x_pad, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        packed_y = pack_padded_sequence(
            y_pad, lengths.cpu(), batch_first=True, enforce_sorted=True
        )

        packed_pred = model(packed_x)

        pred = packed_pred.data
        gt = packed_y.data

        loss = criterion(pred, gt)
        total_loss += loss.item()
        total_batches += 1

        pos_err = torch.norm(pred - gt, dim=1)  # meters
        total_pos_error += pos_err.sum().item()
        total_points += pos_err.numel()

    avg_loss = total_loss / max(total_batches, 1)
    mean_pos_error_m = total_pos_error / max(total_points, 1)
    mean_pos_error_cm = mean_pos_error_m * 100.0

    return avg_loss, mean_pos_error_cm


def compute_epoch_train_loss(model, loader, criterion, optimizer, scheduler, clip_val, batch_size_for_norm):
    model.train()

    running = 0.0
    batches = 0

    pbar = tqdm(loader, desc="Train", ncols=120)
    for x_pad, y_pad, lengths in pbar:
        x_pad = x_pad.to(DEVICE)
        y_pad = y_pad.to(DEVICE)

        optimizer.zero_grad()

        packed_x = pack_padded_sequence(
            x_pad, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        packed_y = pack_padded_sequence(
            y_pad, lengths.cpu(), batch_first=True, enforce_sorted=True
        )

        packed_pred = model(packed_x)

        pred = packed_pred.data
        gt = packed_y.data

        loss = criterion(pred, gt) / batch_size_for_norm
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        optimizer.step()
        scheduler.step()

        running += loss.item()
        batches += 1
        pbar.set_postfix({"Loss": f"{running / max(batches, 1):.6f}"})

    return running / max(batches, 1)


@torch.no_grad()
def compute_epoch_val(model, loader, criterion, batch_size_for_norm):
    model.eval()

    running = 0.0
    batches = 0
    total_points = 0
    total_pos_error = 0.0

    for x_pad, y_pad, lengths in loader:
        x_pad = x_pad.to(DEVICE)
        y_pad = y_pad.to(DEVICE)

        packed_x = pack_padded_sequence(
            x_pad, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        packed_y = pack_padded_sequence(
            y_pad, lengths.cpu(), batch_first=True, enforce_sorted=True
        )

        packed_pred = model(packed_x)

        pred = packed_pred.data
        gt = packed_y.data

        loss = criterion(pred, gt) / batch_size_for_norm
        running += loss.item()
        batches += 1

        pos_err = torch.norm(pred - gt, dim=1)
        total_pos_error += pos_err.sum().item()
        total_points += pos_err.numel()

    avg_loss = running / max(batches, 1)
    mean_pos_error_cm = (total_pos_error / max(total_points, 1)) * 100.0

    return avg_loss, mean_pos_error_cm


# ------------------------------------------------------------
# W&B helpers
# ------------------------------------------------------------
def init_wandb(args, target):
    if not args.use_wandb:
        return None

    if not WANDB_AVAILABLE:
        raise ImportError("wandb is not installed. Run: pip install wandb")

    tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
    config = vars(args).copy()
    config["device"] = str(DEVICE)
    config["target"] = target
    config["local_sensor_idx"] = TARGET_CONFIG[target]["sensor_idx"]

    run_name = args.wandb_run_name if args.wandb_run_name else target

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity if args.wandb_entity else None,
        name=run_name,
        group=args.wandb_group if args.wandb_group else None,
        tags=tags,
        config=config,
        reinit=True,
    )
    return run


def log_wandb_epoch(epoch, train_loss, val_loss, val_pos_error_cm, lr):
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/position_error_cm": val_pos_error_cm,
            "train/lr": lr,
        })


def log_wandb_final_metrics(val_loss, val_pos_error_cm, test_loss, test_pos_error_cm):
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({
            "final/val_loss": val_loss,
            "final/val_position_error_cm": val_pos_error_cm,
            "final/test_loss": test_loss,
            "final/test_position_error_cm": test_pos_error_cm,
        })


# ------------------------------------------------------------
# Train
# ------------------------------------------------------------
def train_target(args, target: str, train_gt_files, test_gt_files):
    set_seed(args.seed)

    print("\n" + "=" * 78)
    print(f"TRAINING DISTRIBUTED POSE-S1 BRANCH: {target}")
    print("=" * 78)
    print("DEVICE:", DEVICE)

    train_gt_files = list(train_gt_files)
    random.shuffle(train_gt_files)

    total_len = len(train_gt_files)
    val_len = max(1, int(total_len * args.val_ratio))
    if val_len >= total_len:
        val_len = total_len - 1

    train_files = train_gt_files[:-val_len]
    val_files = train_gt_files[-val_len:]

    print(f"[Data] Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_gt_files)}")

    train_ds = PoseS1DistributedNoClassifierDataset(
        gt_file_list=train_files,
        target_name=target,
        root_sensor_idx=args.root_sensor_idx,
    )
    val_ds = PoseS1DistributedNoClassifierDataset(
        gt_file_list=val_files,
        target_name=target,
        root_sensor_idx=args.root_sensor_idx,
    )
    test_ds = PoseS1DistributedNoClassifierDataset(
        gt_file_list=test_gt_files,
        target_name=target,
        root_sensor_idx=args.root_sensor_idx,
    )

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
    )

    pretrained_state_dict = None
    if args.pretrained_checkpoint_dir:
        pretrained_path = os.path.join(
            args.pretrained_checkpoint_dir,
            target,
            f"best_pose_s1_{target}.pth",
        )
        print(f"[Fine-tune] Loading pretrained checkpoint: {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location=DEVICE, weights_only=False)

        if ckpt.get("target") != target:
            print(
                f"  [warning] checkpoint target='{ckpt.get('target')}' does not match "
                f"target='{target}'"
            )

        for key in ("proj_dim", "rnn_hidden", "rnn_layers", "dropout"):
            if key in ckpt and getattr(args, key) != ckpt[key]:
                print(
                    f"  [Fine-tune] overriding --{key}={getattr(args, key)} with "
                    f"checkpoint value {ckpt[key]} (architecture must match to load weights)"
                )
                setattr(args, key, ckpt[key])

        pretrained_state_dict = ckpt["model_state_dict"]

        if args.learning_rate == LEARNING_RATE:
            print(
                f"  [Fine-tune] using the from-scratch default learning rate "
                f"({LEARNING_RATE}) -- consider passing a lower --learning_rate "
                f"(e.g. 1e-4) when fine-tuning on a small dataset."
            )

    model = DistributedPoseS1NoClassifier(
        input_dim=24,
        proj_dim=args.proj_dim,
        rnn_hidden=args.rnn_hidden,
        rnn_layers=args.rnn_layers,
        dropout=args.dropout,
        bidirectional=True,
        output_dim=3,
    ).to(DEVICE)

    if pretrained_state_dict is not None:
        model.load_state_dict(pretrained_state_dict)
        print("[Fine-tune] Pretrained weights loaded.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.decay_steps,
        gamma=args.decay_rate
    )

    criterion = nn.MSELoss(reduction="sum")

    save_dir_target = os.path.join(args.save_dir, target)
    os.makedirs(save_dir_target, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    best_path = os.path.join(save_dir_target, f"best_pose_s1_{target}.pth")

    run = init_wandb(args, target)
    if WANDB_AVAILABLE and run is not None:
        wandb.watch(model, log="gradients", log_freq=200)

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")

        train_loss = compute_epoch_train_loss(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            clip_val=args.clip_val,
            batch_size_for_norm=args.batch_size,
        )

        val_loss, val_pos_error_cm = compute_epoch_val(
            model=model,
            loader=val_loader,
            criterion=criterion,
            batch_size_for_norm=args.batch_size,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1} -> TrainLoss: {train_loss:.6f} | "
            f"ValLoss: {val_loss:.6f} | ValPosErr(cm): {val_pos_error_cm:.4f} | "
            f"LR: {current_lr:.8f}"
        )

        log_wandb_epoch(
            epoch=epoch + 1,
            train_loss=train_loss,
            val_loss=val_loss,
            val_pos_error_cm=val_pos_error_cm,
            lr=current_lr,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "target": target,
                    "root_sensor_idx": args.root_sensor_idx,
                    "local_sensor_idx": TARGET_CONFIG[target]["sensor_idx"],
                    "gt_key": TARGET_CONFIG[target]["gt_key"],
                    "input_dim": 24,
                    "output_dim": 3,
                    "proj_dim": args.proj_dim,
                    "rnn_hidden": args.rnn_hidden,
                    "rnn_layers": args.rnn_layers,
                    "dropout": args.dropout,
                },
                best_path
            )
            print(f"  [*] Saved best model to: {best_path}")

            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.summary["best_val_loss"] = best_val_loss
                wandb.summary["best_checkpoint"] = best_path
        else:
            patience_counter += 1
            print(f"  [!] Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print("Early stopping.")
                break

    print("\nLoading best model for final evaluation...")
    ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    print("\nRunning final evaluation on VAL split...")
    final_val_loss, final_val_pos_error_cm = evaluate(model, val_loader, criterion)
    print(
        f"VAL -> Loss: {final_val_loss / args.batch_size:.6f} | "
        f"PositionError(cm): {final_val_pos_error_cm:.4f}"
    )

    print("\nRunning final evaluation on TEST split...")
    final_test_loss, final_test_pos_error_cm = evaluate(model, test_loader, criterion)
    print(
        f"TEST -> Loss: {final_test_loss / args.batch_size:.6f} | "
        f"PositionError(cm): {final_test_pos_error_cm:.4f}"
    )

    log_wandb_final_metrics(
        val_loss=final_val_loss / args.batch_size,
        val_pos_error_cm=final_val_pos_error_cm,
        test_loss=final_test_loss / args.batch_size,
        test_pos_error_cm=final_test_pos_error_cm,
    )

    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()

    return {
        "val_position_error_cm": final_val_pos_error_cm,
        "test_position_error_cm": final_test_pos_error_cm,
    }


def main():
    args = build_argparser().parse_args()

    train_gt_files = load_list(args.train_gt_list_file)
    test_gt_files = load_list(args.test_gt_list_file)

    if len(train_gt_files) == 0:
        raise RuntimeError("Training GT list file is empty or invalid.")
    if len(test_gt_files) == 0:
        raise RuntimeError("Test GT list file is empty or invalid.")

    targets = TARGET_ORDER if args.target == "all" else [args.target]
    results = {}

    for target in targets:
        results[target] = train_target(args, target, train_gt_files, test_gt_files)

    print("\n" + "=" * 78)
    print("FINAL FIVE-BRANCH POSE-S1 RESULTS")
    print("=" * 78)

    for target in targets:
        metrics = results[target]
        print(
            f"{target:>10}: "
            f"ValPos={metrics['val_position_error_cm']:.4f} cm | "
            f"TestPos={metrics['test_position_error_cm']:.4f} cm"
        )


def build_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=[*TARGET_ORDER, "all"],
        help="Target sub-network to train: left_arm, right_arm, left_leg, right_leg, head, or all"
    )

    parser.add_argument(
        "--train_gt_list_file",
        type=str,
        default=DEFAULT_TRAIN_GT_LIST,
        help="Path to train Pose-S1 GT list"
    )
    parser.add_argument(
        "--test_gt_list_file",
        type=str,
        default=DEFAULT_TEST_GT_LIST,
        help="Path to test Pose-S1 GT list"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default=DEFAULT_SAVE_DIR,
        help="Directory where weights will be saved"
    )

    parser.add_argument(
        "--pretrained_checkpoint_dir",
        type=str,
        default="",
        help=(
            "Root dir of existing Pose-S1 checkpoints to fine-tune from, e.g. "
            "checkpoints/dbran_pose_s1. The per-target path is built "
            "automatically as <dir>/<target>/best_pose_s1_<target>.pth "
            "-- this avoids accidentally pointing every branch at the same "
            "checkpoint when using --target all. Leave empty to train from "
            "scratch (default behavior, unchanged)."
        ),
    )

    parser.add_argument(
        "--root_sensor_idx",
        type=int,
        default=ROOT_SENSOR_IDX,
        help="Index of the root / pelvis IMU"
    )

    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--clip_val", type=float, default=CLIP_VAL)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--decay_rate", type=float, default=DECAY_RATE)
    parser.add_argument("--decay_steps", type=int, default=DECAY_STEPS)
    parser.add_argument("--val_ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)

    parser.add_argument("--proj_dim", type=int, default=32)
    parser.add_argument("--rnn_hidden", type=int, default=32)
    parser.add_argument("--rnn_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="dbran-pose-s1")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--wandb_group", type=str, default="")
    parser.add_argument("--wandb_tags", type=str, default="pose_s1_distributed")

    return parser


if __name__ == "__main__":
    main()