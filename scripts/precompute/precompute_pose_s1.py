
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
from main_path import POSE_S1_CHECKPOINTS_DIR


import os
import argparse
import hashlib
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_SENSOR_IDX = 5
LEFT_ARM_SENSOR_IDX = 0
RIGHT_ARM_SENSOR_IDX = 1
LEFT_LEG_SENSOR_IDX = 2
RIGHT_LEG_SENSOR_IDX = 3
HEAD_SENSOR_IDX = 4
LEAF_TARGETS = [("left_leg", LEFT_LEG_SENSOR_IDX), ("right_leg", RIGHT_LEG_SENSOR_IDX), ("head", HEAD_SENSOR_IDX), ("left_arm", LEFT_ARM_SENSOR_IDX), ("right_arm", RIGHT_ARM_SENSOR_IDX)]

class DistributedPoseS1(nn.Module):
    def __init__(self, input_dim=24, proj_dim=32, rnn_hidden=32, rnn_layers=2, dropout=0.2, output_dim=3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc_in = nn.Linear(input_dim, proj_dim)
        self.rnn = nn.LSTM(proj_dim, rnn_hidden, rnn_layers, batch_first=False, bidirectional=True, dropout=dropout if rnn_layers > 1 else 0.0)
        self.fc_out = nn.Linear(rnn_hidden * 2, output_dim)
    def forward(self, x):
        data = torch.relu(self.fc_in(self.dropout(x.data)))
        packed = PackedSequence(data, x.batch_sizes, x.sorted_indices, x.unsorted_indices)
        y, _ = self.rnn(packed)
        out = self.fc_out(y.data)
        return PackedSequence(out, y.batch_sizes, y.sorted_indices, y.unsorted_indices)

def load_list(path):
    with open(path, "r") as f:
        return [x.strip() for x in f if x.strip() and os.path.exists(x.strip())]

def safe_name(path):
    return os.path.splitext(os.path.basename(path))[0] + "_" + hashlib.md5(path.encode("utf-8")).hexdigest()[:10]

def ensure_acc(acc):
    if acc.dim() == 3: return acc.float()
    return acc.view(acc.shape[0], 6, 3).float()

def ensure_ori(ori):
    if ori.dim() == 4: return ori.float()
    return ori.view(ori.shape[0], 6, 3, 3).float()

def build_two_imu(acc, ori, root_idx, local_idx):
    acc = ensure_acc(acc); ori = ensure_ori(ori)
    return torch.cat([acc[:, root_idx, :], ori[:, root_idx].reshape(-1, 9), acc[:, local_idx, :], ori[:, local_idx].reshape(-1, 9)], dim=1)

@torch.no_grad()
def run_model(model, x):
    lengths = torch.tensor([x.shape[0]], dtype=torch.long)
    packed = pack_padded_sequence(x.unsqueeze(0).to(DEVICE), lengths.cpu(), batch_first=True, enforce_sorted=True)
    out = model(packed)
    pad, _ = pad_packed_sequence(out, batch_first=True)
    return pad[0].detach().cpu()

def load_model(distributed_root_dir, target_name):
    import glob

    candidate_paths = [
        os.path.join(
            distributed_root_dir,
            target_name,
            f"best_pose_s1_{target_name}.pth",
        ),
    ]

    ckpt_path = None

    for p in candidate_paths:
        if os.path.exists(p):
            ckpt_path = p
            break

    if ckpt_path is None:
        pattern = os.path.join(distributed_root_dir, "**", f"*{target_name}*.pth")
        matches = sorted(glob.glob(pattern, recursive=True))

        if len(matches) == 0:
            raise FileNotFoundError(
                f"No checkpoint found for target '{target_name}' inside {distributed_root_dir}"
            )

        ckpt_path = matches[0]

    print(f"[info] Loading {target_name} from: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    model = DistributedPoseS1(
        input_dim=ckpt.get("input_dim", 24),
        proj_dim=ckpt.get("proj_dim", 32),
        rnn_hidden=ckpt.get("rnn_hidden", 32),
        rnn_layers=ckpt.get("rnn_layers", 2),
        dropout=ckpt.get("dropout", 0.2),
        output_dim=ckpt.get("output_dim", 3),
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_list_file", required=True)
    p.add_argument("--distributed_root_dir", default=str(POSE_S1_CHECKPOINTS_DIR))
    p.add_argument("--output_dir", required=True)
    p.add_argument("--output_list_file", required=True)
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    files = load_list(args.raw_list_file)
    models = {name: load_model(args.distributed_root_dir, name) for name, _ in LEAF_TARGETS}
    saved = []
    print("DEVICE:", DEVICE)
    print("Files:", len(files))
    for src in tqdm(files, desc="Pose-S1 predictions", ncols=120):
        raw = torch.load(src, weights_only=False)
        outs = []
        data = {"source_file": src, "leaf_order": [x[0] for x in LEAF_TARGETS], "leaf_joint_indices": [7, 8, 12, 20, 21]}
        for name, sensor_idx in LEAF_TARGETS:
            x = build_two_imu(raw["acc"], raw["ori"], ROOT_SENSOR_IDX, sensor_idx)
            y = run_model(models[name], x)
            data[f"{name}_p_pred"] = y.float()
            outs.append(y)
        data["p_leaf_pred"] = torch.cat(outs, dim=1).float()
        data["num_frames"] = data["p_leaf_pred"].shape[0]
        dst = os.path.join(args.output_dir, safe_name(src) + "_pose_s1_pred.pt")
        torch.save(data, dst)
        saved.append(dst)
    with open(args.output_list_file, "w") as f:
        for s in saved: f.write(s + "\n")
    print("Saved:", len(saved))

if __name__ == "__main__": main()
