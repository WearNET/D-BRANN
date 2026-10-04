
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

import os
import argparse
import hashlib
import torch
from tqdm import tqdm
import articulate as art
from config import paths

FULL_JOINTS = list(range(1, 24))
LOWER_BODY_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]
UPPER_BODY_JOINTS = [13, 14, 16, 17, 18, 19, 20, 21, 22, 23]
TRUNK_HEAD_JOINTS = [3, 6, 9, 12, 15]


def load_list(list_file):
    with open(list_file, "r") as f:
        return [x.strip() for x in f if x.strip() and os.path.exists(x.strip())]


def safe_name(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    h = hashlib.md5(path.encode("utf-8")).hexdigest()[:10]
    return f"{stem}_{h}"


def cols_for_joints(joints):
    cols = []
    for j in joints:
        c = (j - 1) * 3
        cols += [c, c + 1, c + 2]
    return cols


def fk_joints(body_model, pose_aa, tran):
    pose_rot = art.math.axis_angle_to_rotation_matrix(pose_aa).view(-1, 24, 3, 3)
    out = body_model.forward_kinematics(pose_rot, tran=tran, calc_mesh=False)
    if isinstance(out, (tuple, list)):
        if len(out) == 3:
            _, joints, _ = out
        elif len(out) == 2:
            _, joints = out
        else:
            raise RuntimeError(f"Unexpected FK outputs: {len(out)}")
    else:
        raise RuntimeError("FK did not return tuple/list")
    return joints[:, :24].contiguous()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_list_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_list_file", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = load_list(args.raw_list_file)
    body_model = art.ParametricModel(paths.smpl_file)

    lower_cols = cols_for_joints(LOWER_BODY_JOINTS)
    upper_cols = cols_for_joints(UPPER_BODY_JOINTS)
    trunk_cols = cols_for_joints(TRUNK_HEAD_JOINTS)

    saved = []
    print(f"Files: {len(files)}")
    for src in tqdm(files, desc="Pose-S2 GT", ncols=120):
        raw = torch.load(src, weights_only=False)
        joints = fk_joints(body_model, raw["pose"], raw["tran"])
        root = joints[:, 0:1, :]
        rel = joints - root
        full = rel[:, FULL_JOINTS, :].reshape(joints.shape[0], -1).float()
        out = {
            "source_file": src,
            "num_frames": full.shape[0],
            "full_joints": FULL_JOINTS,
            "lower_body_joints": LOWER_BODY_JOINTS,
            "upper_body_joints": UPPER_BODY_JOINTS,
            "trunk_head_joints": TRUNK_HEAD_JOINTS,
            "full_joint_position_gt": full,
            "lower_body_gt": full[:, lower_cols].contiguous(),
            "upper_body_gt": full[:, upper_cols].contiguous(),
            "trunk_head_gt": full[:, trunk_cols].contiguous(),
        }
        dst = os.path.join(args.output_dir, safe_name(src) + "_pose_s2_gt.pt")
        torch.save(out, dst)
        saved.append(dst)

    with open(args.output_list_file, "w") as f:
        for p in saved:
            f.write(p + "\n")
    print(f"Saved: {len(saved)}")
    print(f"List: {args.output_list_file}")


if __name__ == "__main__":
    main()
