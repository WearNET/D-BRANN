r"""
Convert the aligned OptiTrack+Xsens dataset (dbran_optitrack/train.pt, from
align_and_package_dataset.py) into individual per-sequence raw files
matching the schema used everywhere else in the repo -- acc, ori, pose,
tran, joints, shape, source, index (e.g. data/dataset_train/AMASS/amass_*.pt)
-- so it can be fed unchanged into prepare_pose_s1_gt.py / prepare_pose_s2_*
/ prepare_pose_s3_* and the training scripts.

`shape` is set to the SMPL mean shape (all zeros; forward_kinematics treats
None and zeros identically) for every sequence, since we don't have a
subject-specific body-shape fit from the OptiTrack capture. This only
affects the bone *lengths* used to compute `joints` -- the acc/ori/pose/tran
ground truth itself is unaffected.

Usage:
    python scripts/data/finalize_dbran_optitrack_dataset.py \
        --aligned data/dataset_work/dbran_optitrack/train.pt \
        --output_dir data/dataset_train/DBRAN_OptiTrack \
        --output_list_file data/dataset_train/train_pose_dbran_optitrack.txt
"""

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
from pathlib import Path

import torch

import articulate as art
from config import paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aligned",
        type=str,
        default=str(PROJECT_ROOT / "data" / "dataset_work" / "dbran_optitrack" / "train.pt"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "dataset_train" / "DBRAN_OptiTrack"),
    )
    parser.add_argument(
        "--output_list_file",
        type=str,
        default=str(PROJECT_ROOT / "data" / "dataset_train" / "train_pose_dbran_optitrack.txt"),
    )
    args = parser.parse_args()

    data = torch.load(args.aligned, weights_only=False)
    body_model = art.ParametricModel(paths.smpl_file)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    sequences = zip(data["acc"], data["ori"], data["pose"], data["tran"], data["names"])
    for i, (acc, ori, pose, tran, name) in enumerate(sequences):
        shape = torch.zeros(10)
        rotmats = art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
        _, joints = body_model.forward_kinematics(rotmats, shape, tran, calc_mesh=False)

        out = {
            "acc": acc.float(),
            "ori": ori.float(),
            "pose": pose.float(),
            "tran": tran.float(),
            "joints": joints.float(),
            "shape": shape,
            "source": "DBRAN_OptiTrack",
            "index": i,
            "take_name": name,
        }

        dst = output_dir / f"optitrack_{i:03d}.pt"
        torch.save(out, dst)
        saved_paths.append(str(dst))
        print(f"{name}: {pose.shape[0]} frames -> {dst}")

    with open(args.output_list_file, "w") as f:
        for p in saved_paths:
            f.write(p + "\n")

    print(f"\nSaved {len(saved_paths)} sequences. Manifest: {args.output_list_file}")


if __name__ == "__main__":
    main()
