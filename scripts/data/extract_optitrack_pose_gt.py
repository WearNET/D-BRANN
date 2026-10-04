r"""
Extract SMPL-format pose/translation ground truth from Motive skeleton CSV
exports, using the direct Motive-bone -> SMPL-joint rotation mapping
("fast path", as opposed to a marker-based MoSh-style fit).

Expects CSV exports with:
    - Rotation Type: Quaternion
    - Coordinate Space: Local (i.e. "Use World Coordinates" turned OFF)
    - Units: Meters
    - Skeleton and Markerset Bones enabled, Exclude Fingers enabled

Usage:
    python scripts/data/extract_optitrack_pose_gt.py \
        --csv_dir data/dataset_raw/dbran_optitrack/motive_csv \
        --out data/dataset_raw/dbran_optitrack/pose_gt.pt
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
import csv
from pathlib import Path

import numpy as np
import torch

import articulate as art

# Motive bone name (without the "SkeletonName:" prefix) -> SMPL joint index.
# SMPL 24-joint order: 0 Pelvis, 1 L_Hip, 2 R_Hip, 3 Spine1, 4 L_Knee, 5 R_Knee,
# 6 Spine2, 7 L_Ankle, 8 R_Ankle, 9 Spine3, 10 L_Foot, 11 R_Foot, 12 Neck,
# 13 L_Collar, 14 R_Collar, 15 Head, 16 L_Shoulder, 17 R_Shoulder, 18 L_Elbow,
# 19 R_Elbow, 20 L_Wrist, 21 R_Wrist, 22 L_Hand, 23 R_Hand.
# Joints 9, 22, 23 have no Motive counterpart in this 21-bone skeleton
# (no separate spine3, fingers excluded) and are left at identity rotation.
MOTIVE_BONE_TO_SMPL_JOINT = {
    "SkeletonKevin": 0,
    "LThigh": 1,
    "RThigh": 2,
    "Ab": 3,
    "LShin": 4,
    "RShin": 5,
    "Chest": 6,
    "LFoot": 7,
    "RFoot": 8,
    "LToe": 10,
    "RToe": 11,
    "Neck": 12,
    "LShoulder": 13,
    "RShoulder": 14,
    "Head": 15,
    "LUArm": 16,
    "RUArm": 17,
    "LFArm": 18,
    "RFArm": 19,
    "LHand": 20,
    "RHand": 21,
}


def _parse_header(lines):
    r"""
    Parse the 8 Motive CSV header lines.

    Columns 0/1 are always Frame/Time in every header row (rows that show a
    row label like "Type"/"Name"/"Parent" put it in column 1 instead of
    leaving it blank, but the column *indices* still line up with the data
    rows). Bone-attribute columns start at index 2.

    :return: list of (bone_name, attribute, component) for each data column
        after Frame/Time.
    """
    name_row = next(csv.reader([lines[3]]))[2:]
    attr_row = next(csv.reader([lines[6]]))[2:]
    label_row = next(csv.reader([lines[7]]))[2:]

    columns = []
    for name, attr, label in zip(name_row, attr_row, label_row):
        bone = name.split(":", 1)[1] if ":" in name else name
        columns.append((bone, attr, label))
    return columns


def _interpolate_missing_rows(data, take_name):
    r"""
    Linearly interpolate any fully/partially blank rows -- e.g. from a
    momentary Motive skeleton-solve failure (marker occlusion) -- using the
    nearest valid rows before and after. Linear interpolation of quaternion
    components over a single 60Hz frame (~16.7ms) is an excellent
    approximation of true SLERP, since adjacent frames are nearly
    identical rotations, and quaternion_to_axis_angle() renormalizes
    downstream regardless.

    :param data: (N, num_columns) raw CSV data, NaN where a cell was blank.
    :return: (N, num_columns) with no NaNs left.
    """
    missing = np.isnan(data).any(axis=1)
    if not missing.any():
        return data

    data = data.copy()
    n = data.shape[0]
    missing_idx = np.where(missing)[0]
    print(
        f"  [warning] {take_name}: {len(missing_idx)} blank frame(s) "
        f"detected (likely a momentary tracking-solve failure), "
        f"interpolating: {missing_idx.tolist()}"
    )

    i = 0
    while i < n:
        if not missing[i]:
            i += 1
            continue
        j = i
        while j < n and missing[j]:
            j += 1
        before, after = i - 1, j
        if before < 0 and after >= n:
            raise RuntimeError(f"{take_name}: entire take is missing -- cannot interpolate.")
        elif before < 0:
            data[i:j] = data[after]
        elif after >= n:
            data[i:j] = data[before]
        else:
            for k in range(i, j):
                t = (k - before) / (after - before)
                data[k] = data[before] * (1 - t) + data[after] * t
        i = j

    return data


def _zero_rest_pose(pose, ref_frame_idx):
    r"""
    Re-express joints 1..23 relative to their own rotation at
    ``ref_frame_idx``, removing a constant per-joint offset between
    Motive's rest-pose bone reference and SMPL's zero pose. The root
    joint (0) is left untouched, since it must stay in the same global
    reference frame across every take (matching the Xsens calibration).

    :param pose: (N, 24, 3) axis-angle pose.
    :param ref_frame_idx: frame index used as the per-joint zero
        reference; should fall within a still portion of the take.
    :return: (N, 24, 3) corrected axis-angle pose.
    """
    ref_frame_idx = min(ref_frame_idx, pose.shape[0] - 1)
    n_frames = pose.shape[0]

    r = art.math.axis_angle_to_rotation_matrix(
        torch.from_numpy(pose[:, 1:].reshape(-1, 3))
    ).view(n_frames, 23, 3, 3)
    r_ref = r[ref_frame_idx]  # (23, 3, 3)
    r_corrected = torch.matmul(r_ref.transpose(-1, -2).unsqueeze(0), r)
    aa = art.math.rotation_matrix_to_axis_angle(
        r_corrected.reshape(-1, 3, 3)
    ).view(n_frames, 23, 3)

    pose = pose.copy()
    pose[:, 1:] = aa.numpy()
    return pose


def load_take_csv(path, ref_frame_idx=30):
    r"""
    Load a single Motive skeleton CSV export.

    :param ref_frame_idx: frame used as the per-joint rest-pose reference
        for ``_zero_rest_pose`` (default 30 = 0.5 s at 60 Hz).
    :return: (pose, tran)
        pose: (N, 24, 3) float32 axis-angle SMPL pose.
        tran: (N, 3) float32 root translation in meters, relative to frame 0.
    """
    path = Path(path)
    with open(path, "r", newline="") as f:
        lines = [f.readline() for _ in range(8)]
        columns = _parse_header(lines)
        # genfromtxt (unlike loadtxt) turns blank cells into NaN instead of
        # raising, so a momentary Motive tracking-solve failure doesn't
        # crash the whole take.
        data = np.genfromtxt(f, delimiter=",", dtype=np.float64, filling_values=np.nan)

    if data.ndim == 1:
        data = data[None, :]

    data = _interpolate_missing_rows(data, path.name)

    n_frames = data.shape[0]
    bone_data = data[:, 2:]  # drop Frame, Time

    bone_cols = {}
    for idx, (bone, attr, label) in enumerate(columns):
        bone_cols.setdefault(bone, {}).setdefault(attr, {})[label] = idx

    pose = np.zeros((n_frames, 24, 3), dtype=np.float32)
    tran = np.zeros((n_frames, 3), dtype=np.float32)

    unmapped = []
    for bone, attrs in bone_cols.items():
        joint_idx = MOTIVE_BONE_TO_SMPL_JOINT.get(bone)
        if joint_idx is None:
            unmapped.append(bone)
            continue

        rot = attrs["Rotation"]
        q_xyzw = bone_data[:, [rot["X"], rot["Y"], rot["Z"], rot["W"]]]
        q_wxyz = torch.from_numpy(q_xyzw[:, [3, 0, 1, 2]]).float()

        # Canonicalize sign: q and -q represent the same rotation, but
        # acos() in quaternion_to_axis_angle does not know that. Without
        # this, a near-identity rotation exported with w < 0 comes back as
        # an axis-angle near 360 degrees instead of near 0.
        sign = torch.where(q_wxyz[:, 0:1] < 0, -1.0, 1.0)
        q_wxyz = q_wxyz * sign

        pose[:, joint_idx] = art.math.quaternion_to_axis_angle(q_wxyz).numpy()

        if "Position" in attrs:
            pos = attrs["Position"]
            xyz = bone_data[:, [pos["X"], pos["Y"], pos["Z"]]].astype(np.float32)
            tran[:] = xyz - xyz[0]

    if unmapped:
        print(f"  [warning] unmapped bones in {path.name} (ignored): {unmapped}")

    pose = _zero_rest_pose(pose, ref_frame_idx)

    return pose, tran


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "dataset_raw" / "dbran_optitrack" / "motive_csv"),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(PROJECT_ROOT / "data" / "dataset_raw" / "dbran_optitrack" / "pose_gt.pt"),
    )
    parser.add_argument(
        "--ref_frame",
        type=int,
        default=30,
        help="Frame used as the per-joint rest-pose zero reference (default: 30 = 0.5s at 60Hz).",
    )
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    files = sorted(csv_dir.glob("Take_*.csv"))
    if not files:
        raise RuntimeError(f"No Take_*.csv files found in {csv_dir}")

    poses, trans, names = [], [], []
    for f in files:
        pose, tran = load_take_csv(f, ref_frame_idx=args.ref_frame)
        poses.append(torch.from_numpy(pose))
        trans.append(torch.from_numpy(tran))
        names.append(f.stem)

        flat_idx = np.abs(pose[0]).argmax()
        joint_idx, _ = np.unravel_index(flat_idx, pose[0].shape)
        frame0_deg = np.degrees(np.abs(pose[0]).flat[flat_idx])
        print(
            f"{f.name}: {pose.shape[0]} frames, "
            f"max joint rotation at frame 0 = {frame0_deg:.1f} deg (SMPL joint {joint_idx}) "
            f"(should be small if the take starts near neutral stance)"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"pose": poses, "tran": trans, "names": names}, out_path)
    print(f"\nSaved {len(poses)} sequences to {out_path}")


if __name__ == "__main__":
    main()
