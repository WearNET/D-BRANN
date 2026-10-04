r"""
Align OptiTrack-derived SMPL pose ground truth (pose_gt.pt, one entry per
Motive Take) with the matching Xsens live captures (captura_XXX.pt, one file
per run of run_dbran_live.py --save_pt), and package the result into the
same {'acc', 'ori', 'pose', 'tran'} format used by DIP-IMU/TotalCapture.

The two recordings are started by hand on two different processes, so they
do not share a common absolute clock. Instead, each take starts and ends
with the same sharp, marked gesture (performed while both systems are
already recording). This script finds that gesture's peak in a movement
-energy signal derived from each stream near the start and near the end,
uses the two matched pairs as anchors, and fits a linear (offset + rate)
mapping from Xsens frame index to Motive frame index -- correcting for
both the start-time offset and any small clock-rate mismatch over the
length of the take.

Usage:
    python scripts/data/align_and_package_dataset.py \
        --pose_gt data/dataset_raw/dbran_optitrack/pose_gt.pt \
        --capture_dir data/dataset_raw/dbran_optitrack \
        --out data/dataset_work/dbran_optitrack/train.pt
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
import re
from pathlib import Path

import numpy as np
import torch

import articulate as art

XSENS_HZ = 60.0
MOTIVE_HZ = 60.0


def _capture_name_for_take(take_name):
    r"""'Take_001' -> 'captura_001.pt'"""
    match = re.search(r"(\d+)", take_name)
    if not match:
        raise ValueError(f"Could not extract a sequence number from '{take_name}'")
    return f"captura_{match.group(1)}.pt"


def _smooth(x, window):
    if window <= 1:
        return x
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(x, kernel, mode="same")


def _find_peak(energy, lo, hi, smooth_window):
    lo = max(lo, 0)
    hi = min(hi, len(energy))
    segment = _smooth(energy[lo:hi], smooth_window)
    return lo + int(np.argmax(segment))


def _motive_energy(pose):
    r"""Per-frame proxy for whole-body angular velocity: (N,) numpy array."""
    pose = pose.numpy() if torch.is_tensor(pose) else pose
    delta = np.diff(pose, axis=0)  # (N-1, 24, 3)
    energy = np.linalg.norm(delta.reshape(delta.shape[0], -1), axis=1)
    return np.concatenate([[0.0], energy])  # (N,), aligned to frame index


def _xsens_energy(acc_calibrated):
    r"""Per-frame proxy for whole-body movement from IMU acceleration: (N,)."""
    acc = acc_calibrated.numpy() if torch.is_tensor(acc_calibrated) else acc_calibrated
    return np.linalg.norm(acc.reshape(acc.shape[0], -1), axis=1)


def _slerp_rotmats(r_before, r_after, n):
    r_rel = torch.matmul(r_before.transpose(-1, -2), r_after)
    aa_rel = art.math.rotation_matrix_to_axis_angle(
        r_rel.reshape(-1, 3, 3)
    ).view(r_before.shape[0], 3)
    out = []
    for k in range(1, n + 1):
        t = k / (n + 1)
        r_t = art.math.axis_angle_to_rotation_matrix(
            (aa_rel * t).reshape(-1, 3)
        ).view_as(r_before)
        out.append(torch.matmul(r_before, r_t))
    return out


def _lerp(a_before, a_after, n):
    return [
        a_before * (1 - k / (n + 1)) + a_after * (k / (n + 1))
        for k in range(1, n + 1)
    ]


def fill_missing_capture(acc, ori, is_missing, label):
    r"""
    Fill NaN-marked missing frames (dropped Xsens packets) via linear
    interpolation (acceleration) and geodesic/spherical interpolation
    (orientation), using the valid frames on both sides of each gap.
    Mirrors fill_missing_xsens_frames.py so this always runs, instead of
    depending on that script having been run manually beforehand.

    :param acc: (N, 6, 3)
    :param ori: (N, 6, 3, 3)
    :param is_missing: length-N sequence of bool
    :return: (acc_filled, ori_filled)
    """
    missing = list(is_missing)
    n_missing = sum(missing)
    if n_missing == 0:
        return acc, ori

    print(f"  [info] {label}: filling {n_missing}/{len(missing)} missing Xsens frames")
    acc = acc.clone()
    ori = ori.clone()
    n = acc.shape[0]

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
            raise RuntimeError(f"{label}: entire capture is missing -- cannot fill.")
        elif before < 0:
            for k in range(i, j):
                acc[k] = acc[after]
                ori[k] = ori[after]
        elif after >= n:
            for k in range(i, j):
                acc[k] = acc[before]
                ori[k] = ori[before]
        else:
            gap = j - i
            acc_filled = _lerp(acc[before], acc[after], gap)
            ori_filled = _slerp_rotmats(ori[before], ori[after], gap)
            for offset, k in enumerate(range(i, j)):
                acc[k] = acc_filled[offset]
                ori[k] = ori_filled[offset]
        i = j

    return acc, ori


def align_take(pose, tran, acc, ori, gesture_window_s, smooth_window, trim_s, has_end_gesture=True):
    r"""
    :param pose: (Nm, 24, 3) Motive-derived SMPL pose.
    :param tran: (Nm, 3) Motive-derived root translation.
    :param acc: (Nx, 6, 3) calibrated Xsens acceleration.
    :param ori: (Nx, 6, 3, 3) calibrated Xsens orientation.
    :param has_end_gesture: if False, only the start gesture is used as an
        anchor (no drift/rate correction possible with a single anchor);
        the take is simply trimmed to whichever stream runs out first.
    :return: dict with aligned 'acc', 'ori', 'pose', 'tran' (all length Nx'),
        plus diagnostic peak times in seconds for manual sanity-checking.
    """
    n_motive = pose.shape[0]
    n_xsens = acc.shape[0]

    m_energy = _motive_energy(pose)
    x_energy = _xsens_energy(acc)

    win_m = int(gesture_window_s * MOTIVE_HZ)
    win_x = int(gesture_window_s * XSENS_HZ)
    trim = int(trim_s * XSENS_HZ)

    m_start = _find_peak(m_energy, 0, win_m, smooth_window)
    x_start = _find_peak(x_energy, 0, win_x, smooth_window)

    if has_end_gesture:
        m_end = _find_peak(m_energy, n_motive - win_m, n_motive, smooth_window)
        x_end = _find_peak(x_energy, n_xsens - win_x, n_xsens, smooth_window)

        # Fit motive_frame = a * xsens_frame + b from the two anchor pairs.
        if x_end == x_start:
            raise RuntimeError("Start and end gesture peaks collapsed to the same Xsens frame.")
        a = (m_end - m_start) / (x_end - x_start)
        b = m_start - a * x_start
        x_hi = min(x_end - trim, n_xsens)
    else:
        # No end gesture (recording cut off before it): a single anchor
        # only fixes the offset, not the rate, so assume rate_ratio = 1
        # and just run until whichever stream ends first.
        m_end = None
        x_end = None
        a = 1.0
        b = m_start - a * x_start
        x_hi = min(n_xsens, int(np.floor((n_motive - 1 - b) / a)))

    x_lo = max(x_start + trim, 0)
    x_hi = min(x_hi, n_xsens)
    if x_hi <= x_lo:
        raise RuntimeError("Nothing left to align after trimming around the gesture peak(s).")

    x_indices = np.arange(x_lo, x_hi)
    m_indices = np.round(a * x_indices + b).astype(int)
    valid = (m_indices >= 0) & (m_indices < n_motive)
    x_indices = x_indices[valid]
    m_indices = m_indices[valid]

    diagnostics = {
        "motive_start_peak_s": m_start / MOTIVE_HZ,
        "xsens_start_peak_s": x_start / XSENS_HZ,
        "motive_end_peak_s": m_end / MOTIVE_HZ if m_end is not None else None,
        "xsens_end_peak_s": x_end / XSENS_HZ if x_end is not None else None,
        "rate_ratio": a,
        "aligned_frames": len(x_indices),
    }

    return {
        "acc": acc[x_indices],
        "ori": ori[x_indices],
        "pose": pose[m_indices],
        "tran": tran[m_indices],
    }, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pose_gt",
        type=str,
        default=str(PROJECT_ROOT / "data" / "dataset_raw" / "dbran_optitrack" / "pose_gt.pt"),
    )
    parser.add_argument(
        "--capture_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "dataset_raw" / "dbran_optitrack"),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(PROJECT_ROOT / "data" / "dataset_work" / "dbran_optitrack" / "train.pt"),
    )
    parser.add_argument(
        "--gesture_window_s",
        type=float,
        default=8.0,
        help="How many seconds from the start/end to search for the marked gesture peak.",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=5,
        help="Moving-average window (frames) applied before peak-picking.",
    )
    parser.add_argument(
        "--trim_s",
        type=float,
        default=0.5,
        help="Seconds to trim inward from each detected gesture peak before keeping data.",
    )
    parser.add_argument(
        "--no_end_gesture_takes",
        type=str,
        default="",
        help=(
            "Comma-separated take numbers (or names) that have no closing "
            "gesture, e.g. '10' or 'Take_010,Take_007'. Those takes are "
            "aligned using only the start gesture and trimmed to whichever "
            "stream (Motive or Xsens) runs out first."
        ),
    )
    args = parser.parse_args()

    no_end_gesture = {
        re.search(r"(\d+)", tok).group(1).lstrip("0") or "0"
        for tok in args.no_end_gesture_takes.split(",")
        if tok.strip()
    }

    pose_gt = torch.load(args.pose_gt)
    capture_dir = Path(args.capture_dir)

    accs, oris, poses, trans, names = [], [], [], [], []

    for take_name, pose, tran in zip(pose_gt["names"], pose_gt["pose"], pose_gt["tran"]):
        capture_path = capture_dir / _capture_name_for_take(take_name)
        if not capture_path.is_file():
            print(f"  [warning] no matching capture for {take_name} at {capture_path}, skipping")
            continue

        take_number = re.search(r"(\d+)", take_name).group(1).lstrip("0") or "0"
        has_end_gesture = take_number not in no_end_gesture

        capture = torch.load(capture_path)
        acc_capture, ori_capture = fill_missing_capture(
            capture["acc_calibrated"],
            capture["ori_calibrated"],
            capture.get("is_missing", []),
            take_name,
        )
        aligned, diag = align_take(
            pose,
            tran,
            acc_capture,
            ori_capture,
            args.gesture_window_s,
            args.smooth_window,
            args.trim_s,
            has_end_gesture=has_end_gesture,
        )

        accs.append(aligned["acc"])
        oris.append(aligned["ori"])
        poses.append(aligned["pose"])
        trans.append(aligned["tran"])
        names.append(take_name)

        start_str = (
            f"start gesture at motive={diag['motive_start_peak_s']:.2f}s / "
            f"xsens={diag['xsens_start_peak_s']:.2f}s"
        )
        if has_end_gesture:
            end_str = (
                f"end gesture at motive={diag['motive_end_peak_s']:.2f}s / "
                f"xsens={diag['xsens_end_peak_s']:.2f}s"
            )
        else:
            end_str = "no end gesture (trimmed to shorter stream)"

        print(
            f"{take_name}: {start_str} | {end_str} | "
            f"rate_ratio={diag['rate_ratio']:.5f} | "
            f"{diag['aligned_frames']} aligned frames "
            f"({diag['aligned_frames'] / XSENS_HZ:.1f}s)"
        )

        if has_end_gesture and abs(diag["rate_ratio"] - 1.0) > 0.01:
            print(
                f"  [warning] rate_ratio far from 1.0 -- check that both gesture "
                f"peaks were correctly detected for {take_name}"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"acc": accs, "ori": oris, "pose": poses, "tran": trans, "names": names}, out_path)
    print(f"\nSaved {len(accs)} aligned sequences to {out_path}")


if __name__ == "__main__":
    main()
