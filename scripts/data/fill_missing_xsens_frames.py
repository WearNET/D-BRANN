r"""
Fill NaN-marked missing frames saved by xsensDataCapture.py, using proper
interpolation between the valid frames immediately before and after each
gap: linear for acceleration, spherical/geodesic (via axis-angle) for
orientation, since naive linear interpolation of rotation matrix entries
does not produce a valid rotation.

This is deliberately a separate post-processing step, not done live during
capture, so the fill can use the frame *after* the gap too -- something the
live loop cannot know yet when the gap is first detected.

Usage:
    python scripts/data/fill_missing_xsens_frames.py \
        --input data/dataset_raw/dbran_optitrack/captura_001.pt \
        --output data/dataset_raw/dbran_optitrack/captura_001_filled.pt
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
# END D-BRAN PROJECT BOOTSTRAP

import argparse
from pathlib import Path

import torch

import articulate as art


def _slerp_rotmats(r_before: torch.Tensor, r_after: torch.Tensor, n: int):
    r"""Geodesic interpolation between two (6,3,3) rotation-matrix sets."""
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


def _lerp(a_before: torch.Tensor, a_after: torch.Tensor, n: int):
    return [
        a_before * (1 - k / (n + 1)) + a_after * (k / (n + 1))
        for k in range(1, n + 1)
    ]


def fill_missing(acc: torch.Tensor, ori: torch.Tensor, is_missing):
    r"""
    :param acc: (N, 6, 3)
    :param ori: (N, 6, 3, 3)
    :param is_missing: length-N sequence of bool
    :return: (acc_filled, ori_filled), same shapes as input.
    """
    acc = acc.clone()
    ori = ori.clone()
    n = acc.shape[0]
    missing = list(is_missing)

    i = 0
    while i < n:
        if not missing[i]:
            i += 1
            continue

        j = i
        while j < n and missing[j]:
            j += 1

        gap = j - i
        before_idx = i - 1
        after_idx = j

        if before_idx < 0 and after_idx >= n:
            raise RuntimeError("Entire capture is missing -- cannot fill.")
        elif before_idx < 0:
            print(
                f"  [warning] leading gap of {gap} frame(s), no prior frame -- "
                f"holding the first valid frame"
            )
            for k in range(i, j):
                acc[k] = acc[after_idx]
                ori[k] = ori[after_idx]
        elif after_idx >= n:
            print(
                f"  [warning] trailing gap of {gap} frame(s), no following frame -- "
                f"holding the last valid frame"
            )
            for k in range(i, j):
                acc[k] = acc[before_idx]
                ori[k] = ori[before_idx]
        else:
            acc_filled = _lerp(acc[before_idx], acc[after_idx], gap)
            ori_filled = _slerp_rotmats(ori[before_idx], ori[after_idx], gap)
            for offset, k in enumerate(range(i, j)):
                acc[k] = acc_filled[offset]
                ori[k] = ori_filled[offset]

        i = j

    return acc, ori


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Defaults to overwriting --input.",
    )
    args = parser.parse_args()

    data = torch.load(args.input, weights_only=False)
    is_missing = data.get("is_missing")
    if is_missing is None:
        raise RuntimeError(
            f"{args.input} has no 'is_missing' field -- was it captured "
            "with an older version of xsensDataCapture.py?"
        )

    n_missing = int(sum(is_missing))
    n_total = len(is_missing)
    print(
        f"{args.input}: {n_missing}/{n_total} frames missing "
        f"({100.0 * n_missing / n_total:.2f}%)"
    )

    out_path = Path(args.output) if args.output else Path(args.input)

    if n_missing == 0:
        print("Nothing to fill.")
    else:
        acc_filled, ori_filled = fill_missing(
            data["acc_calibrated"], data["ori_calibrated"], is_missing
        )
        data["acc_calibrated"] = acc_filled
        data["ori_calibrated"] = ori_filled
        data["is_missing"] = [False] * n_total
        data["frames_missing"] = 0
        data["frames_filled_by_postprocessing"] = n_missing

    torch.save(data, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
