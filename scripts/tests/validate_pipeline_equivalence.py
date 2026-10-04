"""
Validate that `dbran.pipeline.DBranPipeline` reproduces the current profiler.

The script compares the new reusable implementation against
`scripts/profile/profile_full_pipeline.py` using the same sequence
and checkpoint files.
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
# END D-BRAN PROJECT BOOTSTRAP

import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import torch

from dbran.pipeline import DBranPipeline
from main_path import (
    POSE_S1_CHECKPOINTS_DIR,
    POSE_S2_CHECKPOINTS_DIR,
    POSE_S3_CHECKPOINTS_DIR,
    FUSION_CHECKPOINT,
    TEST_POSE_LIST,
    TRANSPOSE_WEIGHTS_FILE,
)
from scripts.profile import profile_full_pipeline as reference


def _first_existing_path(list_file: Path) -> Path:
    if not list_file.is_file():
        raise FileNotFoundError(f"Sequence list not found: {list_file}")

    missing = []
    with list_file.open("r", encoding="utf-8") as file:
        for line in file:
            value = line.strip()
            if not value:
                continue
            path = Path(value).expanduser()
            if path.is_file():
                return path.resolve()
            missing.append(path)

    raise RuntimeError(
        f"No existing sequence was found in {list_file}. "
        f"Checked {len(missing)} non-empty entries."
    )


def _load_raw_sequence(
    sequence_file: Optional[str],
    raw_list_file: str,
) -> Tuple[Path, Dict]:
    if sequence_file is not None:
        path = Path(sequence_file).expanduser().resolve()
    else:
        path = _first_existing_path(Path(raw_list_file).expanduser().resolve())

    if not path.is_file():
        raise FileNotFoundError(f"Sequence file not found: {path}")

    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise RuntimeError(
            f"Expected a dictionary in {path}, received {type(raw).__name__}."
        )
    for required_key in ("acc", "ori"):
        if required_key not in raw:
            raise KeyError(f"Sequence {path} does not contain key {required_key!r}.")
    return path, raw


def _max_abs_difference(actual: torch.Tensor, expected: torch.Tensor) -> float:
    if actual.numel() == 0 and expected.numel() == 0:
        return 0.0
    return float((actual.detach().cpu() - expected.detach().cpu()).abs().max())


def _assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
) -> None:
    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.detach().cpu()

    if actual_cpu.shape != expected_cpu.shape:
        raise AssertionError(
            f"{name}: shape mismatch: "
            f"new={tuple(actual_cpu.shape)}, "
            f"reference={tuple(expected_cpu.shape)}"
        )

    torch.testing.assert_close(
        actual_cpu,
        expected_cpu,
        atol=atol,
        rtol=rtol,
        msg=lambda message: f"{name} failed:\n{message}",
    )
    print(
        f"[PASS] {name:<24} "
        f"shape={tuple(actual_cpu.shape)} | "
        f"max_abs_diff={_max_abs_difference(actual_cpu, expected_cpu):.9g}"
    )


def _slice_raw(raw: Dict, frame_count: Optional[int]) -> Dict:
    if frame_count is None:
        return raw

    sliced = dict(raw)
    for key in ("acc", "ori", "pose", "tran"):
        value = raw.get(key)
        if isinstance(value, torch.Tensor) and value.ndim >= 1:
            sliced[key] = value[:frame_count]
    return sliced


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_file", type=str, default=None)
    parser.add_argument(
        "--raw_list_file",
        type=str,
        default=str(TEST_POSE_LIST),
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_past_frame", type=int, default=20)
    parser.add_argument("--num_future_frame", type=int, default=5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument(
        "--check_online",
        action="store_true",
        help="Also compare the frame-by-frame real-time path.",
    )
    parser.add_argument(
        "--online_frames",
        type=int,
        default=120,
        help="Maximum number of source frames used by the online comparison.",
    )
    args = parser.parse_args()

    sequence_path, raw = _load_raw_sequence(
        args.sequence_file,
        args.raw_list_file,
    )

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is unavailable.")

    print("=" * 72)
    print("D-BRAN PIPELINE EQUIVALENCE VALIDATION")
    print("=" * 72)
    print(f"Sequence: {sequence_path}")
    print(f"Frames:   {raw['acc'].shape[0]}")
    print(f"Device:   {device}")

    # The reference profiler uses a module-level DEVICE variable.
    reference.DEVICE = device

    print("\nLoading reference profiler models...")
    reference_translation_net = reference.load_transpose_net(
        str(TRANSPOSE_WEIGHTS_FILE),
        args.num_past_frame,
        args.num_future_frame,
    )
    reference_s1 = reference.load_all_distributed_s1(
        str(POSE_S1_CHECKPOINTS_DIR)
    )
    reference_s2 = reference.load_all_pose_s2_full(
        str(POSE_S2_CHECKPOINTS_DIR)
    )
    reference_s3 = reference.load_all_pose_s3_five_branch_regions(
        str(POSE_S3_CHECKPOINTS_DIR)
    )
    reference_fusion = reference.load_fusion(
        str(FUSION_CHECKPOINT)
    )

    print("\nLoading reusable DBranPipeline...")
    pipeline = DBranPipeline(
        device=device,
        num_past_frame=args.num_past_frame,
        num_future_frame=args.num_future_frame,
        use_cuda_streams=False,
        verbose=True,
    )

    print("\nRunning offline reference...")
    reference_imu, reference_leaf, reference_full, reference_reduced = (
        reference.distributed_pose_forward(
            raw,
            reference_s1,
            reference_s2,
            reference_s3,
            reference_fusion,
        )
    )
    reference_pose, reference_translation = reference.distributed_offline(
        reference_translation_net,
        raw,
        reference_s1,
        reference_s2,
        reference_s3,
        reference_fusion,
    )

    print("Running offline reusable pipeline...")
    new_output = pipeline.forward_offline(raw["acc"], raw["ori"])

    print("\nOffline comparison")
    _assert_close("normalized IMU", new_output.imu, reference_imu, args.atol, args.rtol)
    _assert_close("Pose-S1 leaf positions", new_output.leaf_positions, reference_leaf, args.atol, args.rtol)
    _assert_close("Pose-S2 full positions", new_output.full_positions, reference_full, args.atol, args.rtol)
    _assert_close("Pose-S3 reduced 6D", new_output.reduced_pose_6d, reference_reduced, args.atol, args.rtol)
    _assert_close("full local pose", new_output.pose, reference_pose, args.atol, args.rtol)
    _assert_close("root translation", new_output.translation, reference_translation, args.atol, args.rtol)

    counts = pipeline.parameter_counts()
    print("\nParameter counts")
    for key, value in counts.items():
        print(f"  {key:<12}: {value:,}")

    if args.check_online:
        online_frame_count = min(
            int(args.online_frames),
            int(raw["acc"].shape[0]),
        )
        if online_frame_count <= 0:
            raise ValueError("--online_frames must be positive.")

        online_raw = _slice_raw(raw, online_frame_count)
        profile_args = SimpleNamespace(
            num_past_frame=args.num_past_frame,
            num_future_frame=args.num_future_frame,
        )

        print(f"\nRunning online comparison with {online_frame_count} frames...")
        reference_online_pose, reference_online_translation, _, _, _ = (
            reference.distributed_online_profiled(
                reference_translation_net,
                online_raw,
                reference_s1,
                reference_s2,
                reference_s3,
                reference_fusion,
                None,
                None,
                None,
                profile_args,
                latency_warmup_frames=0,
            )
        )
        new_online = pipeline.forward_online_sequence(
            online_raw["acc"],
            online_raw["ori"],
            pad_future=True,
        )

        print("\nOnline comparison")
        _assert_close(
            "online full local pose",
            new_online.pose,
            reference_online_pose,
            args.atol,
            args.rtol,
        )
        _assert_close(
            "online root translation",
            new_online.translation,
            reference_online_translation,
            args.atol,
            args.rtol,
        )

    print("\n" + "=" * 72)
    print("VALIDATION PASSED")
    print("The reusable pipeline reproduces the current profiler output.")
    print("=" * 72)


if __name__ == "__main__":
    main()
