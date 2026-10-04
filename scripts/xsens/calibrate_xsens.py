"""Interactively create an Xsens-to-D-BRAN static calibration."""

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
import socket
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import torch

from dbran.xsens.calibration import (
    XsensCalibration,
    rotation_capture_spread_degrees,
)
from dbran.xsens.protocol import SENSOR_ROLES
from dbran.xsens.receiver import XsensTorchFrame, XsensUdpReceiver
from main_path import PROJECT_ROOT


def _wait_for_frame(receiver: XsensUdpReceiver) -> XsensTorchFrame:
    while True:
        try:
            return receiver.receive(device="cpu")
        except socket.timeout:
            print("[waiting] No Xsens UDP frame received yet...")


def _receive_for_one_second(receiver: XsensUdpReceiver) -> int:
    end_time = time.perf_counter() + 1.0
    count = 0
    while time.perf_counter() < end_time:
        try:
            receiver.receive(device="cpu")
            count += 1
        except socket.timeout:
            pass
    return count


def _countdown_while_draining(
    receiver: XsensUdpReceiver,
    seconds: int,
    message: str,
) -> None:
    if seconds <= 0:
        return
    for remaining in range(seconds, 0, -1):
        print(f"\r{message} Starting in {remaining:2d} s...", end="", flush=True)
        _receive_for_one_second(receiver)
    print("\r" + " " * 88 + "\r", end="", flush=True)


def _collect_frames(
    receiver: XsensUdpReceiver,
    seconds: float,
    update_rate_hz: int,
    label: str,
    expected_device_ids: Sequence[str],
) -> List[XsensTorchFrame]:
    target_frames = max(1, round(float(seconds) * int(update_rate_hz)))
    frames: List[XsensTorchFrame] = []
    started = time.perf_counter()

    print(f"{label}: collecting {target_frames} frames ({seconds:.1f} s)...")
    while len(frames) < target_frames:
        try:
            frame = receiver.receive(device="cpu")
        except socket.timeout:
            print("\n[waiting] Xsens stream timed out during capture...")
            continue

        if tuple(frame.device_ids) != tuple(expected_device_ids):
            raise RuntimeError(
                "MTw assignment changed during calibration. "
                f"Expected {tuple(expected_device_ids)}, received {frame.device_ids}."
            )
        if frame.update_rate_hz != update_rate_hz:
            raise RuntimeError(
                "Xsens update rate changed during calibration: "
                f"expected {update_rate_hz}, received {frame.update_rate_hz}."
            )

        frames.append(frame)
        if len(frames) == 1 or len(frames) % update_rate_hz == 0:
            elapsed = time.perf_counter() - started
            print(
                f"\r  {len(frames):4d}/{target_frames} frames | "
                f"elapsed {elapsed:5.1f} s",
                end="",
                flush=True,
            )

    print()
    return frames


def _stack_frames(
    frames: Sequence[XsensTorchFrame],
) -> Tuple[torch.Tensor, torch.Tensor]:
    acc = torch.stack([frame.acc for frame in frames])
    ori = torch.stack([frame.ori for frame in frames])
    return acc, ori


def _print_sensor_assignment(frame: XsensTorchFrame) -> None:
    print("Connected sensor assignment:")
    for index, (role, device_id) in enumerate(zip(SENSOR_ROLES, frame.device_ids)):
        print(f"  [{index}] {role:<10} -> {device_id}")
    print()


def _confirm_static_capture(
    name: str,
    rotations: torch.Tensor,
    max_allowed_degrees: float,
    allow_unstable: bool,
) -> None:
    spread = rotation_capture_spread_degrees(rotations)
    print(
        f"{name} orientation spread: mean={spread['mean_deg']:.3f} deg | "
        f"max={spread['max_deg']:.3f} deg"
    )
    if spread["max_deg"] > max_allowed_degrees:
        message = (
            f"{name} was not sufficiently static: maximum rotation spread "
            f"{spread['max_deg']:.3f} deg exceeds "
            f"{max_allowed_degrees:.3f} deg."
        )
        if allow_unstable:
            print("[warning] " + message)
        else:
            raise RuntimeError(message + " Repeat the calibration.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9763)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "xsens_calibration.json"),
    )
    parser.add_argument(
        "--reference_role",
        choices=SENSOR_ROLES,
        default="root",
        help="MTw used to define +X left, +Y up, +Z forward.",
    )
    parser.add_argument("--reference_seconds", type=float, default=5.0)
    parser.add_argument("--tpose_seconds", type=float, default=5.0)
    parser.add_argument("--countdown", type=int, default=8)
    parser.add_argument(
        "--max_static_rotation_deg",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--allow_unstable",
        action="store_true",
        help="Save even when a static capture exceeds the rotation threshold.",
    )
    parser.add_argument(
        "--save_capture",
        type=str,
        default=None,
        help="Optional .pt file for the raw reference and T-pose captures.",
    )
    args = parser.parse_args()

    if args.reference_seconds <= 0 or args.tpose_seconds <= 0:
        raise ValueError("Capture durations must be positive.")
    if args.countdown < 0:
        raise ValueError("--countdown cannot be negative.")
    if args.max_static_rotation_deg <= 0:
        raise ValueError("--max_static_rotation_deg must be positive.")

    reference_index = SENSOR_ROLES.index(args.reference_role)

    print("=" * 76)
    print("D-BRAN XSENS STATIC CALIBRATION")
    print("=" * 76)
    print(f"Listening on {args.host}:{args.port}")
    print("Start the native Xsens bridge and begin its measurement mode.")
    print("The bridge must stream all six MTw sensors during both steps.\n")

    with XsensUdpReceiver(
        host=args.host,
        port=args.port,
        timeout_s=args.timeout,
    ) as receiver:
        first_frame = _wait_for_frame(receiver)
        _print_sensor_assignment(first_frame)
        update_rate_hz = first_frame.update_rate_hz
        device_ids = first_frame.device_ids

        print("STEP 1 OF 2 — GLOBAL REFERENCE")
        print(f"Use the MTw assigned to: {args.reference_role}")
        print("Align the SENSOR-FIXED coordinate axes as follows:")
        print("  +X = subject's left")
        print("  +Y = up")
        print("  +Z = forward")
        print("Use the Xsens axis markings or MT Manager coordinate display.")
        print("Do not infer the axes only from the logo or the charging orientation.")
        input(
            "Press ENTER, then align the reference MTw. "
            f"You will have {args.countdown} seconds before capture starts. "
        )

        _countdown_while_draining(
            receiver,
            args.countdown,
            "Align the reference MTw and hold it still.",
        )
        reference_frames = _collect_frames(
            receiver,
            args.reference_seconds,
            update_rate_hz,
            "Reference capture",
            device_ids,
        )
        _, reference_ori_all = _stack_frames(reference_frames)
        reference_ori = reference_ori_all[:, reference_index]
        _confirm_static_capture(
            "Reference capture",
            reference_ori,
            args.max_static_rotation_deg,
            args.allow_unstable,
        )

        print("\nSTEP 2 OF 2 — SIX-SENSOR T-POSE")
        print("Wear the sensors using the fixed D-BRAN assignments:")
        print("  left_arm, right_arm, left_leg, right_leg, head, root")
        print("Secure every MTw so it cannot move relative to its body segment.")
        print("Stand upright in a neutral T-pose: arms horizontal, palms down,")
        print("legs straight, feet forward, head and torso facing forward.")
        input(
            "Press ENTER, then move into the neutral T-pose. "
            f"You will have {args.countdown} seconds before capture starts. "
        )

        _countdown_while_draining(
            receiver,
            args.countdown,
            "Move into the T-pose and hold still.",
        )
        tpose_frames = _collect_frames(
            receiver,
            args.tpose_seconds,
            update_rate_hz,
            "T-pose capture",
            device_ids,
        )
        tpose_acc, tpose_ori = _stack_frames(tpose_frames)

        for index, role in enumerate(SENSOR_ROLES):
            _confirm_static_capture(
                f"T-pose {role}",
                tpose_ori[:, index],
                args.max_static_rotation_deg,
                args.allow_unstable,
            )

    calibration = XsensCalibration.from_captures(
        reference_orientations=reference_ori,
        tpose_orientations=tpose_ori,
        tpose_acc_local=tpose_acc,
        device_ids=device_ids,
        reference_role=args.reference_role,
        update_rate_hz=update_rate_hz,
        metadata={
            "calibration_method": "reference_alignment_plus_static_tpose",
            "reference_seconds": float(args.reference_seconds),
            "tpose_seconds": float(args.tpose_seconds),
            "body_frame": {
                "x": "left",
                "y": "up",
                "z": "forward",
            },
            "raw_orientation_convention": "sensor_local_to_xsens_global",
            "raw_acceleration_convention": "sensor_local_m_per_s2",
        },
    )
    output_path = calibration.save(args.output)

    if args.save_capture:
        capture_path = Path(args.save_capture).expanduser().resolve()
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "sensor_roles": SENSOR_ROLES,
                "device_ids": device_ids,
                "update_rate_hz": update_rate_hz,
                "reference_role": args.reference_role,
                "reference_ori": reference_ori,
                "tpose_acc_local": tpose_acc,
                "tpose_ori_sensor_to_global": tpose_ori,
            },
            capture_path,
        )
        print(f"Raw calibration capture: {capture_path}")

    print("\n" + "=" * 76)
    print("CALIBRATION SAVED")
    print("=" * 76)
    print(f"File: {output_path}")
    print(f"Reference role: {calibration.reference_role}")
    print(f"Update rate: {calibration.update_rate_hz} Hz")
    print("\nT-pose calibration diagnostics:")
    angle_errors = calibration.metadata[
        "tpose_calibrated_orientation_mean_error_deg"
    ]
    acc_norms = calibration.metadata[
        "tpose_calibrated_acceleration_mean_norm_mps2"
    ]
    for role in SENSOR_ROLES:
        print(
            f"  {role:<10} orientation error={angle_errors[role]:7.3f} deg | "
            f"mean acc norm={acc_norms[role]:8.5f} m/s^2"
        )

    print("\nKeep the sensors in the T-pose and run:")
    print(
        "  python scripts/xsens/test_xsens_calibration.py "
        "--calibration configs/xsens_calibration.json"
    )


if __name__ == "__main__":
    main()
