"""Validate a saved Xsens calibration while the subject holds a T-pose."""

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

import torch

from dbran.xsens.calibration import XsensCalibration, rotation_angle_degrees
from dbran.xsens.protocol import SENSOR_ROLES
from dbran.xsens.receiver import XsensUdpReceiver
from main_path import PROJECT_ROOT
from utils import normalize_and_concat


def _receive_for_one_second(receiver: XsensUdpReceiver) -> int:
    """Drain incoming UDP frames for approximately one second."""
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
    """Show a countdown while discarding frames captured during preparation."""
    if seconds <= 0:
        return

    for remaining in range(seconds, 0, -1):
        print(
            f"\r{message} Capture starts in {remaining:2d} s...",
            end="",
            flush=True,
        )
        _receive_for_one_second(receiver)

    print("\r" + " " * 88 + "\r", end="", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9763)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument(
        "--countdown",
        type=int,
        default=8,
        help="Preparation time after pressing ENTER and before capture starts.",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "xsens_calibration.json"),
    )
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--print_every", type=int, default=60)
    parser.add_argument("--max_mean_angle_deg", type=float, default=10.0)
    parser.add_argument("--max_mean_acc_norm", type=float, default=1.5)
    parser.add_argument(
        "--save_pt",
        type=str,
        default=None,
        help="Optional output containing raw and calibrated test frames.",
    )
    args = parser.parse_args()

    if args.max_frames <= 0:
        raise ValueError("--max_frames must be positive.")
    if args.countdown < 0:
        raise ValueError("--countdown cannot be negative.")
    if args.print_every <= 0:
        raise ValueError("--print_every must be positive.")

    calibration = XsensCalibration.load(args.calibration)

    print("=" * 76)
    print("D-BRAN XSENS CALIBRATION TEST")
    print("=" * 76)
    print(f"Calibration: {Path(args.calibration).expanduser().resolve()}")
    print(f"Listening on {args.host}:{args.port}")
    print("You will have time to move into the same neutral T-pose used during calibration.")
    print(f"Preparation countdown: {args.countdown} seconds.")
    print(f"Frames to collect after the countdown: {args.max_frames}.\n")

    raw_acc_frames = []
    raw_ori_frames = []
    calibrated_acc_frames = []
    calibrated_ori_frames = []
    normalized_frames = []
    received = 0
    first_time = None
    last_time = None

    with XsensUdpReceiver(
        host=args.host,
        port=args.port,
        timeout_s=args.timeout,
    ) as receiver:
        input(
            "Press ENTER, then move into the neutral T-pose and hold still. "
        )
        _countdown_while_draining(
            receiver,
            args.countdown,
            "Move into the T-pose and hold still.",
        )
        print(f"Collecting {args.max_frames} frames...\n")

        while received < args.max_frames:
            try:
                frame = receiver.receive(device="cpu")
            except socket.timeout:
                print("[waiting] No Xsens UDP frame received...")
                continue

            calibrated = calibration.apply_frame(
                frame,
                output_device="cpu",
                strict_device_ids=True,
            )
            data_nn = normalize_and_concat(
                calibrated.acc.unsqueeze(0),
                calibrated.ori.unsqueeze(0),
            )
            if data_nn.shape != (1, 72) or not torch.isfinite(data_nn).all():
                raise RuntimeError(
                    f"Invalid normalized D-BRAN input: {tuple(data_nn.shape)}"
                )

            now = time.perf_counter()
            if first_time is None:
                first_time = now
            last_time = now
            received += 1

            raw_acc_frames.append(frame.acc)
            raw_ori_frames.append(frame.ori)
            calibrated_acc_frames.append(calibrated.acc)
            calibrated_ori_frames.append(calibrated.ori)
            normalized_frames.append(data_nn.squeeze(0))

            if received % args.print_every == 0:
                angles = rotation_angle_degrees(calibrated.ori).float()
                acc_norms = calibrated.acc.norm(dim=1)
                print(
                    f"Frame {received:4d} | sequence={frame.sequence} | "
                    f"sequence_gaps={receiver.sequence_gaps}"
                )
                for index, role in enumerate(SENSOR_ROLES):
                    print(
                        f"  {role:<10} angle-to-I={angles[index]:7.3f} deg | "
                        f"acc norm={acc_norms[index]:7.3f} m/s^2"
                    )
                print()

    acc = torch.stack(calibrated_acc_frames)
    ori = torch.stack(calibrated_ori_frames)
    normalized = torch.stack(normalized_frames)
    angles = rotation_angle_degrees(ori).float()

    elapsed = (
        last_time - first_time
        if first_time is not None and last_time is not None
        else 0.0
    )
    rate = (received - 1) / elapsed if received > 1 and elapsed > 0 else 0.0

    print("\nCalibration test summary")
    print(f"  Frames received:       {received}")
    print(f"  Receive rate:          {rate:.3f} Hz")
    print(f"  Normalized input:      {tuple(normalized.shape)}")
    print(f"  All values finite:     {bool(torch.isfinite(normalized).all())}")
    print()

    passed = True
    for index, role in enumerate(SENSOR_ROLES):
        mean_angle = float(angles[:, index].mean().item())
        max_angle = float(angles[:, index].max().item())
        mean_acc = acc[:, index].mean(dim=0)
        mean_acc_norm = float(mean_acc.norm().item())
        acc_std_norm = float(acc[:, index].std(dim=0, unbiased=False).norm().item())

        angle_ok = mean_angle <= args.max_mean_angle_deg
        acc_ok = mean_acc_norm <= args.max_mean_acc_norm
        passed = passed and angle_ok and acc_ok
        status = "PASS" if angle_ok and acc_ok else "WARN"
        print(
            f"[{status}] {role:<10} mean angle={mean_angle:7.3f} deg | "
            f"max angle={max_angle:7.3f} deg | "
            f"mean acc norm={mean_acc_norm:7.3f} | "
            f"acc std norm={acc_std_norm:7.3f}"
        )

    if args.save_pt:
        output_path = Path(args.save_pt).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "sensor_roles": SENSOR_ROLES,
                "raw_acc_local": torch.stack(raw_acc_frames),
                "raw_ori_sensor_to_global": torch.stack(raw_ori_frames),
                "acc_calibrated": acc,
                "ori_calibrated": ori,
                "normalized_imu": normalized,
            },
            output_path,
        )
        print(f"\nSaved test capture: {output_path}")

    print()
    if passed:
        print("CALIBRATION TEST PASSED")
        print("The calibrated stream is ready for DBranPipeline.forward_online().")
    else:
        print("CALIBRATION TEST PRODUCED WARNINGS")
        print("Repeat the static calibration before running the model live.")


if __name__ == "__main__":
    main()
