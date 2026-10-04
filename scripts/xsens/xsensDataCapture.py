"""Calibrate and capture Xsens MTw data in one session, with no D-BRAN
dependency at all.

This combines the two steps we used to run separately:

    1. Two-step static calibration (same procedure and math as
       calibrate_xsens.py: global reference alignment, then six-sensor
       T-pose).
    2. An 8-second countdown, then a straight capture-and-save loop --
       only UDP receive + calibration.apply_frame() per frame, no
       DBranPipeline, no CUDA, no neural network at all. That keeps
       per-frame cost to ~1ms, comfortably under the 16.67ms budget for a
       true, consistent 60Hz (unlike run_dbran_live.py, which spends
       ~15-16ms/frame on inference alone and only sustained ~50Hz on long
       sessions).

Capture length is bounded by --max_seconds (wall-clock, from real frame
timestamps) rather than --max_frames, so the take stops at the same
real-world instant as a fixed-duration Motive Take even if a few packets
are dropped along the way -- it will simply end up with a few frames fewer
than a perfect 60Hz stream, instead of running past when Motive already
stopped recording.

Usage:
    python .\\scripts\\xsens\\xsensDataCapture.py `
      --reference_role root `
      --countdown 8 `
      --max_seconds 180 `
      --save_pt .\\data\\dataset_raw\\dbran_optitrack\\captura_001.pt
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
import socket
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from dbran.xsens.calibration import XsensCalibration, rotation_capture_spread_degrees
from dbran.xsens.protocol import SENSOR_ROLES
from dbran.xsens.receiver import XsensTorchFrame, XsensUdpReceiver
from main_path import PROJECT_ROOT


# ------------------------------------------------------------------
# Shared helpers (mirrors calibrate_xsens.py)
# ------------------------------------------------------------------
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


def _countdown_while_draining(receiver: XsensUdpReceiver, seconds: int, message: str) -> None:
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
            print(f"\r  {len(frames):4d}/{target_frames} frames | elapsed {elapsed:5.1f} s", end="", flush=True)

    print()
    return frames


def _stack_frames(frames: Sequence[XsensTorchFrame]) -> Tuple[torch.Tensor, torch.Tensor]:
    acc = torch.stack([frame.acc for frame in frames])
    ori = torch.stack([frame.ori for frame in frames])
    return acc, ori


def _print_sensor_assignment(frame: XsensTorchFrame) -> None:
    print("Connected sensor assignment:")
    for index, (role, device_id) in enumerate(zip(SENSOR_ROLES, frame.device_ids)):
        print(f"  [{index}] {role:<10} -> {device_id}")
    print()


def _confirm_static_capture(name: str, rotations: torch.Tensor, max_allowed_degrees: float, allow_unstable: bool) -> None:
    spread = rotation_capture_spread_degrees(rotations)
    print(f"{name} orientation spread: mean={spread['mean_deg']:.3f} deg | max={spread['max_deg']:.3f} deg")
    if spread["max_deg"] > max_allowed_degrees:
        message = (
            f"{name} was not sufficiently static: maximum rotation spread "
            f"{spread['max_deg']:.3f} deg exceeds {max_allowed_degrees:.3f} deg."
        )
        if allow_unstable:
            print("[warning] " + message)
        else:
            raise RuntimeError(message + " Repeat the calibration.")


def _stats(values):
    if not values:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    t = torch.tensor(values, dtype=torch.float64)
    return {"mean": float(t.mean().item()), "p95": float(torch.quantile(t, 0.95).item()), "max": float(t.max().item())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9763)
    parser.add_argument("--timeout", type=float, default=1.0)

    # Calibration step
    parser.add_argument(
        "--calibration_output",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "xsens_calibration.json"),
    )
    parser.add_argument("--reference_role", choices=SENSOR_ROLES, default="root")
    parser.add_argument("--reference_seconds", type=float, default=5.0)
    parser.add_argument("--tpose_seconds", type=float, default=5.0)
    parser.add_argument("--max_static_rotation_deg", type=float, default=5.0)
    parser.add_argument("--allow_unstable", action="store_true")

    # Shared countdown (calibration steps and the pre-capture countdown)
    parser.add_argument("--countdown", type=int, default=8)

    # Capture step
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help=(
            "Frame-count cap after the countdown; 0 = no cap (default). "
            "Use --max_seconds instead so Xsens and Motive cover the same "
            "real-world time window even if a few frames are dropped."
        ),
    )
    parser.add_argument(
        "--max_seconds",
        type=float,
        default=180.0,
        help=(
            "Wall-clock seconds to capture, measured from real frame "
            "timestamps (not frame count); 0 = no time cap. This is the "
            "recommended way to bound a take, so it stops at the same "
            "real-world instant as a fixed-duration Motive Take regardless "
            "of a few dropped packets."
        ),
    )
    parser.add_argument("--print_every", type=int, default=300)
    parser.add_argument("--save_pt", type=str, required=True)

    args = parser.parse_args()

    if args.reference_seconds <= 0 or args.tpose_seconds <= 0:
        raise ValueError("Capture durations must be positive.")
    if args.countdown < 0:
        raise ValueError("--countdown cannot be negative.")
    if args.max_static_rotation_deg <= 0:
        raise ValueError("--max_static_rotation_deg must be positive.")

    reference_index = SENSOR_ROLES.index(args.reference_role)

    print("=" * 76)
    print("XSENS DATA CAPTURE (calibration + recording, no D-BRAN)")
    print("=" * 76)
    print(f"Listening on {args.host}:{args.port}")
    print("Start the native Xsens bridge and begin its measurement mode.\n")

    with XsensUdpReceiver(host=args.host, port=args.port, timeout_s=args.timeout) as receiver:
        first_frame = _wait_for_frame(receiver)
        _print_sensor_assignment(first_frame)
        update_rate_hz = first_frame.update_rate_hz
        device_ids = first_frame.device_ids

        # ---------------- STEP 1 OF 2 -- GLOBAL REFERENCE ----------------
        print("STEP 1 OF 2 -- GLOBAL REFERENCE")
        print(f"Use the MTw assigned to: {args.reference_role}")
        print("Align the SENSOR-FIXED coordinate axes as follows:")
        print("  +X = subject's left")
        print("  +Y = up")
        print("  +Z = forward")
        print("Use the Xsens axis markings or MT Manager coordinate display.")
        input(
            "Press ENTER, then align the reference MTw. "
            f"You will have {args.countdown} seconds before capture starts. "
        )
        _countdown_while_draining(receiver, args.countdown, "Align the reference MTw and hold it still.")
        reference_frames = _collect_frames(
            receiver, args.reference_seconds, update_rate_hz, "Reference capture", device_ids
        )
        _, reference_ori_all = _stack_frames(reference_frames)
        reference_ori = reference_ori_all[:, reference_index]
        _confirm_static_capture("Reference capture", reference_ori, args.max_static_rotation_deg, args.allow_unstable)

        # ---------------- STEP 2 OF 2 -- SIX-SENSOR T-POSE ----------------
        print("\nSTEP 2 OF 2 -- SIX-SENSOR T-POSE")
        print("Wear the sensors using the fixed D-BRAN assignments:")
        print("  left_arm, right_arm, left_leg, right_leg, head, root")
        print("Stand upright in a neutral T-pose: arms horizontal, palms down,")
        print("legs straight, feet forward, head and torso facing forward.")
        input(
            "Press ENTER, then move into the neutral T-pose. "
            f"You will have {args.countdown} seconds before capture starts. "
        )
        _countdown_while_draining(receiver, args.countdown, "Move into the T-pose and hold still.")
        tpose_frames = _collect_frames(receiver, args.tpose_seconds, update_rate_hz, "T-pose capture", device_ids)
        tpose_acc, tpose_ori = _stack_frames(tpose_frames)
        for index, role in enumerate(SENSOR_ROLES):
            _confirm_static_capture(f"T-pose {role}", tpose_ori[:, index], args.max_static_rotation_deg, args.allow_unstable)

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
                "body_frame": {"x": "left", "y": "up", "z": "forward"},
                "raw_orientation_convention": "sensor_local_to_xsens_global",
                "raw_acceleration_convention": "sensor_local_m_per_s2",
            },
        )
        calibration_output_path = calibration.save(args.calibration_output)

        print("\n" + "=" * 76)
        print("CALIBRATION SAVED")
        print("=" * 76)
        print(f"File: {calibration_output_path}")
        angle_errors = calibration.metadata["tpose_calibrated_orientation_mean_error_deg"]
        acc_norms = calibration.metadata["tpose_calibrated_acceleration_mean_norm_mps2"]
        for role in SENSOR_ROLES:
            print(f"  {role:<10} orientation error={angle_errors[role]:7.3f} deg | mean acc norm={acc_norms[role]:8.5f} m/s^2")

        # ---------------- PRE-CAPTURE COUNTDOWN ----------------
        print()
        input(
            "Calibration done. Press ENTER, then move into a neutral T-pose. "
            f"You will have {args.countdown} seconds before the take starts. "
        )
        _countdown_while_draining(receiver, args.countdown, "Move into a neutral T-pose. Take starts soon.")
        receiver.sequence_gaps = 0

        # ---------------- CAPTURE AND SAVE ----------------
        print("Capture started. Perform the marked start gesture now.\n")

        store: Dict[str, List[object]] = {
            "sequence": [],
            "host_unix_time_ns": [],
            "acc_calibrated": [],
            "ori_calibrated": [],
            "is_missing": [],
        }
        receive_ms: List[float] = []
        frames_received = 0
        frames_missing = 0
        first_time = None
        last_sequence = None
        last_host_time = None

        while True:
            if args.max_frames > 0 and frames_received >= args.max_frames:
                print(f"\nReached --max_frames={args.max_frames}, stopping capture.")
                break

            try:
                t0 = time.perf_counter()
                frame = receiver.receive(device="cpu")
            except socket.timeout:
                print("[waiting] Xsens UDP timeout...")
                continue

            calibrated = calibration.apply_frame(frame, output_device="cpu", strict_device_ids=True)
            receive_ms.append((time.perf_counter() - t0) * 1000.0)

            # If one or more synchronized slots were missing (the bridge
            # now assigns sequence numbers from a counter that advances on
            # every callback attempt, complete or not -- see
            # xsens_stream_bridge.cpp -- so a gap here is a real missing
            # slot), leave the hole as NaN instead of guessing a value.
            # Timestamps are still linearly interpolated, so a later
            # post-processing pass knows exactly when each hole occurred
            # and can interpolate acc/ori properly using the frames on
            # both sides, not just the one before.
            if last_sequence is not None and frame.sequence > last_sequence + 1:
                gap = frame.sequence - last_sequence - 1
                nan_acc = torch.full((6, 3), float("nan"))
                nan_ori = torch.full((6, 3, 3), float("nan"))
                for k in range(1, gap + 1):
                    filled_time = last_host_time + round(
                        (frame.host_unix_time_ns - last_host_time) * k / (gap + 1)
                    )
                    store["sequence"].append(last_sequence + k)
                    store["host_unix_time_ns"].append(int(filled_time))
                    store["acc_calibrated"].append(nan_acc)
                    store["ori_calibrated"].append(nan_ori)
                    store["is_missing"].append(True)
                    frames_missing += 1
                    frames_received += 1
                    if first_time is None:
                        first_time = filled_time

            store["sequence"].append(int(frame.sequence))
            store["host_unix_time_ns"].append(int(frame.host_unix_time_ns))
            store["acc_calibrated"].append(calibrated.acc)
            store["ori_calibrated"].append(calibrated.ori)
            store["is_missing"].append(False)

            frames_received += 1
            if first_time is None:
                first_time = frame.host_unix_time_ns

            last_sequence = frame.sequence
            last_host_time = frame.host_unix_time_ns

            elapsed_s = (frame.host_unix_time_ns - first_time) / 1e9

            if frames_received % args.print_every == 0:
                rate = frames_received / elapsed_s if elapsed_s > 0 else 0.0
                print(
                    f"Frame {frames_received:6d} (missing={frames_missing}) | elapsed={elapsed_s:7.2f}s | "
                    f"effective rate={rate:6.2f} Hz | sequence_gaps={receiver.sequence_gaps}"
                )

            if args.max_seconds > 0 and elapsed_s >= args.max_seconds:
                print(f"\nReached --max_seconds={args.max_seconds:.1f}s, stopping capture.")
                break

        sequence_gaps = receiver.sequence_gaps

    total_elapsed_s = (
        (store["host_unix_time_ns"][-1] - store["host_unix_time_ns"][0]) / 1e9
        if len(store["host_unix_time_ns"]) > 1
        else 0.0
    )
    measured_rate = (frames_received - 1) / total_elapsed_s if total_elapsed_s > 0 else 0.0
    r_stats = _stats(receive_ms)

    print("\n" + "=" * 76)
    print("CAPTURE SUMMARY")
    print("=" * 76)
    print(f"  Frames captured:        {frames_received} ({frames_missing} missing, saved as NaN placeholders)")
    print(f"  Wall-clock duration:    {total_elapsed_s:.2f} s")
    print(f"  Measured effective rate:{measured_rate:7.3f} Hz (target 60 Hz)")
    print(f"  Sequence gaps:          {sequence_gaps}")
    print(
        f"  Receive+calibrate time: mean={r_stats['mean']:.3f} ms | "
        f"p95={r_stats['p95']:.3f} ms | max={r_stats['max']:.3f} ms"
    )
    if frames_received > 0:
        print(f"  Missing fraction:       {100.0 * frames_missing / frames_received:.2f}%")
    if frames_missing > 0:
        print(
            "  Run scripts/data/fill_missing_xsens_frames.py before using "
            "this capture in the alignment pipeline."
        )

    destination = Path(args.save_pt).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    saved = {
        "format": "dbran_xsens_calibrated_capture_v1",
        "sensor_roles": SENSOR_ROLES,
        "calibration_file": str(calibration_output_path),
        "arguments": vars(args),
        "sequence": store["sequence"],
        "host_unix_time_ns": store["host_unix_time_ns"],
        "is_missing": store["is_missing"],
        "frames_missing": frames_missing,
        "acc_calibrated": torch.stack(store["acc_calibrated"]),
        "ori_calibrated": torch.stack(store["ori_calibrated"]),
    }
    torch.save(saved, destination)
    print(f"\nSaved calibrated capture: {destination}")

    if measured_rate < 55.0:
        print(
            "\n[warning] measured rate is still notably below 60 Hz -- "
            "something other than D-BRAN inference may be the bottleneck."
        )


if __name__ == "__main__":
    main()
