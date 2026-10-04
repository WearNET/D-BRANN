"""Receive and validate the native Xsens UDP stream without running D-BRAN."""

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
import math
import socket
import time

import torch

from dbran.xsens.protocol import SENSOR_ROLES
from dbran.xsens.receiver import XsensUdpReceiver


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9763)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--print_every", type=int, default=25)
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Stop after this many frames; 0 means run until Ctrl+C.",
    )
    args = parser.parse_args()

    if args.print_every <= 0:
        raise ValueError("--print_every must be positive.")
    if args.max_frames < 0:
        raise ValueError("--max_frames cannot be negative.")

    print("=" * 72)
    print("D-BRAN XSENS UDP RECEIVER TEST")
    print("=" * 72)
    print(f"Listening on {args.host}:{args.port}")
    print("Expected order: " + ", ".join(SENSOR_ROLES))
    print("Start the native bridge, then connect all six MTw sensors.")
    print("Press Ctrl+C to stop.\n")

    received = 0
    counter_mismatches = 0
    sample_time_mismatches = 0
    invalid_rotations = 0
    first_receive_time = None
    last_receive_time = None

    try:
        with XsensUdpReceiver(
            host=args.host,
            port=args.port,
            timeout_s=args.timeout,
        ) as receiver:
            while args.max_frames == 0 or received < args.max_frames:
                try:
                    frame = receiver.receive(device="cpu")
                except socket.timeout:
                    print(
                        f"[waiting] No UDP frame received for {args.timeout:.1f} s..."
                    )
                    continue

                now = time.perf_counter()
                received += 1
                if first_receive_time is None:
                    first_receive_time = now
                    print("Connected sensor assignment:")
                    for index, (role, device_id) in enumerate(
                        zip(SENSOR_ROLES, frame.device_ids)
                    ):
                        print(f"  [{index}] {role:<10} -> {device_id}")
                    print()
                last_receive_time = now

                if len(set(frame.packet_counters)) != 1:
                    counter_mismatches += 1
                if len(set(frame.sample_times_fine)) != 1:
                    sample_time_mismatches += 1

                identity = torch.eye(3).expand(6, -1, -1)
                orthogonality_error = (
                    frame.ori.transpose(1, 2).bmm(frame.ori) - identity
                ).abs().amax().item()
                determinants = torch.linalg.det(frame.ori)
                determinant_error = (determinants - 1.0).abs().amax().item()
                if (
                    not torch.isfinite(frame.acc).all()
                    or not torch.isfinite(frame.ori).all()
                    or orthogonality_error > 1e-3
                    or determinant_error > 1e-3
                ):
                    invalid_rotations += 1

                if received % args.print_every == 0:
                    elapsed = max(now - first_receive_time, 1e-9)
                    measured_rate = (received - 1) / elapsed if received > 1 else 0.0
                    host_age_ms = max(
                        0.0,
                        (time.time_ns() - frame.host_unix_time_ns) / 1e6,
                    )
                    print(
                        f"Frame {received} | sequence={frame.sequence} | "
                        f"rate={measured_rate:.2f} Hz | host age={host_age_ms:.2f} ms"
                    )
                    print(
                        f"  packet_counter={frame.packet_counters[0]} | "
                        f"sample_time_fine={frame.sample_times_fine[0]} | "
                        f"sequence_gaps={receiver.sequence_gaps}"
                    )
                    print(
                        f"  acc shape={tuple(frame.acc.shape)} | "
                        f"ori shape={tuple(frame.ori.shape)} | "
                        f"orth_err={orthogonality_error:.3e} | "
                        f"det_err={determinant_error:.3e}"
                    )
                    for index, role in enumerate(SENSOR_ROLES):
                        ax, ay, az = frame.acc[index].tolist()
                        print(
                            f"  [{index}] {role:<10} Acc: "
                            f"[{ax: .3f}, {ay: .3f}, {az: .3f}]"
                        )
                    print()

    except KeyboardInterrupt:
        print("\nStop requested by user.")

    elapsed = (
        (last_receive_time - first_receive_time)
        if first_receive_time is not None and last_receive_time is not None
        else 0.0
    )
    measured_rate = (received - 1) / elapsed if received > 1 and elapsed > 0 else 0.0

    print("\nReceiver summary")
    print(f"  Frames received:             {received}")
    print(f"  Measured receive rate:       {measured_rate:.3f} Hz")
    print(f"  Packet-counter mismatches:   {counter_mismatches}")
    print(f"  Sample-time mismatches:      {sample_time_mismatches}")
    print(f"  Invalid rotation frames:     {invalid_rotations}")


if __name__ == "__main__":
    main()
