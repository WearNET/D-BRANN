"""Calibrate Xsens for the current session and stream D-BRAN to Unity.

This script performs the complete validated Xsens calibration in memory for
Every live take, then starts D-BRAN inference and streams the original
TransPose Unity TCP protocol:

    72 local SMPL axis-angle values # 3 root-translation values $

Normal workflow:

    1. Start the native Xsens bridge and measurement mode.
    2. Run this script.
    3. Press Play in the original TransPose Unity Example scene.
    4. Complete the global-reference capture.
    5. Complete the six-sensor neutral T-pose capture.
    6. Streaming starts automatically and continues until Ctrl+C.

The session calibration is not saved by default. Use --save_calibration only
when a diagnostic copy is intentionally needed.

By default the five anatomical pose branches each run in their own OS
process (DBranParallelPipeline). The original single-process pipeline
(DBranPipeline, optionally with --cuda_streams) remains available as a
variant via --pipeline sequential.
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
import atexit
import socket
from collections import deque
import time
from pathlib import Path
from typing import Sequence

import articulate.math as M
import torch

from dbran.pipeline import DBranPipeline
from dbran.pipeline_parallel import DBranParallelPipeline
from dbran.xsens.calibration import (
    XsensCalibration,
    rotation_capture_spread_degrees,
)
from dbran.xsens.protocol import SENSOR_ROLES
from dbran.xsens.receiver import XsensTorchFrame, XsensUdpReceiver

JOINT_COUNT = 24
POSE_VALUE_COUNT = JOINT_COUNT * 3
TRANSLATION_VALUE_COUNT = 3
EXPECTED_UPDATE_RATE_HZ = 60


def resolve_device(value: str) -> torch.device:
    """Resolve a requested PyTorch device and validate CUDA availability."""
    if value.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is unavailable.")
    return device


def synchronize(device: torch.device) -> None:
    """Synchronize CUDA only when timing GPU work."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def stats(values: Sequence[float]) -> dict[str, float]:
    """Return basic timing statistics."""
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}

    tensor = torch.tensor(list(values), dtype=torch.float64)
    return {
        "mean": float(tensor.mean().item()),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p95": float(torch.quantile(tensor, 0.95).item()),
        "max": float(tensor.max().item()),
    }


def print_stats(label: str, values: Sequence[float]) -> None:
    """Print one formatted timing-statistics row."""
    result = stats(values)
    print(
        f"  {label:<30} mean={result['mean']:8.3f} ms | "
        f"p50={result['p50']:8.3f} | "
        f"p95={result['p95']:8.3f} | "
        f"max={result['max']:8.3f}"
    )


def receive_one(receiver: XsensUdpReceiver) -> XsensTorchFrame:
    """Wait until one synchronized Xsens frame is available."""
    while True:
        try:
            return receiver.receive(device="cpu")
        except socket.timeout:
            print("[waiting] No Xsens UDP frame. Is the native bridge streaming?")


def drain_one_second(receiver: XsensUdpReceiver) -> int:
    """Drain incoming frames for approximately one second."""
    deadline = time.perf_counter() + 1.0
    count = 0
    while time.perf_counter() < deadline:
        try:
            receiver.receive(device="cpu")
            count += 1
        except socket.timeout:
            pass
    return count


def countdown_while_draining(
    receiver: XsensUdpReceiver,
    seconds: int,
    message: str,
) -> None:
    """Display a countdown while discarding preparation frames."""
    if seconds <= 0:
        return

    for remaining in range(seconds, 0, -1):
        print(
            f"\r{message} Capture starts in {remaining:2d} s...",
            end="",
            flush=True,
        )
        drain_one_second(receiver)

    print("\r" + " " * 96 + "\r", end="", flush=True)


def collect_frames(
    receiver: XsensUdpReceiver,
    seconds: float,
    update_rate_hz: int,
    label: str,
    expected_device_ids: Sequence[str],
) -> list[XsensTorchFrame]:
    """Collect a fixed-duration synchronized Xsens capture."""
    target_frames = max(1, round(float(seconds) * int(update_rate_hz)))
    frames: list[XsensTorchFrame] = []
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
                f"Expected {tuple(expected_device_ids)}, "
                f"received {tuple(frame.device_ids)}."
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


def stack_frames(
    frames: Sequence[XsensTorchFrame],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack raw acceleration and orientation tensors from captured frames."""
    acc = torch.stack([frame.acc for frame in frames])
    ori = torch.stack([frame.ori for frame in frames])
    return acc, ori


def confirm_static_capture(
    name: str,
    rotations: torch.Tensor,
    max_allowed_degrees: float,
    allow_unstable: bool,
) -> None:
    """Reject a calibration capture containing excessive sensor motion."""
    spread = rotation_capture_spread_degrees(rotations)
    print(
        f"{name} orientation spread: mean={spread['mean_deg']:.3f} deg | "
        f"max={spread['max_deg']:.3f} deg"
    )

    if spread["max_deg"] <= max_allowed_degrees:
        return

    message = (
        f"{name} was not sufficiently static: maximum rotation spread "
        f"{spread['max_deg']:.3f} deg exceeds "
        f"{max_allowed_degrees:.3f} deg."
    )
    if allow_unstable:
        print("[warning] " + message)
    else:
        raise RuntimeError(message + " Restart the script and repeat calibration.")


def print_sensor_assignment(frame: XsensTorchFrame) -> None:
    """Print and validate the fixed six-sensor role assignment."""
    if len(frame.device_ids) != len(SENSOR_ROLES):
        raise RuntimeError(
            f"Expected {len(SENSOR_ROLES)} sensors, received "
            f"{len(frame.device_ids)}."
        )
    if frame.update_rate_hz != EXPECTED_UPDATE_RATE_HZ:
        raise RuntimeError(
            f"Expected a {EXPECTED_UPDATE_RATE_HZ} Hz Xsens stream, "
            f"received {frame.update_rate_hz} Hz."
        )

    print("Connected sensor assignment:")
    for index, (role, device_id) in enumerate(zip(SENSOR_ROLES, frame.device_ids)):
        print(f"  [{index}] {role:<10} -> {device_id}")
    print()


def create_session_calibration(
    receiver: XsensUdpReceiver,
    first_frame: XsensTorchFrame,
    *,
    reference_role: str,
    reference_seconds: float,
    tpose_seconds: float,
    countdown_seconds: int,
    max_static_rotation_deg: float,
    allow_unstable: bool,
) -> XsensCalibration:
    """Perform the validated two-stage calibration and keep it in memory."""
    update_rate_hz = first_frame.update_rate_hz
    device_ids = tuple(first_frame.device_ids)
    reference_index = SENSOR_ROLES.index(reference_role)

    print("=" * 80)
    print("SESSION CALIBRATION — STEP 1 OF 2: GLOBAL REFERENCE")
    print("=" * 80)
    print(f"Reference sensor role: {reference_role}")
    print("Align the SENSOR-FIXED coordinate axes as follows:")
    print("  +X = subject's left")
    print("  +Y = up")
    print("  +Z = forward")
    print("Use the Xsens axis markings or MT Manager coordinate display.")
    input(
        "Press ENTER, then align the reference MTw and hold it still. "
        f"You have {countdown_seconds} seconds. "
    )

    countdown_while_draining(
        receiver,
        countdown_seconds,
        "Align the reference MTw and hold it still.",
    )
    reference_frames = collect_frames(
        receiver,
        reference_seconds,
        update_rate_hz,
        "Reference capture",
        device_ids,
    )
    _, reference_ori_all = stack_frames(reference_frames)
    reference_ori = reference_ori_all[:, reference_index]
    confirm_static_capture(
        "Reference capture",
        reference_ori,
        max_static_rotation_deg,
        allow_unstable,
    )

    print("\n" + "=" * 80)
    print("SESSION CALIBRATION — STEP 2 OF 2: NEUTRAL T-POSE")
    print("=" * 80)
    print("Wear and secure all six sensors in their fixed D-BRAN assignments.")
    print("Stand upright with arms horizontal, palms down, legs straight,")
    print("feet forward, and head and torso facing forward.")
    print("This is the final calibration immediately before live streaming.")
    input(
        "Press ENTER, then adopt the neutral T-pose and hold still. "
        f"You have {countdown_seconds} seconds. "
    )

    countdown_while_draining(
        receiver,
        countdown_seconds,
        "Move into the neutral T-pose and hold still.",
    )
    tpose_frames = collect_frames(
        receiver,
        tpose_seconds,
        update_rate_hz,
        "T-pose capture",
        device_ids,
    )
    tpose_acc, tpose_ori = stack_frames(tpose_frames)

    for index, role in enumerate(SENSOR_ROLES):
        confirm_static_capture(
            f"T-pose {role}",
            tpose_ori[:, index],
            max_static_rotation_deg,
            allow_unstable,
        )

    calibration = XsensCalibration.from_captures(
        reference_orientations=reference_ori,
        tpose_orientations=tpose_ori,
        tpose_acc_local=tpose_acc,
        device_ids=device_ids,
        reference_role=reference_role,
        update_rate_hz=update_rate_hz,
        metadata={
            "calibration_method": (
                "in_memory_reference_alignment_plus_static_tpose"
            ),
            "reference_seconds": float(reference_seconds),
            "tpose_seconds": float(tpose_seconds),
            "body_frame": {
                "x": "left",
                "y": "up",
                "z": "forward",
            },
            "raw_orientation_convention": "sensor_local_to_xsens_global",
            "raw_acceleration_convention": "sensor_local_m_per_s2",
            "session_only": True,
        },
    )

    print("\n" + "=" * 80)
    print("SESSION CALIBRATION READY")
    print("=" * 80)
    print("The calibration is active in memory and has not been saved.")
    print("\nT-pose diagnostics:")
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
    print()

    return calibration


def warm_up(pipeline: DBranPipeline, frames: int) -> None:
    """Warm up the online pipeline without retaining temporal state."""
    if frames <= 0:
        pipeline.reset_online_state()
        return

    print(f"Warming up D-BRAN with {frames} synthetic frames...")
    acc = torch.zeros(6, 3)
    ori = torch.eye(3).repeat(6, 1, 1)

    with torch.inference_mode():
        for _ in range(frames):
            pipeline.forward_online(acc, ori)

    synchronize(pipeline.device)
    pipeline.reset_online_state()
    print("D-BRAN warm-up complete.\n")


def validate_first_output(
    pose: torch.Tensor,
    translation: torch.Tensor,
) -> None:
    """Validate the first full-window D-BRAN output."""
    if tuple(pose.shape) != (JOINT_COUNT, 3, 3):
        raise RuntimeError(
            f"Expected pose ({JOINT_COUNT}, 3, 3), got {tuple(pose.shape)}."
        )

    if translation.numel() != TRANSLATION_VALUE_COUNT:
        raise RuntimeError(
            f"Expected 3 translation values, got shape {tuple(translation.shape)}."
        )

    if not torch.isfinite(pose).all() or not torch.isfinite(translation).all():
        raise RuntimeError("The first full-window output contains non-finite values.")

    pose_cpu = pose.detach().to(device="cpu", dtype=torch.float32)
    identity = torch.eye(3).expand_as(pose_cpu)
    orthogonality_error = float(
        (pose_cpu.transpose(-1, -2) @ pose_cpu - identity).abs().amax().item()
    )
    determinant_error = float(
        (torch.linalg.det(pose_cpu) - 1.0).abs().amax().item()
    )

    print(
        f"[validated] First output: orth_err={orthogonality_error:.3e}, "
        f"det_err={determinant_error:.3e}\n"
    )


def build_transpose_message(
    local_pose: torch.Tensor,
    translation: torch.Tensor,
) -> bytes:
    """Encode one frame using the original TransPose Unity text protocol."""
    if tuple(local_pose.shape) != (JOINT_COUNT, 3, 3):
        raise ValueError(
            f"Expected local pose ({JOINT_COUNT}, 3, 3), "
            f"got {tuple(local_pose.shape)}."
        )

    axis_angle = M.rotation_matrix_to_axis_angle(local_pose).reshape(-1)
    if axis_angle.numel() != POSE_VALUE_COUNT:
        raise RuntimeError(
            f"Expected {POSE_VALUE_COUNT} axis-angle values, "
            f"got {axis_angle.numel()}."
        )

    pose_values = (
        axis_angle.detach().to(device="cpu", dtype=torch.float32).numpy()
    )
    translation_values = (
        translation.reshape(-1)
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .numpy()
    )

    if translation_values.size != TRANSLATION_VALUE_COUNT:
        raise RuntimeError(
            f"Expected {TRANSLATION_VALUE_COUNT} translation values, "
            f"got {translation_values.size}."
        )

    pose_text = ",".join(f"{float(value):g}" for value in pose_values)
    translation_text = ",".join(
        f"{float(value):g}" for value in translation_values
    )
    return f"{pose_text}#{translation_text}$".encode("utf-8")


def create_unity_server(host: str, port: int) -> socket.socket:
    """Create the TCP server expected by the TransPose Unity example."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    return server


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create an in-memory Xsens session calibration, run D-BRAN, "
            "and stream to the original TransPose Unity example."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9763)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--countdown", type=int, default=8)
    parser.add_argument(
        "--reference_role",
        choices=SENSOR_ROLES,
        default="root",
    )
    parser.add_argument("--reference_seconds", type=float, default=5.0)
    parser.add_argument("--tpose_seconds", type=float, default=5.0)
    parser.add_argument("--max_static_rotation_deg", type=float, default=5.0)
    parser.add_argument(
        "--allow_unstable",
        action="store_true",
        help="Continue even when a calibration capture is not sufficiently static.",
    )
    parser.add_argument(
        "--save_calibration",
        default=None,
        help=(
            "Optional diagnostic JSON output. By default, the current-session "
            "calibration remains only in memory."
        ),
    )
    parser.add_argument("--print_every", type=int, default=120)
    parser.add_argument("--stats_window", type=int, default=3600)
    parser.add_argument("--warmup_frames", type=int, default=30)
    parser.add_argument("--num_past_frame", type=int, default=20)
    parser.add_argument("--num_future_frame", type=int, default=5)
    parser.add_argument(
        "--pipeline",
        choices=["parallel", "sequential"],
        default="parallel",
        help=(
            "parallel (default): each of the 5 pose branches runs in its own "
            "OS process (DBranParallelPipeline). sequential: the original "
            "single-process pipeline (DBranPipeline), kept as a variant."
        ),
    )
    parser.add_argument(
        "--worker_device",
        default=None,
        help="Device for the 5 branch processes with --pipeline parallel. Defaults to --device.",
    )
    parser.add_argument(
        "--cuda_streams",
        dest="cuda_streams",
        action="store_true",
        default=True,
        help="With --pipeline sequential: run anatomical branches on separate CUDA streams.",
    )
    parser.add_argument(
        "--no_cuda_streams",
        dest="cuda_streams",
        action="store_false",
    )
    parser.add_argument("--unity_host", default="127.0.0.1")
    parser.add_argument("--unity_port", type=int, default=8888)
    args = parser.parse_args()

    if args.countdown < 0:
        raise ValueError("countdown cannot be negative.")
    if args.reference_seconds <= 0 or args.tpose_seconds <= 0:
        raise ValueError("Calibration capture durations must be positive.")
    if args.max_static_rotation_deg <= 0:
        raise ValueError("max_static_rotation_deg must be positive.")
    if args.print_every <= 0 or args.stats_window <= 0 or args.warmup_frames < 0:
        raise ValueError(
            "print_every and stats_window must be positive; "
            "warmup_frames must be non-negative."
        )
    if not 1 <= args.port <= 65535 or not 1 <= args.unity_port <= 65535:
        raise ValueError("TCP and UDP ports must be between 1 and 65535.")

    device = resolve_device(args.device)

    print("=" * 80)
    print("D-BRAN SESSION CALIBRATION -> TRANSPOSE UNITY")
    print("=" * 80)
    print("Calibration:       New in-memory calibration for this run")
    print(f"Device:            {device}")
    if device.type == "cuda":
        print(f"GPU:               {torch.cuda.get_device_name(device)}")
    print(f"Xsens bridge UDP:  {args.host}:{args.port}")
    print(f"Unity TCP server:  {args.unity_host}:{args.unity_port}")
    print("Stream duration:   Continuous until Ctrl+C")
    print()

    print(f"Pipeline:          {args.pipeline}")
    if args.pipeline == "parallel":
        pipeline = DBranParallelPipeline(
            device=device,
            num_past_frame=args.num_past_frame,
            num_future_frame=args.num_future_frame,
            worker_device=args.worker_device,
            verbose=True,
        )
        atexit.register(pipeline.shutdown)
    else:
        pipeline = DBranPipeline(
            device=device,
            num_past_frame=args.num_past_frame,
            num_future_frame=args.num_future_frame,
            use_cuda_streams=args.cuda_streams,
            verbose=True,
        )
    print(f"[D-BRAN] Total parameters: {pipeline.parameter_counts()['total']:,}\n")
    warm_up(pipeline, args.warmup_frames)

    processing_times: deque[float] = deque(maxlen=args.stats_window)
    host_age_times: deque[float] = deque(maxlen=args.stats_window)
    effective_age_times: deque[float] = deque(maxlen=args.stats_window)

    frames_received = 0
    messages_sent = 0
    send_failures = 0
    over_budget = 0
    receive_timeouts = 0
    sequence_gaps = 0
    first_checked = False
    first_receive_time: float | None = None
    last_receive_time: float | None = None
    frame_period_ms = 1000.0 / EXPECTED_UPDATE_RATE_HZ
    algorithmic_delay_ms = args.num_future_frame * frame_period_ms

    unity_server: socket.socket | None = None
    unity_connection: socket.socket | None = None

    try:
        with XsensUdpReceiver(args.host, args.port, args.timeout) as receiver:
            print("Waiting for the synchronized Xsens bridge...")
            initial_frame = receive_one(receiver)
            print_sensor_assignment(initial_frame)

            # Connect Unity before calibration. The original client blocks while
            # waiting for the first message, so no pose is transmitted until the
            # final T-pose calibration has completed successfully.
            unity_server = create_unity_server(args.unity_host, args.unity_port)
            print("Unity server started. Press Play in the TransPose Example scene.")
            print("Unity will remain connected while the session is calibrated.")
            print("Waiting for Unity to connect...")
            unity_connection, unity_address = unity_server.accept()
            unity_connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"Unity connected from {unity_address[0]}:{unity_address[1]}.\n")

            calibration = create_session_calibration(
                receiver,
                initial_frame,
                reference_role=args.reference_role,
                reference_seconds=args.reference_seconds,
                tpose_seconds=args.tpose_seconds,
                countdown_seconds=args.countdown,
                max_static_rotation_deg=args.max_static_rotation_deg,
                allow_unstable=args.allow_unstable,
            )

            if args.save_calibration:
                saved_path = calibration.save(
                    Path(args.save_calibration).expanduser().resolve()
                )
                print(f"Diagnostic calibration saved to: {saved_path}\n")

            receiver.sequence_gaps = 0
            pipeline.reset_online_state()
            print("Calibration complete. Live transmission starts now.")
            print("Hold the T-pose until the 26-frame temporal window is ready.\n")

            with torch.inference_mode():
                while True:
                    try:
                        frame = receiver.receive(device="cpu")
                    except socket.timeout:
                        receive_timeouts += 1
                        continue

                    receive_time = time.perf_counter()
                    if first_receive_time is None:
                        first_receive_time = receive_time
                    last_receive_time = receive_time
                    processing_start = time.perf_counter()

                    calibrated = calibration.apply_frame(
                        frame,
                        output_device=device,
                        strict_device_ids=True,
                    )
                    output = pipeline.forward_online(
                        calibrated.acc,
                        calibrated.ori,
                    )
                    frames_received += 1
                    sequence_gaps = receiver.sequence_gaps

                    if not output.has_full_window:
                        continue

                    if not first_checked:
                        validate_first_output(output.pose, output.translation)
                        first_checked = True
                        print(
                            f"[ready] Full temporal window at input frame "
                            f"{output.input_frame_index}. Sending to Unity.\n"
                        )

                    message = build_transpose_message(
                        output.pose,
                        output.translation,
                    )

                    try:
                        unity_connection.sendall(message)
                        messages_sent += 1
                    except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                        send_failures += 1
                        raise RuntimeError(
                            "The TransPose Unity client disconnected. Stop Play "
                            "mode and restart the complete session."
                        ) from exc

                    synchronize(device)
                    processing_ms = (
                        time.perf_counter() - processing_start
                    ) * 1000.0
                    host_age_ms = (
                        time.time_ns() - frame.host_unix_time_ns
                    ) / 1_000_000.0
                    effective_age_ms = host_age_ms + algorithmic_delay_ms

                    processing_times.append(processing_ms)
                    host_age_times.append(host_age_ms)
                    effective_age_times.append(effective_age_ms)

                    if processing_ms > frame_period_ms:
                        over_budget += 1

                    if messages_sent % args.print_every == 0:
                        elapsed = receive_time - first_receive_time
                        measured_rate = (
                            (frames_received - 1) / elapsed if elapsed > 0 else 0.0
                        )
                        root = output.translation.detach().reshape(-1).to("cpu")
                        print(
                            f"Sent {messages_sent:5d} | "
                            f"rate={measured_rate:6.2f} Hz | "
                            f"processing={processing_ms:7.3f} ms | "
                            f"effective age={effective_age_ms:7.3f} ms"
                        )
                        print(
                            f"  Root=[{float(root[0]): .4f}, "
                            f"{float(root[1]): .4f}, "
                            f"{float(root[2]): .4f}] | "
                            f"gaps={receiver.sequence_gaps}\n"
                        )

    except KeyboardInterrupt:
        print("\nStreaming stopped by user.")
    finally:
        if unity_connection is not None:
            try:
                unity_connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            unity_connection.close()

        if unity_server is not None:
            unity_server.close()

    elapsed = (
        last_receive_time - first_receive_time
        if first_receive_time is not None and last_receive_time is not None
        else 0.0
    )
    measured_rate = (
        (frames_received - 1) / elapsed
        if frames_received > 1 and elapsed > 0
        else 0.0
    )

    print("\n" + "=" * 80)
    print("TRANSPOSE UNITY SESSION SUMMARY")
    print("=" * 80)
    print(f"  Xsens frames received:       {frames_received}")
    print(f"  Unity messages sent:         {messages_sent}")
    print(f"  Unity send failures:         {send_failures}")
    print(f"  Xsens sequence gaps:         {sequence_gaps}")
    print(f"  Xsens receive timeouts:      {receive_timeouts}")
    print(f"  Measured receive rate:       {measured_rate:.3f} Hz")
    print(
        f"  Messages over {frame_period_ms:.3f} ms budget: "
        f"{over_budget}/{messages_sent}"
    )
    print()
    print_stats("Inference + encode + TCP send", processing_times)
    print_stats("Newest-frame host age", host_age_times)
    print_stats("Effective output age", effective_age_times)
    print("\nD-BRAN TRANSPOSE UNITY SESSION COMPLETED")


if __name__ == "__main__":
    main()
