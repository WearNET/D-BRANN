"""Run calibrated Xsens MTw data through D-BRAN in real time.

This is the first end-to-end live test before adding Unity:

    six Xsens MTw sensors
        -> native synchronized UDP bridge
        -> Xsens static calibration
        -> DBranParallelPipeline.forward_online() (default) or
           DBranPipeline.forward_online() (--pipeline sequential)
        -> pose [24, 3, 3] and root translation [3]

By default the five anatomical pose branches each run in their own OS
process (DBranParallelPipeline). The original single-process pipeline
(DBranPipeline, optionally with --cuda_streams) remains available as a
variant via --pipeline sequential.

The script measures stream integrity, calibration time, inference time,
end-to-end frame age, output validity, and the 60 Hz processing budget.
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
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from dbran.pipeline import DBranPipeline
from dbran.pipeline_parallel import DBranParallelPipeline
from dbran.xsens.calibration import XsensCalibration
from dbran.xsens.protocol import SENSOR_ROLES
from dbran.xsens.receiver import XsensTorchFrame, XsensUdpReceiver
from main_path import PROJECT_ROOT


def _resolve_device(value: str) -> torch.device:
    if value.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False."
        )
    return device


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _stats(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }

    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean().item()),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p95": float(torch.quantile(tensor, 0.95).item()),
        "max": float(tensor.max().item()),
    }


def _print_stats(label: str, values: Sequence[float], unit: str = "ms") -> None:
    values_stats = _stats(values)
    print(
        f"  {label:<25} "
        f"mean={values_stats['mean']:8.3f} {unit} | "
        f"p50={values_stats['p50']:8.3f} | "
        f"p95={values_stats['p95']:8.3f} | "
        f"max={values_stats['max']:8.3f}"
    )


def _rotation_errors(rotation: torch.Tensor) -> Tuple[float, float]:
    """Return maximum orthogonality and determinant errors."""
    rotation = rotation.detach().cpu().float()
    identity = torch.eye(3, dtype=rotation.dtype).expand_as(rotation)
    orthogonality_error = float(
        (
            rotation.transpose(-1, -2) @ rotation - identity
        ).abs().amax().item()
    )
    determinant_error = float(
        (torch.linalg.det(rotation) - 1.0).abs().amax().item()
    )
    return orthogonality_error, determinant_error


def _receive_one_frame(receiver: XsensUdpReceiver) -> XsensTorchFrame:
    while True:
        try:
            return receiver.receive(device="cpu")
        except socket.timeout:
            print("[waiting] No Xsens UDP frame received. Is the bridge streaming?")


def _drain_for_one_second(receiver: XsensUdpReceiver) -> int:
    """Discard incoming frames for approximately one second."""
    deadline = time.perf_counter() + 1.0
    drained = 0
    while time.perf_counter() < deadline:
        try:
            receiver.receive(device="cpu")
            drained += 1
        except socket.timeout:
            pass
    return drained


def _countdown_while_draining(
    receiver: XsensUdpReceiver,
    seconds: int,
) -> None:
    """Give the subject preparation time without leaving stale UDP frames."""
    if seconds <= 0:
        return

    for remaining in range(seconds, 0, -1):
        print(
            f"\rMove into a neutral T-pose. Live inference starts in "
            f"{remaining:2d} s...",
            end="",
            flush=True,
        )
        _drain_for_one_second(receiver)

    print("\r" + " " * 88 + "\r", end="", flush=True)


def _warm_up_pipeline(
    pipeline: DBranPipeline,
    warmup_frames: int,
) -> None:
    if warmup_frames <= 0:
        pipeline.reset_online_state()
        return

    print(f"Warming up D-BRAN with {warmup_frames} synthetic frames...")
    acc = torch.zeros(6, 3, dtype=torch.float32)
    ori = torch.eye(3, dtype=torch.float32).repeat(6, 1, 1)

    for _ in range(warmup_frames):
        pipeline.forward_online(acc, ori)

    _synchronize(pipeline.device)
    pipeline.reset_online_state()
    print("D-BRAN warm-up complete.\n")


def _validate_initial_frame(
    frame: XsensTorchFrame,
    calibration: XsensCalibration,
) -> None:
    calibration.validate_device_ids(frame.device_ids)

    if frame.update_rate_hz != calibration.update_rate_hz:
        raise RuntimeError(
            "The Xsens stream rate does not match the calibration: "
            f"stream={frame.update_rate_hz} Hz, "
            f"calibration={calibration.update_rate_hz} Hz."
        )
    if frame.update_rate_hz != 60:
        raise RuntimeError(
            "The current D-BRAN live pipeline assumes 60 Hz, but the bridge "
            f"is streaming at {frame.update_rate_hz} Hz."
        )

    print("Connected sensor assignment:")
    for index, role in enumerate(SENSOR_ROLES):
        print(f"  [{index}] {role:<10} -> {frame.device_ids[index]}")
    print()


def _new_capture_store() -> Dict[str, List[object]]:
    return {
        "sequence": [],
        "host_unix_time_ns": [],
        "packet_counters": [],
        "raw_acc_local": [],
        "raw_ori_sensor_to_global": [],
        "acc_calibrated": [],
        "ori_calibrated": [],
        "pose": [],
        "translation": [],
        "input_frame_index": [],
        "output_frame_index": [],
        "has_future_context": [],
        "has_full_window": [],
        "calibration_ms": [],
        "inference_ms": [],
        "processing_ms": [],
        "end_to_end_age_ms": [],
        "effective_output_age_ms": [],
    }


def _append_capture(
    store: Dict[str, List[object]],
    frame: XsensTorchFrame,
    calibrated_acc: torch.Tensor,
    calibrated_ori: torch.Tensor,
    output: object,
    calibration_ms: float,
    inference_ms: float,
    processing_ms: float,
    end_to_end_age_ms: float,
    effective_output_age_ms: float,
) -> None:
    store["sequence"].append(int(frame.sequence))
    store["host_unix_time_ns"].append(int(frame.host_unix_time_ns))
    store["packet_counters"].append(tuple(int(v) for v in frame.packet_counters))
    store["raw_acc_local"].append(frame.acc.detach().cpu())
    store["raw_ori_sensor_to_global"].append(frame.ori.detach().cpu())
    store["acc_calibrated"].append(calibrated_acc.detach().cpu())
    store["ori_calibrated"].append(calibrated_ori.detach().cpu())
    store["pose"].append(output.pose.detach().cpu())
    store["translation"].append(output.translation.detach().cpu())
    store["input_frame_index"].append(int(output.input_frame_index))
    store["output_frame_index"].append(int(output.output_frame_index))
    store["has_future_context"].append(bool(output.has_future_context))
    store["has_full_window"].append(bool(output.has_full_window))
    store["calibration_ms"].append(float(calibration_ms))
    store["inference_ms"].append(float(inference_ms))
    store["processing_ms"].append(float(processing_ms))
    store["end_to_end_age_ms"].append(float(end_to_end_age_ms))
    store["effective_output_age_ms"].append(float(effective_output_age_ms))


def _save_capture(
    path: str,
    store: Dict[str, List[object]],
    calibration_path: Path,
    args: argparse.Namespace,
    parameter_counts: Dict[str, int],
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    tensor_keys = (
        "raw_acc_local",
        "raw_ori_sensor_to_global",
        "acc_calibrated",
        "ori_calibrated",
        "pose",
        "translation",
    )
    saved: Dict[str, object] = {
        "format": "dbran_xsens_live_capture_v1",
        "sensor_roles": SENSOR_ROLES,
        "calibration_file": str(calibration_path),
        "arguments": vars(args),
        "parameter_counts": parameter_counts,
    }

    for key, values in store.items():
        if key in tensor_keys:
            saved[key] = torch.stack(values) if values else torch.empty(0)
        else:
            saved[key] = values

    torch.save(saved, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9763)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument(
        "--calibration",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "xsens_calibration.json"),
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--countdown", type=int, default=8)
    parser.add_argument(
        "--max_frames",
        type=int,
        default=600,
        help="Frames to process after the countdown; 0 runs until Ctrl+C.",
    )
    parser.add_argument("--print_every", type=int, default=60)
    parser.add_argument("--warmup_frames", type=int, default=5)
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
        type=str,
        default=None,
        help="Device for the 5 branch processes with --pipeline parallel. Defaults to --device.",
    )
    parser.add_argument(
        "--cuda_streams",
        action="store_true",
        help="With --pipeline sequential: run anatomical branches on separate CUDA streams.",
    )
    parser.add_argument(
        "--save_pt",
        type=str,
        default=None,
        help="Optional path for raw/calibrated inputs, model outputs, and timing.",
    )
    parser.add_argument(
        "--rotation_tolerance",
        type=float,
        default=1e-3,
        help="Maximum accepted pose orthogonality and determinant errors.",
    )
    args = parser.parse_args()

    if args.countdown < 0:
        raise ValueError("--countdown cannot be negative.")
    if args.max_frames < 0:
        raise ValueError("--max_frames cannot be negative.")
    if args.print_every <= 0:
        raise ValueError("--print_every must be positive.")
    if args.warmup_frames < 0:
        raise ValueError("--warmup_frames cannot be negative.")

    device = _resolve_device(args.device)
    calibration_path = Path(args.calibration).expanduser().resolve()
    calibration = XsensCalibration.load(calibration_path)

    if device.type == "cuda":
        # The online window shape (num_past_frame + 1 + num_future_frame,
        # fixed per run) never changes between frames, so let cuDNN benchmark
        # and cache the fastest algorithm for it instead of picking heuristically
        # on every call.
        torch.backends.cudnn.benchmark = True

    print("=" * 80)
    print("D-BRAN LIVE XSENS INFERENCE")
    print("=" * 80)
    print(f"Calibration: {calibration_path}")
    print(f"Device:      {device}")
    if device.type == "cuda":
        print(f"GPU:         {torch.cuda.get_device_name(device)}")
        print(f"cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    print(f"UDP input:   {args.host}:{args.port}")
    print(
        f"Window:      {args.num_past_frame} past + current + "
        f"{args.num_future_frame} future"
    )
    print()

    print(f"Pipeline:    {args.pipeline}")
    print("Loading D-BRAN checkpoints...")
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
    parameter_counts = pipeline.parameter_counts()
    print(f"[D-BRAN] Total parameters: {parameter_counts['total']:,}\n")

    _warm_up_pipeline(pipeline, args.warmup_frames)

    stream_period_ms = 1000.0 / calibration.update_rate_hz
    store = _new_capture_store()

    frames_received = 0
    valid_outputs = 0
    packet_counter_mismatches = 0
    nonfinite_outputs = 0
    invalid_rotation_outputs = 0
    receive_timeouts = 0
    over_budget_frames = 0
    window_ready_announced = False

    calibration_times: List[float] = []
    inference_times: List[float] = []
    processing_times: List[float] = []
    age_times: List[float] = []
    effective_age_times: List[float] = []

    first_capture_time: float | None = None
    last_capture_time: float | None = None
    start_sequence_gaps = 0

    try:
        with XsensUdpReceiver(
            host=args.host,
            port=args.port,
            timeout_s=args.timeout,
        ) as receiver:
            print("Waiting for the synchronized Xsens bridge...")
            initial_frame = _receive_one_frame(receiver)
            _validate_initial_frame(initial_frame, calibration)

            input(
                "Press ENTER, then move into a neutral T-pose. "
                f"You will have {args.countdown} seconds before inference starts. "
            )
            _countdown_while_draining(receiver, args.countdown)

            # Count only gaps that occur during the actual live inference.
            receiver.sequence_gaps = 0
            start_sequence_gaps = receiver.sequence_gaps
            pipeline.reset_online_state()

            print(
                "Live inference started. Hold the T-pose until the temporal "
                "window is ready, then make slow controlled movements.\n"
            )

            while args.max_frames == 0 or frames_received < args.max_frames:
                try:
                    frame = receiver.receive(device="cpu")
                except socket.timeout:
                    receive_timeouts += 1
                    print("[waiting] Xsens UDP timeout...")
                    continue

                receive_time = time.perf_counter()
                if first_capture_time is None:
                    first_capture_time = receive_time
                last_capture_time = receive_time

                if len(set(frame.packet_counters)) != 1:
                    packet_counter_mismatches += 1

                calibration_start = time.perf_counter()
                calibrated = calibration.apply_frame(
                    frame,
                    output_device=device,
                    strict_device_ids=True,
                )
                _synchronize(device)
                calibration_end = time.perf_counter()

                inference_start = calibration_end
                output = pipeline.forward_online(
                    acc_frame=calibrated.acc,
                    ori_frame=calibrated.ori,
                )
                _synchronize(device)
                inference_end = time.perf_counter()

                calibration_ms = (
                    calibration_end - calibration_start
                ) * 1000.0
                inference_ms = (
                    inference_end - inference_start
                ) * 1000.0
                processing_ms = (
                    inference_end - calibration_start
                ) * 1000.0
                end_to_end_age_ms = (
                    time.time_ns() - frame.host_unix_time_ns
                ) / 1_000_000.0
                algorithmic_delay_ms = (
                    args.num_future_frame * stream_period_ms
                )
                effective_output_age_ms = (
                    end_to_end_age_ms + algorithmic_delay_ms
                )

                calibration_times.append(calibration_ms)
                inference_times.append(inference_ms)
                processing_times.append(processing_ms)
                age_times.append(end_to_end_age_ms)
                effective_age_times.append(effective_output_age_ms)

                if processing_ms > stream_period_ms:
                    over_budget_frames += 1

                finite = bool(
                    torch.isfinite(output.pose).all()
                    and torch.isfinite(output.translation).all()
                    and torch.isfinite(output.reduced_pose_6d).all()
                )
                if not finite:
                    nonfinite_outputs += 1

                orth_error, determinant_error = _rotation_errors(output.pose)
                if (
                    orth_error > args.rotation_tolerance
                    or determinant_error > args.rotation_tolerance
                ):
                    invalid_rotation_outputs += 1

                frames_received += 1
                if output.has_full_window:
                    valid_outputs += 1
                    if not window_ready_announced:
                        window_ready_announced = True
                        print(
                            f"[ready] Temporal window filled at input frame "
                            f"{output.input_frame_index}. Model output now has "
                            f"the validated {args.num_future_frame}-frame delay.\n"
                        )

                if args.save_pt:
                    _append_capture(
                        store,
                        frame,
                        calibrated.acc,
                        calibrated.ori,
                        output,
                        calibration_ms,
                        inference_ms,
                        processing_ms,
                        end_to_end_age_ms,
                        effective_output_age_ms,
                    )

                if frames_received % args.print_every == 0:
                    elapsed = (
                        receive_time - first_capture_time
                        if first_capture_time is not None
                        else 0.0
                    )
                    receive_rate = (
                        (frames_received - 1) / elapsed
                        if frames_received > 1 and elapsed > 0
                        else 0.0
                    )
                    root = output.translation
                    window_state = (
                        "READY" if output.has_full_window else
                        f"{pipeline.online_received_frames}/{pipeline.window_size}"
                    )
                    print(
                        f"Frame {frames_received:5d} | sequence={frame.sequence} | "
                        f"rate={receive_rate:6.2f} Hz | window={window_state}"
                    )
                    print(
                        f"  input={output.input_frame_index} | "
                        f"output={output.output_frame_index} | "
                        f"future_context={output.has_future_context}"
                    )
                    print(
                        f"  calibration={calibration_ms:7.3f} ms | "
                        f"inference={inference_ms:7.3f} ms | "
                        f"processing={processing_ms:7.3f} ms | "
                        f"host age={end_to_end_age_ms:7.3f} ms"
                    )
                    print(
                        f"  effective output age including "
                        f"{args.num_future_frame}-frame look-ahead: "
                        f"{effective_output_age_ms:7.3f} ms"
                    )
                    print(
                        f"  root translation=[{root[0]: .4f}, "
                        f"{root[1]: .4f}, {root[2]: .4f}] m"
                    )
                    print(
                        f"  pose orth_err={orth_error:.3e} | "
                        f"det_err={determinant_error:.3e} | "
                        f"sequence_gaps={receiver.sequence_gaps}"
                    )
                    print()

            sequence_gaps = receiver.sequence_gaps - start_sequence_gaps

    except KeyboardInterrupt:
        print("\nLive inference stopped by user.")
        sequence_gaps = 0

    elapsed = (
        last_capture_time - first_capture_time
        if (
            first_capture_time is not None
            and last_capture_time is not None
        )
        else 0.0
    )
    measured_rate = (
        (frames_received - 1) / elapsed
        if frames_received > 1 and elapsed > 0
        else 0.0
    )

    print("\n" + "=" * 80)
    print("LIVE INFERENCE SUMMARY")
    print("=" * 80)
    print(f"  Frames received:             {frames_received}")
    print(f"  Full-window outputs:         {valid_outputs}")
    print(f"  Measured receive rate:       {measured_rate:.3f} Hz")
    print(f"  Sequence gaps:               {sequence_gaps}")
    print(f"  Packet-counter mismatches:   {packet_counter_mismatches}")
    print(f"  Receive timeouts:            {receive_timeouts}")
    print(f"  Non-finite model outputs:    {nonfinite_outputs}")
    print(f"  Invalid pose rotations:      {invalid_rotation_outputs}")
    print(
        f"  Frames over {stream_period_ms:.3f} ms budget: "
        f"{over_budget_frames}/{frames_received}"
    )
    print()

    _print_stats("Calibration", calibration_times)
    _print_stats("D-BRAN inference", inference_times)
    _print_stats("Total processing", processing_times)
    _print_stats("Newest-frame host age", age_times)
    _print_stats("Effective output age", effective_age_times)

    if args.save_pt and store["pose"]:
        saved_path = _save_capture(
            args.save_pt,
            store,
            calibration_path,
            args,
            parameter_counts,
        )
        print(f"\nSaved live capture: {saved_path}")

    processing_p95 = _stats(processing_times)["p95"]
    passed = (
        frames_received > 0
        and valid_outputs > 0
        and sequence_gaps == 0
        and packet_counter_mismatches == 0
        and nonfinite_outputs == 0
        and invalid_rotation_outputs == 0
        and processing_p95 <= stream_period_ms
    )

    print()
    if passed:
        print("D-BRAN LIVE INFERENCE TEST PASSED")
        print(
            "The calibrated Xsens stream is producing valid D-BRAN pose and "
            "translation outputs within the 60 Hz processing budget."
        )
    else:
        print("D-BRAN LIVE INFERENCE TEST PRODUCED WARNINGS")
        if processing_p95 > stream_period_ms:
            print(
                f"  Processing p95 is {processing_p95:.3f} ms, above the "
                f"{stream_period_ms:.3f} ms frame budget."
            )
        if sequence_gaps:
            print(f"  UDP sequence gaps detected: {sequence_gaps}.")
        if packet_counter_mismatches:
            print(
                "  Some frames contained mismatched Xsens packet counters."
            )
        if nonfinite_outputs:
            print(f"  Non-finite outputs detected: {nonfinite_outputs}.")
        if invalid_rotation_outputs:
            print(
                f"  Invalid pose rotation outputs: {invalid_rotation_outputs}."
            )


if __name__ == "__main__":
    main()
