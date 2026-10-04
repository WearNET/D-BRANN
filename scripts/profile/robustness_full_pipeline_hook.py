import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import torch

HOOK_VERSION = "2026-06-15-exclusive-burst-v3"


SENSOR_NAME_TO_INDEX = {
    "left_arm": 0,
    "right_arm": 1,
    "left_leg": 2,
    "right_leg": 3,
    "head": 4,
    "root": 5,
}

SENSOR_ALIASES = {
    "left_lower_arm": "left_arm",
    "right_lower_arm": "right_arm",
    "left_lower_leg": "left_leg",
    "right_lower_leg": "right_leg",
    "pelvis": "root",
}


@dataclass(frozen=True)
class FaultInjectionConfig:
    enabled: bool
    method: str
    pattern: str
    sensors: Tuple[str, ...]
    sensor_indices: Tuple[int, ...]
    loss_rate: float
    burst_length: int
    seed: int
    repeat: int
    mask_mode: str
    manifest_path: Optional[str]
    verbose: bool


_ORIGINAL_TORCH_LOAD = None
_INSTALLED_CONFIG = None
_LOGGED_SOURCES = set()
_MANIFEST_KEYS = set()


def _parse_sensor_names(value: str) -> Tuple[str, ...]:
    raw_names = [
        item.strip().lower()
        for item in value.split(",")
        if item.strip()
    ]

    if not raw_names:
        raise ValueError("At least one sensor must be specified.")

    expanded = []

    for raw_name in raw_names:
        name = SENSOR_ALIASES.get(raw_name, raw_name)

        if name == "all_peripheral":
            expanded.extend([
                "left_arm",
                "right_arm",
                "left_leg",
                "right_leg",
                "head",
            ])
        elif name == "all":
            expanded.extend([
                "left_arm",
                "right_arm",
                "left_leg",
                "right_leg",
                "head",
                "root",
            ])
        elif name in SENSOR_NAME_TO_INDEX:
            expanded.append(name)
        else:
            valid = sorted(
                list(SENSOR_NAME_TO_INDEX)
                + list(SENSOR_ALIASES)
                + ["all_peripheral", "all"]
            )
            raise ValueError(
                f"Unknown sensor '{raw_name}'. Valid values: {valid}"
            )

    unique = []
    seen = set()

    for name in expanded:
        if name not in seen:
            unique.append(name)
            seen.add(name)

    return tuple(unique)


def _parse_hook_arguments(argv: Sequence[str]):
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "--fault_enable",
        action="store_true",
    )
    parser.add_argument(
        "--fault_method",
        choices=["failure_zero", "zoh"],
        default="zoh",
    )
    parser.add_argument(
        "--fault_pattern",
        choices=["iid", "burst"],
        default="burst",
    )
    parser.add_argument(
        "--fault_sensors",
        type=str,
        default="left_arm",
    )
    parser.add_argument(
        "--fault_loss_rate",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--fault_burst_length",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--fault_seed",
        type=int,
        default=1234,
    )
    parser.add_argument(
        "--fault_repeat",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--fault_mask_mode",
        choices=["shared", "independent", "exclusive"],
        default="shared",
    )
    parser.add_argument(
        "--fault_manifest_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--fault_verbose",
        action="store_true",
    )

    known, remaining = parser.parse_known_args(argv)

    sensor_names = _parse_sensor_names(
        known.fault_sensors
    )
    sensor_indices = tuple(
        SENSOR_NAME_TO_INDEX[name]
        for name in sensor_names
    )

    if not 0.0 <= known.fault_loss_rate <= 1.0:
        raise ValueError(
            "--fault_loss_rate must be in the interval [0, 1]."
        )

    if known.fault_burst_length <= 0:
        raise ValueError(
            "--fault_burst_length must be greater than zero."
        )

    config = FaultInjectionConfig(
        enabled=known.fault_enable,
        method=known.fault_method,
        pattern=known.fault_pattern,
        sensors=sensor_names,
        sensor_indices=sensor_indices,
        loss_rate=known.fault_loss_rate,
        burst_length=known.fault_burst_length,
        seed=known.fault_seed,
        repeat=known.fault_repeat,
        mask_mode=known.fault_mask_mode,
        manifest_path=known.fault_manifest_path,
        verbose=known.fault_verbose,
    )

    return config, remaining


def _stable_seed(
    config: FaultInjectionConfig,
    source_name: str,
    sensor_index: Optional[int] = None,
) -> int:
    payload = (
        f"{source_name}|{config.seed}|{config.repeat}|"
        f"{config.pattern}|{config.loss_rate}|"
        f"{config.burst_length}|{config.mask_mode}|"
        f"{sensor_index}"
    )

    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).digest()

    hashed_value = int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    )

    return hashed_value % (2**63 - 1)


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _build_iid_temporal_mask(
    sequence_length: int,
    loss_rate: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if loss_rate <= 0.0:
        return torch.zeros(
            sequence_length,
            dtype=torch.bool,
        )

    temporal_mask = (
        torch.rand(
            sequence_length,
            generator=generator,
        )
        < loss_rate
    )

    # ZOH requires a previous valid observation.
    temporal_mask[0] = False
    return temporal_mask


def _build_burst_temporal_mask(
    sequence_length: int,
    loss_rate: float,
    burst_length: int,
    generator: torch.Generator,
) -> torch.Tensor:
    temporal_mask = torch.zeros(
        sequence_length,
        dtype=torch.bool,
    )

    if loss_rate <= 0.0 or sequence_length <= 1:
        return temporal_mask

    target_missing = int(
        round(sequence_length * loss_rate)
    )
    target_missing = max(
        1,
        min(target_missing, sequence_length - 1),
    )

    burst_length = max(
        1,
        min(burst_length, sequence_length - 1),
    )

    attempts = 0
    maximum_attempts = max(
        1000,
        target_missing * 20,
    )

    while (
        int(temporal_mask.sum().item()) < target_missing
        and attempts < maximum_attempts
    ):
        maximum_start = max(
            1,
            sequence_length - burst_length,
        )

        start = int(
            torch.randint(
                low=1,
                high=maximum_start + 1,
                size=(1,),
                generator=generator,
            ).item()
        )

        end = min(
            sequence_length,
            start + burst_length,
        )

        temporal_mask[start:end] = True
        attempts += 1

    current_missing = int(
        temporal_mask.sum().item()
    )

    if current_missing > target_missing:
        missing_indices = torch.where(
            temporal_mask
        )[0]

        keep_indices = missing_indices[
            :target_missing
        ]

        temporal_mask.zero_()
        temporal_mask[keep_indices] = True

    temporal_mask[0] = False
    return temporal_mask


def _build_sensor_mask(
    sequence_length: int,
    config: FaultInjectionConfig,
    source_name: str,
) -> torch.Tensor:
    mask = torch.zeros(
        sequence_length,
        6,
        dtype=torch.bool,
    )

    def build_temporal(sensor_index: Optional[int]):
        seed = _stable_seed(
            config,
            source_name,
            sensor_index,
        )
        generator = _generator(seed)

        if config.pattern == "iid":
            return _build_iid_temporal_mask(
                sequence_length,
                config.loss_rate,
                generator,
            )

        return _build_burst_temporal_mask(
            sequence_length,
            config.loss_rate,
            config.burst_length,
            generator,
        )

    if config.mask_mode == "shared":
        temporal_mask = build_temporal(None)

        for sensor_index in config.sensor_indices:
            mask[:, sensor_index] = temporal_mask

    elif config.mask_mode == "independent":
        for sensor_index in config.sensor_indices:
            mask[:, sensor_index] = build_temporal(
                sensor_index
            )

    elif config.mask_mode == "exclusive":
        generator = _generator(
            _stable_seed(
                config,
                source_name,
                None,
            )
        )

        if config.pattern == "iid":
            # Each affected frame contains exactly one failed sensor.
            temporal_mask = _build_iid_temporal_mask(
                sequence_length,
                config.loss_rate,
                generator,
            )

            failure_frames = torch.where(
                temporal_mask
            )[0]

            if failure_frames.numel() > 0:
                selected_sensor_positions = torch.randint(
                    low=0,
                    high=len(config.sensor_indices),
                    size=(failure_frames.numel(),),
                    generator=generator,
                )

                for frame_index, sensor_position in zip(
                    failure_frames.tolist(),
                    selected_sensor_positions.tolist(),
                ):
                    sensor_index = config.sensor_indices[
                        sensor_position
                    ]
                    mask[frame_index, sensor_index] = True

        elif config.pattern == "burst":
            # Create prolonged, non-overlapping failures.
            # loss_rate is the total fraction of affected frames.
            target_affected = int(
                round(sequence_length * config.loss_rate)
            )
            target_affected = max(
                1,
                min(target_affected, sequence_length - 1),
            )

            burst_length = max(
                1,
                min(config.burst_length, target_affected),
            )

            burst_lengths = []
            remaining = target_affected

            while remaining > 0:
                current_length = min(
                    burst_length,
                    remaining,
                )
                burst_lengths.append(current_length)
                remaining -= current_length

            burst_count = len(burst_lengths)
            available_valid_frames = (
                sequence_length - target_affected
            )

            # Keep frame 0 valid for causal ZOH.
            minimum_internal_gap = (
                1
                if available_valid_frames >= burst_count
                else 0
            )

            base_gaps = [1]
            base_gaps.extend(
                [minimum_internal_gap] * (burst_count - 1)
            )
            base_gaps.append(0)

            used_valid_frames = sum(base_gaps)
            extra_valid_frames = max(
                0,
                available_valid_frames - used_valid_frames,
            )

            gap_weights = torch.rand(
                burst_count + 1,
                generator=generator,
            )
            gap_weights = gap_weights / gap_weights.sum()

            extra_gaps = torch.floor(
                gap_weights * extra_valid_frames
            ).to(torch.long)

            unassigned = (
                extra_valid_frames
                - int(extra_gaps.sum().item())
            )

            if unassigned > 0:
                order = torch.randperm(
                    burst_count + 1,
                    generator=generator,
                )
                extra_gaps[order[:unassigned]] += 1

            gaps = [
                base_gaps[index]
                + int(extra_gaps[index].item())
                for index in range(burst_count + 1)
            ]

            cursor = gaps[0]
            previous_sensor_position = None

            for burst_index, current_length in enumerate(
                burst_lengths
            ):
                if len(config.sensor_indices) > 1:
                    candidates = list(
                        range(len(config.sensor_indices))
                    )

                    if previous_sensor_position in candidates:
                        candidates.remove(
                            previous_sensor_position
                        )

                    candidate_index = int(
                        torch.randint(
                            low=0,
                            high=len(candidates),
                            size=(1,),
                            generator=generator,
                        ).item()
                    )
                    sensor_position = candidates[
                        candidate_index
                    ]
                else:
                    sensor_position = 0

                sensor_index = config.sensor_indices[
                    sensor_position
                ]

                end = cursor + current_length
                mask[cursor:end, sensor_index] = True

                previous_sensor_position = sensor_position
                cursor = end + gaps[burst_index + 1]

        else:
            raise ValueError(
                f"Unsupported fault pattern for exclusive mode: "
                f"{config.pattern}"
            )

    else:
        raise ValueError(
            f"Unsupported fault_mask_mode: {config.mask_mode}"
        )

    return mask


def _canonicalize_acceleration(
    acceleration: torch.Tensor,
):
    original_shape = tuple(acceleration.shape)

    if acceleration.dim() == 3:
        canonical = acceleration.reshape(
            acceleration.shape[0],
            6,
            3,
        )
    elif (
        acceleration.dim() == 2
        and acceleration.shape[1] == 18
    ):
        canonical = acceleration.reshape(
            acceleration.shape[0],
            6,
            3,
        )
    else:
        raise ValueError(
            "Unsupported acceleration shape for fault injection: "
            f"{original_shape}"
        )

    return canonical.clone(), original_shape


def _canonicalize_orientation(
    orientation: torch.Tensor,
):
    original_shape = tuple(orientation.shape)

    if orientation.dim() == 4:
        canonical = orientation.reshape(
            orientation.shape[0],
            6,
            3,
            3,
        )
    elif (
        orientation.dim() == 3
        and tuple(orientation.shape[1:]) == (6, 9)
    ):
        canonical = orientation.reshape(
            orientation.shape[0],
            6,
            3,
            3,
        )
    elif (
        orientation.dim() == 2
        and orientation.shape[1] == 54
    ):
        canonical = orientation.reshape(
            orientation.shape[0],
            6,
            3,
            3,
        )
    else:
        raise ValueError(
            "Unsupported orientation shape for fault injection: "
            f"{original_shape}"
        )

    return canonical.clone(), original_shape


def _restore_shape(
    tensor: torch.Tensor,
    original_shape: Tuple[int, ...],
) -> torch.Tensor:
    return tensor.reshape(original_shape)


def _apply_zero_failure(
    acceleration: torch.Tensor,
    orientation: torch.Tensor,
    mask: torch.Tensor,
):
    acceleration[mask] = 0.0
    orientation[mask] = 0.0


def _apply_zoh(
    acceleration: torch.Tensor,
    orientation: torch.Tensor,
    mask: torch.Tensor,
):
    sequence_length = acceleration.shape[0]
    frame_indices = torch.arange(
        sequence_length,
        dtype=torch.long,
    )

    for sensor_index in range(6):
        sensor_mask = mask[:, sensor_index]

        if not bool(sensor_mask.any()):
            continue

        valid_source_indices = torch.where(
            sensor_mask,
            torch.full_like(frame_indices, -1),
            frame_indices,
        )

        last_valid_indices = torch.cummax(
            valid_source_indices,
            dim=0,
        ).values

        # The first frame is always valid by construction.
        acceleration[:, sensor_index] = (
            acceleration[
                last_valid_indices,
                sensor_index,
            ]
        )

        orientation[:, sensor_index] = (
            orientation[
                last_valid_indices,
                sensor_index,
            ]
        )


def _looks_like_raw_sequence(data: Any) -> bool:
    if not isinstance(data, dict):
        return False

    required_keys = {"acc", "ori", "pose"}

    if not required_keys.issubset(data.keys()):
        return False

    acceleration = data["acc"]
    orientation = data["ori"]

    return (
        torch.is_tensor(acceleration)
        and torch.is_tensor(orientation)
        and acceleration.dim() >= 2
        and orientation.dim() >= 2
    )


def _source_name_from_load_argument(
    load_argument: Any,
) -> str:
    if isinstance(load_argument, (str, os.PathLike)):
        return str(Path(load_argument).resolve())

    name = getattr(load_argument, "name", None)

    if name is not None:
        return str(Path(name).resolve())

    return "<in-memory-torch-load>"


def _write_manifest(
    config: FaultInjectionConfig,
    source_name: str,
    sequence_length: int,
    mask: torch.Tensor,
):
    if config.manifest_path is None:
        return

    manifest_key = (
        source_name,
        config.method,
        config.pattern,
        config.sensors,
        config.loss_rate,
        config.burst_length,
        config.seed,
        config.repeat,
        config.mask_mode,
    )

    if manifest_key in _MANIFEST_KEYS:
        return

    _MANIFEST_KEYS.add(manifest_key)

    manifest_path = Path(
        config.manifest_path
    )
    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_mask = mask[
        :,
        list(config.sensor_indices),
    ]

    affected_frame_rate = float(
        selected_mask.any(dim=1).float().mean().item()
    )
    effective_packet_loss_rate = float(
        selected_mask.float().mean().item()
    )
    maximum_simultaneous_failures = int(
        selected_mask.sum(dim=1).max().item()
    )

    entry = {
        **asdict(config),
        "hook_version": HOOK_VERSION,
        "source_file": source_name,
        "sequence_length": sequence_length,
        "affected_frame_rate": affected_frame_rate,
        "effective_packet_loss_rate": effective_packet_loss_rate,
        "maximum_simultaneous_failures": maximum_simultaneous_failures,
        "missing_frames_per_sensor": {
            sensor_name: int(
                mask[
                    :,
                    SENSOR_NAME_TO_INDEX[sensor_name],
                ].sum().item()
            )
            for sensor_name in config.sensors
        },
    }

    with manifest_path.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(entry, sort_keys=True)
            + "\n"
        )


def _inject_faults(
    data: Dict[str, Any],
    config: FaultInjectionConfig,
    source_name: str,
) -> Dict[str, Any]:
    output = dict(data)

    acceleration, acceleration_shape = (
        _canonicalize_acceleration(
            data["acc"].float()
        )
    )

    orientation, orientation_shape = (
        _canonicalize_orientation(
            data["ori"].float()
        )
    )

    sequence_length = min(
        acceleration.shape[0],
        orientation.shape[0],
    )

    acceleration = acceleration[
        :sequence_length
    ]
    orientation = orientation[
        :sequence_length
    ]

    mask = _build_sensor_mask(
        sequence_length,
        config,
        source_name,
    )

    if config.method == "failure_zero":
        _apply_zero_failure(
            acceleration,
            orientation,
            mask,
        )
    elif config.method == "zoh":
        _apply_zoh(
            acceleration,
            orientation,
            mask,
        )
    else:
        raise ValueError(
            f"Unsupported fault method: {config.method}"
        )

    output["acc"] = _restore_shape(
        acceleration,
        (
            sequence_length,
            *acceleration_shape[1:],
        ),
    )

    output["ori"] = _restore_shape(
        orientation,
        (
            sequence_length,
            *orientation_shape[1:],
        ),
    )

    _write_manifest(
        config,
        source_name,
        sequence_length,
        mask,
    )

    if (
        config.verbose
        and source_name not in _LOGGED_SOURCES
    ):
        selected_mask = mask[
            :,
            list(config.sensor_indices),
        ]

        affected_frame_rate = float(
            selected_mask.any(dim=1).float().mean().item()
        )
        effective_packet_loss_rate = float(
            selected_mask.float().mean().item()
        )
        maximum_simultaneous_failures = int(
            selected_mask.sum(dim=1).max().item()
        )

        print(
            "[fault hook]",
            Path(source_name).name,
            f"version={HOOK_VERSION}",
            f"method={config.method}",
            f"pattern={config.pattern}",
            f"sensors={','.join(config.sensors)}",
            f"requested_frame_failure_rate={config.loss_rate:.4f}",
            f"affected_frame_rate={affected_frame_rate:.4f}",
            f"effective_packet_loss_rate={effective_packet_loss_rate:.4f}",
            f"max_simultaneous={maximum_simultaneous_failures}",
            f"burst_length={config.burst_length}",
        )

        _LOGGED_SOURCES.add(source_name)

    return output


def install_fault_injection() -> FaultInjectionConfig:
    """
    Install a transparent torch.load hook.

    Add exactly this line after the project root is inserted into sys.path:

        from robustness_full_pipeline_hook import install_fault_injection; install_fault_injection()

    Custom --fault_* arguments are parsed and removed from sys.argv before
    the profile's own ArgumentParser sees them.
    """
    global _ORIGINAL_TORCH_LOAD
    global _INSTALLED_CONFIG

    if _INSTALLED_CONFIG is not None:
        return _INSTALLED_CONFIG

    config, remaining_arguments = (
        _parse_hook_arguments(
            sys.argv[1:]
        )
    )

    sys.argv = [
        sys.argv[0],
        *remaining_arguments,
    ]

    _INSTALLED_CONFIG = config

    if not config.enabled:
        return config

    if _ORIGINAL_TORCH_LOAD is None:
        _ORIGINAL_TORCH_LOAD = torch.load

    original_load = _ORIGINAL_TORCH_LOAD

    def hooked_torch_load(*args, **kwargs):
        data = original_load(
            *args,
            **kwargs,
        )

        if not _looks_like_raw_sequence(data):
            return data

        source_name = (
            _source_name_from_load_argument(
                args[0] if args else "<unknown>"
            )
        )

        return _inject_faults(
            data,
            config,
            source_name,
        )

    hooked_torch_load._sensor_fault_hook_installed = True
    torch.load = hooked_torch_load

    print(
        "\n==================== SENSOR FAULT INJECTION ===================="
    )
    print(f"Hook version:     {HOOK_VERSION}")
    print("Enabled:          True")
    print(f"Method:           {config.method}")
    print(f"Pattern:          {config.pattern}")
    print(
        "Sensors:          "
        + ",".join(config.sensors)
    )
    print(f"Loss rate:        {config.loss_rate:.4f}")
    print(f"Burst length:     {config.burst_length} frames")
    print(f"Mask mode:        {config.mask_mode}")
    if config.mask_mode == "exclusive":
        print(
            "Exclusive mode:   at most one selected sensor fails in any frame"
        )
        print(
            "Loss-rate meaning: total fraction of frames affected by one sensor failure"
        )
    print(f"Seed:             {config.seed}")
    print(f"Repeat:           {config.repeat}")
    print(
        "Manifest:         "
        + (
            config.manifest_path
            if config.manifest_path
            else "disabled"
        )
    )

    return config



def _make_test_config(pattern: str, burst_length: int):
    sensor_names = (
        "left_arm",
        "right_arm",
        "left_leg",
        "right_leg",
        "head",
        "root",
    )

    return FaultInjectionConfig(
        enabled=True,
        method="zoh",
        pattern=pattern,
        sensors=sensor_names,
        sensor_indices=tuple(
            SENSOR_NAME_TO_INDEX[name]
            for name in sensor_names
        ),
        loss_rate=0.20,
        burst_length=burst_length,
        seed=1234,
        repeat=0,
        mask_mode="exclusive",
        manifest_path=None,
        verbose=False,
    )


def _longest_affected_run(mask: torch.Tensor) -> int:
    longest = 0
    current = 0

    for affected in mask.any(dim=1).tolist():
        if affected:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def _run_exclusive_self_test():
    print(f"Hook version: {HOOK_VERSION}")

    iid_config = _make_test_config(
        pattern="iid",
        burst_length=30,
    )
    iid_mask = _build_sensor_mask(
        sequence_length=10000,
        config=iid_config,
        source_name="exclusive-iid-self-test",
    )

    burst_config = _make_test_config(
        pattern="burst",
        burst_length=30,
    )
    burst_mask = _build_sensor_mask(
        sequence_length=10000,
        config=burst_config,
        source_name="exclusive-burst-self-test",
    )

    for label, mask, expected_tolerance in (
        ("IID", iid_mask, 0.03),
        ("BURST", burst_mask, 0.01),
    ):
        failures_per_frame = mask.sum(dim=1)
        affected_rate = float(
            failures_per_frame.gt(0).float().mean().item()
        )
        maximum_simultaneous = int(
            failures_per_frame.max().item()
        )

        print(f"{label} affected frame rate: {affected_rate:.4f}")
        print(
            f"{label} maximum simultaneous failures: "
            f"{maximum_simultaneous}"
        )

        if maximum_simultaneous != 1:
            raise RuntimeError(
                f"{label} test failed: simultaneous failures detected."
            )

        if abs(affected_rate - 0.20) > expected_tolerance:
            raise RuntimeError(
                f"{label} test failed: affected-frame rate is unexpected."
            )

    longest_run = _longest_affected_run(
        burst_mask
    )
    print(
        f"BURST longest affected run: "
        f"{longest_run} frames"
    )

    if longest_run < burst_config.burst_length:
        raise RuntimeError(
            "BURST test failed: prolonged dropout was not created."
        )

    print("EXCLUSIVE IID + BURST SELF-TEST PASSED")


if __name__ == "__main__":
    if "--self-test-exclusive" in sys.argv:
        _run_exclusive_self_test()
    else:
        print(
            "Run: python robustness_full_pipeline_hook.py "
            "--self-test-exclusive"
        )
