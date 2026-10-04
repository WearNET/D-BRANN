"""Static Xsens-to-D-BRAN calibration utilities.

The calibration follows the structure used by the original TransPose live
example, adapted to Xsens MTw measurements:

1. A reference MTw is aligned with the D-BRAN body frame
   (+X left, +Y up, +Z forward) to estimate the global-frame transform.
2. All six MTw sensors are worn and a static T-pose is captured to estimate
   each device-to-segment rotation.
3. Xsens calibrated acceleration is converted from sensor-local coordinates
   to the Xsens global frame before applying the D-BRAN global transform.
4. The static T-pose acceleration is stored as an offset and removed online.

The calibrated output is compatible with ``DBranPipeline.forward_online``:

    acc: [6, 3]
    ori: [6, 3, 3]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch

from .protocol import SENSOR_ROLES
from .receiver import XsensTorchFrame

CALIBRATION_FORMAT_VERSION = 1


class XsensCalibrationError(ValueError):
    """Raised when calibration data is invalid or incompatible."""


def _as_float_tensor(value: Any, shape: Tuple[int, ...], name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float64)
    if tuple(tensor.shape) != shape:
        raise XsensCalibrationError(
            f"{name} must have shape {shape}, received {tuple(tensor.shape)}."
        )
    if not torch.isfinite(tensor).all():
        raise XsensCalibrationError(f"{name} contains non-finite values.")
    return tensor


def project_to_rotation_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Project one or more 3x3 matrices to the nearest proper rotation."""
    matrix = torch.as_tensor(matrix, dtype=torch.float64)
    if matrix.shape[-2:] != (3, 3):
        raise XsensCalibrationError(
            f"Expected matrices ending in [3, 3], received {tuple(matrix.shape)}."
        )

    u, _, vh = torch.linalg.svd(matrix)
    uvh = u @ vh
    determinant = torch.linalg.det(uvh)

    correction = torch.eye(3, dtype=matrix.dtype, device=matrix.device)
    correction = correction.expand(*matrix.shape[:-2], 3, 3).clone()
    correction[..., 2, 2] = torch.where(
        determinant < 0,
        torch.tensor(-1.0, dtype=matrix.dtype, device=matrix.device),
        torch.tensor(1.0, dtype=matrix.dtype, device=matrix.device),
    )
    return u @ correction @ vh


def average_rotation_matrices(rotations: torch.Tensor) -> torch.Tensor:
    """Average rotations along the first dimension and project to SO(3)."""
    rotations = torch.as_tensor(rotations, dtype=torch.float64)
    if rotations.ndim < 3 or rotations.shape[-2:] != (3, 3):
        raise XsensCalibrationError(
            "rotations must have shape [N, ..., 3, 3], "
            f"received {tuple(rotations.shape)}."
        )
    if rotations.shape[0] == 0:
        raise XsensCalibrationError("Cannot average an empty rotation capture.")
    return project_to_rotation_matrix(rotations.mean(dim=0))


def rotation_angle_degrees(rotation: torch.Tensor) -> torch.Tensor:
    """Return the unsigned angle of one or more rotation matrices in degrees."""
    rotation = torch.as_tensor(rotation, dtype=torch.float64)
    if rotation.shape[-2:] != (3, 3):
        raise XsensCalibrationError(
            f"rotation must end in [3, 3], received {tuple(rotation.shape)}."
        )
    cosine = ((torch.diagonal(rotation, dim1=-2, dim2=-1).sum(dim=-1) - 1.0) / 2.0)
    cosine = cosine.clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def rotation_capture_spread_degrees(
    rotations: torch.Tensor,
    mean_rotation: torch.Tensor | None = None,
) -> Dict[str, float]:
    """Summarize orientation variation around the capture mean."""
    rotations = torch.as_tensor(rotations, dtype=torch.float64)
    if mean_rotation is None:
        mean_rotation = average_rotation_matrices(rotations)
    relative = mean_rotation.transpose(-1, -2).unsqueeze(0) @ rotations
    angles = rotation_angle_degrees(relative)
    return {
        "mean_deg": float(angles.mean().item()),
        "max_deg": float(angles.max().item()),
        "std_deg": float(angles.std(unbiased=False).item()),
    }


@dataclass(frozen=True)
class CalibratedXsensFrame:
    """One synchronized frame after Xsens-to-D-BRAN calibration."""

    sequence: int
    host_unix_time_ns: int
    update_rate_hz: int
    device_ids: Tuple[str, ...]
    packet_counters: Tuple[int, ...]
    acc: torch.Tensor
    ori: torch.Tensor


@dataclass
class XsensCalibration:
    """Serializable static calibration for six Xsens MTw sensors."""

    global_to_dbran: torch.Tensor
    device_to_bone: torch.Tensor
    acceleration_offsets: torch.Tensor
    device_ids: Tuple[str, ...]
    reference_role: str
    update_rate_hz: int
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    sensor_roles: Tuple[str, ...] = SENSOR_ROLES
    metadata: Dict[str, Any] = field(default_factory=dict)
    format_version: int = CALIBRATION_FORMAT_VERSION

    def __post_init__(self) -> None:
        self.global_to_dbran = _as_float_tensor(
            self.global_to_dbran,
            (3, 3),
            "global_to_dbran",
        )
        self.device_to_bone = _as_float_tensor(
            self.device_to_bone,
            (6, 3, 3),
            "device_to_bone",
        )
        self.acceleration_offsets = _as_float_tensor(
            self.acceleration_offsets,
            (6, 3),
            "acceleration_offsets",
        )
        self.device_ids = tuple(str(value).upper() for value in self.device_ids)
        self.sensor_roles = tuple(str(value) for value in self.sensor_roles)
        self.reference_role = str(self.reference_role)
        self.update_rate_hz = int(self.update_rate_hz)
        self.format_version = int(self.format_version)

        if self.format_version != CALIBRATION_FORMAT_VERSION:
            raise XsensCalibrationError(
                f"Unsupported calibration version {self.format_version}; "
                f"expected {CALIBRATION_FORMAT_VERSION}."
            )
        if self.sensor_roles != SENSOR_ROLES:
            raise XsensCalibrationError(
                "Calibration sensor order does not match D-BRAN: "
                f"{self.sensor_roles} vs {SENSOR_ROLES}."
            )
        if len(self.device_ids) != 6:
            raise XsensCalibrationError(
                f"device_ids must contain six entries, received {len(self.device_ids)}."
            )
        if self.reference_role not in SENSOR_ROLES:
            raise XsensCalibrationError(
                f"Unknown reference role {self.reference_role!r}."
            )
        if self.update_rate_hz <= 0:
            raise XsensCalibrationError("update_rate_hz must be positive.")

        for name, rotations in (
            ("global_to_dbran", self.global_to_dbran.unsqueeze(0)),
            ("device_to_bone", self.device_to_bone),
        ):
            identity = torch.eye(3, dtype=torch.float64).expand_as(rotations)
            orthogonality_error = (
                rotations.transpose(-1, -2) @ rotations - identity
            ).abs().amax().item()
            determinant_error = (
                torch.linalg.det(rotations) - 1.0
            ).abs().amax().item()
            if orthogonality_error > 1e-5 or determinant_error > 1e-5:
                raise XsensCalibrationError(
                    f"{name} is not a valid rotation set: "
                    f"orthogonality_error={orthogonality_error:.3e}, "
                    f"determinant_error={determinant_error:.3e}."
                )

    @classmethod
    def from_captures(
        cls,
        reference_orientations: torch.Tensor,
        tpose_orientations: torch.Tensor,
        tpose_acc_local: torch.Tensor,
        device_ids: Sequence[str],
        reference_role: str,
        update_rate_hz: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> "XsensCalibration":
        """Build a calibration from reference and T-pose captures."""
        reference_orientations = torch.as_tensor(
            reference_orientations,
            dtype=torch.float64,
        )
        tpose_orientations = torch.as_tensor(
            tpose_orientations,
            dtype=torch.float64,
        )
        tpose_acc_local = torch.as_tensor(
            tpose_acc_local,
            dtype=torch.float64,
        )

        if reference_orientations.ndim != 3 or reference_orientations.shape[1:] != (3, 3):
            raise XsensCalibrationError(
                "reference_orientations must have shape [N, 3, 3], "
                f"received {tuple(reference_orientations.shape)}."
            )
        if tpose_orientations.ndim != 4 or tpose_orientations.shape[1:] != (6, 3, 3):
            raise XsensCalibrationError(
                "tpose_orientations must have shape [N, 6, 3, 3], "
                f"received {tuple(tpose_orientations.shape)}."
            )
        if tpose_acc_local.ndim != 3 or tpose_acc_local.shape[1:] != (6, 3):
            raise XsensCalibrationError(
                "tpose_acc_local must have shape [N, 6, 3], "
                f"received {tuple(tpose_acc_local.shape)}."
            )
        if tpose_orientations.shape[0] != tpose_acc_local.shape[0]:
            raise XsensCalibrationError(
                "T-pose orientation and acceleration lengths do not match: "
                f"{tpose_orientations.shape[0]} vs {tpose_acc_local.shape[0]}."
            )

        reference_mean = average_rotation_matrices(reference_orientations)
        global_to_dbran = reference_mean.transpose(-1, -2)

        tpose_orientation_mean = average_rotation_matrices(tpose_orientations)
        orientation_in_dbran = global_to_dbran @ tpose_orientation_mean
        device_to_bone = orientation_in_dbran.transpose(-1, -2)

        # Xsens calibratedAcceleration() is sensor-local for this bridge.
        # Convert local acceleration to Xsens global, then to D-BRAN global.
        acc_xsens_global = (
            tpose_orientations @ tpose_acc_local.unsqueeze(-1)
        ).squeeze(-1)
        acc_dbran_global = (
            global_to_dbran @ acc_xsens_global.unsqueeze(-1)
        ).squeeze(-1)
        acceleration_offsets = acc_dbran_global.mean(dim=0)

        calibration = cls(
            global_to_dbran=global_to_dbran,
            device_to_bone=device_to_bone,
            acceleration_offsets=acceleration_offsets,
            device_ids=tuple(device_ids),
            reference_role=reference_role,
            update_rate_hz=update_rate_hz,
            metadata=dict(metadata or {}),
        )

        reference_spread = rotation_capture_spread_degrees(
            reference_orientations,
            reference_mean,
        )
        tpose_spread = {}
        for index, role in enumerate(SENSOR_ROLES):
            tpose_spread[role] = rotation_capture_spread_degrees(
                tpose_orientations[:, index],
                tpose_orientation_mean[index],
            )

        tpose_acc_std = acc_dbran_global.std(dim=0, unbiased=False)
        calibrated_tpose_ori = (
            global_to_dbran @ tpose_orientations @ device_to_bone
        )
        calibrated_tpose_acc = acc_dbran_global - acceleration_offsets
        identity = torch.eye(3, dtype=torch.float64)
        tpose_angle = rotation_angle_degrees(
            calibrated_tpose_ori @ identity
        )

        calibration.metadata.update(
            {
                "reference_capture_frames": int(reference_orientations.shape[0]),
                "tpose_capture_frames": int(tpose_orientations.shape[0]),
                "reference_orientation_spread_deg": reference_spread,
                "tpose_orientation_spread_deg": tpose_spread,
                "tpose_acceleration_std_mps2": {
                    role: [float(value) for value in tpose_acc_std[index].tolist()]
                    for index, role in enumerate(SENSOR_ROLES)
                },
                "tpose_calibrated_orientation_mean_error_deg": {
                    role: float(tpose_angle[:, index].mean().item())
                    for index, role in enumerate(SENSOR_ROLES)
                },
                "tpose_calibrated_acceleration_mean_norm_mps2": {
                    role: float(
                        calibrated_tpose_acc[:, index].mean(dim=0).norm().item()
                    )
                    for index, role in enumerate(SENSOR_ROLES)
                },
            }
        )
        return calibration

    def validate_device_ids(self, device_ids: Sequence[str]) -> None:
        received = tuple(str(value).upper() for value in device_ids)
        if received != self.device_ids:
            details = []
            for index, role in enumerate(SENSOR_ROLES):
                expected = self.device_ids[index]
                actual = received[index] if index < len(received) else "<missing>"
                if expected != actual:
                    details.append(f"{role}: expected {expected}, received {actual}")
            raise XsensCalibrationError(
                "Connected MTw assignment does not match the calibration. "
                + "; ".join(details)
            )

    def apply_tensors(
        self,
        acc_local: torch.Tensor,
        ori_sensor_to_global: torch.Tensor,
        output_device: str | torch.device | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply calibration to one frame or a sequence.

        Accepted shapes:
            acc_local: [6, 3] or [T, 6, 3]
            ori_sensor_to_global: [6, 3, 3] or [T, 6, 3, 3]
        """
        acc_local = torch.as_tensor(acc_local)
        ori_sensor_to_global = torch.as_tensor(ori_sensor_to_global)

        single_frame = acc_local.ndim == 2
        if single_frame:
            acc_local = acc_local.unsqueeze(0)
            ori_sensor_to_global = ori_sensor_to_global.unsqueeze(0)

        if acc_local.ndim != 3 or acc_local.shape[1:] != (6, 3):
            raise XsensCalibrationError(
                "acc_local must have shape [6, 3] or [T, 6, 3], "
                f"received {tuple(acc_local.shape)}."
            )
        if ori_sensor_to_global.ndim != 4 or ori_sensor_to_global.shape[1:] != (6, 3, 3):
            raise XsensCalibrationError(
                "ori_sensor_to_global must have shape [6, 3, 3] or "
                f"[T, 6, 3, 3], received {tuple(ori_sensor_to_global.shape)}."
            )
        if acc_local.shape[0] != ori_sensor_to_global.shape[0]:
            raise XsensCalibrationError(
                "Acceleration and orientation lengths do not match: "
                f"{acc_local.shape[0]} vs {ori_sensor_to_global.shape[0]}."
            )

        target_device = (
            torch.device(output_device)
            if output_device is not None
            else ori_sensor_to_global.device
        )
        dtype = torch.float32
        acc_local = acc_local.to(device=target_device, dtype=dtype)
        ori_sensor_to_global = ori_sensor_to_global.to(
            device=target_device,
            dtype=dtype,
        )
        global_to_dbran = self.global_to_dbran.to(
            device=target_device,
            dtype=dtype,
        )
        device_to_bone = self.device_to_bone.to(
            device=target_device,
            dtype=dtype,
        )
        acceleration_offsets = self.acceleration_offsets.to(
            device=target_device,
            dtype=dtype,
        )

        acc_xsens_global = (
            ori_sensor_to_global @ acc_local.unsqueeze(-1)
        ).squeeze(-1)
        acc_dbran_global = (
            global_to_dbran @ acc_xsens_global.unsqueeze(-1)
        ).squeeze(-1)
        acc_calibrated = acc_dbran_global - acceleration_offsets

        ori_calibrated = (
            global_to_dbran @ ori_sensor_to_global @ device_to_bone
        )

        if single_frame:
            return acc_calibrated[0], ori_calibrated[0]
        return acc_calibrated, ori_calibrated

    def apply_frame(
        self,
        frame: XsensTorchFrame,
        output_device: str | torch.device | None = None,
        strict_device_ids: bool = True,
    ) -> CalibratedXsensFrame:
        """Apply calibration to a decoded UDP frame."""
        if strict_device_ids:
            self.validate_device_ids(frame.device_ids)
        acc, ori = self.apply_tensors(
            frame.acc,
            frame.ori,
            output_device=output_device,
        )
        return CalibratedXsensFrame(
            sequence=frame.sequence,
            host_unix_time_ns=frame.host_unix_time_ns,
            update_rate_hz=frame.update_rate_hz,
            device_ids=frame.device_ids,
            packet_counters=frame.packet_counters,
            acc=acc,
            ori=ori,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_version": self.format_version,
            "created_utc": self.created_utc,
            "reference_role": self.reference_role,
            "update_rate_hz": self.update_rate_hz,
            "sensor_roles": list(self.sensor_roles),
            "device_ids": list(self.device_ids),
            "global_to_dbran": self.global_to_dbran.tolist(),
            "device_to_bone": self.device_to_bone.tolist(),
            "acceleration_offsets": self.acceleration_offsets.tolist(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "XsensCalibration":
        required = (
            "format_version",
            "created_utc",
            "reference_role",
            "update_rate_hz",
            "sensor_roles",
            "device_ids",
            "global_to_dbran",
            "device_to_bone",
            "acceleration_offsets",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise XsensCalibrationError(
                "Calibration file is missing fields: " + ", ".join(missing)
            )
        return cls(
            format_version=int(data["format_version"]),
            created_utc=str(data["created_utc"]),
            reference_role=str(data["reference_role"]),
            update_rate_hz=int(data["update_rate_hz"]),
            sensor_roles=tuple(data["sensor_roles"]),
            device_ids=tuple(data["device_ids"]),
            global_to_dbran=data["global_to_dbran"],
            device_to_bone=data["device_to_bone"],
            acceleration_offsets=data["acceleration_offsets"],
            metadata=dict(data.get("metadata", {})),
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "XsensCalibration":
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Xsens calibration file not found: {source}")
        data = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise XsensCalibrationError(
                f"Calibration root must be a JSON object: {source}"
            )
        return cls.from_dict(data)
