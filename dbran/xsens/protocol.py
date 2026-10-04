"""Binary protocol shared by the native Xsens bridge and Python."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Final, Tuple

MAGIC: Final[bytes] = b"DBRN"
PROTOCOL_VERSION: Final[int] = 1
SENSOR_COUNT: Final[int] = 6
FRAME_COMPLETE_FLAG: Final[int] = 1

SENSOR_ROLES: Final[Tuple[str, ...]] = (
    "left_arm",
    "right_arm",
    "left_leg",
    "right_leg",
    "head",
    "root",
)

HEADER_STRUCT: Final[struct.Struct] = struct.Struct("<4sHHQQII")
SENSOR_STRUCT: Final[struct.Struct] = struct.Struct("<Iqq4d9d3d")
PACKET_SIZE: Final[int] = HEADER_STRUCT.size + SENSOR_COUNT * SENSOR_STRUCT.size


class XsensProtocolError(ValueError):
    """Raised when a UDP datagram does not match the D-BRAN protocol."""


@dataclass(frozen=True)
class XsensSensorSample:
    role: str
    device_id: int
    packet_counter: int
    sample_time_fine: int
    quaternion_wxyz: Tuple[float, float, float, float]
    rotation_matrix: Tuple[float, ...]
    acceleration: Tuple[float, float, float]

    @property
    def device_id_hex(self) -> str:
        return f"{self.device_id:08X}"


@dataclass(frozen=True)
class XsensSynchronizedFrame:
    sequence: int
    host_unix_time_ns: int
    update_rate_hz: int
    flags: int
    sensors: Tuple[XsensSensorSample, ...]

    @property
    def is_complete(self) -> bool:
        return bool(self.flags & FRAME_COMPLETE_FLAG)


def decode_frame(datagram: bytes) -> XsensSynchronizedFrame:
    """Decode and validate one fixed-size UDP frame."""
    if len(datagram) != PACKET_SIZE:
        raise XsensProtocolError(
            f"Invalid datagram size: received {len(datagram)}, expected {PACKET_SIZE}."
        )

    (
        magic,
        version,
        sensor_count,
        sequence,
        host_unix_time_ns,
        update_rate_hz,
        flags,
    ) = HEADER_STRUCT.unpack_from(datagram, 0)

    if magic != MAGIC:
        raise XsensProtocolError(f"Invalid magic: {magic!r}.")
    if version != PROTOCOL_VERSION:
        raise XsensProtocolError(
            f"Unsupported protocol version: {version}; expected {PROTOCOL_VERSION}."
        )
    if sensor_count != SENSOR_COUNT:
        raise XsensProtocolError(
            f"Invalid sensor count: {sensor_count}; expected {SENSOR_COUNT}."
        )

    samples = []
    offset = HEADER_STRUCT.size
    for sensor_index, role in enumerate(SENSOR_ROLES):
        values = SENSOR_STRUCT.unpack_from(datagram, offset)
        offset += SENSOR_STRUCT.size

        device_id = int(values[0])
        packet_counter = int(values[1])
        sample_time_fine = int(values[2])
        quaternion = tuple(float(value) for value in values[3:7])
        rotation = tuple(float(value) for value in values[7:16])
        acceleration = tuple(float(value) for value in values[16:19])

        samples.append(
            XsensSensorSample(
                role=role,
                device_id=device_id,
                packet_counter=packet_counter,
                sample_time_fine=sample_time_fine,
                quaternion_wxyz=quaternion,  # type: ignore[arg-type]
                rotation_matrix=rotation,
                acceleration=acceleration,  # type: ignore[arg-type]
            )
        )

    if offset != len(datagram):
        raise XsensProtocolError(
            f"Decoder offset mismatch: decoded {offset}, packet has {len(datagram)} bytes."
        )

    return XsensSynchronizedFrame(
        sequence=int(sequence),
        host_unix_time_ns=int(host_unix_time_ns),
        update_rate_hz=int(update_rate_hz),
        flags=int(flags),
        sensors=tuple(samples),
    )
