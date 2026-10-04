"""UDP receiver for synchronized six-sensor Xsens frames."""

from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .protocol import PACKET_SIZE, XsensSynchronizedFrame, decode_frame


@dataclass(frozen=True)
class XsensTorchFrame:
    sequence: int
    host_unix_time_ns: int
    update_rate_hz: int
    device_ids: Tuple[str, ...]
    packet_counters: Tuple[int, ...]
    sample_times_fine: Tuple[int, ...]
    acc: torch.Tensor
    ori: torch.Tensor


class XsensUdpReceiver:
    """Receive D-BRAN Xsens frames from the native localhost bridge."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9763,
        timeout_s: Optional[float] = 2.0,
        receive_buffer_bytes: int = 1 << 20,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_RCVBUF,
            int(receive_buffer_bytes),
        )
        self.socket.bind((self.host, self.port))
        self.socket.settimeout(timeout_s)
        self.last_sequence: Optional[int] = None
        self.sequence_gaps = 0

    def close(self) -> None:
        self.socket.close()

    def __enter__(self) -> "XsensUdpReceiver":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def receive_raw(self) -> XsensSynchronizedFrame:
        datagram, _ = self.socket.recvfrom(PACKET_SIZE + 64)
        frame = decode_frame(datagram)

        if self.last_sequence is not None and frame.sequence > self.last_sequence + 1:
            self.sequence_gaps += frame.sequence - self.last_sequence - 1
        self.last_sequence = frame.sequence
        return frame

    def receive(self, device: str | torch.device = "cpu") -> XsensTorchFrame:
        frame = self.receive_raw()

        acc = torch.tensor(
            [sample.acceleration for sample in frame.sensors],
            dtype=torch.float32,
            device=device,
        )
        ori = torch.tensor(
            [sample.rotation_matrix for sample in frame.sensors],
            dtype=torch.float32,
            device=device,
        ).reshape(6, 3, 3)

        return XsensTorchFrame(
            sequence=frame.sequence,
            host_unix_time_ns=frame.host_unix_time_ns,
            update_rate_hz=frame.update_rate_hz,
            device_ids=tuple(sample.device_id_hex for sample in frame.sensors),
            packet_counters=tuple(sample.packet_counter for sample in frame.sensors),
            sample_times_fine=tuple(sample.sample_time_fine for sample in frame.sensors),
            acc=acc,
            ori=ori,
        )
