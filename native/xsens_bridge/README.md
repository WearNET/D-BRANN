# Native Xsens bridge

This directory contains the Windows C++ bridge that acquires six synchronized
Xsens MTw2 samples through the Xsens Device API and transmits one fixed-size UDP
datagram per synchronized frame to the Python D-BRAN process.

## Sensor order

1. LeftArm
2. RightArm
3. LeftLeg
4. RightLeg
5. Head
6. Hip/root

## Default stream

- Destination: `127.0.0.1:9763`
- Rate: `60 Hz`
- Datagram size: `920 bytes`
- Protocol version: `1`

The C++ bridge performs acquisition and synchronization only. The 26-frame
online window remains inside `dbran.pipeline.DBranPipeline`.

## Startup stabilization

On entering measurement mode, the bridge waits `STABILIZATION_TIME_MS` (2000 ms)
before opening the capture gate, then discards anything buffered during that
window. This happens once, at startup, to let the wireless synchronization settle
before trusting the stream — it is not a recurring per-frame delay and does not
affect throughput once streaming has started.

## Sequence numbering and dropped frames

`frame.callbackSequence` (sent to Python as `sequence`) is assigned from
`m_callbackAttempts`, a counter that increments on **every** synchronized-callback
attempt from the Xsens Device API, whether or not all six sensors reported in time.
Only complete frames are ever sent over UDP, so a gap in the received sequence
numbers means the SDK attempted a synchronized frame at that slot but it was
incomplete (missing sensor data) — this is how `dbran/xsens/receiver.py`'s
`sequence_gaps` counter, and the gap-filling in
`scripts/xsens/xsensDataCapture.py`, detect real dropped data.

This matters because the naive alternative — a counter that only increments on
success — makes dropped frames structurally invisible: the received sequence
always looks perfectly contiguous no matter how much was actually lost upstream.
If you are reading this bridge's protocol from another client, do not assume
`sequence` is a simple frame-arrival counter; it specifically encodes gaps.

## Diagnostic summary

Press **ENTER** in the bridge's terminal to stop measurement. On stop, it prints:

```text
Synchronized frames processed: ...    # complete frames actually sent
Synchronized callback attempts: ...   # complete + incomplete attempts (the sequence-number space)
Incomplete multi-device callbacks: ...# callbacks missing at least one sensor
Application queue frames dropped: ... # complete frames dropped because the internal
                                       # queue (MAX_SYNCHRONIZED_QUEUE_SIZE = 300, ~5s
                                       # buffer) filled up before being sent
Missed packets reported by XDA: ...   # wireless packet loss reported by the Xsens SDK itself
```

A non-zero "Missed packets reported by XDA" together with non-zero "Incomplete
multi-device callbacks" indicates real wireless-link packet loss (radio
interference, distance/line-of-sight to the Awinda station) — not something
fixable in this code, only in the physical setup. "Application queue frames
dropped" being non-zero instead would point to the UDP-sending side not keeping
up, which is a different (and currently unseen) failure mode.

## Build

The versioned files in this package should be copied to their active names:

```text
src/xsens_stream_bridge/xsens_stream_bridge.cpp
build.ps1
run.ps1
```

Then run:

```powershell
.\build.ps1 xsens_stream_bridge
.\run.ps1 xsens_stream_bridge
```

Start the Python receiver before opening the capture gate so the first streamed
frame can be observed.
