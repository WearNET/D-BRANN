# Validated Xsens Calibration Artifacts

This directory stores the calibration artifacts used for the first validated
D-BRAN Xsens integration.

Recommended tracked files:

```text
xsens_calibration_capture.pt
xsens_calibration_test.pt
```

The active calibration remains at:

```text
configs/xsens_calibration.json
```

These files are hardware- and placement-specific. They contain the assigned
MTw device IDs and a static T-pose calibration for the current six-sensor
setup. A new calibration should be created whenever sensor placement changes.

The calibration can be (re)generated two ways: standalone via
`scripts/xsens/calibrate_xsens.py`, or as part of a single combined session via
`scripts/xsens/xsensDataCapture.py`, which runs the same two-step calibration
and then immediately captures data with it — useful when recording a custom
session, since it avoids a stale calibration silently being reused across a
capture session.
