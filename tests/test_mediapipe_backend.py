"""Tests for the MediaPipe backend, focused on the image vs world 3D modes.

Skipped when MediaPipe or the sample stills are unavailable, so the core suite
stays green without the optional heavy dependency.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

mp = pytest.importorskip("mediapipe")
cv2 = pytest.importorskip("cv2")

from rehubai import joints as joints_mod
from rehubai.angles import calculate_angle
from rehubai.backends.mediapipe_backend import MediaPipePoseBackend

ASSET = Path(__file__).resolve().parents[1] / "artifacts" / "assets" / "dead_bug" / "phase_0.png"
needs_asset = pytest.mark.skipif(not ASSET.exists(), reason="dead_bug sample still not present")


def _kp(name: str) -> int:
    return joints_mod.resolve("blazepose33", name)


@needs_asset
def test_world_mode_returns_metric_coordinates():
    frame = cv2.imread(str(ASSET))
    with MediaPipePoseBackend(use_world_landmarks=True, static_image_mode=True) as be:
        kps, conf = be.detect(frame)
    assert kps.shape == (33, 3)
    # World landmarks are metres about the mid-hip: a standing/lying human spans
    # ~2 m, so every coordinate must be small — unlike pixel coords (hundreds).
    assert np.abs(kps).max() < 3.0
    # z must carry real signal (not all zero) — this is the whole point of 3D.
    assert np.abs(kps[:, 2]).max() > 0.05
    assert 0.0 <= conf.max() <= 1.0


@needs_asset
def test_image_mode_returns_pixel_scale():
    frame = cv2.imread(str(ASSET))
    h, w = frame.shape[:2]
    with MediaPipePoseBackend(use_world_landmarks=False, static_image_mode=True) as be:
        kps, _ = be.detect(frame)
    # x/y are in pixels, so they live on the frame scale (hundreds), not metres.
    assert kps[:, 0].max() > 1.5  # far larger than any metric value
    assert kps[:, 0].max() <= w
    assert kps[:, 1].max() <= h


@needs_asset
def test_3d_angle_differs_from_2d_for_depth_bent_limb():
    # The left elbow in phase_0 looks near-straight in the image but is bent in
    # depth; the 3D angle must diverge materially from the 2D one.
    frame = cv2.imread(str(ASSET))
    with MediaPipePoseBackend(use_world_landmarks=True, static_image_mode=True) as be:
        kps, _ = be.detect(frame)
    s, e, wr = kps[_kp("left_shoulder")], kps[_kp("left_elbow")], kps[_kp("left_wrist")]
    a3 = calculate_angle(s, e, wr)
    a2 = calculate_angle(s[:2], e[:2], wr[:2])
    assert abs(a3 - a2) > 10.0