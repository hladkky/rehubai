"""Tests for the UI-PRMD dataset adapter.

Tests that need the (4.6 GB, not-in-git) archive are skipped when it is absent,
so the suite stays green on machines without the data.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rehubai import joints as joints_mod
from rehubai.angles import calculate_angle
from rehubai.datasets import uiprmd

DATA_ROOT = uiprmd.default_root()
_HAS_DATA = DATA_ROOT.exists()
needs_data = pytest.mark.skipif(not _HAS_DATA, reason="UI-PRMD archive not present")


# --------------------------------------------------------------------------
# Pure-logic tests (no data required)
# --------------------------------------------------------------------------
def test_scheme_registered_with_core_joints():
    assert uiprmd.SCHEME in joints_mod.available_schemes()
    core = [
        "left_shoulder", "left_elbow", "left_wrist",
        "right_shoulder", "right_elbow", "right_wrist",
        "left_hip", "left_knee", "left_ankle",
        "right_hip", "right_knee", "right_ankle",
    ]
    assert joints_mod.supports(uiprmd.SCHEME, core)
    # A few index spot-checks against the documented joint order.
    assert joints_mod.resolve(uiprmd.SCHEME, "spine_base") == 0
    assert joints_mod.resolve(uiprmd.SCHEME, "left_elbow") == 8
    assert joints_mod.resolve(uiprmd.SCHEME, "right_knee") == 19


@pytest.mark.parametrize(
    "name,expected",
    [
        ("m01_s01_positions.txt",
         {"movement": "m01", "subject": "s01", "episode": None, "correct": True}),
        ("m10_s07_positions_inc.txt",
         {"movement": "m10", "subject": "s07", "episode": None, "correct": False}),
        ("m04_s06_e03_positions.txt",
         {"movement": "m04", "subject": "s06", "episode": "e03", "correct": True}),
        ("m02_s02_e10_positions_inc.txt",
         {"movement": "m02", "subject": "s02", "episode": "e10", "correct": False}),
    ],
)
def test_parse_filename(name, expected):
    assert uiprmd.parse_filename(name) == expected


def test_parse_filename_rejects_angles_and_junk():
    assert uiprmd.parse_filename("m01_s01_angles.txt") is None
    assert uiprmd.parse_filename("readme.txt") is None


def test_euler_matrix_is_orthonormal():
    R = uiprmd.euler_to_matrix(np.deg2rad([37.0, -12.0, 88.0]))
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert math.isclose(np.linalg.det(R), 1.0, abs_tol=1e-9)


def test_fk_preserves_bone_lengths():
    # A synthetic sequence with fixed offsets but time-varying angles must keep
    # every bone rigid (rotation is length-preserving).
    rng = np.random.default_rng(0)
    offsets = rng.normal(size=(uiprmd.N_JOINTS, 3))
    offsets[0] = [10.0, 20.0, 30.0]  # absolute root
    frames = np.stack([offsets] * 8)
    angles = rng.uniform(-90, 90, size=(8, uiprmd.N_JOINTS, 3))
    world = uiprmd.forward_kinematics(frames, angles)

    for child, parent in uiprmd._PARENT.items():
        rest = np.linalg.norm(offsets[child])
        lengths = np.linalg.norm(world[:, child] - world[:, parent], axis=1)
        assert np.allclose(lengths, rest, atol=1e-6)


def test_fk_shape_validation():
    with pytest.raises(ValueError):
        uiprmd.forward_kinematics(np.zeros((3, 21, 3)), np.zeros((3, 21, 3)))


def test_mirror_lr_is_involution():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, uiprmd.N_JOINTS, 3))
    assert np.allclose(uiprmd.mirror_lr(uiprmd.mirror_lr(x)), x)


def test_mirror_lr_swaps_left_and_right():
    lh = joints_mod.resolve(uiprmd.SCHEME, "left_hip")
    rh = joints_mod.resolve(uiprmd.SCHEME, "right_hip")
    x = np.zeros((uiprmd.N_JOINTS, 3))
    x[lh] = [1.0, 2.0, 3.0]
    x[rh] = [4.0, 5.0, 6.0]
    m = uiprmd.mirror_lr(x)
    assert np.allclose(m[rh], [1.0, 2.0, 3.0])  # left moved into right slot
    assert np.allclose(m[lh], [4.0, 5.0, 6.0])


# --------------------------------------------------------------------------
# Data-backed tests
# --------------------------------------------------------------------------
@needs_data
def test_discovery_counts_segmented_kinect():
    ds = uiprmd.UIPRMDDataset(segmented=True, system="kinect")
    correct = ds.samples(correct=True)
    incorrect = ds.samples(correct=False)
    # 10 movements x 10 subjects x 10 episodes, correct and incorrect.
    assert len(correct) == 1000
    assert len(incorrect) == 1000
    assert ds.subjects() == [f"s{i:02d}" for i in range(1, 11)]


@needs_data
def test_loso_splits_are_disjoint_and_complete():
    ds = uiprmd.UIPRMDDataset(segmented=True, movements=["m01"])
    total = len(ds.samples())
    folds = list(ds.loso_splits())
    assert len(folds) == 10
    for held_out, train, test in folds:
        assert all(s.subject == held_out for s in test)
        assert all(s.subject != held_out for s in train)
        assert len(train) + len(test) == total


@needs_data
def test_deep_squat_knee_flexes():
    # m01 deep squat: the knee interior angle must swing from near-straight to
    # deeply bent — a functional check that FK reconstruction is correct.
    ds = uiprmd.UIPRMDDataset(segmented=True, movements=["m01"])
    sample = next(s for s in ds.samples(correct=True) if s.subject == "s01")
    kps = sample.keypoints()
    assert kps.shape[1:] == (uiprmd.N_JOINTS, 3)

    hip = joints_mod.resolve(uiprmd.SCHEME, "left_hip")
    knee = joints_mod.resolve(uiprmd.SCHEME, "left_knee")
    ankle = joints_mod.resolve(uiprmd.SCHEME, "left_ankle")
    angles = np.array(
        [calculate_angle(f[hip], f[knee], f[ankle]) for f in kps]
    )
    assert angles.max() > 150.0          # near-straight standing
    assert angles.min() < 90.0           # bent while squatting
    assert angles.max() - angles.min() > 60.0