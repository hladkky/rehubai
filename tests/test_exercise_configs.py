"""Tests for the UI-PRMD exercise configs (deep squat, sit-to-stand, leg raise,
shoulder abduction).

Pure-schema tests always run; the cycle-detection tests need the UI-PRMD archive
and are skipped without it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rehubai import joints as joints_mod
from rehubai.config import ExerciseConfig
from rehubai.datasets import uiprmd
from rehubai.validator import ExerciseValidator

CFG_DIR = Path(__file__).resolve().parents[1] / "configs" / "exercises"

# config name -> UI-PRMD movement code it is authored against
UIPRMD_CONFIGS = {
    "deep_squat": "m01",
    "sit_to_stand": "m05",
    "leg_raise": "m06",
    "shoulder_abduction": "m07",
}

DATA_ROOT = uiprmd.default_root()
needs_data = pytest.mark.skipif(not DATA_ROOT.exists(), reason="UI-PRMD archive not present")


# --------------------------------------------------------------------------
# Schema tests (no data)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(UIPRMD_CONFIGS))
def test_config_loads(name):
    cfg = ExerciseConfig.load(CFG_DIR / f"{name}.json")
    assert cfg.exercise_id == name
    assert len(cfg.stages) >= 2
    assert cfg.cycle_sequence
    assert set(cfg.cycle_sequence) <= set(cfg.stages)


@pytest.mark.parametrize("name", list(UIPRMD_CONFIGS))
def test_config_is_scheme_universal(name):
    # The core novelty: one config resolves on both the dataset skeleton
    # (Kinect-22) and the deployment skeleton (BlazePose-33) with no changes.
    cfg = ExerciseConfig.load(CFG_DIR / f"{name}.json")
    joints_needed = cfg.all_joint_names()
    assert joints_needed  # config actually references joints
    assert joints_mod.supports("uiprmd_kinect22", joints_needed)
    assert joints_mod.supports("blazepose33", joints_needed)


@pytest.mark.parametrize("name", list(UIPRMD_CONFIGS))
def test_config_is_angle_only_hence_frame_invariant(name):
    # These configs must avoid coordinate/global checks so they are invariant to
    # the coordinate frame (UI-PRMD +y up, mm vs MediaPipe conventions).
    cfg = ExerciseConfig.load(CFG_DIR / f"{name}.json")
    assert not cfg.global_constraints
    for stage in cfg.stages.values():
        assert not stage.coordinate_checks
        assert stage.angle_constraints


# --------------------------------------------------------------------------
# Data-backed behaviour
# --------------------------------------------------------------------------
@needs_data
@pytest.mark.parametrize("name,mv", list(UIPRMD_CONFIGS.items()))
def test_config_completes_cycle_on_correct(name, mv):
    cfg = ExerciseConfig.load(CFG_DIR / f"{name}.json")
    ds = uiprmd.UIPRMDDataset(segmented=True, movements=[mv])
    left = uiprmd.LEFT_WORKING_SUBJECTS.get(mv, frozenset())

    detected = total = 0
    for s in ds.samples(correct=True):
        if s.episode != "e01":  # one episode per subject keeps the test fast
            continue
        kps = s.keypoints()
        if s.subject in left:
            kps = uiprmd.mirror_lr(kps)
        v = ExerciseValidator(cfg, scheme=uiprmd.SCHEME)
        for f in kps:
            v.process_frame(f, confidences=None)
        detected += v.completed_cycles >= 1
        total += 1

    assert total > 0
    # Stage FSM fires and a full cycle completes for the large majority of
    # correct repetitions across all 10 subjects.
    assert detected / total >= 0.7