"""Unit tests for the FSM ExerciseValidator (the core algorithm)."""

import math
from pathlib import Path

import numpy as np
import pytest

from rehubai.config import ExerciseConfig
from rehubai.validator import DYNAMIC, GEOMETRIC, GLOBAL, SEQUENTIAL, TRANSITION, ExerciseValidator

DEAD_BUG = Path(__file__).resolve().parents[1] / "configs" / "exercises" / "dead_bug.json"

# Synthetic 2-stage config using raw integer joint indices (scheme-independent).
TOY = {
    "exercise_id": "toy",
    "name": "Toy",
    "fps": 10,
    "detection_threshold": 0.7,
    "stages": [
        {"id": "A", "angle_constraints": [
            {"joints": [0, 1, 2], "ideal": 180, "tolerance": 10, "name": "straight"}]},
        {"id": "B", "angle_constraints": [
            {"joints": [0, 1, 2], "ideal": 90, "tolerance": 10, "name": "bent"}]},
    ],
    "transitions": [
        {"from": "A", "to": "B", "min_sec": 0.5, "max_sec": 2.0},
        {"from": "B", "to": "A", "min_sec": 0.5, "max_sec": 2.0},
    ],
    "cycle_sequence": ["A", "B", "A"],
}

STRAIGHT = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])   # angle at 1 == 180
BENT = np.array([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]])       # angle at 1 == 90
_A150 = [math.cos(math.radians(150)), math.sin(math.radians(150))]
SLIGHT = np.array([_A150, [0.0, 0.0], [1.0, 0.0]])          # angle at 1 == 150


def make_validator(cfg_dict=None, **kw):
    cfg = ExerciseConfig.from_dict(cfg_dict or TOY)
    return ExerciseValidator(cfg, smoothing_window=1, **kw)


def test_detect_stage():
    v = make_validator()
    assert v.detect_stage(STRAIGHT) == "A"
    assert v.detect_stage(BENT) == "B"
    assert v.detect_stage(SLIGHT) == TRANSITION  # matches neither band


def test_first_stage_entry_is_valid():
    v = make_validator()
    r = v.process_frame(STRAIGHT, frame_idx=0)
    assert r.stage == "A"
    assert r.violations == []


def test_geometric_violation_in_current_stage():
    v = make_validator()
    v.process_frame(STRAIGHT, frame_idx=0)        # enter A
    r = v.process_frame(SLIGHT, frame_idx=1)      # still A (150 not a stage), but off-band
    geo = [x for x in r.violations if x["type"] == GEOMETRIC]
    assert geo and geo[0]["constraint"] == "straight"
    assert geo[0]["deviation"] < 0  # need to open the joint (increase angle)


def test_completed_cycle():
    v = make_validator()
    v.process_frame(STRAIGHT, frame_idx=0)    # A
    v.process_frame(BENT, frame_idx=6)        # -> B (0.6s, in tempo)
    r = v.process_frame(STRAIGHT, frame_idx=12)  # -> A, completes A,B,A
    assert r.completed_cycles == 1
    assert v.completed_cycles == 1


def test_dynamic_too_fast_violation():
    v = make_validator()
    v.process_frame(STRAIGHT, frame_idx=0)     # A at t=0
    r = v.process_frame(BENT, frame_idx=2)     # -> B after 0.2s < 0.5s min
    dyn = [x for x in r.violations if x["type"] == DYNAMIC]
    assert dyn, "expected a DYNAMIC (too fast) violation"


def test_forbidden_transition_is_sequential_violation():
    cfg = dict(TOY)
    cfg = {**TOY, "forbidden_transitions": [{"from": "A", "to": "B"}]}
    v = make_validator(cfg)
    v.process_frame(STRAIGHT, frame_idx=0)     # A
    r = v.process_frame(BENT, frame_idx=6)     # A -> B is forbidden
    seq = [x for x in r.violations if x["type"] == SEQUENTIAL]
    assert seq, "expected a SEQUENTIAL (forbidden transition) violation"


def test_global_symmetry_violation():
    cfg = {
        "exercise_id": "g",
        "fps": 10,
        "stages": [{"id": "S", "angle_constraints": [
            {"joints": [0, 1, 2], "ideal": 180, "tolerance": 180, "name": "any"}]}],
        "global_constraints": [
            {"type": "symmetry", "points": [3, 4], "axis": "y", "threshold": 5, "name": "level"}],
        "cycle_sequence": ["S"],
    }
    v = make_validator(cfg)
    kps = np.array([[0, 0], [1, 0], [2, 0], [0, 0], [0, 20]], dtype=float)  # y diff 20 > 5
    r = v.process_frame(kps, frame_idx=0)
    glob = [x for x in r.violations if x["type"] == GLOBAL]
    assert glob and glob[0]["subtype"] == "symmetry"


def test_report_structure():
    v = make_validator()
    v.process_frame(STRAIGHT, frame_idx=0)
    v.process_frame(BENT, frame_idx=6)
    v.process_frame(STRAIGHT, frame_idx=12)
    report = v.generate_report()
    assert report["exercise"] == "toy"
    assert report["completed_cycles"] == 1
    assert "quality_score" in report
    assert report["timeline"][0]["stage"] == "A"


def test_reset_clears_state():
    v = make_validator()
    v.process_frame(STRAIGHT, frame_idx=0)
    v.reset()
    assert v.current_stage is None
    assert v.stage_history == []


# --- cross-scheme universality: one config, three skeleton schemes ---------
def test_dead_bug_builds_on_multiple_schemes():
    cfg = ExerciseConfig.load(DEAD_BUG)
    for scheme in ("blazepose33", "coco17", "kinect25"):
        val = ExerciseValidator(cfg, scheme=scheme)
        assert val.scheme == scheme


def test_scheme_missing_joint_raises():
    cfg = {
        "exercise_id": "needs_neck",
        "stages": [{"id": "S", "angle_constraints": [
            {"joints": ["neck", "left_shoulder", "left_hip"], "ideal": 90, "tolerance": 90, "name": "n"}]}],
        "cycle_sequence": ["S"],
    }
    c = ExerciseConfig.from_dict(cfg)
    with pytest.raises(ValueError):
        ExerciseValidator(c, scheme="coco17")  # coco17 has no neck
