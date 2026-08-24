"""Unit tests for exercise-config parsing and validation."""

from pathlib import Path

import pytest

from rehubai.config import AngleConstraint, ConfigError, ExerciseConfig

DEAD_BUG = Path(__file__).resolve().parents[1] / "configs" / "exercises" / "dead_bug.json"


def test_load_dead_bug_config():
    cfg = ExerciseConfig.load(DEAD_BUG)
    assert cfg.exercise_id == "dead_bug"
    assert set(cfg.stages) == {"REST", "LEFT_PHASE", "RIGHT_PHASE"}
    assert cfg.cycle_sequence == ["REST", "LEFT_PHASE", "REST", "RIGHT_PHASE", "REST"]
    assert cfg.is_forbidden("LEFT_PHASE", "RIGHT_PHASE") is True
    assert cfg.transition("REST", "LEFT_PHASE").max_sec == 2.0


def test_dead_bug_joints_are_semantic():
    cfg = ExerciseConfig.load(DEAD_BUG)
    names = cfg.all_joint_names()
    assert "left_elbow" in names and "right_ankle" in names


def test_angle_constraint_ideal_tolerance():
    ac = AngleConstraint.from_dict(
        {"joints": ["a", "b", "c"], "ideal": 180, "tolerance": 10, "name": "x"}, "ctx"
    )
    assert ac.min_angle == 170 and ac.max_angle == 190
    assert ac.satisfies(175) and not ac.satisfies(160)
    assert ac.deviation(160) == pytest.approx(-10.0)  # below band => open joint
    assert ac.deviation(200) == pytest.approx(10.0)   # above band => close joint
    assert ac.deviation(180) == 0.0


def test_angle_constraint_min_max_band():
    ac = AngleConstraint.from_dict(
        {"joints": ["a", "b", "c"], "min": 80, "max": 100, "name": "y"}, "ctx"
    )
    assert ac.center == 90
    assert ac.satisfies(90) and not ac.satisfies(120)


def test_missing_exercise_id_raises():
    with pytest.raises(ConfigError):
        ExerciseConfig.from_dict({"stages": [{"id": "S"}], "cycle_sequence": ["S"]})


def test_transition_to_unknown_stage_raises():
    with pytest.raises(ConfigError):
        ExerciseConfig.from_dict({
            "exercise_id": "e",
            "stages": [{"id": "A"}],
            "transitions": [{"from": "A", "to": "GHOST"}],
            "cycle_sequence": ["A"],
        })


def test_cycle_references_unknown_stage_raises():
    with pytest.raises(ConfigError):
        ExerciseConfig.from_dict({
            "exercise_id": "e",
            "stages": [{"id": "A"}],
            "cycle_sequence": ["A", "B"],
        })


def test_duplicate_stage_raises():
    with pytest.raises(ConfigError):
        ExerciseConfig.from_dict({
            "exercise_id": "e",
            "stages": [{"id": "A"}, {"id": "A"}],
            "cycle_sequence": ["A"],
        })
