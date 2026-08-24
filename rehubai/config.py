"""Exercise configuration model.

An exercise is described entirely by a declarative JSON config — no per-exercise
model training. This module parses and validates that config into typed objects
consumed by :class:`rehubai.validator.ExerciseValidator`.

The schema mirrors the formal model ``E = (S, Theta, T, G)`` from
``MainAlgorithm.md``:

* ``stages``               -> S (ordered discrete poses)
* ``angle_constraints``    -> Theta (geometric constraints)
* ``transitions`` / ``forbidden_transitions`` -> T (allowed/forbidden, with tempo)
* ``global_constraints``   -> G (symmetry / stability / alignment)
* ``cycle_sequence``       -> sigma (the ideal cycle)

Joints are referenced by *semantic name* (see :mod:`rehubai.joints`) so one
config works across skeleton schemes. Raw integer indices are also accepted for
convenience/back-compat.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

JointRef = Union[str, int]


class ConfigError(ValueError):
    """Raised when an exercise config is missing fields or is malformed."""


def _require(d: dict, key: str, ctx: str):
    if key not in d:
        raise ConfigError(f"{ctx}: missing required field '{key}'")
    return d[key]


@dataclass(frozen=True)
class AngleConstraint:
    """Geometric constraint on the angle at ``joints[1]``.

    Accepts either an ``ideal`` + ``tolerance`` pair or an explicit ``[min, max]``
    acceptance band (``min_angle``/``max_angle``). Internally both are normalised
    to a band so ``satisfies`` / ``deviation`` are uniform.
    """

    joints: Tuple[JointRef, JointRef, JointRef]
    name: str
    min_angle: float
    max_angle: float

    @property
    def center(self) -> float:
        return (self.min_angle + self.max_angle) / 2.0

    def deviation(self, angle: float) -> float:
        """Signed distance to the acceptance band (0 inside, else nearest edge)."""
        if angle < self.min_angle:
            return angle - self.min_angle  # negative: open the joint
        if angle > self.max_angle:
            return angle - self.max_angle  # positive: close the joint
        return 0.0

    def satisfies(self, angle: float) -> bool:
        return self.min_angle <= angle <= self.max_angle

    @staticmethod
    def from_dict(d: dict, ctx: str) -> "AngleConstraint":
        joints = _require(d, "joints", ctx)
        if len(joints) != 3:
            raise ConfigError(f"{ctx}: 'joints' must have exactly 3 entries")
        name = d.get("name", d.get("description", "angle"))
        if "min" in d and "max" in d:
            lo, hi = float(d["min"]), float(d["max"])
        else:
            ideal = float(_require(d, "ideal", ctx))
            tol = float(_require(d, "tolerance", ctx))
            lo, hi = ideal - tol, ideal + tol
        if lo > hi:
            raise ConfigError(f"{ctx}: angle band has min>max for '{name}'")
        return AngleConstraint(tuple(joints), name, lo, hi)


@dataclass(frozen=True)
class CoordinateCheck:
    """Non-angular positional check within a stage.

    Supported ``type`` values:
        * ``y_above`` — ``points[0].y < points[1].y - min_diff`` (image y grows down,
          so smaller y is higher on screen).
        * ``y_below`` — ``points[0].y > points[1].y + min_diff``.
        * ``distance_gt`` / ``distance_lt`` — Euclidean distance vs ``threshold``.
    """

    type: str
    points: Tuple[JointRef, ...]
    name: str
    min_diff: float = 0.0
    threshold: float = 0.0

    @staticmethod
    def from_dict(d: dict, ctx: str) -> "CoordinateCheck":
        ctype = _require(d, "type", ctx)
        points = tuple(_require(d, "points", ctx))
        name = d.get("name", d.get("description", ctype))
        return CoordinateCheck(
            type=ctype,
            points=points,
            name=name,
            min_diff=float(d.get("min_diff", 0.0)),
            threshold=float(d.get("threshold", 0.0)),
        )


@dataclass(frozen=True)
class GlobalConstraint:
    """Constraint that holds throughout the whole exercise (set G).

    ``type`` is one of ``symmetry`` | ``stability`` | ``alignment``.
    """

    type: str
    points: Tuple[JointRef, ...]
    name: str
    threshold: float
    axis: str = "y"  # for symmetry: which coordinate to compare

    @staticmethod
    def from_dict(d: dict, ctx: str) -> "GlobalConstraint":
        ctype = _require(d, "type", ctx)
        points = tuple(_require(d, "points", ctx))
        name = d.get("name", d.get("description", ctype))
        threshold = float(_require(d, "threshold", ctx))
        return GlobalConstraint(ctype, points, name, threshold, d.get("axis", "y"))


@dataclass(frozen=True)
class Transition:
    """Allowed transition between two stages with tempo bounds (seconds)."""

    src: str
    dst: str
    min_sec: float
    max_sec: float

    @staticmethod
    def from_dict(d: dict, ctx: str) -> "Transition":
        return Transition(
            src=_require(d, "from", ctx),
            dst=_require(d, "to", ctx),
            min_sec=float(d.get("min_sec", 0.0)),
            max_sec=float(d.get("max_sec", float("inf"))),
        )


@dataclass
class Stage:
    """A discrete biomechanical stage (element of S)."""

    id: str
    angle_constraints: List[AngleConstraint] = field(default_factory=list)
    coordinate_checks: List[CoordinateCheck] = field(default_factory=list)
    min_hold_sec: float = 0.0
    max_hold_sec: Optional[float] = None

    @property
    def total_checks(self) -> int:
        return len(self.angle_constraints) + len(self.coordinate_checks)

    @staticmethod
    def from_dict(d: dict, ctx: str) -> "Stage":
        sid = _require(d, "id", ctx)
        actx = f"{ctx}.stage[{sid}]"
        angles = [
            AngleConstraint.from_dict(a, actx)
            for a in d.get("angle_constraints", [])
        ]
        coords = [
            CoordinateCheck.from_dict(c, actx)
            for c in d.get("coordinate_checks", [])
        ]
        return Stage(
            id=sid,
            angle_constraints=angles,
            coordinate_checks=coords,
            min_hold_sec=float(d.get("min_hold_sec", 0.0)),
            max_hold_sec=(
                float(d["max_hold_sec"]) if d.get("max_hold_sec") is not None else None
            ),
        )


@dataclass
class ExerciseConfig:
    """Parsed, validated exercise configuration."""

    exercise_id: str
    name: str
    stages: Dict[str, Stage]
    transitions: List[Transition]
    forbidden_transitions: List[Tuple[str, str]]
    global_constraints: List[GlobalConstraint]
    cycle_sequence: List[str]
    default_scheme: str = "blazepose33"
    fps: int = 30
    detection_threshold: float = 0.7
    min_keypoint_visibility: float = 0.5
    quality_weights: Tuple[float, float] = (1.0, 0.5)  # (w_v, w_t)

    # -- lookups -----------------------------------------------------------
    def transition(self, src: str, dst: str) -> Optional[Transition]:
        for t in self.transitions:
            if t.src == src and t.dst == dst:
                return t
        return None

    def is_forbidden(self, src: str, dst: str) -> bool:
        return (src, dst) in self.forbidden_transitions

    def all_joint_names(self) -> List[str]:
        """Every semantic joint name referenced anywhere in the config."""
        names: set = set()
        for stage in self.stages.values():
            for ac in stage.angle_constraints:
                names.update(j for j in ac.joints if isinstance(j, str))
            for cc in stage.coordinate_checks:
                names.update(j for j in cc.points if isinstance(j, str))
        for gc in self.global_constraints:
            names.update(j for j in gc.points if isinstance(j, str))
        return sorted(names)

    # -- construction ------------------------------------------------------
    @staticmethod
    def from_dict(d: dict) -> "ExerciseConfig":
        ctx = "config"
        exercise_id = _require(d, "exercise_id", ctx)
        stages_raw = _require(d, "stages", ctx)
        if not stages_raw:
            raise ConfigError(f"{ctx}: 'stages' must be non-empty")
        stages = {}
        for s in stages_raw:
            stage = Stage.from_dict(s, ctx)
            if stage.id in stages:
                raise ConfigError(f"{ctx}: duplicate stage id '{stage.id}'")
            stages[stage.id] = stage

        transitions = [
            Transition.from_dict(t, ctx) for t in d.get("transitions", [])
        ]
        forbidden = [
            (_require(t, "from", ctx), _require(t, "to", ctx))
            for t in d.get("forbidden_transitions", [])
        ]
        globals_ = [
            GlobalConstraint.from_dict(g, ctx)
            for g in d.get("global_constraints", [])
        ]
        cycle = _require(d, "cycle_sequence", ctx)

        # Referential integrity: every stage named in transitions/cycle exists.
        known = set(stages)
        for t in transitions:
            for sid in (t.src, t.dst):
                if sid not in known:
                    raise ConfigError(f"{ctx}: transition references unknown stage '{sid}'")
        for sid in cycle:
            if sid not in known:
                raise ConfigError(f"{ctx}: cycle_sequence references unknown stage '{sid}'")

        qw = d.get("quality_weights", {})
        return ExerciseConfig(
            exercise_id=exercise_id,
            name=d.get("name", exercise_id),
            stages=stages,
            transitions=transitions,
            forbidden_transitions=forbidden,
            global_constraints=globals_,
            cycle_sequence=list(cycle),
            default_scheme=d.get("default_scheme", "blazepose33"),
            fps=int(d.get("fps", 30)),
            detection_threshold=float(d.get("detection_threshold", 0.7)),
            min_keypoint_visibility=float(d.get("min_keypoint_visibility", 0.5)),
            quality_weights=(
                float(qw.get("w_v", 1.0)),
                float(qw.get("w_t", 0.5)),
            ),
        )

    @staticmethod
    def load(path: Union[str, Path]) -> "ExerciseConfig":
        """Load and validate an exercise config from a JSON file."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return ExerciseConfig.from_dict(data)
