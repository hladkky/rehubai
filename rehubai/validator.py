"""Finite-state-machine exercise validator (the core algorithm).

Implements the per-frame validation function ``F: V x C -> R`` from
``MainAlgorithm.md``: given a stream of pose keypoints and an
:class:`~rehubai.config.ExerciseConfig`, it tracks the current stage, detects the
three deviation classes simultaneously, counts completed cycles, and produces
per-frame feedback.

Deviation classes (all checked every frame):
    * **GEOMETRIC**  — a joint angle / coordinate check violates its stage band.
    * **SEQUENTIAL** — a forbidden or out-of-order stage transition.
    * **DYNAMIC**    — a stage transition faster/slower than its tempo bounds.

The validator is skeleton-scheme agnostic: joint names in the config are resolved
to keypoint indices via :mod:`rehubai.joints`, so the same config validates
BlazePose, Kinect, or COCO keypoint arrays.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import joints as joints_mod
from .angles import calculate_angle, confidence_ok
from .config import (
    AngleConstraint,
    CoordinateCheck,
    ExerciseConfig,
    GlobalConstraint,
    JointRef,
    Stage,
)
from .quality import compute_quality

TRANSITION = "TRANSITION"

# Violation category tags.
GEOMETRIC = "GEOMETRIC"
SEQUENTIAL = "SEQUENTIAL"
DYNAMIC = "DYNAMIC"
GLOBAL = "GLOBAL"


@dataclass
class FrameResult:
    """Outcome of validating a single frame."""

    frame_idx: int
    stage: Optional[str]
    violations: List[dict] = field(default_factory=list)
    feedback: List[str] = field(default_factory=list)
    completed_cycles: int = 0
    paused: bool = False  # True when skipped due to low keypoint visibility


class ExerciseValidator:
    """Stateful validator for one exercise config over a keypoint stream."""

    def __init__(
        self,
        config: ExerciseConfig,
        scheme: Optional[str] = None,
        smoothing_window: int = 5,
    ) -> None:
        """Create a validator.

        Args:
            config: Parsed exercise configuration.
            scheme: Skeleton scheme of the incoming keypoints. Defaults to
                ``config.default_scheme``.
            smoothing_window: Length of the majority-vote window that stabilises
                stage detection against per-frame jitter.
        """
        self.config = config
        self.scheme = scheme or config.default_scheme
        # Fail fast if the scheme cannot express this exercise's joints.
        missing = [
            n for n in config.all_joint_names()
            if not joints_mod.supports(self.scheme, [n])
        ]
        if missing:
            raise ValueError(
                f"Scheme '{self.scheme}' lacks joints required by "
                f"'{config.exercise_id}': {missing}"
            )
        self.smoothing_window = smoothing_window
        self.reset()

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all accumulated state (call before re-processing a stream)."""
        self.current_stage: Optional[str] = None
        self.stage_start_frame: int = 0
        self.stage_history: List[tuple] = []  # (stage_id, frame_idx)
        self.completed_cycles: int = 0
        self.violations: List[dict] = []
        self.feedback_log: List[tuple] = []  # (frame_idx, level, message)
        self.stage_durations: Dict[str, List[float]] = {}
        self._stage_buffer: deque = deque(maxlen=self.smoothing_window)
        self._baselines: Dict[str, float] = {}
        self._frame_counter: int = 0

    # ------------------------------------------------------------------
    # Keypoint helpers
    # ------------------------------------------------------------------
    def _index(self, ref: JointRef) -> int:
        return ref if isinstance(ref, int) else joints_mod.resolve(self.scheme, ref)

    def _point(self, kps: np.ndarray, ref: JointRef) -> np.ndarray:
        return np.asarray(kps[self._index(ref)], dtype=float)

    def _angle_of(self, kps: np.ndarray, c: AngleConstraint) -> float:
        a, b, d = (self._point(kps, j) for j in c.joints)
        return calculate_angle(a, b, d)

    def _indices(self, refs: Sequence[JointRef]) -> List[int]:
        return [self._index(r) for r in refs]

    # ------------------------------------------------------------------
    # Stage detection
    # ------------------------------------------------------------------
    def _coordinate_ok(self, kps: np.ndarray, check: CoordinateCheck) -> bool:
        pts = [self._point(kps, p) for p in check.points]
        if check.type == "y_above":
            return pts[0][1] < pts[1][1] - check.min_diff
        if check.type == "y_below":
            return pts[0][1] > pts[1][1] + check.min_diff
        dist = float(np.linalg.norm(pts[0] - pts[1]))
        if check.type == "distance_gt":
            return dist > check.threshold
        if check.type == "distance_lt":
            return dist < check.threshold
        raise ValueError(f"Unknown coordinate check type '{check.type}'")

    def _stage_score(
        self,
        kps: np.ndarray,
        stage: Stage,
        confidences: Optional[Sequence[float]],
    ) -> float:
        """Fraction of this stage's checks that the pose satisfies (0..1).

        Checks whose keypoints are below the visibility threshold are ignored
        (neither satisfied nor counted), so occlusion lowers confidence rather
        than fabricating a mismatch.
        """
        satisfied = 0
        total = 0
        vis = self.config.min_keypoint_visibility
        for ac in stage.angle_constraints:
            if not confidence_ok(confidences, self._indices(ac.joints), vis):
                continue
            total += 1
            if ac.satisfies(self._angle_of(kps, ac)):
                satisfied += 1
        for cc in stage.coordinate_checks:
            if not confidence_ok(confidences, self._indices(cc.points), vis):
                continue
            total += 1
            if self._coordinate_ok(kps, cc):
                satisfied += 1
        return satisfied / total if total else 0.0

    def detect_stage(
        self, kps: np.ndarray, confidences: Optional[Sequence[float]] = None
    ) -> str:
        """Return the best-matching stage id, or ``TRANSITION`` if none is confident."""
        best_stage = TRANSITION
        best_score = self.config.detection_threshold
        for sid, stage in self.config.stages.items():
            score = self._stage_score(kps, stage, confidences)
            if score >= best_score:
                # Strictly greater keeps the first-declared stage on ties.
                if score > best_score or best_stage == TRANSITION:
                    best_score = score
                    best_stage = sid
        return best_stage

    # ------------------------------------------------------------------
    # Per-stage geometric validation
    # ------------------------------------------------------------------
    def validate_current_stage(
        self, kps: np.ndarray, confidences: Optional[Sequence[float]] = None
    ) -> List[dict]:
        """Return GEOMETRIC violations for the pose against the current stage."""
        if self.current_stage is None or self.current_stage == TRANSITION:
            return []
        stage = self.config.stages[self.current_stage]
        vis = self.config.min_keypoint_visibility
        out: List[dict] = []
        for ac in stage.angle_constraints:
            if not confidence_ok(confidences, self._indices(ac.joints), vis):
                continue
            angle = self._angle_of(kps, ac)
            if not ac.satisfies(angle):
                dev = ac.deviation(angle)
                direction = "зменшити" if dev > 0 else "збільшити"
                out.append({
                    "type": GEOMETRIC,
                    "constraint": ac.name,
                    "stage": self.current_stage,
                    "measured": round(angle, 1),
                    "band": [ac.min_angle, ac.max_angle],
                    "deviation": round(dev, 1),
                    "feedback": (
                        f"Кут '{ac.name}': {direction} на {abs(dev):.0f}° "
                        f"(зараз {angle:.0f}°, треба {ac.min_angle:.0f}–{ac.max_angle:.0f}°)"
                    ),
                })
        for cc in stage.coordinate_checks:
            if not confidence_ok(confidences, self._indices(cc.points), vis):
                continue
            if not self._coordinate_ok(kps, cc):
                out.append({
                    "type": GEOMETRIC,
                    "constraint": cc.name,
                    "stage": self.current_stage,
                    "feedback": f"Порушено умову '{cc.name}'.",
                })
        return out

    # ------------------------------------------------------------------
    # Global constraints (symmetry / stability / alignment)
    # ------------------------------------------------------------------
    def validate_global_constraints(
        self, kps: np.ndarray, confidences: Optional[Sequence[float]] = None
    ) -> List[dict]:
        vis = self.config.min_keypoint_visibility
        out: List[dict] = []
        for gc in self.config.global_constraints:
            if not confidence_ok(confidences, self._indices(gc.points), vis):
                continue
            violation = self._check_global(kps, gc)
            if violation is not None:
                out.append(violation)
        return out

    def _check_global(
        self, kps: np.ndarray, gc: GlobalConstraint
    ) -> Optional[dict]:
        pts = [self._point(kps, p) for p in gc.points]
        if gc.type == "symmetry":
            axis = 1 if gc.axis == "y" else 0
            diff = abs(pts[0][axis] - pts[1][axis])
            if diff > gc.threshold:
                return {
                    "type": GLOBAL, "subtype": "symmetry", "constraint": gc.name,
                    "value": round(float(diff), 1), "threshold": gc.threshold,
                    "feedback": f"Асиметрія '{gc.name}': {diff:.0f} (допуск {gc.threshold:.0f}). Вирівняйте.",
                }
        elif gc.type == "stability":
            dist = float(np.linalg.norm(pts[0] - pts[1]))
            base = self._baselines.setdefault(gc.name, dist)
            if base > 1e-8:
                dev = abs(dist - base) / base
                if dev > gc.threshold:
                    return {
                        "type": GLOBAL, "subtype": "stability", "constraint": gc.name,
                        "value": round(dev, 3), "threshold": gc.threshold,
                        "feedback": f"Нестабільність '{gc.name}': {dev*100:.0f}% (допуск {gc.threshold*100:.0f}%). Утримуйте позицію.",
                    }
        elif gc.type == "alignment":
            v1 = pts[1] - pts[0]
            v2 = pts[2] - pts[0]
            n = float(np.linalg.norm(v1) * np.linalg.norm(v2))
            score = abs(float(np.dot(v1, v2)) / n) if n > 1e-8 else 0.0
            if score < gc.threshold:
                return {
                    "type": GLOBAL, "subtype": "alignment", "constraint": gc.name,
                    "value": round(score, 3), "threshold": gc.threshold,
                    "feedback": f"Недостатнє вирівнювання '{gc.name}'. Тримайте пряму лінію.",
                }
        else:
            raise ValueError(f"Unknown global constraint type '{gc.type}'")
        return None

    # ------------------------------------------------------------------
    # Transition validation (sequential + dynamic)
    # ------------------------------------------------------------------
    def validate_transition(
        self, src: Optional[str], dst: str, elapsed_frames: int
    ) -> dict:
        """Validate a stage change ``src -> dst``.

        Returns a dict with ``valid`` and, if invalid, ``type``/``feedback``.
        The first transition (``src is None``) is always accepted.
        """
        if src is None:
            return {"valid": True}
        if self.config.is_forbidden(src, dst):
            return {
                "valid": False, "type": SEQUENTIAL,
                "feedback": f"Недозволений перехід {src} → {dst}. Спершу поверніться в початкову позицію.",
            }
        transition = self.config.transition(src, dst)
        if transition is None:
            return {
                "valid": False, "type": SEQUENTIAL,
                "feedback": f"Неочікуваний перехід {src} → {dst}. Дотримуйтесь послідовності етапів.",
            }
        elapsed = elapsed_frames / max(self.config.fps, 1)
        if elapsed < transition.min_sec:
            return {
                "valid": False, "type": DYNAMIC,
                "feedback": f"Занадто швидко ({elapsed:.1f}с): уповільніть перехід {src} → {dst}.",
                "elapsed": round(elapsed, 2),
            }
        if elapsed > transition.max_sec:
            return {
                "valid": False, "type": DYNAMIC,
                "feedback": f"Занадто повільно ({elapsed:.1f}с): пришвидшіть перехід {src} → {dst}.",
                "elapsed": round(elapsed, 2),
            }
        return {"valid": True, "elapsed": round(elapsed, 2)}

    # ------------------------------------------------------------------
    # Cycle completion
    # ------------------------------------------------------------------
    def check_cycle_completion(self) -> bool:
        """True if the tail of the (dedup) stage history equals the ideal cycle."""
        seq = self.config.cycle_sequence
        if len(self.stage_history) < len(seq):
            return False
        recent = [s for s, _ in self.stage_history[-len(seq):]]
        return recent == seq

    # ------------------------------------------------------------------
    # Main per-frame entry point
    # ------------------------------------------------------------------
    def process_frame(
        self,
        kps: np.ndarray,
        confidences: Optional[Sequence[float]] = None,
        frame_idx: Optional[int] = None,
    ) -> FrameResult:
        """Validate one frame and advance the FSM. Returns a :class:`FrameResult`."""
        if frame_idx is None:
            frame_idx = self._frame_counter
        self._frame_counter = frame_idx + 1

        result = FrameResult(frame_idx=frame_idx, stage=self.current_stage)

        # Step 1: visibility gate — pause when too many critical joints are lost.
        if confidences is not None and self._too_occluded(confidences):
            result.paused = True
            result.stage = self.current_stage
            result.feedback.append("Втрата видимості опорних точок — переконайтесь, що все тіло в кадрі.")
            self.feedback_log.append((frame_idx, "WARNING", result.feedback[-1]))
            return result

        # Step 2/3: detect + temporally smooth the stage.
        raw = self.detect_stage(kps, confidences)
        self._stage_buffer.append(raw)
        smoothed = Counter(self._stage_buffer).most_common(1)[0][0]

        # Step 4: handle a settled transition into a *named* stage.
        if smoothed != TRANSITION and smoothed != self.current_stage:
            elapsed_frames = frame_idx - self.stage_start_frame
            check = self.validate_transition(self.current_stage, smoothed, elapsed_frames)
            if not check["valid"]:
                self._record(frame_idx, check["type"], check, result)
            self._advance_stage(smoothed, frame_idx)
            if self.check_cycle_completion():
                self.completed_cycles += 1
                msg = f"Цикл {self.completed_cycles} завершено успішно!"
                result.feedback.append(msg)
                self.feedback_log.append((frame_idx, "SUCCESS", msg))

        result.stage = self.current_stage

        # Step 5/6: validate the current pose (geometry + global constraints).
        for v in self.validate_current_stage(kps, confidences):
            self._record(frame_idx, GEOMETRIC, v, result)
        for v in self.validate_global_constraints(kps, confidences):
            self._record(frame_idx, GLOBAL, v, result)

        result.completed_cycles = self.completed_cycles
        return result

    # ------------------------------------------------------------------
    def _too_occluded(self, confidences: Sequence[float]) -> bool:
        idxs = self._indices(self.config.all_joint_names())
        if not idxs:
            return False
        visible = sum(
            1 for i in idxs if float(confidences[i]) >= self.config.min_keypoint_visibility
        )
        return visible < 0.5 * len(idxs)

    def _advance_stage(self, stage_id: str, frame_idx: int) -> None:
        if self.current_stage is not None:
            dur = (frame_idx - self.stage_start_frame) / max(self.config.fps, 1)
            self.stage_durations.setdefault(self.current_stage, []).append(dur)
        self.current_stage = stage_id
        self.stage_start_frame = frame_idx
        self.stage_history.append((stage_id, frame_idx))

    def _record(self, frame_idx: int, level: str, violation: dict, result: FrameResult) -> None:
        entry = {"frame": frame_idx, **violation}
        self.violations.append(entry)
        result.violations.append(entry)
        if "feedback" in violation:
            result.feedback.append(violation["feedback"])
            self.feedback_log.append((frame_idx, level, violation["feedback"]))

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def generate_report(self) -> dict:
        """Aggregate a JSON-serialisable report over the processed stream."""
        by_type: Counter = Counter(v.get("type", "UNKNOWN") for v in self.violations)
        transitions_done = max(len(self.stage_history) - 1, 0)
        quality = compute_quality(
            n_cycles=self.completed_cycles,
            n_violations=len(self.violations),
            transitions=transitions_done,
            weights=self.config.quality_weights,
        )
        durations = {
            sid: {
                "mean": round(float(np.mean(v)), 2),
                "std": round(float(np.std(v)), 2),
                "count": len(v),
            }
            for sid, v in self.stage_durations.items()
        }
        return {
            "exercise": self.config.exercise_id,
            "frames_processed": self._frame_counter,
            "fps": self.config.fps,
            "completed_cycles": self.completed_cycles,
            "total_violations": len(self.violations),
            "violations_by_type": dict(by_type),
            "stage_durations": durations,
            "quality_score": round(quality * 100, 1),
            "timeline": [{"stage": s, "frame": f} for s, f in self.stage_history],
        }
