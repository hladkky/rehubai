"""Geometric primitives: joint angles from keypoints.

Implements the relative-angle formulation that underpins the geometric-deviation
check (see ``MainAlgorithm.md`` §3.6). Using the angle between two bone vectors,
rather than absolute coordinates, makes the check invariant to camera position
and to patient anthropometrics.

For three points ``a, b, c`` the angle at the middle vertex ``b`` is::

    v1 = a - b
    v2 = c - b
    theta = arccos( (v1 . v2) / (|v1| * |v2|) )

All functions accept 2D ``(x, y)`` or 3D ``(x, y, z)`` points as array-likes and
work identically; supplying the MediaPipe z-coordinate yields a true 3D angle.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# Guard against divide-by-zero when a bone vector has (near) zero length.
_EPS = 1e-8


def calculate_angle(
    point1: Sequence[float],
    point2: Sequence[float],
    point3: Sequence[float],
) -> float:
    """Angle (degrees) at ``point2`` formed by ``point1``->``point2``->``point3``.

    Args:
        point1: First point (bone endpoint).
        point2: Middle point (the vertex the angle is measured at).
        point3: Third point (bone endpoint).

    Returns:
        Angle in degrees in ``[0, 180]``. Returns ``0.0`` if either bone vector
        is degenerate (zero length), matching the reference implementation.
    """
    p1 = np.asarray(point1, dtype=float)
    p2 = np.asarray(point2, dtype=float)
    p3 = np.asarray(point3, dtype=float)

    v1 = p1 - p2
    v2 = p3 - p2

    norm1 = float(np.linalg.norm(v1))
    norm2 = float(np.linalg.norm(v2))
    if norm1 < _EPS or norm2 < _EPS:
        return 0.0

    cos_angle = float(np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def vector_angle(
    origin: Sequence[float],
    tip_a: Sequence[float],
    tip_b: Sequence[float],
) -> float:
    """Angle (degrees) between vectors ``origin->tip_a`` and ``origin->tip_b``.

    Convenience wrapper for checks framed as "arm relative to torso" where the two
    rays share an origin (e.g. shoulder) but go to non-adjacent joints. Equivalent
    to :func:`calculate_angle` with the shared point in the middle.
    """
    return calculate_angle(tip_a, origin, tip_b)


def angle_deviation(measured: float, ideal: float) -> float:
    """Signed deviation ``measured - ideal`` in degrees.

    A positive result means the joint is *more open* than the reference and should
    be closed; negative means it should be opened further.
    """
    return float(measured) - float(ideal)


def within_tolerance(measured: float, ideal: float, tolerance: float) -> bool:
    """Return True if ``|measured - ideal| <= tolerance`` (the geometric check)."""
    return abs(angle_deviation(measured, ideal)) <= tolerance


def confidence_ok(
    confidences: Optional[Sequence[float]],
    indices: Sequence[int],
    min_visibility: float,
) -> bool:
    """Return True if every keypoint in ``indices`` is visible enough.

    Used to skip angle checks that rely on occluded/low-confidence joints, which
    reduces false-positive violations. If ``confidences`` is ``None`` the check is
    treated as passing (e.g. datasets that provide clean skeletons without scores).

    Args:
        confidences: Per-keypoint visibility/confidence in ``[0, 1]`` or ``None``.
        indices: Keypoint indices the angle depends on.
        min_visibility: Minimum acceptable confidence for each involved keypoint.
    """
    if confidences is None:
        return True
    return all(float(confidences[i]) >= min_visibility for i in indices)
