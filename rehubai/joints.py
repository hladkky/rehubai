"""Semantic joint abstraction for cross-scheme pose validation.

The exercise-validation algorithm is designed to be independent of any specific
pose-estimation backend or skeleton convention. Exercise configs reference joints
by *semantic name* (e.g. ``"left_elbow"``); this module maps those names to the
integer keypoint indices used by a concrete skeleton scheme.

Supporting several schemes with one config is a core novelty of the method: a
single ``dead_bug.json`` validates equally on MediaPipe BlazePose video, on the
Kinect skeletons shipped with KIMORE, or on the reduced UI-PRMD marker set.

Schemes currently supported:
    * ``blazepose33`` — MediaPipe Pose / BlazePose (33 keypoints).
    * ``kinect25``    — Microsoft Kinect v2 body skeleton (25 joints).
    * ``coco17``      — COCO / RTMPose / YOLO-Pose top-down (17 keypoints).

Additional dataset-specific schemes (e.g. ``uiprmd_kinect22``) are registered by
the corresponding :mod:`rehubai.datasets` adapter via :func:`register_scheme`.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

# ---------------------------------------------------------------------------
# Canonical semantic joint vocabulary
# ---------------------------------------------------------------------------
# Every scheme map below uses names drawn from this set. Configs must reference
# joints using these names so that a constraint is portable across schemes.
SEMANTIC_JOINTS: List[str] = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    # Optional / scheme-dependent joints (not present in every scheme).
    "left_heel",
    "right_heel",
    "left_foot",
    "right_foot",
    "neck",
    "spine_mid",
    "spine_base",
    "head",
]

# ---------------------------------------------------------------------------
# Scheme -> {semantic_name: index}
# ---------------------------------------------------------------------------
# MediaPipe BlazePose (33 keypoints).
_BLAZEPOSE33: Dict[str, int] = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot": 31,
    "right_foot": 32,
}

# Microsoft Kinect v2 (25 joints). Used by KIMORE / IntelliRehabDS.
_KINECT25: Dict[str, int] = {
    "spine_base": 0,
    "spine_mid": 1,
    "neck": 2,
    "head": 3,
    "left_shoulder": 4,
    "left_elbow": 5,
    "left_wrist": 6,
    "right_shoulder": 8,
    "right_elbow": 9,
    "right_wrist": 10,
    "left_hip": 12,
    "left_knee": 13,
    "left_ankle": 14,
    "left_foot": 15,
    "right_hip": 16,
    "right_knee": 17,
    "right_ankle": 18,
    "right_foot": 19,
}

# COCO 17 keypoints (RTMPose / YOLO-Pose / OpenPose body).
_COCO17: Dict[str, int] = {
    "nose": 0,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

_SCHEMES: Dict[str, Dict[str, int]] = {
    "blazepose33": _BLAZEPOSE33,
    "kinect25": _KINECT25,
    "coco17": _COCO17,
}


def available_schemes() -> List[str]:
    """Return the names of all registered skeleton schemes."""
    return sorted(_SCHEMES)


def register_scheme(name: str, mapping: Dict[str, int]) -> None:
    """Register a new skeleton scheme (e.g. a dataset-specific marker set).

    Args:
        name: Scheme identifier used by :class:`~rehubai.validator.ExerciseValidator`.
        mapping: Map of semantic joint name -> integer keypoint index. Every key
            must belong to :data:`SEMANTIC_JOINTS`.

    Raises:
        ValueError: If ``mapping`` contains a name outside the canonical vocabulary.
    """
    unknown = set(mapping) - set(SEMANTIC_JOINTS)
    if unknown:
        raise ValueError(
            f"Scheme '{name}' references unknown semantic joints: {sorted(unknown)}"
        )
    _SCHEMES[name] = dict(mapping)


def scheme_map(scheme: str) -> Dict[str, int]:
    """Return the full ``{semantic_name: index}`` map for a scheme.

    Raises:
        KeyError: If ``scheme`` is not registered.
    """
    try:
        return _SCHEMES[scheme]
    except KeyError:
        raise KeyError(
            f"Unknown skeleton scheme '{scheme}'. Available: {available_schemes()}"
        )


def resolve(scheme: str, name: str) -> int:
    """Resolve one semantic joint name to a keypoint index within a scheme.

    Args:
        scheme: Registered scheme name (see :func:`available_schemes`).
        name: Semantic joint name (see :data:`SEMANTIC_JOINTS`).

    Returns:
        The integer keypoint index for ``name`` under ``scheme``.

    Raises:
        KeyError: If the scheme is unknown or lacks the requested joint.
    """
    mapping = scheme_map(scheme)
    try:
        return mapping[name]
    except KeyError:
        raise KeyError(
            f"Scheme '{scheme}' does not define joint '{name}'. "
            f"Defined joints: {sorted(mapping)}"
        )


def resolve_many(scheme: str, names: Iterable[str]) -> List[int]:
    """Resolve a sequence of semantic joint names to indices (order preserved)."""
    return [resolve(scheme, n) for n in names]


def supports(scheme: str, names: Iterable[str]) -> bool:
    """Return True if ``scheme`` defines every joint in ``names``."""
    mapping = scheme_map(scheme)
    return all(n in mapping for n in names)
