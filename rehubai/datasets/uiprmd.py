"""UI-PRMD dataset adapter (University of Idaho — Physical Rehabilitation
Movement Dataset, Vakanski et al., 2018).

The dataset ships **skeletal/kinematic data only** (no RGB video): 10 movements
(``m01``–``m10``) x 10 subjects (``s01``–``s10``), each performed *correctly* and
*incorrectly* (suffix ``_inc``), captured with Vicon (39 markers) and Kinect
(22 joints), split into per-repetition episodes (``e01``–``e10``).

Coordinate representation (Kinect)
----------------------------------
The Kinect ``Positions`` files are **not** absolute joint coordinates: the root
(waist) is absolute, but every other joint is stored as a *bone offset expressed
in its parent's local frame*. Reconstructing world coordinates therefore requires
**forward kinematics**, combining those offsets with the per-joint YXZ Euler
angles from the accompanying ``Angles`` files down the skeleton hierarchy.

This module performs that reconstruction (:func:`forward_kinematics`, ported from
the reference visualisation of Vakanski's MATLAB code) and exposes the resulting
absolute joints under the semantic scheme ``uiprmd_kinect22`` so the same exercise
config that runs on MediaPipe video also validates UI-PRMD skeletons.

Notes
-----
* Reconstructed coordinates are in **millimetres**, with **+y up** (opposite to
  image-space conventions where y grows downward). Angle constraints are frame /
  scale invariant, so this does not affect geometric validation; positional
  ``y_above`` / ``y_below`` coordinate checks, however, are authored for
  image-space and would need their sense flipped for this frame.
* Single-limb movements (leg raise ``m06``; shoulder abduction/extension/rotation
  ``m07``–``m09``) are performed with the **right** limb in this dataset.
* File delimiter is inconsistent across the archive (whitespace in ``Movements/``,
  commas in ``Segmented Movements/``); :func:`load_matrix` handles both.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from .. import joints as joints_mod

# ---------------------------------------------------------------------------
# Skeleton scheme: UI-PRMD Kinect, 22 joints
# ---------------------------------------------------------------------------
# Joint order as stored in each row of a Kinect file (22 joints x 3 values):
#   0 Waist  1 Spine  2 Chest  3 Neck  4 Head  5 HeadTip
#   6 L_Collar  7 L_UpperArm  8 L_Forearm  9 L_Hand
#  10 R_Collar 11 R_UpperArm 12 R_Forearm 13 R_Hand
#  14 L_UpperLeg 15 L_LowerLeg 16 L_Foot 17 L_Toes
#  18 R_UpperLeg 19 R_LowerLeg 20 R_Foot 21 R_Toes
# In this skeletal model a limb *segment* joint carries the position of the joint
# at its proximal end, so e.g. "L_UpperArm" is the left shoulder joint and
# "L_Forearm" is the left elbow joint.
SCHEME = "uiprmd_kinect22"

_KINECT22: Dict[str, int] = {
    "spine_base": 0,     # Waist
    "spine_mid": 1,      # Spine
    "neck": 3,           # Neck   (2 = Chest / spine-shoulder has no semantic name)
    "head": 4,           # Head
    "left_shoulder": 7,  # L_UpperArm
    "left_elbow": 8,     # L_Forearm
    "left_wrist": 9,     # L_Hand
    "right_shoulder": 11,
    "right_elbow": 12,
    "right_wrist": 13,
    "left_hip": 14,      # L_UpperLeg
    "left_knee": 15,     # L_LowerLeg
    "left_ankle": 16,    # L_Foot
    "left_foot": 17,     # L_Toes
    "right_hip": 18,
    "right_knee": 19,
    "right_ankle": 20,
    "right_foot": 21,
}

N_JOINTS = 22

MOVEMENT_NAMES: Dict[str, str] = {
    "m01": "Deep squat",
    "m02": "Hurdle step",
    "m03": "Inline lunge",
    "m04": "Side lunge",
    "m05": "Sit to stand",
    "m06": "Standing active straight leg raise",
    "m07": "Standing shoulder abduction",
    "m08": "Standing shoulder extension",
    "m09": "Standing shoulder internal-external rotation",
    "m10": "Standing shoulder scaption",
}

# Register the scheme on import so ``ExerciseValidator(config, scheme=SCHEME)``
# resolves semantic joint names to this skeleton.
if SCHEME not in joints_mod.available_schemes():
    joints_mod.register_scheme(SCHEME, _KINECT22)


# ---------------------------------------------------------------------------
# Forward kinematics (hierarchical offsets + YXZ Euler -> world coordinates)
# ---------------------------------------------------------------------------
def _rotx(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _roty(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rotz(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler_to_matrix(angles_rad: np.ndarray) -> np.ndarray:
    """Rotation matrix for one joint's ``(x, y, z)`` Euler triplet (radians).

    Composition is ``Rz . Ry . Rx`` (the YXZ convention used by the dataset's
    reference code), where ``angles_rad = (rx, ry, rz)``.
    """
    rx, ry, rz = angles_rad
    return _rotz(rz).dot(_roty(ry)).dot(_rotx(rx))


# Kinematic chains: (base_angle_joint, [ordered joint indices]). Each chain is
# rooted at ``parent[chain[0]]`` and its accumulated rotation is seeded from the
# Euler angles of ``base_angle_joint`` (waist for spine/legs, chest for arms).
_CHAINS: Tuple[Tuple[int, Tuple[int, ...]], ...] = (
    (0, (1, 2, 3, 4, 5)),      # spine -> neck -> head -> head-tip
    (2, (6, 7, 8, 9)),         # left arm
    (2, (10, 11, 12, 13)),     # right arm
    (0, (14, 15, 16, 17)),     # left leg
    (0, (18, 19, 20, 21)),     # right leg
)
_PARENT: Dict[int, int] = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4,
    6: 2, 7: 6, 8: 7, 9: 8,
    10: 2, 11: 10, 12: 11, 13: 12,
    14: 0, 15: 14, 16: 15, 17: 16,
    18: 0, 19: 18, 20: 19, 21: 20,
}


def _fk_frame(offsets: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """Reconstruct absolute joint positions for a single frame.

    Args:
        offsets: ``(22, 3)`` hierarchical bone offsets (waist row is absolute).
        angles_deg: ``(22, 3)`` per-joint YXZ Euler angles in degrees.

    Returns:
        ``(22, 3)`` absolute joint positions in the Kinect frame.
    """
    ang = np.deg2rad(angles_deg)
    pos = offsets.copy()
    for base, chain in _CHAINS:
        rot = euler_to_matrix(ang[base])
        for i, joint in enumerate(chain):
            if i > 0:
                # Extend the accumulated rotation by the previous joint's Euler.
                rot = rot.dot(euler_to_matrix(ang[joint - 1]))
            pos[joint] = rot.dot(offsets[joint]) + pos[_PARENT[joint]]
    return pos


def forward_kinematics(offsets: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """Reconstruct absolute joint positions for a whole sequence.

    Args:
        offsets: ``(T, 22, 3)`` hierarchical bone offsets.
        angles_deg: ``(T, 22, 3)`` per-joint Euler angles (degrees).

    Returns:
        ``(T, 22, 3)`` absolute joint positions.
    """
    if offsets.shape[1:] != (N_JOINTS, 3) or angles_deg.shape[1:] != (N_JOINTS, 3):
        raise ValueError(
            f"expected (T, {N_JOINTS}, 3) arrays, got "
            f"{offsets.shape} and {angles_deg.shape}"
        )
    t = min(len(offsets), len(angles_deg))
    return np.stack([_fk_frame(offsets[i], angles_deg[i]) for i in range(t)])


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------
def load_matrix(path: Path) -> np.ndarray:
    """Load a UI-PRMD ``.txt`` matrix, tolerating whitespace- or comma-separated
    values and reshaping ``(T, 66)`` to ``(T, 22, 3)``.
    """
    text = Path(path).read_text().replace(",", " ")
    rows = [
        [float(x) for x in line.split()]
        for line in text.splitlines()
        if line.strip()
    ]
    arr = np.asarray(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != N_JOINTS * 3:
        raise ValueError(
            f"{path}: expected {N_JOINTS * 3} columns, got {arr.shape}"
        )
    return arr.reshape(-1, N_JOINTS, 3)


_FILENAME_RE = re.compile(
    r"^m(?P<m>\d+)_s(?P<s>\d+)(?:_e(?P<e>\d+))?_positions(?P<inc>_inc)?\.txt$"
)


@dataclass(frozen=True)
class Sample:
    """One recorded sequence (a full session or a single-repetition episode)."""

    movement: str           # e.g. "m01"
    subject: str            # e.g. "s01"
    episode: Optional[str]  # e.g. "e01", or None for a full session
    correct: bool
    system: str             # "kinect"
    pos_path: Path
    ang_path: Path

    @property
    def label(self) -> int:
        """Binary label for the correct/incorrect classification task."""
        return 1 if self.correct else 0

    @property
    def movement_name(self) -> str:
        return MOVEMENT_NAMES.get(self.movement, self.movement)

    def keypoints(self) -> np.ndarray:
        """Absolute ``(T, 22, 3)`` joint positions under scheme ``uiprmd_kinect22``.

        The skeletons are clean mocap output, so no confidence array is produced;
        pass ``confidences=None`` to :class:`~rehubai.validator.ExerciseValidator`.
        """
        offsets = load_matrix(self.pos_path)
        angles = load_matrix(self.ang_path)
        return forward_kinematics(offsets, angles)


# Subjects that perform the single-limb movements with the LEFT limb (all others
# use the right). Single-limb exercise configs are authored for the canonical
# (right) working side; these subjects must be mirrored via :func:`mirror_lr`
# before validation. Determined empirically from per-subject joint ROM.
LEFT_WORKING_SUBJECTS: Dict[str, frozenset] = {
    "m06": frozenset({"s07", "s10"}),  # standing straight leg raise
    "m07": frozenset({"s07", "s10"}),  # standing shoulder abduction
    "m08": frozenset({"s07", "s10"}),  # standing shoulder extension
    "m09": frozenset({"s07", "s10"}),  # standing shoulder rotation
}


def mirror_lr(kps: np.ndarray, scheme: str = SCHEME) -> np.ndarray:
    """Swap left/right joints so a left-limb execution matches a right-side config.

    A single-limb config references e.g. ``right_hip``; for a subject who performs
    the movement with the left limb, this permutes the keypoint rows so the working
    (left) limb occupies the right-side indices. Joint angles are scalar functions
    of three points, so this label swap is exact — no coordinate reflection needed.

    Args:
        kps: ``(..., J, D)`` keypoint array.
        scheme: Skeleton scheme whose ``left_*``/``right_*`` pairs define the swap.

    Returns:
        A view/copy of ``kps`` with left/right joint rows exchanged.
    """
    kps = np.asarray(kps)
    perm = np.arange(kps.shape[-2])
    mapping = joints_mod.scheme_map(scheme)
    for name, idx in mapping.items():
        if name.startswith("left_"):
            right = "right_" + name[len("left_"):]
            if right in mapping:
                perm[idx], perm[mapping[right]] = mapping[right], idx
    return kps[..., perm, :]


def parse_filename(name: str) -> Optional[dict]:
    """Parse a ``*_positions*.txt`` filename into its metadata, or ``None``."""
    m = _FILENAME_RE.match(name)
    if not m:
        return None
    return {
        "movement": f"m{int(m.group('m')):02d}",
        "subject": f"s{int(m.group('s')):02d}",
        "episode": f"e{int(m.group('e')):02d}" if m.group("e") else None,
        "correct": m.group("inc") is None,
    }


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------
# (correct, segmented) -> top-level folder name within the dataset root.
_FOLDERS = {
    (True, False): "Movements",
    (False, False): "Incorrect Movements",
    (True, True): "Segmented Movements",
    (False, True): "Incorrect Segmented Movements",
}


def default_root() -> Path:
    """Repo-relative default location of the extracted dataset."""
    return Path(__file__).resolve().parents[2] / "datasets" / "uiprmd" / "raw"


class UIPRMDDataset:
    """Index over the UI-PRMD archive, yielding :class:`Sample` objects.

    Args:
        root: Folder that directly contains ``Movements/``, ``Segmented
            Movements/``, etc. Defaults to :func:`default_root`.
        system: Capture system; only ``"kinect"`` is supported (Vicon stores 39
            markers, which lack a semantic joint mapping — future work).
        segmented: Use per-repetition episode files (recommended for
            classification) rather than full-session files.
        movements: Restrict to these movement codes (e.g. ``["m01", "m05"]``).
    """

    def __init__(
        self,
        root: Optional[Path] = None,
        system: str = "kinect",
        segmented: bool = True,
        movements: Optional[List[str]] = None,
    ) -> None:
        if system != "kinect":
            raise NotImplementedError(
                "Only the Kinect skeleton is supported; Vicon (39 markers) has no "
                "semantic joint map yet."
            )
        self.root = Path(root) if root is not None else default_root()
        if not self.root.exists():
            raise FileNotFoundError(
                f"UI-PRMD root not found: {self.root}. Point `root=` at the folder "
                "containing 'Movements/', 'Segmented Movements/', ..."
            )
        self.system = system
        self.segmented = segmented
        self.movements = set(movements) if movements else None

    # -- discovery ---------------------------------------------------------
    def _positions_dir(self, correct: bool) -> Path:
        folder = _FOLDERS[(correct, self.segmented)]
        return self.root / folder / self.system.capitalize() / "Positions"

    def _samples_for(self, correct: bool) -> List[Sample]:
        pos_dir = self._positions_dir(correct)
        if not pos_dir.exists():
            return []
        ang_dir = pos_dir.parent / "Angles"
        out: List[Sample] = []
        for pos_path in sorted(pos_dir.glob("*_positions*.txt")):
            meta = parse_filename(pos_path.name)
            if meta is None or meta["correct"] != correct:
                continue
            if self.movements and meta["movement"] not in self.movements:
                continue
            ang_path = ang_dir / pos_path.name.replace("positions", "angles")
            if not ang_path.exists():
                continue
            out.append(
                Sample(
                    movement=meta["movement"],
                    subject=meta["subject"],
                    episode=meta["episode"],
                    correct=correct,
                    system=self.system,
                    pos_path=pos_path,
                    ang_path=ang_path,
                )
            )
        return out

    def samples(self, correct: Optional[bool] = None) -> List[Sample]:
        """All samples, optionally filtered to only correct / only incorrect."""
        if correct is True:
            return self._samples_for(True)
        if correct is False:
            return self._samples_for(False)
        return self._samples_for(True) + self._samples_for(False)

    def __len__(self) -> int:
        return len(self.samples())

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples())

    # -- cross-validation --------------------------------------------------
    def subjects(self) -> List[str]:
        return sorted({s.subject for s in self.samples()})

    def by_subject(self) -> Dict[str, List[Sample]]:
        groups: Dict[str, List[Sample]] = {}
        for s in self.samples():
            groups.setdefault(s.subject, []).append(s)
        return groups

    def loso_splits(self) -> Iterator[Tuple[str, List[Sample], List[Sample]]]:
        """Leave-one-subject-out folds.

        Yields ``(held_out_subject, train_samples, test_samples)`` for each
        subject — the cross-subject protocol required for UI-PRMD classification.
        """
        groups = self.by_subject()
        for held_out in sorted(groups):
            test = groups[held_out]
            train = [s for subj, xs in groups.items() if subj != held_out for s in xs]
            yield held_out, train, test