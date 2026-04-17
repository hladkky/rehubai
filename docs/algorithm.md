# Validation Algorithm — Technical Reference

**Source:** Hladkyi et al. (2026). "A universal real-time validation algorithm for
rehabilitation exercises based on spatial pose configuration analysis." IEEE SIST-2026, Astana.

## Core Concept

Each rehabilitation exercise is formalized as an **ideal cycle** — an ordered sequence
of discrete biomechanical stages. The system validates whether the patient correctly
executes each stage using MediaPipe Pose (33 keypoints).

## Exercise Representation

An exercise is described as a tuple:

```
E = (S, T, V)
```

- `S` — ordered set of stages (static poses)
- `T` — temporal constraints on transitions between stages
- `V` — constraint vectors for each keypoint (permissible angles)

## Angle Calculation

For three consecutive keypoints A, B, C (e.g. shoulder → elbow → wrist):

```
u = A - B
v = C - B
θ = arccos( (u·v) / (‖u‖ · ‖v‖) )
```

Deviation registered if: `|θ_computed - θ_reference| > ε`

**Key advantage:** uses relative angles, not absolute coordinates — invariant to
camera position (stable at ±15-20° viewing angle variation) and user anthropometrics.

## 4-Step Real-Time Validation Loop

**Step 1: Frame Acquisition**
Get (x, y, z) coordinates of 33 keypoints from MediaPipe Pose.

**Step 2: State Check**
Wait for body to enter the tolerance zone of the target stage.

**Step 3: Sequence Control**
If user attempts stage N while skipping stage N-1 → error notification.

**Step 4: Temporal Analysis**
Measure time between stage transitions. If outside `[T_min, T_max]` → pace recommendation.

## Three Error Categories (Biofeedback)

| Category   | What is checked          | Example feedback                  |
|------------|--------------------------|-----------------------------------|
| Geometric  | Joint angles vs reference | "Straighten your elbow"          |
| Sequential | Stage order              | "Return to start position first"  |
| Dynamic    | Transition speed         | "Slow down the movement"          |

## JSON Config Format

Each exercise is a JSON file — no model retraining needed:

```json
{
  "exercise_id": "dead_bug",
  "stages": [
    {
      "id": "REST",
      "constraints": [
        {
          "points": [11, 13, 15],
          "ideal_angle": 175,
          "tolerance": 15,
          "description": "left_elbow_extended"
        },
        {
          "points": [23, 25, 27],
          "ideal_angle": 90,
          "tolerance": 20,
          "description": "left_knee_bent"
        }
      ],
      "min_hold_sec": 0.3
    }
  ],
  "transitions": [
    {"from": "REST", "to": "LEFT_PHASE", "min_sec": 1.0, "max_sec": 2.0},
    {"from": "LEFT_PHASE", "to": "REST", "min_sec": 1.0, "max_sec": 2.0}
  ],
  "forbidden_transitions": [
    {"from": "LEFT_PHASE", "to": "RIGHT_PHASE"}
  ],
  "general_requirements": {
    "torso_stability_tolerance": 0.10,
    "shoulder_symmetry_px": 20,
    "min_keypoint_visibility": 0.5
  }
}
```

## MediaPipe BlazePose Key Indices

```
11: left_shoulder    12: right_shoulder
13: left_elbow       14: right_elbow
15: left_wrist       16: right_wrist
23: left_hip         24: right_hip
25: left_knee        26: right_knee
27: left_ankle       28: right_ankle
```

## 7-Step Universal Processing Function

1. Retrieve keypoints from pose estimation model
2. Check visibility of critical keypoints (pause if confidence < 0.5)
3. Determine current stage based on config
4. Validate transitions between stages
5. Validate current stage (angles + coordinates)
6. Check general requirements (stability, symmetry)
7. Visualize (green/red skeleton + text prompt)

## Performance Results (Dead Bug exercise test)

| Parameter                  | Reference    | Tolerance  | Accuracy |
|----------------------------|-------------|------------|----------|
| Elbow angle (extension)    | 180°        | ±10°       | 97.4%    |
| Knee angle (extension)     | 90°         | ±7°        | 96.2%    |
| Phase transition time      | 2.5 sec/stage | ±0.5 sec | 94.8%    |

## Implementation Notes

- **Adaptive smoothing:** low-pass filter for landmark jitter reduction
- **Frame skipping:** biomechanical computation on every 2nd frame (reduces CPU without quality loss)
- **Z-coordinate:** used for depth to improve 3D angle accuracy
- **FSM filter interval:** transition fires only after N consecutive True frames (avoids noise false positives)
- **Performance:** >30 FPS on average mobile device

## Known Limitation

Severe occlusion when one limb overlaps another (e.g. strict lateral camera view).
Mitigation: guide user to optimal camera placement before session start.
