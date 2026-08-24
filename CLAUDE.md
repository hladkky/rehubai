# RehubAI — Dissertation Project Context

## Project Overview
PhD dissertation project at Taras Shevchenko National University of Kyiv.
Specialty: 126 — Information Systems and Technologies.
Supervisor: Druzhynin Volodymyr Anatoliyovych.

Topic: "Intelligent Information Technology for Rehabilitation of Military
Personnel and Civilians After Injuries and Wounds Using Artificial Intelligence"

## Core Technical Stack
- MediaPipe Pose (33 body keypoints, BlazePose)
- Python, OpenCV, NumPy
- Architecture: finite-state machine with JSON exercise configs
- Deviation types: Geometric (joint angles), Sequential (stage order), Dynamic (tempo)
- On-device processing (client-side, no server required)

## Key Algorithms
- Angle between keypoints: θ = arccos((u·v) / (‖u‖·‖v‖))
- Validation: compare θ_computed vs θ_reference with tolerance ε
- Temporal control: measure time between stage transitions
- Feature space: statistical + dynamic + spectral + correlation (dimensionality n = 8m + m(m-1)/2)
- Adaptive control: quality functional J(t) minimization

## Repository Structure
This is the main software repository for the dissertation.
Exercise configs are JSON files — adding a new exercise requires ONLY a new config, no model retraining.

## Coding Standards
- Python with type hints and docstrings (Google style)
- Unit tests for all validation logic
- Reproducibility: fix random seeds, document environment
- Keep detection pipeline / logic pipeline / feedback module separated

## Published Papers (for reference in comments/docs)
1. Hladkyi et al. (2021). An IoT Solution: A Fitness Trainer. IT&I-2021, CEUR-WS [Scopus]
2. Hladkyi et al. (2025). Hardware-Configuration Space of Intelligent IT for Rehabilitation. CEUR-2025 [Scopus]
3. Hladkyi et al. (2026). Universal real-time validation algorithm. IEEE SIST-2026, Astana
4. Hladkyi (2026). Intelligent Rehabilitation Technologies: Current State. (in press)

## Notion (research notes)
https://www.notion.so/127e9718b1f1806b8bb7e3abcab2cb3a

## IIT docs
All related docs are in `docs/` folder.
