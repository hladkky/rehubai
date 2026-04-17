# Literature Review — Intelligent Rehabilitation 2021-2026

**Source:** Hladkyi Y. (2026). "Intelligent Rehabilitation Technologies for Trauma
and Combat Injury Recovery: Current State, Challenges, and Development Trends." (in press)

## Bibliometric Profile

- 31 total sources reviewed; 24 with explicit dates
- **79%+ of sources from 2025-2026** — reflects rapid field emergence
- Dominant categories: AI/ML general, Telerehabilitation/IoMT, LLM/GenAI,
  Computer Vision, Digital Twins

## Technological Evolution by Period

| Period    | Core Technology              | Key Objective                            |
|-----------|------------------------------|------------------------------------------|
| 2021-2022 | Computer Vision (CNNs)       | Remote position monitoring               |
| 2023-2024 | Sensor Fusion & Gamification | Enhanced accuracy and motivation         |
| 2025-2026 | Digital Twins & GenAI        | Full personalization & autonomous coaching |

## Key Technology Areas

### 1. Computer Vision & Markerless Motion Analysis (2021-2023)
- MediaPipe and OpenPose for exercise technique assessment (no expensive sensors)
- Deep learning achieves accuracy near motion capture systems (Vicon)
- Enables transition from clinical to home rehab (smartphone camera sufficient)
- RMSE metric: `RMSE = sqrt(1/n · Σ(θ_ref,i - θ_pat,i)²)`

### 2. Gamification & Sensor Integration (2023-2024)
- AI-based platforms double patient engagement vs conventional methods (RCT)
- IMU + EMG fusion: captures movement AND muscle activation/fatigue in real-time
- SMART telerehabilitation: AI + ontologies + biofeedback

### 3. Generative AI & Multimodal LLMs (2025-2026)
- LLMs as "coaches": adapt exercise instructions, respond to pain questions,
  generate structured reports for clinicians
- Specialized LLMs (lower back pain): recommendations comparable to senior medical students
- Requires physician oversight
- RL reward function for adaptive load:
  `R_t = ω_1·ΔP + ω_2·ΔV - ω_3·F`
  - ΔP = progress in exercise performance
  - ΔV = patient engagement (heart rate, facial expression)
  - F  = fatigue level (EMG signal)
  - ω  = personalization weights

### 4. Human Digital Twins (HDT) — 2025-2026
- Virtual patient model synchronized with physical state via IoT sensors
- Predicts rehabilitation outcomes before prescribing to real patient
- Exoskeleton integration: automatic support adjustment based on neuromuscular response
- Ukraine active contributor ("Human Digital Twin in Ukraine" framework)

## Technical Tools Comparison

| Category           | Primary Advantage         | Primary Disadvantage              |
|--------------------|---------------------------|-----------------------------------|
| Robotics           | Precise load control      | High cost, large form factor      |
| VR/AR systems      | High patient motivation   | Cybersickness risk                |
| BCIs               | Direct CNS connection     | Complex setup & calibration       |
| Telehealth platforms | Home accessibility      | Limited physician oversight       |

## Verified Performance Indicators

- CV-based exercise assessment accuracy: **92-95%** vs motion capture gold standard
- Telerehabilitation: **15-25% reduction** in recovery time vs clinic-based
- AI-driven platforms: **2× patient adherence** improvement (RCT)

## Critical Challenges (2026)

### Technical
- "Black box" problem: clinicians distrust opaque AI decisions
- No unified data exchange standards across device manufacturers
- IoMT cybersecurity vulnerabilities

### Economic
- Exoskeletons inaccessible to most public hospitals
- Shortage of specialists at medicine-robotics-data analysis intersection

### Ukraine-Specific
- Large-scale blast injuries and complex amputations require rapid scaling
- Rehabilitation of physical injuries complicated by PTSD
- Large-scale reform of prosthetics system launched in early 2026

## Research Gaps (This Dissertation's Contribution)

- Most CV systems narrowly specialized OR require large computational resources
- No flexible "configuration space" for arbitrary exercises without reprogramming core
- Lack of systems specifically targeting military/post-trauma rehabilitation
- **Gap addressed:** simultaneous verification of geometry + sequence + tempo

## Key Papers to Cite

1. Priyadarshi et al. (2026). Smart healthcare technologies in rehabilitation.
   *Connection Science*, 38(1). https://doi.org/10.1080/09540091.2025.2612458 [Scopus]

2. Sumner et al. (2023). AI in physical rehabilitation: systematic review.
   *Artif Intell Med*, 146:102693. doi:10.1016/j.artmed.2023.102693 [Scopus]

3. Namdar et al. (2025). AI-based digital rehabilitation and adherence.
   *JMIR Rehabil Assist Technol*, 12:e69763. doi:10.2196/69763

4. Gao et al. (2025). Systematic survey on human pose estimation.
   *Artif Intell Rev*. https://doi.org/10.1007/s10462-024-11060-2

5. Michou et al. (2026). AI-Powered Physiotherapy: LLMs vs Students.
   *Appl Sci*, 16:1165. https://doi.org/10.3390/app16031165

6. Olawade et al. (2026). Digital twin technology in physiotherapy.
   *Virtual Reality & Intelligent Hardware*, 8(1):71-86.
   https://doi.org/10.1016/j.vrih.2026.01.002

7. Lisnevskyi et al. (2025). IoT-based health monitoring system.
   CEUR-WS Vol-3966. https://ceur-ws.org/Vol-3966/W2Paper1.pdf [Scopus]
