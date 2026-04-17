# Hardware-Configuration Space — Mathematical Framework

**Source:** Hladkyi et al. (2025). "Hardware-Configuration Space of an Intelligent
Information Technology for Rehabilitation Support." CEUR-2025 [Scopus].

## System Tuple

The hardware-configuration space is formalized as:

```
Ω = ⟨H, C, S, T⟩
```

- `H = {h_1, ..., h_k}` — set of hardware components (sensors, actuators, computing nodes)
- `C = {c_1, ..., c_l}` — set of configuration parameters (network topologies, protocols)
- `S` — set of software-logic modules
- `T` — temporal space of system operation

## Three-Layer Architecture

### Hardware Layer (5 groups)
1. Wearable sensors & embedded systems (EMG sensors, smart insoles)
2. Rehabilitation robotics (limb-rehab robots, exoskeletons)
3. Telemetry & remote monitoring (wireless vital-sign monitors, IoT devices)
4. VR/AR systems (360° headsets for immersive exercise environments)
5. Neuromodulation equipment (spinal-cord stimulators)

### Configuration Layer
- Network settings
- Virtualization and containerization

### Software-Logic Layer
- Patient data analysis algorithms
- Machine learning models
- User interfaces
- Adaptation, workload, and feedback modules

## Sensor Data Model

Sensor data as multidimensional stochastic process:

```
D(t) = [d_1(t), d_2(t), ..., d_m(t)]^T ∈ ℝ^m
```

Each component: `d_i(t) = s_i(t) + ε_i(t)` (useful signal + additive noise)

Data aggregation from multiple IoT sources:

```
D_a(t) = Σ v_i · φ_i(d_i(t))
```

where `v_i` = reliability-weighted coefficients, `φ_i` = normalization function.

## Feature Space Construction

Built within sliding window of length L:

```
W_t = {D_a(t), D_a(t-1), ..., D_a(t-L+1)}
```

Feature vector as composition of operators:

```
X(t) = Φ_norm ∘ Φ_stat ∘ Φ_dyn ∘ Φ_spec (W_t)
```

### Statistical Features X_stat(t) ∈ ℝ^{3m}

Per channel i:
- Mean:     μ_i = (1/L) Σ d_i(t-k)
- Variance: σ_i² = (1/L-1) Σ (d_i(t-k) - μ_i)²
- Skewness: γ_i

### Dynamic (Kinematic) Features X_dyn(t) ∈ ℝ^{3m}

- Velocity:     ḋ_i(t) = (d_i(t) - d_i(t-1)) / Δt
- Acceleration: d̈_i(t) = (ḋ_i(t) - ḋ_i(t-1)) / Δt
- Signal energy: E_i = Σ d_i²(t-k)

### Spectral Features X_spec(t) ∈ ℝ^{2m}

- DFT:           D_i(ω) = Σ d_i(t-k) · e^{-jωk}
- Band power:    P_i = ∫[ω_a to ω_b] |D_i(ω)|² dω
- Spectral entropy: H_i = -Σ p_ik · log(p_ik)

### Correlation Features X_corr(t)

Pearson coefficient between channels i and j:

```
ρ_ij = Σ(d_i(t-k) - μ_i)(d_j(t-k) - μ_j) / ((L-1)·σ_i·σ_j)
```

Number of pairs: `m(m-1)/2`

### Full Feature Space Dimensionality

```
n = 8m + m(m-1)/2
```

where m = number of sensor channels.

## Patient State Model

Latent state vector:

```
S(t) = F(X(t), Θ)
where F = f_k ∘ f_{k-1} ∘ ... ∘ f_1   (composition of ML models)
Θ = {θ_1, θ_2, ..., θ_k}              (model parameters)
```

State dynamics:

```
S(t+1) = A·S(t) + B·U(t) + W(t)
```

where A, B = system dynamics matrices, W(t) = disturbance term.

## Adaptive Control (Optimization)

Control input as optimization problem:

```
U*(t) = argmin_U J(t)
```

Quality functional:

```
J(t) = ∫_t^{t+T} [‖S(τ) - S_ref‖²_Q + ‖U(τ)‖²_R] dτ
```

- `S_ref` — reference (target) patient state
- `Q, R`  — weighting matrices
- Prevents overload, adapts session intensity in real-time

## Self-Adaptation (Online Learning)

Model parameter update:

```
Θ(t+1) = Θ(t) - η · ∇_Θ L(S(t), S_obs(t))
```

Configuration space adaptation:

```
C(t+1) = C(t) ⊕ ΔC(t)
```

Overall recurrent mapping:

```
Ω(t+1) = M(Ω(t), D(t), U(t))
```

## Practical Results

- Noise reduction from preprocessing: **15-20% improvement** vs raw signals
- Detects microtremor and spastic manifestations missed by classical amplitude methods
- Modular design: new sensor types added without redesigning logic layer
- Dimensionality reduction needed for m > 10 sensors on mobile (future work)

## Integration Standards
- Data exchange: HL7 FHIR (for EHR integration)
- Security: encrypted transmission for medical data privacy compliance
