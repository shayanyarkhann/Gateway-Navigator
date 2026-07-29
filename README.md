# Gateway-Navigator
# Gateway Navigator

A CR3BP-based closed-loop orbit maintenance simulator for NASA's Gateway lunar station.

## Overview
Gateway Navigator models spacecraft dynamics in the Circular Restricted Three-Body Problem (CR3BP) framework, propagates the Near-Rectilinear Halo Orbit (NRHO), and implements two station-keeping controllers — LQR and PID — with a Kalman filter for state estimation. The goal is to compare how well each controller maintains the orbit under realistic perturbations.

## Features
- NRHO trajectory propagation using CR3BP dynamics
- LQR (Linear Quadratic Regulator) station-keeping controller
- PID station-keeping controller
- Kalman filter for state estimation
- Performance comparison between control strategies

## Tech Stack
- Python
- NumPy, SciPy
- Matplotlib

## Project Status
Current Status
done:
Module 1 — NRHO Propagation
Module 2 — Sensor and Actuator Noise
Module 3 — Extended Kalman Filter
Module 4 — PID Station-Keeping Controller
Module 5 — LQR Station-Keeping Controller

🚧 Module 6 — Controller Comparison and Monte Carlo Evaluation (in progress)

## Author
Shayan yar khan
Tested with Python 3.13

## Installation

Clone the repository

```bash
git clone https://github.com/yourname/Gateway-Navigator.git
cd Gateway-Navigator
```

Install dependencies

```bash
pip install -e .
```
## Running

Run the Module 1 validation

```bash
pytest tests/test_m1_validation.py
```

Run the complete test suite

```bash
pytest
```
The main research question:Which station-keeping strategy provides the best balance of orbital accuracy and propellant consumption for NASA Gateway Near Rectilinear Halo Orbit operations under realistic navigation uncertainty?

This repository accompanies the Gateway Navigator Engineering Companion and the associated research paper describing the development and evaluation of autonomous station-keeping algorithms for the NASA Gateway NRHO.

# Gateway Navigator — RMS v1.0 implementation

26 files. Full suite: **114 passed**. Layered on `c05d85a` plus the earlier `gn-patch`
(GN-002/009/022), which is **included here and must be applied** — the estimator is not
RMS-admissible without it.

## Install

```bash
pip install -e .
pytest -m "not slow"                                  # fast suite
pytest                                                # 114 tests, ~10 s
python -m modules.m6_campaign configs/v1-smoke.yaml   # end-to-end smoke run
python -m modules.m6_campaign configs/v1.yaml         # production campaign
```

## New files

| File | Role | RMS clause |
|---|---|---|
| `core/config.py` | Schema, validation, fingerprinting | App. C |
| `configs/v1.yaml` `v1-smoke.yaml` `v1-fault.yaml` | Campaign configs | App. C |
| `modules/m6_montecarlo.py` | Ensemble engine, CRN, seeds, convergence | 6.6, 6.7 |
| `modules/m6_metrics.py` | Bootstrap, Wilcoxon, effect size, Pareto | 6.5, 6.8 |
| `modules/m6_targeting.py` | x-axis-crossing benchmark policy | 6.3 |
| `modules/m6_faults.py` | Missed / partial maneuver | 7 |
| `modules/m6_export.py` | CSV, JSON, provenance | App. C |
| `modules/m6_figures.py` | Publication figures from saved records | 6.5 |
| `modules/m6_campaign.py` | One-command regeneration | App. C |

## Changed files

- `core/nrho_ics.py` — adds `correct_to_jacobi`, `resolve_orbit`
- `core/closed_loop.py` — adds `initial_dispersion`, `fault`, `q_accel`; filter now
  initialises on the reference rather than the dispersed truth
- `modules/m3_kalman.py`, `modules/m5_lqr.py` — from `gn-patch` (GN-002/022)

## Deliberate deviations from your brief

**Fault scope.** You listed sensor degradation, actuator degradation, estimator
mismatch and scheduling. RMS §7 specifies **one** primary mode (missed/partial
maneuver, randomized epoch and severity) and §3.3 records "a general fault-injection
framework" as explicitly not pursued in V1. I followed the RMS. Sensor degradation is
already covered by the noise-level sweep (§6.6).

**Module 7 not built.** RMS §3.2 puts interactive visualization out of scope for V1;
§3.5 makes the static page the first thing cut.

**Orbit anchor is configurable, not chosen.** RMS Appendix A's four anchors are mutually
inconsistent in the CR3BP. `configs/v1.yaml` defaults to `nine_two_period` with the
reasoning recorded inline; `jacobi_3p0498` and `period_6p50` are available. Closing
§5.3 to digits still needs Lee (2019).

## Open — needs resolution before the production campaign

**Targeting diverges at coast horizons 2 and 3.** Smoke run (6 trials, 8 revolutions,
MEDIUM noise):

```
Targeting  coast=1     0.191 m/s/yr    0.892 km    0% diverged
Targeting  coast=2     7.659 m/s/yr   27.883 km    0%
Targeting  coast=3 42045.624 m/s/yr 23696.925 km  100%
Targeting  coast=4     0.281 m/s/yr    1.687 km    0%
Targeting  coast=5     0.157 m/s/yr    1.207 km    0%
```

Not an ill-conditioned solve — the targeting map is well behaved throughout
(cond 21–310 over N = 1..6). Likely the horizon/cadence mismatch: RMS §6.2 fixes
one maneuver per revolution while §6.3 names coast duration as targeting's knob, so an
N-revolution correction is re-applied N times before it matures. But the
non-monotonicity (1, 4, 5 fine; 2, 3 not) doesn't fit simple over-correction. Documented
in the module docstring; treat N > 1 as provisional.

**Δv sits below the RMS §5.6 sanity band.** Targeting at coast=1 gives 0.191 m/s/yr
against an expected "few m/s/year". Below rather than above, and CR3BP omits solar
gravity and SRP which drive real maintenance cost — but worth confirming against
Guzzetti et al. (2017) before publishing.

**Smoke-run headline (6 trials, provisional):** LQR frontier dominates PID
(0.064–0.108 vs 1.0–30 m/s/yr) and sits below targeting. LQR beating the operational
benchmark is surprising and deserves scrutiny before it goes in a paper.