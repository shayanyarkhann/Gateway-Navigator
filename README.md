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