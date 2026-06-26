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
In active development — Milestone 1 (NRHO propagation) in progress.

## Author
Shayan yar khan
