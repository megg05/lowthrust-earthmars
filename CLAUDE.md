# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Low-thrust trajectory optimization for Earth-to-Mars transfer (MIT 16.346 coursework). Investigates Solar Electric Propulsion (SEP) using NASA's Psyche spacecraft (Hall-effect thrusters, xenon propellant) as the reference model. Compares continuous-thrust "spiral" trajectories against traditional Hohmann impulsive transfers to quantify propellant savings vs. increased complexity/duration.

The core problem is optimal control: determining the most fuel-efficient thrust direction and coast/burn scheduling within a heliocentric two-body framework. Uses Q-law Lyapunov feedback control to generate initial guesses, intended to feed into a collocation-based optimizer. The code is a work in progress — the optimizer and Q-law gradient computation are incomplete.

## Running

```bash
python scripts/run_optimizer.py
```

No build system, no tests, no linter configured. Pure Python with NumPy and SciPy dependencies.

## Architecture

The pipeline flows: **Q-law initial guess → collocation optimization** (collocation not yet implemented).

- `src/eom.py` — Two-body equations of motion with thrust (Cartesian state + mass: 7-element state vector `[r(3), v(3), m]`)
- `src/qlaw.py` — Q-law Lyapunov feedback controller operating on **modified equinoctial elements** (MEE: `a, f, g, h, k, L`). Computes a proximity quotient Q and steers thrust to descend Q. Supports multi-phase targeting via `PhaseTarget`. Gradient computation (`calculate_qgrad`) is a stub.
- `src/optimizer.py` — Wraps propagator + Q-law to build initial guesses and solve (collocation TODO). Takes `cfg, spacecraft, earth, mars, target_orbit`.
- `src/propagator.py` — Thin wrapper around `scipy.integrate.solve_ivp` (RK45)
- `src/models/` — Data classes for `Body` (gravitational parameter μ, radius), `Spacecraft` (mass, thrust limit, Isp), and `Orbit` representations (Keplerian `Orbit_Kep` and equinoctial `Orbit_Eq` with Q-law helper methods)
- `src/config.py` — `SolverConfig` dataclass (node count, max iterations, tolerance, method)
- `scripts/run_optimizer.py` — Entry point / smoke test; adds `src/` to `sys.path` manually

## Key Conventions

- All units are SI (meters, seconds, kg) unless suffixed (e.g., `alt_km`, `rp_alt` in km)
- Orbital elements use modified equinoctial elements (MEE) for Q-law, not classical Keplerian
- `Body.r` is the body's physical radius (meters); `Body.mu` is the gravitational parameter
- `Spacecraft.Tlim` is max thrust (Newtons); note `run_optimizer.py` uses `Tmax` — there is a naming inconsistency
- The code references a `utils` module (`cart2eq`, `cart2oe`, `oe2cart`, `wrap_angle`) in `qlaw.py` that does not yet exist in the repo
- Imports within `src/` use relative imports (e.g., `from .eom import`), but `run_optimizer.py` uses absolute imports after path manipulation
