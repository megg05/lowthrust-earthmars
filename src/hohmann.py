"""Hohmann transfer with plane change, patched-conic parking orbit maneuvers,
and HCI (Heliocentric Inertial) frame propagation."""

import numpy as np
from dataclasses import dataclass


@dataclass
class HohmannResult:
    # Heliocentric transfer geometry
    a_transfer: float           # transfer ellipse SMA [m]
    e_transfer: float           # transfer ellipse eccentricity
    tof: float                  # heliocentric transfer time of flight [s]
    delta_i: float              # plane change angle [rad]

    # Heliocentric velocities
    v_earth_helio: float        # Earth heliocentric circular speed [m/s]
    v_mars_helio: float         # Mars heliocentric circular speed [m/s]
    v_dep_transfer: float       # transfer orbit speed at periapsis (departure) [m/s]
    v_arr_transfer: float       # transfer orbit speed at apoapsis (arrival) [m/s]

    # Hyperbolic excess velocities (v-infinity)
    v_inf_dep: float            # v-infinity at Earth departure [m/s]
    v_inf_arr: float            # v-infinity at Mars arrival [m/s]

    # Parking orbit burns (patched conic)
    dv_depart: float            # LEO escape burn [m/s]
    dv_capture: float           # MSO capture burn [m/s]
    dv_total: float             # total mission delta-v [m/s]

    # Parking orbit parameters
    r_leo: float                # LEO radius [m]
    r_mso: float                # MSO radius [m]
    v_circ_leo: float           # circular speed in LEO [m/s]
    v_circ_mso: float           # circular speed in MSO [m/s]

    # HCI energy states
    energy_leo: float           # specific energy in LEO (Earth-centered) [J/kg]
    energy_escape: float        # specific energy on escape hyperbola [J/kg]
    energy_transfer: float      # specific orbital energy on transfer orbit [J/kg]
    energy_capture: float       # specific energy on capture hyperbola [J/kg]
    energy_mso: float           # specific energy in MSO (Mars-centered) [J/kg]

    # HCI propagation states (position, velocity at key points)
    state_dep: np.ndarray       # [r(3), v(3)] at departure in HCI [m, m/s]
    state_arr: np.ndarray       # [r(3), v(3)] at arrival in HCI [m, m/s]


def hohmann_transfer(r1, r2, mu_sun, mu_earth, mu_mars,
                     R_earth, R_mars,
                     alt_leo_km=400.0, alt_mso_km=300.0,
                     delta_i_deg=1.85):
    """Full 2-impulse Hohmann transfer with plane change and parking orbits.

    Parameters
    ----------
    r1 : float — Earth heliocentric orbital radius [m]
    r2 : float — Mars heliocentric orbital radius [m]
    mu_sun : float — Sun gravitational parameter [m^3/s^2]
    mu_earth : float — Earth gravitational parameter [m^3/s^2]
    mu_mars : float — Mars gravitational parameter [m^3/s^2]
    R_earth : float — Earth physical radius [m]
    R_mars : float — Mars physical radius [m]
    alt_leo_km : float — LEO altitude [km]
    alt_mso_km : float — MSO altitude [km]
    delta_i_deg : float — orbital plane change angle [deg]

    Returns
    -------
    HohmannResult
    """
    delta_i = np.radians(delta_i_deg)

    # --- Heliocentric Hohmann geometry ---
    a_t = (r1 + r2) / 2.0
    e_t = (r2 - r1) / (r2 + r1)
    tof = np.pi * np.sqrt(a_t**3 / mu_sun)

    # Heliocentric circular velocities
    v_earth = np.sqrt(mu_sun / r1)
    v_mars = np.sqrt(mu_sun / r2)

    # Transfer orbit velocities (vis-viva)
    v_dep_t = np.sqrt(mu_sun * (2.0 / r1 - 1.0 / a_t))
    v_arr_t = np.sqrt(mu_sun * (2.0 / r2 - 1.0 / a_t))

    # --- Hyperbolic excess velocities ---
    # Departure: tangential burn, no plane change
    v_inf_dep = v_dep_t - v_earth

    # Arrival: combined speed change + plane change (vector triangle)
    # Spacecraft arrives at v_arr_t in ecliptic, must match v_mars in Mars's plane
    v_inf_arr = np.sqrt(v_arr_t**2 + v_mars**2 - 2.0 * v_arr_t * v_mars * np.cos(delta_i))

    # --- Parking orbit maneuvers (patched conic) ---
    r_leo = R_earth + alt_leo_km * 1e3
    r_mso = R_mars + alt_mso_km * 1e3

    v_circ_leo = np.sqrt(mu_earth / r_leo)
    v_circ_mso = np.sqrt(mu_mars / r_mso)

    # Escape from LEO: vis-viva on departure hyperbola at r_leo
    v_esc = np.sqrt(v_inf_dep**2 + 2.0 * mu_earth / r_leo)
    dv_depart = v_esc - v_circ_leo

    # Capture into MSO: vis-viva on approach hyperbola at r_mso
    v_approach = np.sqrt(v_inf_arr**2 + 2.0 * mu_mars / r_mso)
    dv_capture = v_approach - v_circ_mso

    dv_total = dv_depart + dv_capture

    # --- Specific orbital energies ---
    # LEO (circular, Earth-centered)
    energy_leo = -mu_earth / (2.0 * r_leo)

    # Escape hyperbola (Earth-centered, energy = v_inf^2 / 2)
    energy_escape = v_inf_dep**2 / 2.0

    # Heliocentric transfer orbit (Sun-centered)
    energy_transfer = -mu_sun / (2.0 * a_t)

    # Capture hyperbola (Mars-centered, energy = v_inf_arr^2 / 2)
    energy_capture = v_inf_arr**2 / 2.0

    # MSO (circular, Mars-centered)
    energy_mso = -mu_mars / (2.0 * r_mso)

    # --- HCI frame states ---
    # Departure: spacecraft at Earth's position, velocity = v_dep_t tangent to orbit
    # Place Earth at (r1, 0, 0) with velocity in +y direction (HCI ecliptic frame)
    r_dep = np.array([r1, 0.0, 0.0])
    v_dep = np.array([0.0, v_dep_t, 0.0])
    state_dep = np.concatenate([r_dep, v_dep])

    # Arrival: spacecraft at Mars position (opposite side, 180° transfer)
    # After half-orbit, position is (-r2, 0, 0) in ecliptic
    # Velocity is in -y direction (tangent at apoapsis) but rotated into Mars's plane
    # The plane change rotates the velocity vector by delta_i about the radial direction
    r_arr = np.array([-r2, 0.0, 0.0])
    v_arr = np.array([0.0, -v_arr_t * np.cos(delta_i), v_arr_t * np.sin(delta_i)])
    state_arr = np.concatenate([r_arr, v_arr])

    return HohmannResult(
        a_transfer=a_t,
        e_transfer=e_t,
        tof=tof,
        delta_i=delta_i,
        v_earth_helio=v_earth,
        v_mars_helio=v_mars,
        v_dep_transfer=v_dep_t,
        v_arr_transfer=v_arr_t,
        v_inf_dep=v_inf_dep,
        v_inf_arr=v_inf_arr,
        dv_depart=dv_depart,
        dv_capture=dv_capture,
        dv_total=dv_total,
        r_leo=r_leo,
        r_mso=r_mso,
        v_circ_leo=v_circ_leo,
        v_circ_mso=v_circ_mso,
        energy_leo=energy_leo,
        energy_escape=energy_escape,
        energy_transfer=energy_transfer,
        energy_capture=energy_capture,
        energy_mso=energy_mso,
        state_dep=state_dep,
        state_arr=state_arr,
    )


def propagate_hci(state0, mu_sun, tof, n_points=500):
    """Propagate a heliocentric state using Kepler's equation (2-body).

    Parameters
    ----------
    state0 : ndarray(6) — [r(3), v(3)] in HCI frame
    mu_sun : float — Sun gravitational parameter [m^3/s^2]
    tof : float — total time of flight [s]
    n_points : int — number of output points

    Returns
    -------
    times : ndarray(n_points) — time array [s]
    states : ndarray(n_points, 6) — [r(3), v(3)] at each time
    energies : ndarray(n_points) — specific orbital energy at each time [J/kg]
    """
    from kepler import solve_kepler, eccentric_to_true, elements_to_cartesian

    r0 = state0[:3]
    v0 = state0[3:]
    r0_mag = np.linalg.norm(r0)
    v0_mag = np.linalg.norm(v0)

    # Orbital energy and SMA
    energy = v0_mag**2 / 2.0 - mu_sun / r0_mag
    a = -mu_sun / (2.0 * energy)

    # Angular momentum
    h = np.cross(r0, v0)
    h_mag = np.linalg.norm(h)

    # Eccentricity vector
    e_vec = np.cross(v0, h) / mu_sun - r0 / r0_mag
    e = np.linalg.norm(e_vec)

    # Orbital frame unit vectors
    # Node vector
    k_hat = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(k_hat, h)
    n_mag = np.linalg.norm(n_vec)

    # Inclination
    i = np.arccos(np.clip(h[2] / h_mag, -1, 1))

    # RAAN
    if n_mag > 1e-10:
        Omega = np.arccos(np.clip(n_vec[0] / n_mag, -1, 1))
        if n_vec[1] < 0:
            Omega = 2 * np.pi - Omega
    else:
        Omega = 0.0

    # Argument of periapsis
    if n_mag > 1e-10 and e > 1e-10:
        omega = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n_mag * e), -1, 1))
        if e_vec[2] < 0:
            omega = 2 * np.pi - omega
    elif e > 1e-10:
        omega = np.arctan2(e_vec[1], e_vec[0])
    else:
        omega = 0.0

    # True anomaly at epoch
    if e > 1e-10:
        nu0 = np.arccos(np.clip(np.dot(e_vec, r0) / (e * r0_mag), -1, 1))
        if np.dot(r0, v0) < 0:
            nu0 = 2 * np.pi - nu0
    else:
        nu0 = np.arccos(np.clip(np.dot(n_vec, r0) / (n_mag * r0_mag), -1, 1))
        if r0[2] < 0:
            nu0 = 2 * np.pi - nu0

    # Eccentric anomaly at epoch
    E0 = 2 * np.arctan2(np.sqrt(1 - e) * np.sin(nu0 / 2),
                         np.sqrt(1 + e) * np.cos(nu0 / 2))
    M0 = E0 - e * np.sin(E0)

    # Mean motion
    n = np.sqrt(mu_sun / a**3)

    # Propagate
    times = np.linspace(0, tof, n_points)
    states = np.zeros((n_points, 6))
    energies = np.full(n_points, energy)

    for idx, t in enumerate(times):
        M = M0 + n * t
        E = solve_kepler(M, e)
        nu = eccentric_to_true(E, e)
        r_vec, v_vec = elements_to_cartesian(a, e, i, Omega, omega, nu, mu_sun)
        states[idx, :3] = r_vec
        states[idx, 3:] = v_vec

    return times, states, energies


def propellant_mass(dv, m0, Isp, g0=9.80665):
    """Propellant mass via Tsiolkovsky equation."""
    return m0 * (1.0 - np.exp(-dv / (Isp * g0)))
