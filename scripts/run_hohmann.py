"""Compute and display the full Hohmann transfer with plane change, parking orbits, and HCI propagation."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.body import Earth, Mars, Sun
from models.spacecraft import Psyche
from hohmann import hohmann_transfer, propagate_hci, propellant_mass


def main():
    result = hohmann_transfer(
        r1=Earth.a_sun,
        r2=Mars.a_sun,
        mu_sun=Sun.mu,
        mu_earth=Earth.mu,
        mu_mars=Mars.mu,
        R_earth=Earth.r,
        R_mars=Mars.r,
        alt_leo_km=400.0,
        alt_mso_km=300.0,
        delta_i_deg=1.85,
    )

    tof_days = result.tof / 86400.0

    print("=" * 60)
    print("  HOHMANN TRANSFER: EARTH → MARS")
    print("  2-impulse with 1.85° plane change, HCI frame")
    print("=" * 60)

    print("\n--- Parking Orbits ---")
    print(f"  LEO altitude:    400 km  (r = {result.r_leo / 1e6:.3f} Mm)")
    print(f"  MSO altitude:    300 km  (r = {result.r_mso / 1e6:.3f} Mm)")
    print(f"  v_circ LEO:      {result.v_circ_leo:.1f} m/s  ({result.v_circ_leo / 1e3:.3f} km/s)")
    print(f"  v_circ MSO:      {result.v_circ_mso:.1f} m/s  ({result.v_circ_mso / 1e3:.3f} km/s)")

    print("\n--- Heliocentric Transfer Geometry ---")
    print(f"  Departure radius (Earth): {Earth.a_sun:.4e} m  (1.000 AU)")
    print(f"  Arrival radius (Mars):    {Mars.a_sun:.4e} m  ({Mars.a_sun / 1.496e11:.3f} AU)")
    print(f"  Transfer SMA:             {result.a_transfer:.4e} m  ({result.a_transfer / 1.496e11:.3f} AU)")
    print(f"  Transfer eccentricity:    {result.e_transfer:.6f}")
    print(f"  Plane change:             {result.delta_i * 180 / 3.141592653589793:.2f}°")
    print(f"  Time of flight:           {tof_days:.1f} days  ({tof_days / 30.44:.1f} months)")

    print("\n--- Heliocentric Velocities ---")
    print(f"  v_Earth (circular):       {result.v_earth_helio / 1e3:.3f} km/s")
    print(f"  v_Mars (circular):        {result.v_mars_helio / 1e3:.3f} km/s")
    print(f"  v_transfer at departure:  {result.v_dep_transfer / 1e3:.3f} km/s")
    print(f"  v_transfer at arrival:    {result.v_arr_transfer / 1e3:.3f} km/s")

    print("\n--- Hyperbolic Excess Velocities ---")
    print(f"  v_inf departure:          {result.v_inf_dep / 1e3:.3f} km/s")
    print(f"  v_inf arrival:            {result.v_inf_arr / 1e3:.3f} km/s")

    print("\n--- Delta-V Budget ---")
    print(f"  Δv₁ (LEO escape):         {result.dv_depart:.1f} m/s  ({result.dv_depart / 1e3:.3f} km/s)")
    print(f"  Δv₂ (MSO capture):        {result.dv_capture:.1f} m/s  ({result.dv_capture / 1e3:.3f} km/s)")
    print(f"  Δv total:                 {result.dv_total:.1f} m/s  ({result.dv_total / 1e3:.3f} km/s)")

    print("\n--- Specific Orbital Energy States ---")
    print(f"  ε_LEO (Earth-centered):    {result.energy_leo:.4e} J/kg")
    print(f"  ε_escape (Earth-centered): {result.energy_escape:.4e} J/kg")
    print(f"  ε_transfer (Sun-centered): {result.energy_transfer:.4e} J/kg")
    print(f"  ε_capture (Mars-centered): {result.energy_capture:.4e} J/kg")
    print(f"  ε_MSO (Mars-centered):     {result.energy_mso:.4e} J/kg")

    print("\n--- HCI Frame States ---")
    print(f"  Departure position: [{result.state_dep[0]:.4e}, {result.state_dep[1]:.4e}, {result.state_dep[2]:.4e}] m")
    print(f"  Departure velocity: [{result.state_dep[3]:.4e}, {result.state_dep[4]:.4e}, {result.state_dep[5]:.4e}] m/s")
    print(f"  Arrival position:   [{result.state_arr[0]:.4e}, {result.state_arr[1]:.4e}, {result.state_arr[2]:.4e}] m")
    print(f"  Arrival velocity:   [{result.state_arr[3]:.4e}, {result.state_arr[4]:.4e}, {result.state_arr[5]:.4e}] m/s")

    # Propagate transfer in HCI
    print("\n--- HCI Propagation (transfer arc) ---")
    times, states, energies = propagate_hci(result.state_dep, Sun.mu, result.tof, n_points=10)
    print(f"  {'Time [days]':>12}  {'r [AU]':>10}  {'v [km/s]':>10}  {'ε [J/kg]':>14}")
    for t, s, e in zip(times, states, energies):
        r_au = np.linalg.norm(s[:3]) / 1.496e11
        v_kms = np.linalg.norm(s[3:]) / 1e3
        print(f"  {t / 86400:12.1f}  {r_au:10.4f}  {v_kms:10.3f}  {e:14.4e}")

    # Propellant comparison
    print("\n--- Propellant Mass (Tsiolkovsky) ---")
    Isp_chem = 320.0
    mp_chem = propellant_mass(result.dv_total, Psyche.m0, Isp_chem)
    mp_elec = propellant_mass(result.dv_total, Psyche.m0, Psyche.Isp)
    print(f"  Spacecraft wet mass:     {Psyche.m0:.0f} kg")
    print(f"  Chemical (Isp={Isp_chem:.0f} s):  {mp_chem:.1f} kg  ({100 * mp_chem / Psyche.m0:.1f}%)")
    print(f"  SPT-140  (Isp={Psyche.Isp:.0f} s): {mp_elec:.1f} kg  ({100 * mp_elec / Psyche.m0:.1f}%)")
    print(f"  Propellant savings:      {mp_chem - mp_elec:.1f} kg  ({100 * (mp_chem - mp_elec) / mp_chem:.1f}%)")


if __name__ == "__main__":
    import numpy as np
    main()
