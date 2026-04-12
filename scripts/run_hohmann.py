from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.body import Earth, Mars, Sun
from hohmann import hohmann, propellant_mass


def main():
    result = hohmann(Earth.a_sun, Mars.a_sun, Sun.mu)

    print("=== Hohmann Transfer: Earth -> Mars (heliocentric) ===\n")
    print(f"Departure orbit radius:  {Earth.a_sun:.3e} m  ({Earth.a_sun/1.496e11:.3f} AU)")
    print(f"Arrival orbit radius:    {Mars.a_sun:.3e} m  ({Mars.a_sun/1.496e11:.3f} AU)")
    print(f"Transfer semi-major axis:{result.a_transfer:.3e} m  ({result.a_transfer/1.496e11:.3f} AU)")
    print()
    print(f"Departure dv1:  {result.dv1:.1f} m/s")
    print(f"Arrival   dv2:  {result.dv2:.1f} m/s")
    print(f"Total     dv:   {result.dv_total:.1f} m/s")
    print(f"Transfer time:  {result.tof:.1f} s  ({result.tof/86400:.1f} days)")
    print()

    # Propellant comparison: chemical vs electric
    m0 = 1500.0  # kg, same as Psyche reference

    # Chemical (bipropellant ~ 320 s Isp)
    Isp_chem = 320.0
    mp_chem = propellant_mass(result.dv_total, m0, Isp_chem)

    # Electric (Hall-effect ~ 1800 s Isp, Psyche-class)
    Isp_elec = 1800.0
    mp_elec = propellant_mass(result.dv_total, m0, Isp_elec)

    print("=== Propellant comparison (Tsiolkovsky) ===\n")
    print(f"Spacecraft wet mass: {m0:.0f} kg")
    print(f"Chemical (Isp={Isp_chem:.0f} s):  {mp_chem:.1f} kg propellant  ({100*mp_chem/m0:.1f}%)")
    print(f"Electric (Isp={Isp_elec:.0f} s): {mp_elec:.1f} kg propellant  ({100*mp_elec/m0:.1f}%)")
    print(f"Propellant savings:       {mp_chem - mp_elec:.1f} kg  ({100*(mp_chem - mp_elec)/mp_chem:.1f}%)")


if __name__ == "__main__":
    main()
