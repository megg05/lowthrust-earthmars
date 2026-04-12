from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.body import Earth, Mars, Sun
from models.spacecraft import Psyche
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

    # Propellant comparison: chemical vs Psyche SEP
    Isp_chem = 320.0  # bipropellant
    mp_chem = propellant_mass(result.dv_total, Psyche.m0, Isp_chem)
    mp_elec = propellant_mass(result.dv_total, Psyche.m0, Psyche.Isp)

    print("=== Propellant comparison (Tsiolkovsky, Psyche spacecraft) ===\n")
    print(f"Spacecraft wet mass: {Psyche.m0:.0f} kg")
    print(f"Chemical (Isp={Isp_chem:.0f} s):   {mp_chem:.1f} kg propellant  ({100*mp_chem/Psyche.m0:.1f}%)")
    print(f"SPT-140  (Isp={Psyche.Isp:.0f} s): {mp_elec:.1f} kg propellant  ({100*mp_elec/Psyche.m0:.1f}%)")
    print(f"Propellant savings:        {mp_chem - mp_elec:.1f} kg  ({100*(mp_chem - mp_elec)/mp_chem:.1f}%)")


if __name__ == "__main__":
    main()
