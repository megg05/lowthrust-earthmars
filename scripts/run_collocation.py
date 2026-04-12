"""Heliocentric Earth-to-Mars low-thrust transfer optimization."""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.spacecraft import Spacecraft, Psyche
from models.body import Earth, Mars, Sun
from collocation import LowThrustOptimizer


def main():
    sc_dim = Psyche

    r1 = Earth.a_sun
    r2 = Mars.a_sun
    mu = Sun.mu

    # canonical units
    LU = r1
    TU = np.sqrt(r1**3 / mu)
    VU = LU / TU
    MU_sc = sc_dim.m0

    # nondimensional spacecraft
    ve = sc_dim.Isp * sc_dim.g0
    FU = MU_sc * VU / TU
    sc_nd = Spacecraft(m0=1.0, Tlim=sc_dim.Tlim / FU, Isp=ve / VU, g0=1.0)

    r2_nd = r2 / LU
    vf_nd = np.sqrt(1.0 / r2_nd)
    y0 = np.array([1.0, 0.0, 0.0, 1.0, 1.0])  # circular at r=1

    tf_days = 400.0
    tf_nd = tf_days * 86400.0 / TU

    n_seg = 10
    opt = LowThrustOptimizer(mu=1.0, spacecraft=sc_nd, n_segments=n_seg)
    t_nodes = np.linspace(0.0, tf_nd, n_seg + 1)

    print("=== Low-Thrust Transfer: Earth -> Mars (heliocentric, planar) ===\n")
    print(f"Spacecraft: m0={sc_dim.m0} kg, Tlim={sc_dim.Tlim} N, Isp={sc_dim.Isp} s")
    print(f"Transfer: {tf_days:.0f} days,  Segments: {n_seg}")
    print()

    result = opt.solve(y0, r2_nd, vf_nd, t_nodes, tol=1e-3)

    X = result.X
    m_final = X[-1, 4] * MU_sc
    r_final = np.sqrt(X[-1, 0]**2 + X[-1, 1]**2) * LU
    v_final = np.sqrt(X[-1, 2]**2 + X[-1, 3]**2) * VU
    throttles = result.throttles

    print(f"\nSuccess:       {result.success}")
    print(f"Radius error:  {result.r_error:.2e} ({result.r_error/r2_nd*100:.3f}%)")
    print(f"Velocity error:{result.v_error:.2e} ({result.v_error/vf_nd*100:.3f}%)")
    print()
    print(f"Final mass:    {m_final:.1f} kg  (propellant: {sc_dim.m0 - m_final:.1f} kg)")
    print(f"Final radius:  {r_final:.3e} m  (target {r2:.3e} m)")
    print(f"Final velocity:{v_final:.1f} m/s  (target {np.sqrt(mu/r2):.1f} m/s)")
    print(f"Max throttle:  {throttles.max():.3f}")
    print(f"Mean throttle: {throttles.mean():.3f}")
    print(f"Coast fraction:{np.mean(throttles < 0.01):.1%}")


if __name__ == "__main__":
    main()
