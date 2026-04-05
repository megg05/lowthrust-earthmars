from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from config import SolverConfig
from models.spacecraft import Spacecraft
from models.body import Earth, Mars
from models.orbit import Orbit
from optimizer import Optimizer


def make_initial_state(mu, alt_km=300.0):
    r0 = Earth.radius + alt_km * 1000.0
    v0 = np.sqrt(mu / r0)
    return np.array([r0, 0.0, 0.0, 0.0, v0, 0.0, 1500.0])


def main():
    cfg = SolverConfig(n_nodes=20, max_iter=20, tol=1e-6, method="SLSQP")
    sc = Spacecraft(m0=1500.0, Tmax=0.25, Isp=1800.0)

    earth_parking = Orbit(rp_alt=300.0, inc=28.5)
    mars_parking = Orbit(rp_alt=300.0, inc=90.0)

    opt = Optimizer(cfg, sc, Earth, Mars)

    y0 = make_initial_state(Earth.mu, alt_km=earth_parking.rp_alt)

    tf = 60.0 * 24.0 * 3600.0
    z0 = opt.build_initial_guess(y0=y0, t0=0.0, tf=tf)

    assert z0 is not None, "Initial guess was not built."
    assert np.all(np.isfinite(z0)), "Initial guess contains non-finite values."
    assert len(z0) > 0, "Initial guess vector is empty."

    result = opt.solve(y0=y0, t0=0.0, tf=tf)

    assert result is not None, "Optimizer returned no result."
    assert hasattr(result, "success"), "Result does not look like a scipy.optimize result."
    assert hasattr(result, "fun"), "Result missing objective value."

    print("Smoke test passed.")
    print("Success:", result.success)
    print("Message:", result.message)
    print("Objective:", result.fun)


if __name__ == "__main__":
    main()