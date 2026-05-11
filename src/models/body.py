from dataclasses import dataclass, field
import numpy as np
from kepler import solve_kepler, eccentric_to_true, elements_to_cartesian


# J2000 epoch: 2000-01-01 12:00:00 TDB
J2000 = 0.0


@dataclass
class Body:
    name: str
    mu: float               # gravitational parameter [m^3/s^2]
    r: float                # physical radius [m]
    a_sun: float = 0.0      # heliocentric semi-major axis [m]
    e_sun: float = 0.0      # eccentricity
    i_sun: float = 0.0      # inclination [rad]
    Omega_sun: float = 0.0  # RAAN [rad]
    omega_sun: float = 0.0  # argument of periapsis [rad]
    M0_sun: float = 0.0     # mean anomaly at epoch [rad]
    epoch: float = J2000    # reference epoch [s from J2000]
    mu_sun: float = 1.327124400e20  # Sun mu, needed for state computation

    def mean_anomaly_at(self, t):
        """Mean anomaly at time t [seconds from J2000]."""
        n = np.sqrt(self.mu_sun / self.a_sun**3)  # mean motion
        return self.M0_sun + n * (t - self.epoch)

    def elements_at(self, t):
        """Keplerian elements at time t [seconds from J2000].

        Returns dict with keys: a, e, i, Omega, omega, nu, M.
        Only mean anomaly (and thus true anomaly) changes; other elements
        are held constant (no secular perturbations).
        """
        M = self.mean_anomaly_at(t)
        E = solve_kepler(M, self.e_sun)
        nu = eccentric_to_true(E, self.e_sun)
        return {
            "a": self.a_sun,
            "e": self.e_sun,
            "i": self.i_sun,
            "Omega": self.Omega_sun,
            "omega": self.omega_sun,
            "nu": nu,
            "M": M % (2 * np.pi),
        }

    def state_at(self, t):
        """Heliocentric Cartesian state at time t [seconds from J2000].

        Returns
        -------
        r : ndarray(3) — position [m]
        v : ndarray(3) — velocity [m/s]
        """
        elems = self.elements_at(t)
        return elements_to_cartesian(
            elems["a"], elems["e"], elems["i"],
            elems["Omega"], elems["omega"], elems["nu"],
            self.mu_sun,
        )


# --- J2000 mean orbital elements ---
# Source: Standish (1992) / JPL planetary ephemerides

Earth = Body(
    name="earth",
    mu=3.986004418e14,
    r=6378137.0,
    a_sun=1.49598023e11,       # [m]
    e_sun=0.0167086,
    i_sun=np.radians(0.00005),  # ~0 by definition (ecliptic reference)
    Omega_sun=np.radians(-11.26064),
    omega_sun=np.radians(114.20783),
    M0_sun=np.radians(358.617),
)

Mars = Body(
    name="mars",
    mu=4.282837e13,
    r=3389500.0,
    a_sun=2.27939366e11,       # [m]
    e_sun=0.0934,
    i_sun=np.radians(1.85061),
    Omega_sun=np.radians(49.57854),
    omega_sun=np.radians(286.50180),
    M0_sun=np.radians(19.373),
)

Sun = Body(
    name="sun",
    mu=1.327124400e20,
    r=6.957e8,
)
