import numpy as np


def solve_kepler(M, e, tol=1e-12, max_iter=50):
    """Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E.

    Uses Newton-Raphson iteration.

    Parameters
    ----------
    M : float — mean anomaly [rad]
    e : float — eccentricity
    tol : float — convergence tolerance
    max_iter : int — max iterations

    Returns
    -------
    E : float — eccentric anomaly [rad]
    """
    M = M % (2 * np.pi)
    E = M + 0.85 * e * np.sign(np.sin(M))  # smart initial guess
    for _ in range(max_iter):
        dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= dE
        if abs(dE) < tol:
            return E
    raise RuntimeError(f"Kepler solver did not converge: M={M}, e={e}")


def eccentric_to_true(E, e):
    """Convert eccentric anomaly to true anomaly."""
    return 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                          np.sqrt(1 - e) * np.cos(E / 2))


def elements_to_cartesian(a, e, i, Omega, omega, nu, mu):
    """Convert Keplerian elements to heliocentric Cartesian state.

    Parameters
    ----------
    a : float — semi-major axis [m]
    e : float — eccentricity
    i : float — inclination [rad]
    Omega : float — right ascension of ascending node [rad]
    omega : float — argument of periapsis [rad]
    nu : float — true anomaly [rad]
    mu : float — gravitational parameter of central body [m^3/s^2]

    Returns
    -------
    r : ndarray(3) — position [m]
    v : ndarray(3) — velocity [m/s]
    """
    p = a * (1 - e**2)
    r_mag = p / (1 + e * np.cos(nu))

    # perifocal frame
    r_pqw = r_mag * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pqw = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    # rotation matrix: perifocal -> inertial
    cO, sO = np.cos(Omega), np.sin(Omega)
    co, so = np.cos(omega), np.sin(omega)
    ci, si = np.cos(i), np.sin(i)

    R = np.array([
        [cO * co - sO * so * ci, -cO * so - sO * co * ci,  sO * si],
        [sO * co + cO * so * ci, -sO * so + cO * co * ci, -cO * si],
        [so * si,                  co * si,                  ci     ],
    ])

    return R @ r_pqw, R @ v_pqw
