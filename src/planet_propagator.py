import numpy as np

# =========================
# Constants
# =========================
AU = 1.495978707e11  # meters

# Epoch: near a favorable Earth-Mars departure window in 2033
epoch = "2033-04-15T00:00:00"

# Simplified heliocentric orbital elements
earth_a = 1.000000 * AU
earth_e = 0.0167
earth_i = np.deg2rad(0.0)

mars_a  = 1.523679 * AU
mars_e  = 0.0934
mars_i  = np.deg2rad(1.85)

mu_sun = 1327e20  # ensure this is defined in your environment

# =========================
# Kepler solver (Newton)
# =========================
def solve_kepler(M, e, tol=1e-10):
    E = M.copy()
    for _ in range(50):
        f = E - e*np.sin(E) - M
        fp = 1 - e*np.cos(E)
        E_new = E - f/fp
        if np.max(np.abs(E_new - E)) < tol:
            break
        E = E_new
    return E

# =========================
# Orbital state from elements
# =========================
def coe2rv(a, e, i, Omega, omega, nu, mu):
    # perifocal position/velocity
    p = a * (1 - e**2)

    r_pf = np.array([
        p*np.cos(nu)/(1 + e*np.cos(nu)),
        p*np.sin(nu)/(1 + e*np.cos(nu)),
        0.0
    ])

    v_pf = np.array([
        -np.sqrt(mu/p)*np.sin(nu),
        np.sqrt(mu/p)*(e + np.cos(nu)),
        0.0
    ])

    # rotation matrix (PQW -> inertial)
    cO, sO = np.cos(Omega), np.sin(Omega)
    co, so = np.cos(omega), np.sin(omega)
    ci, si = np.cos(i), np.sin(i)

    R3_W = np.array([[cO, -sO, 0],
                     [sO,  cO, 0],
                     [0,   0,  1]])

    R1_i = np.array([[1, 0, 0],
                     [0, ci, -si],
                     [0, si, ci]])

    R3_w = np.array([[co, -so, 0],
                     [so,  co, 0],
                     [0,   0,  1]])

    Q = R3_W @ R1_i @ R3_w

    r = Q @ r_pf
    v = Q @ v_pf

    return r.flatten(), v.flatten()

# =========================
# Earth state
# =========================
def earth_state(t):
    n = np.sqrt(mu_sun / earth_a**3)

    M = n * t
    E = solve_kepler(M, earth_e)
    nu = 2*np.arctan2(np.sqrt(1+earth_e)*np.sin(E/2),
                      np.sqrt(1-earth_e)*np.cos(E/2))

    Omega = 0.0
    omega = 0.0

    return coe2rv(earth_a, earth_e, earth_i, Omega, omega, nu, mu_sun)

# =========================
# Mars state
# =========================
def mars_state(t):
    n = np.sqrt(mu_sun / mars_a**3)

    M = n * t + np.deg2rad(50.0)  # phase offset so Earth/Mars aren't aligned
    E = solve_kepler(M, mars_e)
    nu = 2*np.arctan2(np.sqrt(1+mars_e)*np.sin(E/2),
                      np.sqrt(1-mars_e)*np.cos(E/2))

    Omega = 0.0
    omega = 0.0

    return coe2rv(mars_a, mars_e, mars_i, Omega, omega, nu, mu_sun)