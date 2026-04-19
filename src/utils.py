import numpy as np
from .models.body import Body
from .models.orbit import Orbit_Eq, Orbit_Kep
from .models.spacecraft import Spacecraft

def wrap_angle(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi

def oe2cart(mu: float, oe: Orbit_Eq):
    pass
def cart2oe(mu: float, cart: np.ndarray):
    pass

def cart2eq(cart: np.ndarray):
    mu = 1.327e20 # always heliocentric MEE
    r = np.asarray(cart[0:3], dtype=float)
    v = np.asarray(cart[3:], dtype=float)
    rmag = np.linalg.norm(r)

    hvec = np.cross(r, v)
    hmag = np.linalg.norm(hvec)
    if hmag < 1e-14:
        raise ValueError("Degenerate state: angular momentum too small.")

    evec = (np.cross(v, hvec) / mu) - r / rmag
    e = np.linalg.norm(evec)
    p = hmag**2 / mu
    a = p / (1.0 - e**2) if e < 1.0 else np.inf

    k_hat = np.array([0.0, 0.0, 1.0])
    nvec = np.cross(k_hat, hvec)
    nmag = np.linalg.norm(nvec)
    inc = np.arccos(np.clip(hvec[2] / hmag, -1.0, 1.0))
    raan = 0.0 if nmag < 1e-14 else np.arctan2(nvec[1], nvec[0])

    if nmag < 1e-14 or e < 1e-14:
        argp = 0.0
    else:
        argp = np.arctan2(np.dot(np.cross(nvec, evec), hvec) / hmag, np.dot(nvec, evec))

    if e < 1e-14:
        nu = np.arctan2(np.dot(np.cross(nvec, r), hvec) / hmag, np.dot(nvec, r)) if nmag >= 1e-14 else np.arctan2(r[1], r[0])
    else:
        nu = np.arctan2(np.dot(np.cross(evec, r), hvec) / hmag, np.dot(evec, r))

    # L = wrap_angle(raan + argp + nu)

    # f = e * np.cos(raan + argp)
    # g = e * np.sin(raan + argp)
    # t = np.tan(inc / 2.0)
    # h = t * np.cos(raan)
    # k = t * np.sin(raan)

    k_hat = np.array([0.0, 0.0, 1.0])
    nvec = np.cross(k_hat, hvec)
    nmag = np.linalg.norm(nvec)

    inc = np.arccos(np.clip(hvec[2] / hmag, -1.0, 1.0))
    raan = 0.0 if nmag < 1e-14 else np.arctan2(nvec[1], nvec[0])

    hhat = hvec / hmag
    h = -hhat[1] / (1.0 + hhat[2])
    k =  hhat[0] / (1.0 + hhat[2])
    g = e * np.sin(raan + argp)
    f = e * np.cos(raan + argp)

    L = wrap_angle(raan + argp + nu)

    return [a,f,g,h,k,L]

def eq2cart(mu: float, eq: np.ndarray):
    pass