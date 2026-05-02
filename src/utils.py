import numpy as np
from .models.body import Body
from .models.orbit import Orbit_Eq, Orbit_Kep
from .models.spacecraft import Spacecraft

def wrap_angle(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi

def oe2cart(mu: float, oe: Orbit_Eq):
    pass
def cart2kep(cart: np.ndarray):
    mu = 1.327e20  # heliocentric
    r = np.asarray(cart[0:3], dtype=float)
    v = np.asarray(cart[3:6], dtype=float)

    rmag = np.linalg.norm(r)
    vmag = np.linalg.norm(v)
    if rmag < 1e-14:
        raise ValueError("Degenerate state: position magnitude too small.")

    hvec = np.cross(r, v)
    hmag = np.linalg.norm(hvec)
    if hmag < 1e-14:
        raise ValueError("Degenerate state: angular momentum too small.")

    evec = (np.cross(v, hvec) / mu) - (r / rmag)
    e = np.linalg.norm(evec)

    energy = 0.5 * vmag**2 - mu / rmag
    if abs(energy) < 1e-14:
        a = np.inf
    else:
        a = -mu / (2.0 * energy)

    k_hat = np.array([0.0, 0.0, 1.0])
    nvec = np.cross(k_hat, hvec)
    nmag = np.linalg.norm(nvec)

    inc = np.arccos(np.clip(hvec[2] / hmag, -1.0, 1.0))

    if nmag < 1e-14:
        raan = 0.0
    else:
        raan = np.arctan2(nvec[1], nvec[0])

    if e < 1e-14 or nmag < 1e-14:
        argp = 0.0
    else:
        argp = np.arctan2(
            np.dot(np.cross(nvec, evec), hvec) / hmag,
            np.dot(nvec, evec)
        )

    if e < 1e-14:
        if nmag < 1e-14:
            nu = np.arctan2(r[1], r[0])
        else:
            nu = np.arctan2(
                np.dot(np.cross(nvec, r), hvec) / hmag,
                np.dot(nvec, r)
            )
    else:
        nu = np.arctan2(
            np.dot(np.cross(evec, r), hvec) / hmag,
            np.dot(evec, r)
        )

    raan = wrap_angle(raan)
    argp = wrap_angle(argp)
    nu = wrap_angle(nu)

    return np.array([a, e, inc, raan, argp, nu], dtype=float)

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

def eq2cart(mee: np.ndarray):
    mu = 1.327e20  # heliocentric
    a, f, g, h, k, L = np.asarray(mee, dtype=float)

    e = np.hypot(f, g)
    p = a * (1.0 - e**2)
    print(a)
    print(e)

    s2 = 1.0 + h*h + k*k
    w = 1.0 + f*np.cos(L) + g*np.sin(L)

    if p <= 0.0:
        raise ValueError("Invalid MEE state: p must be positive.")
    if np.abs(w) < 1e-14:
        raise ValueError("Invalid MEE state: w too small.")

    r_pf = p / w
    r_pqw = np.array([
        r_pf * np.cos(L),
        r_pf * np.sin(L),
        0.0
    ])

    sqrt_mu_p = np.sqrt(mu / p)
    v_pqw = sqrt_mu_p * np.array([
        -np.sin(L),
        e + np.cos(L),
        0.0
    ])

    f_hat = np.array([
        (1.0 - h*h + k*k) / s2,
        2.0*h*k / s2,
        -2.0*h / s2
    ])
    g_hat = np.array([
        2.0*h*k / s2,
        (1.0 + h*h - k*k) / s2,
        2.0*k / s2
    ])
    w_hat = np.array([
        2.0*h / s2,
        -2.0*k / s2,
        (1.0 - h*h - k*k) / s2
    ])

    Q = np.column_stack((f_hat, g_hat, w_hat))

    r = Q @ r_pqw
    v = Q @ v_pqw

    return np.concatenate((r, v))