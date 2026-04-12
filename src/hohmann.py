import numpy as np
from dataclasses import dataclass


@dataclass
class HohmannResult:
    a_transfer: float       # transfer ellipse SMA [m]
    dv1: float              # departure dv [m/s]
    dv2: float              # arrival dv [m/s]
    dv_total: float         # total dv [m/s]
    tof: float              # time of flight [s]
    v_dep_circ: float       # circular vel at departure [m/s]
    v_arr_circ: float       # circular vel at arrival [m/s]
    v_dep_transfer: float   # transfer vel at departure [m/s]
    v_arr_transfer: float   # transfer vel at arrival [m/s]


def hohmann(r1, r2, mu):
    """Hohmann transfer between two circular coplanar orbits."""
    a_t = (r1 + r2) / 2.0

    v1_circ = np.sqrt(mu / r1)
    v2_circ = np.sqrt(mu / r2)

    v1_transfer = np.sqrt(mu * (2.0 / r1 - 1.0 / a_t))
    v2_transfer = np.sqrt(mu * (2.0 / r2 - 1.0 / a_t))

    dv1 = abs(v1_transfer - v1_circ)
    dv2 = abs(v2_circ - v2_transfer)

    tof = np.pi * np.sqrt(a_t**3 / mu)

    return HohmannResult(
        a_transfer=a_t,
        dv1=dv1,
        dv2=dv2,
        dv_total=dv1 + dv2,
        tof=tof,
        v_dep_circ=v1_circ,
        v_arr_circ=v2_circ,
        v_dep_transfer=v1_transfer,
        v_arr_transfer=v2_transfer,
    )


def propellant_mass(dv, m0, Isp, g0=9.80665):
    """Propellant mass via Tsiolkovsky equation."""
    return m0 * (1.0 - np.exp(-dv / (Isp * g0)))
