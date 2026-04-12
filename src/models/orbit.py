from dataclasses import dataclass
import numpy as np

@dataclass
class Orbit_Kep:
    a: float
    e: float
    i: float
    raan: float
    omega: float
    theta: float

@dataclass
class Orbit_Eq:
    a: float | None = None
    f: float | None = None
    g: float | None = None
    h: float | None = None
    k: float | None = None
    L: float | None = None

    def __init__(self, *args):
        self.a, self.f, self.g, self.h, self.k, self.L = args

    # def calculate_max_rates(self, mu, thrust) -> np.ndarray:
    #     adot_xx = 2*thrust*self.a*np.sqrt(self.a/mu)*np.sqrt((1+np.sqrt(self.f**2+self.g**2))/(1-np.sqrt(self.f**2+self.g**2)))
    #     fdot_xx = 2*thrust*np.sqrt(np.sqrt(1+np.sqrt(self.f**2+self.g**2))/mu)
    #     gdot_xx = fdot_xx
    #     s2 = 1+self.h**2+self.k**2
    #     hdot_xx = (fdot_xx/4)*s2/(self.f+np.sqrt(1-self.g**2))
    #     kdot_xx = (fdot_xx/4)*s2/(self.g+np.sqrt(1-self.f**2))

    #     return np.array([adot_xx, fdot_xx, gdot_xx, hdot_xx, kdot_xx])
    
    # def calculate_error(self, target) -> np.ndarray:
    #     return np.array([self.a - target.a,
    #                      self.f - target.f,
    #                      self.g - target.g,
    #                      self.h - target.h,
    #                      self.k - target.k])

@dataclass
class PhaseTarget:
    name: str
    frame: str
    target: Orbit_Eq