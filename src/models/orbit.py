from dataclasses import dataclass
import numpy as np

@dataclass
class Orbit:
    a: float
    e: float
    i: float
    raan: float
    omega: float
    theta: float

def oe2cart(mu: float, oe: Orbit):
    pass
def cart2oe(mu: float, cart: np.ndarray):
    pass