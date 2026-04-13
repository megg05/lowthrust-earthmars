from dataclasses import dataclass
from .orbit import Orbit_Eq, Orbit_Kep
import numpy as np

@dataclass
class Spacecraft:
    # TODO:  psyche params
    m0: float = 1648+1085 #dry mass + propellant
    min_thrust: float = 0.0
    max_thrust: float = 0.240
    # min_power: float = 1000.0 #W
    # max_power: float = 4500.0 #W
    Isp: float = 1800
    g0: float = 9.80665 #m/s^2
    
    @property
    def exhaust_vel(self) -> float:
        return self.Isp*self.g0

    def mdot(self, thrust: float) -> float:
        return -abs(thrust)/self.exhaust_vel


# SPT-140 Hall thruster, from Lord & Tilley (2017) IEEE Aerospace
# Psyche = Spacecraft(
#     m0=2400.0,   # wet mass: 1300 kg dry + 1100 kg xenon
#     Tlim=0.24,   # single SPT-140 thrust [N]
#     Isp=1800.0,  # Hall thruster specific impulse [s]
# )
    
