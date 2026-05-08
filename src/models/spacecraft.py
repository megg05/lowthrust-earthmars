from dataclasses import dataclass
from .orbit import Orbit_Eq, Orbit_Kep
import numpy as np

AU = 1.495978707e11

@dataclass
class Spacecraft:
    # TODO:  psyche params
    m0: float = 1648+5000 #dry mass + propellant
    min_thrust: float = 0.0
    max_thrust: float = 0.280*4
    P_1AU: float=21000.0
    eps_eff: float=0.95
    thruster_eff: float=0.55
    P_min: float=900.0
    P_max: float=4500.0
    Isp: float = 1800
    g0: float = 9.80665 #m/s^2
    at_mars: bool=False
    
    @property
    def exhaust_vel(self) -> float:
        return self.Isp*self.g0

    def mdot(self, thrust: float) -> float:
        return -abs(thrust)/self.exhaust_vel

    def solar_power_available(self, r_sun_m: float) -> float:
        if self.at_mars:
            r_sun_m = 1.52*AU
        return self.P_1AU * self.eps_eff * (AU / r_sun_m)**2

    def thrust_mag_from_power(self, P_in: float) -> float:
        T = 2.0 * P_in / self.exhaust_vel
        return np.clip(T, self.min_thrust, self.max_thrust)
