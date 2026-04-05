from dataclasses import dataclass

@dataclass
class Spacecraft:
    # TODO:  psyche params
    m0: float
    Tlim: float
    Isp: float
    g0: float
    
    @property
    def exhaust_vel(self) -> float:
        return self.Isp*self.g0
    
    def mdot(self, thrust: float) -> float:
        return -abs(thrust)/self.exhaust_vel
    
