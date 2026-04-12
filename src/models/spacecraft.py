from dataclasses import dataclass

@dataclass
class Spacecraft:
    m0: float
    Tlim: float
    Isp: float
    g0: float = 9.80665
    
    @property
    def exhaust_vel(self) -> float:
        return self.Isp*self.g0

    def mdot(self, thrust: float) -> float:
        return -abs(thrust)/self.exhaust_vel


# SPT-140 Hall thruster, from Lord & Tilley (2017) IEEE Aerospace
Psyche = Spacecraft(
    m0=2400.0,   # wet mass: 1300 kg dry + 1100 kg xenon
    Tlim=0.24,   # single SPT-140 thrust [N]
    Isp=1800.0,  # Hall thruster specific impulse [s]
)
    
