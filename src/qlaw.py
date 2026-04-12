import numpy as np
from dataclasses import dataclass
from utils import cart2eq, cart2oe, oe2cart, wrap_angle
from models.orbit import Orbit_Eq

@dataclass
class Qlawgains:
    Wp: float=1.0 #ignore for now, penalty for low perigee passages
    W_oe: np.ndarray=np.array([1.0,1.0,1.0,1.0,1.0])
    coast_threshold: float = 0.0
    objective_mode: str = "min_fuel"

@dataclass
class PhaseTarget:
    name: str
    frame: str
    target: Orbit_Eq
    gains: Qlawgains

class Qlaw:
    def __init__(self, gains, spacecraft, phases):
        self.gains = gains
        self.spacecraft = spacecraft
        self.phases = phases
        self.phase_idx = 0

    def calculate_q(self, mu, curr, target, thrust):
        #any necessity of having incomplete target elements?
        oe_xx = curr.calculate_max_rates(mu, np.linalg.norm(thrust))
        e_oe = curr.calculate_error(target)
        S_oe = np.ones(5)
        S_oe[0] = np.sqrt(1.0+(np.abs(curr.a-target.a)/3.0/target.a)**4)
        Q = np.sum(S_oe*self.gains.W_oe*(e_oe/oe_xx)**2)
        return Q

    
    def calculate_qgrad(self, mu, curr, target, thrust):
        pass

    def q_from_cart(self, mu,y):
        r = y[0:3]
        v = y[3:6]
        eq = cart2eq(mu,r,v)
        return self.calculate_q(eq)

    def thrust_dir(self, mu,y):
        r = y[0:3]
        v = y[3:6]
        vnorm = np.linalg.norm(v)
        if np.linalg.norm(v) < 1e-12:
            return np.zeros(3), 0.0

        q = self.q_from_cart(mu, y)
        qgrad = self.calculate_qgrad(mu, y)
        #approximate descent in velocity space
        d = -qgrad
        dnorm = np.linalg.norm(d)
        if dnorm < 1e-12:
            return v / vnorm, q

        return d / dnorm, q
    
    def throttle(self, q, qprev=None):
        # TODO: model in some varying throttle based on available solar power...
        if not qprev:
            return 1.0
        if q>self.gains.coast_threshold:
            return 1.0
        return 0.0
    
    def control(self, mu, y, qprev = None):
        dir, q = self.thrust_direction(mu,y)
        if np.linalg.norm(dir) == 0.0:
            return np.zeros(3), q
        thrust = self.throttle(q, qprev = qprev)*self.spacecraft.Tlim
        return thrust*dir, q
        