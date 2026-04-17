import numpy as np
import sympy as sym
from dataclasses import dataclass, field

from .models.spacecraft import Spacecraft
from .qlaw_sym import symbolic_qlaw
from .utils import cart2eq

@dataclass
class Qlawgains:
    Wp: float=1.0 #ignore for now, penalty for low perigee passages
    W_oe: np.ndarray=field(default_factory=lambda: np.ones(5))
    coast_threshold: float = 0.0

class QlawController:
    def __init__(self, spacecraft, gains = Qlawgains()):
        self.gains = gains
        self.fun_control, self.fun_q_dqdt = symbolic_qlaw()
        self.spacecraft = spacecraft

    def compute_thrust(self, 
                        mu: float,
                      curr: list, 
                      target: list,
                      thrust_mag: float,
                      W_oe: np.ndarray) -> dict:
        
        #TODO: take out thrust mag so that throttle can be calculated after Q

        oe_list = curr
        oeT_list = target
        
        args = [mu, thrust_mag, oe_list, oeT_list,W_oe]
        
        u_r, u_t, u_n, alpha, beta, q = self.fun_control(*args)
        
        print("u_rtn ", u_r, u_t, u_n)

        return {
            'u_rtn': np.array([u_r, u_t, u_n]),  # radial, tangential, normal
            'alpha': alpha,  # in-plane angle
            'beta': beta,    # out-of-plane angle
            'Q': q
        }
    
    def compute_throttle(self, Q, Qprev=None):
        # TODO: model in some varying throttle based on available solar power...
        if not Qprev:
            return 1.0
        if Q>self.gains.coast_threshold:
            return 1.0
        return 0.0
    
    def control(self, mu, y, target, Qprev = None):
        thrust_accel = self.spacecraft.max_thrust/y[6]
        curr_orbit = cart2eq(y[0:6])
        target_orbit = target
        thrust_dict = self.compute_thrust(mu, curr_orbit, target_orbit, thrust_accel, self.gains.W_oe)
        Q = thrust_dict["Q"]
        thrust_mag = self.compute_throttle(Q, Qprev = Qprev)*self.spacecraft.max_thrust
        
        if np.linalg.norm([thrust_dict["alpha"], thrust_dict["beta"]]) == 0.0:
            return np.zeros(3), Q
        
        return thrust_mag*thrust_dict["u_rtn"], Q

if __name__ == "__main__":
    mu_earth = 3.986e14
    
    sc = Spacecraft()
    qlaw = QlawController(sc)
    
    curr = sc.geprint("=== TEST 1: Perfect orbit match ===")
    mu = 3.986e14
    curr = [7000e3, 0.01, 0.0, 0.0, 0.0, 0.0]  # a=7000km, near-circular LEO
    target = curr.copy()  # IDENTICAL
    
    result = qlaw.compute_thrust(mu, curr, target, 1e-3, np.ones(5))
    print(f"u_rtn: {result['u_rtn']}")