import numpy as np
import sympy as sym
from dataclasses import dataclass, field
# from utils import cart2eq, cart2oe, oe2cart, wrap_angle
from .models.orbit import Orbit_Eq, PhaseTarget
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
        self.mu = None
        self.target = None
    
    def set_target(self, mu, target: Orbit_Eq):
        self.mu = mu
        self.target = target

    def compute_thrust(self, 
                      curr: list, 
                      target: list,
                      thrust_mag: float,
                      W_oe: np.ndarray) -> dict:
        
        #TODO: take out thrust mag so that throttle can be calculated after Q

        oe_list = curr
        oeT_list = target
        
        args = [self.mu, thrust_mag, oe_list, oeT_list,W_oe]
        
        u_r, u_t, u_n, alpha, beta, q = self.fun_control(*args)
        
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
        thrust_dict = self.compute_thrust(curr_orbit, target_orbit, thrust_accel, self.gains.W_oe)
        Q = thrust_dict["Q"]
        thrust_mag = self.compute_throttle(Q, Qprev = Qprev)*self.spacecraft.max_thrust
        
        if np.linalg.norm([thrust_dict["alpha"], thrust_dict["beta"]]) == 0.0:
            return np.zeros(3), Q
        
        return thrust_mag*thrust_dict["u_rtn"], Q

if __name__ == "__main__":
    mu_earth = 3.986e14
    
    qgains = Qlawgains()
    sc = Spacecraft()
    qlaw = QlawController(mu_earth,qgains, sc)
    
    curr = sc.get_state_eq()
    target_orbit = Orbit_Eq(a=8000e3, f=0.0, g=0.0, h=0.0, k=0.0)
    target = PhaseTarget(name="target", frame="earth", target = target_orbit)
        
    thrust_cmd = qlaw.compute_thrust(curr, target_orbit, thrust_mag=1e-3, W_oe=qlaw.gains.W_oe)
    print(f"Thrust direction (RTN): {thrust_cmd['u_rtn']}")
    print(f"Alpha: {np.degrees(thrust_cmd['alpha']):.1f}, Beta: {np.degrees(thrust_cmd['beta']):.1f}")