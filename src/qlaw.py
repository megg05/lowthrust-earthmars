import numpy as np
import sympy as sym
from dataclasses import dataclass, field

from .models.spacecraft import Spacecraft
from .qlaw_sym import symbolic_qlaw
from .utils import cart2eq

@dataclass
class Qlawgains:
    Wp: float=1.0 #ignore for now, penalty for low perigee passages
    W_oe: np.ndarray=field(default_factory=lambda: np.array([100.0,100.0,100.0,100.0,100.0,1.0]))
    coast_threshold: float = 0.01
    coast_tolerances: np.ndarray=field(default_factory=lambda: np.array([9e2, 0.008, 0.008, 0.0012, 0.0012]))

class QlawController:
    def __init__(self, spacecraft, gains = Qlawgains()):
        self.gains = gains
        self.fun_control, self.fun_q_dqdt = symbolic_qlaw()
        self.spacecraft = spacecraft
        self.phase = 1

    def compute_thrust(self, 
                        mu: float,
                      curr: list, 
                      target: list,
                      thrust_mag: float,
                      W_oe: np.ndarray) -> dict:
        
        oe_list = curr
        oeT_list = target
        
        args = [mu, thrust_mag, oe_list, oeT_list,W_oe, self.D_mee, self.phase]
        
        u_r, u_t, u_n, alpha, beta, q = self.fun_control(*args)
        _, dqdt = self.fun_q_dqdt(*args)
        
        return {
            'u_rtn': np.array([u_r, u_t, u_n]),  # radial, tangential, normal
            'alpha': alpha,  # in-plane angle
            'beta': beta,    # out-of-plane angle
            'Q': q,
            'Qdot': dqdt
        }
    
    def compute_throttle(self, Qdot, Qprev=None):
        # TODO: model in thrust-coast effectivity
        # TODO: model in some varying throttle based on available solar power...
        if not Qprev:
            return 1.0
        if abs(Qdot)>self.gains.coast_threshold:
            return 1.0
        return 0.0
    
    def control(self, mu, y, target, Qprev = None):
        thrust_accel = self.spacecraft.max_thrust/y[6]
        curr_orbit = cart2eq(y[0:6])
        target_orbit = target
        thrust_dict = self.compute_thrust(mu, curr_orbit, target_orbit, thrust_accel, self.gains.W_oe)
        Q = thrust_dict["Q"]
        Qdot = thrust_dict["Qdot"]
        if self.phase == 2 and self.close_to_biased(curr_orbit, target):
            throttle = 0.0
        else: 
            throttle = self.compute_throttle(Qdot, Qprev=Qprev)
        thrust_mag = throttle*self.spacecraft.max_thrust
        return thrust_mag*thrust_dict["u_rtn"],Q
    
    def close_to_biased(self, curr, target):
        errors = abs(np.array(curr[0:5]) - np.array(target[0:5]))
        if all(errors[i] < self.gains.coast_tolerances[i] for i in range(5)):
            return True
        return False
    
    def compute_mee_max_rates(self, mu, y):
        # NOT USED
        """
        Calculates scaling factors (D_oe) for MEE-a: [a, f, g, h, k]
        """
        # y = [a, f, g, h, k, L]
        a, f, g, h, k, L = cart2eq(y[0:6])
        thrust_accel = self.spacecraft.max_thrust / y[6]
        
        # We still need p for the internal geometry calculations
        p = a * (1 - (f**2 + g**2))
        sqrt_p_mu = np.sqrt(p / mu)
        
        D = np.zeros(5) 
        
        # Sample the orbit to find maximum authority
        for L_sample in np.linspace(0, 2*np.pi, 24):
            sinL, cosL = np.sin(L_sample), np.cos(L_sample)
            q_s = 1 + f * cosL + g * sinL
            
            # Max rate of a (Semi-major axis)
            # da/dt = 2*a^2/h * (e*sin(nu)*fr + p/r*ft)
            # Simplified max authority for a:
            da = (2 * a**2 / np.sqrt(mu * p)) * np.sqrt((f*sinL - g*cosL)**2 + q_s**2)
            
            # Max rates of f and g (Eccentricity vector)
            df = sqrt_p_mu * np.sqrt(sinL**2 + ((q_s + 1) * cosL + f)**2)
            dg = sqrt_p_mu * np.sqrt(cosL**2 + ((q_s + 1) * sinL + g)**2)
            
            # Max rates of h and k (Inclination vector)
            s2 = 1 + h**2 + k**2
            dh = (s2 * abs(cosL) / (2 * q_s)) * sqrt_p_mu
            dk = (s2 * abs(sinL) / (2 * q_s)) * sqrt_p_mu
            
            D = np.maximum(D, [da, df, dg, dh, dk])

        # Avoid division by zero
        self.D_mee = np.where(D > 1e-15, D, 1e-15)
        