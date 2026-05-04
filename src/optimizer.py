import numpy as np
from .propagator import Propagator
from .qlaw import QlawController, Qlawgains
from .utils import cart2eq, cart2kep

AU = 1.496e+11
class Optimizer:
    def __init__(self, cfg, spacecraft, target_orbit):
        self.cfg = cfg #TODO: make config file for solver settings and how many nodes
        self.spacecraft = spacecraft
        self.propagator = Propagator(spacecraft)

        self.controller = QlawController(spacecraft) # arbitrary gains for now
        self.phasing_tolerances = np.array([0.05*AU, 0.02, 0.02, 0.01, 0.01])
        self.final_tolerances = np.array([8e8, 0.002, 0.002, 0.002, 0.002, 0.17])
        self.target = target_orbit
        self.true_target = target_orbit.copy()

    def simulate_forward(self, t0, tf, y0, dt=86400.0):
        t = t0
        y = y0
        ts = [t0]
        ys = [y0]
        t_hists = []
        u_hists = []
        q_hists = []
        kep0 = cart2kep(y0[0:6])
        a_s = [kep0[0]]
        e_s = [kep0[1]]

        self.controller.compute_mee_max_rates(self.propagator.body.mu, y)

        while t < t0+tf:
            t_next = min(t + dt, t0+tf)

            if self.controller.phase == 2 and self.check_converge(y):
                break
            
            self.update_target(t,y)

            if ((t-t0)/86400)%5 == 0:
                self.controller.compute_mee_max_rates(self.propagator.body.mu,y)
            sol, t_hist, u_hist, q_hist = self.propagator.forward(
                y0=y,
                tspan=(t, t_next),
                control=self.qlaw_control()
            )
            y = sol.y[:, -1]
            t = t_next
            ts.extend(sol.t[1:])
            ys.extend(sol.y.T[1:])
            t_hists.extend(t_hist)
            u_hists.extend(u_hist)
            q_hists.extend(q_hist)
            kepy = cart2kep(y[0:6])
            a_s.extend([kepy[0]])
            e_s.extend([kepy[1]])

        return np.array(ts), np.array(ys).T, np.array(a_s), np.array(e_s), np.array(t_hists), np.array(u_hists), np.array(q_hists)
    
    # def simulate_backward(self, t0, tf, y0, dt=86400.0):
    #     # t0 is later time; tf is earlier time; t0 > tf
    #     t = t0
    #     y = y0
    #     ts = [t0]
    #     ys = [y0]
    #     kep0 = cart2kep(y[0:6])
    #     a_s = [kep0[0]]
    #     e_s = [kep0[1]]


    #     while t > tf:   # reverse loop: t decreases
    #         t_next = max(t - dt, tf)
    #         print(t/86400.0)

    #         sol = self.propagator.backward(
    #             y0=y,
    #             tspan=(t, t_next),   # t > t_next
    #             control=self.qlaw_control()
    #         )
    #         y = sol.y[:, -1]
    #         t = t_next
    #         ts.extend(sol.t[1:])
    #         ys.extend(sol.y.T[1:])
    #         kepy = cart2kep(y[0:6])
    #         a_s.append(kepy[0])
    #         e_s.append(kepy[1])

    #     return np.array(ts), np.array(ys).T, np.array(a_s), np.array(e_s)

    def qlaw_control(self):
        qprev = [None]
        def u_fun(t,y):
            u,Q = self.controller.control(self.propagator.body.mu, y, self.target, Qprev=qprev[0])
            qprev[0] = Q
            return u, Q
        return u_fun
    
    def update_target(self, t,y):
        # if self.controller.phase == 1:
        #     errors = abs(y[0:5] - self.target[0:5])
        #     if all(errors[i] > self.phasing_tolerances[i] for i in range(5)):
        #         return
        #     else:
        #         self.controller.phase = 2
        # print("SWITCH")
        # dL = (y[5] - self.target[5] + np.pi) % (2*np.pi) - np.pi
        # n_mars = np.sqrt(self.propagator.body.mu / self.target[0]**3)
        # da_ref = 0.03*self.target[0]
        # dn_ref = 1.5*n_mars/self.target[0]*da_ref
        # tr = np.clip(abs(dL) / dn_ref, 90*86400,500*86400)
        # n_desired = n_mars - dL/tr
        # aT_biased = (self.propagator.body.mu / n_desired**2)**(1/3)
        # aT_biased = np.clip(aT_biased, self.target[0] - 0.04*AU, self.target[0] + 0.04*AU)
        # self.target[0] = aT_biased
        # return
        curr_eq = cart2eq(y[0:6])
        
        # 1. Evaluate "Shape" Errors (a, f, g, h, k)
        # We use a relative error for 'a' because its magnitude is so large (1e11)
        errors = np.zeros(5)
        errors[0] = abs(curr_eq[0] - self.true_target[0]) / self.true_target[0] # Relative error for SMA
        errors[1:5] = abs(np.array(curr_eq[1:5]) - np.array(self.true_target[1:5]))           # Absolute for others
        
        # 2. Check for Phase Transition
        if self.controller.phase == 1:
            # Switch to Phase 2 if we are within ~2-5% of the target shape
            # This prevents the controller from "stalling" out in Phase 1
            shape_converged = all(errors[i] < self.phasing_tolerances[i] for i in range(5))
            
            if shape_converged:
                print("--- SWITCHING TO PHASE 2 (PHASING) ---")
                self.controller.phase = 2
                print(t)
                print(y)
            # elif t>800*86400:
            #     print("forcing phase 2")
            #     self.controller.phase = 2
            else:
                return # Stay in Phase 1, targeting the original Mars orbit

        # 3. Phase 2: Biased Semi-Major Axis (Longitude Targeting)
        # We create a temporary 'a' target to drift into the correct position
        
        # Calculate the wrap-around longitude error
        dL = (curr_eq[5] - self.true_target[5] + np.pi) % (2*np.pi) - np.pi
        
        # Calculate mean motion of the target (Mars)
        n_mars = np.sqrt(self.propagator.body.mu / self.true_target[0]**3)
        
        # Calculate the time required to close the gap (tr)
        # We limit this so we don't try to drift too fast or too slow
        # 1.5*n_mars is a standard heuristic for drift rates
        da_ref = 0.03 * self.true_target[0] # Allow 5% variation in SMA for drift
        dn_ref = 1.5 * n_mars / self.true_target[0] * da_ref
        
        tr = np.clip(abs(dL) / dn_ref, 90*86400, 500*86400) # Between 30 and 300 days
        
        # Desired drift rate
        n_desired = n_mars - dL/tr
        
        # The "Biased" semi-major axis target
        aT_biased = (self.propagator.body.mu / n_desired**2)**(1/3)
        
        # CRITICAL: Increase clipping range. 0.04 AU was too small for your 0.07 AU error.
        # We clip it to ensure the transfer doesn't become parabolic or crash into the Sun.
        self.target[0] = np.clip(aT_biased, 1.2 * AU, 1.8 * AU) 

        return
    

    
    def check_converge(self, y):
        errors = abs(np.array(cart2eq(y[0:6]))-np.array(self.true_target))
        if all(errors[i] < self.final_tolerances[i] for i in range(6)):
            return True
        return False
    
    
        
    # def trajectory(self, y0, t0, tf):
    #     n = self.cfg.num_nodes
    #     t_nodes = np.linspace(t0, tf, n)
    #     u_fun = self.qlaw_control()
    #     sol = self.propagator.forward(y0, (t0, tf), u_fun, t_nodes)
    #     return sol
