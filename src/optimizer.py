import numpy as np
from .propagator import Propagator
from .qlaw import QlawController, Qlawgains
from .utils import cart2eq, cart2kep

class Optimizer:
    def __init__(self, cfg, spacecraft, target_orbit):
        self.cfg = cfg #TODO: make config file for solver settings and how many nodes
        self.spacecraft = spacecraft
        self.propagator = Propagator(spacecraft)

        self.qlaw = QlawController(spacecraft) # arbitrary gains for now
        self.target = target_orbit

    def simulate_forward(self, t0, tf, y0, dt=86400.0):
        t = t0
        y = y0
        ts = [t0]
        ys = [y0]
        kep0 = cart2kep(y0[0:6])
        a_s = [kep0[0]]
        e_s = [kep0[1]]

        while t < t0+tf:
            t_next = min(t + dt, t0+tf)
            sol = self.propagator.forward(
                y0=y,
                tspan=(t, t_next),
                control=self.qlaw_control()
            )
            y = sol.y[:, -1]
            t = t_next
            ts.extend(sol.t[1:])
            ys.extend(sol.y.T[1:])
            kepy = cart2kep(y[0:6])
            a_s.extend([kepy[0]])
            e_s.extend([kepy[1]])

        return np.array(ts), np.array(ys).T, np.array(a_s), np.array(e_s)
    
    def simulate_backward(self, t0, tf, y0, dt=86400.0):
        # t0 is later time; tf is earlier time; t0 > tf
        t = t0
        y = y0
        ts = [t0]
        ys = [y0]
        kep0 = cart2kep(y[0:6])
        a_s = [kep0[0]]
        e_s = [kep0[1]]


        while t > tf:   # reverse loop: t decreases
            t_next = max(t - dt, tf)
            print(t/86400.0)

            sol = self.propagator.backward(
                y0=y,
                tspan=(t, t_next),   # t > t_next
                control=self.qlaw_control()
            )
            y = sol.y[:, -1]
            t = t_next
            ts.extend(sol.t[1:])
            ys.extend(sol.y.T[1:])
            kepy = cart2kep(y[0:6])
            a_s.append(kepy[0])
            e_s.append(kepy[1])

        return np.array(ts), np.array(ys).T, np.array(a_s), np.array(e_s)

    def qlaw_control(self):
        qprev = [None]
        def u_fun(t,y):
            u,Q = self.qlaw.control(self.propagator.body.mu, y, self.target, Qprev=qprev[0])
            qprev[0] = Q
            return u
        return u_fun
    
    
    def trajectory(self, y0, t0, tf):
        n = self.cfg.num_nodes
        t_nodes = np.linspace(t0, tf, n)
        u_fun = self.qlaw_control()
        sol = self.propagator.forward(y0, (t0, tf), u_fun, t_nodes)
        return sol
