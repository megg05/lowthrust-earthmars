import numpy as np
from .propagator import Propagator
from .qlaw import QlawController, Qlawgains
from .utils import cart2eq

class Optimizer:
    def __init__(self, cfg, spacecraft, target_orbit):
        self.cfg = cfg #TODO: make config file for solver settings and how many nodes
        self.spacecraft = spacecraft
        self.propagator = Propagator(spacecraft)

        self.qlaw = QlawController(spacecraft) # arbitrary gains for now
        self.target = target_orbit

    def qlaw_control(self):
        qprev = {"val": None}
        def u_fun(t,y):
            u,Q = self.qlaw.control(self.propagator.body.mu,y, self.target,Qprev = qprev["val"])
            qprev["val"] = Q
            return u
        return u_fun
    
    def trajectory(self, y0, t0, tf):
        n = self.cfg.num_nodes
        t_nodes = np.linspace(t0, tf, n)
        u_fun = self.qlaw_control()
        sol = self.propagator.forward(y0, (t0, tf), u_fun, t_nodes)
        return sol

    def solve(self, y0, t0, tf):
        init = self.initial_guess(y0, t0, tf)
        print(init)
