import numpy as np
from .propagator import Propagator
from .qlaw import Qlaw, Qlawgains, Qlawtarget

class Optimizer:
    def __init__(self, cfg, spacecraft, earth, mars, target_orbit):
        self.cfg = cfg #TODO: make config file for solver settings and how many nodes
        self.spacecraft = spacecraft
        self.earth = earth
        self.mars = mars
        self.propagator = Propagator(earth, spacecraft)
        self.qlaw = Qlaw(spacecraft, target_orbit, Qlawgains())

    def qlaw_control(self, mu):
        qprev = {"val": None}
        def u_fun(t,y):
            u,q = self.qlaw.control(mu,y,qprev = qprev["value"])
            qprev["value"] = q
            return u
        return u_fun
    
    def initial_guess(self, y0, t0, tf):
        n = self.cfg.num_nodes
        t_nodes = np.linspace(t0, tf, n)
        u_fun = self.qlaw_control(self.earth.mu)
        sol = self.propagator.forward(y0, (t0, tf), u_fun, t_nodes)

        #TODO: sample this with collocation nodes

    def solve(self, y0, t0, tf):
        init = self.initial_guess(y0, t0, tf)
        #TODO: run init guess through collocation
