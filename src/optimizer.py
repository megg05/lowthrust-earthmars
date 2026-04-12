import numpy as np
from .collocation import HermiteSimpsonCollocation


class Optimizer:
    def __init__(self, cfg, spacecraft, body_central, rf_target, vf_target):
        self.cfg = cfg
        self.spacecraft = spacecraft
        self.collocation = HermiteSimpsonCollocation(
            body_central.mu, spacecraft, cfg.n_nodes
        )
        self.rf = rf_target
        self.vf = vf_target

    def build_initial_guess(self, y0, t0, tf):
        t_nodes = np.linspace(t0, tf, self.cfg.n_nodes)
        yf = np.array([self.rf, 0.0, 0.0, 0.0, self.vf, 0.0, y0[6] * 0.9])
        return self.collocation.linear_guess(y0, yf, t_nodes)

    def solve(self, y0, t0, tf, z0=None):
        t_nodes = np.linspace(t0, tf, self.cfg.n_nodes)
        return self.collocation.solve(
            y0, self.rf, self.vf, t_nodes, z0=z0,
            method=self.cfg.method,
            maxiter=self.cfg.max_iter,
            tol=self.cfg.tol,
        )
