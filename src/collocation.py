import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp


class LowThrustOptimizer:
    """Single-shooting optimizer for planar low-thrust orbit transfers.

    Thrust is always tangential (prograde); only the throttle schedule
    is optimized. This is nearly optimal for circular-to-circular transfers.
    """

    def __init__(self, mu, spacecraft, n_segments):
        self.mu = mu
        self.sc = spacecraft
        self.n = n_segments

    def dynamics(self, t, x, throttle):
        """Planar two-body + tangential thrust."""
        rx, ry, vx, vy, m = x
        r = np.sqrt(rx**2 + ry**2)
        v = np.sqrt(vx**2 + vy**2)

        # gravity
        ax = -self.mu * rx / r**3
        ay = -self.mu * ry / r**3

        # tangential thrust (prograde)
        T = throttle * self.sc.Tlim
        if v > 1e-14:
            ax += T / m * (vx / v)
            ay += T / m * (vy / v)

        mdot = -T / (self.sc.Isp * self.sc.g0)
        return np.array([vx, vy, ax, ay, mdot])

    def propagate(self, y0, t_nodes, throttles):
        """Forward-propagate with piecewise-constant throttle."""
        X = np.zeros((self.n + 1, 5))
        X[0] = y0

        for k in range(self.n):
            sol = solve_ivp(
                lambda t, x: self.dynamics(t, x, throttles[k]),
                (t_nodes[k], t_nodes[k + 1]), X[k],
                method="RK45", rtol=1e-10, atol=1e-12,
            )
            X[k + 1] = sol.y[:, -1]
        return X

    def cost(self, throttles, y0, t_nodes, rf, vf, weight):
        """Objective: fuel use + weighted terminal orbit error."""
        X = self.propagate(y0, t_nodes, throttles)
        r_f = np.sqrt(X[-1, 0]**2 + X[-1, 1]**2)
        v_f = np.sqrt(X[-1, 2]**2 + X[-1, 3]**2)
        rdot_f = (X[-1, 0] * X[-1, 2] + X[-1, 1] * X[-1, 3])

        fuel = y0[4] - X[-1, 4]
        err_r = (r_f - rf)**2
        err_v = (v_f - vf)**2
        err_circ = rdot_f**2
        return fuel + weight * (err_r + err_v + err_circ)

    def solve(self, y0, rf, vf, t_nodes, tol=1e-4):
        """Two-phase solve: global search then gradient refinement."""
        bounds = [(0.0, 1.0)] * self.n

        # Phase 1: differential evolution for global structure
        print("Phase 1: global search (differential evolution)...")
        de_result = differential_evolution(
            self.cost, bounds,
            args=(y0, t_nodes, rf, vf, 1e4),
            maxiter=200, popsize=10, seed=42, tol=0.01,
            polish=False, disp=True, workers=-1,
        )
        z = de_result.x

        # Phase 2: refine with Powell (gradient-free), increasing penalty
        print("\nPhase 2: gradient-free refinement...")
        for weight in [1e4, 1e5, 1e6, 1e7, 1e8]:
            result = minimize(
                self.cost, z, args=(y0, t_nodes, rf, vf, weight),
                method="Powell",
                options={"maxiter": 2000, "ftol": 1e-15},
            )
            # clip to bounds
            z = np.clip(result.x, 0.0, 1.0)

        # evaluate final solution
        X = self.propagate(y0, t_nodes, z)
        r_f = np.sqrt(X[-1, 0]**2 + X[-1, 1]**2)
        v_f = np.sqrt(X[-1, 2]**2 + X[-1, 3]**2)
        rdot_f = X[-1, 0] * X[-1, 2] + X[-1, 1] * X[-1, 3]

        result.X = X
        result.throttles = z
        result.r_error = abs(r_f - rf)
        result.v_error = abs(v_f - vf)
        result.circ_error = abs(rdot_f)
        result.success = (result.r_error < tol and result.v_error < tol)
        return result
