import numpy as np
from dataclasses import dataclass

@dataclass
class Qlawtarget: #using equinoctial elements
    a: float
    ex: float
    ey: float
    hx: float
    hy: float
    L: float

@dataclass
class Qlawgains:
    Wa: float=1.0
    We: float=1.0
    Wi: float=1.0
    WL: float=0.0
    coast_threshold: float = 0.0

class Qlaw:
    def __init__(self, spacecraft, target: Qlawtarget, gains: Qlawgains):
        self.spacecraft = spacecraft
        self.target = target
        self.gains = gains

    def wrap(ang):
        return (ang+np.pi)%(2*np.pi)-np.pi
    
    def cart2eq(self, mu, r,v, retrograde=False):
        # check this
        r = np.asarray(r, dtype=float)
        v = np.asarray(v, dtype=float)
        rmag = np.linalg.norm(r)
        hvec = np.cross(r, v)
        hmag = np.linalg.norm(hvec)
        evec = (np.cross(v, hvec) / mu) - r / rmag
        e = np.linalg.norm(evec)
        p = hmag**2 / mu
        k_hat = np.array([0., 0., 1.])
        nvec = np.cross(k_hat, hvec)
        nmag = np.linalg.norm(nvec)
        i = np.arccos(np.clip(hvec[2] / hmag, -1.0, 1.0))
        raan = 0.0 if nmag < 1e-14 else np.arctan2(nvec[1], nvec[0])
        if nmag < 1e-14 or e < 1e-14:
            argp = 0.0
        else:
            argp = np.arctan2(np.dot(np.cross(nvec, evec), hvec)/hmag, np.dot(nvec, evec))
        if e < 1e-14:
            nu = np.arctan2(np.dot(np.cross(nvec, r), hvec)/hmag, np.dot(nvec, r)) if nmag >= 1e-14 else np.arctan2(r[1], r[0])
        else:
            nu = np.arctan2(np.dot(np.cross(evec, r), hvec)/hmag, np.dot(evec, r))
        L = self.wrap(raan + argp + nu)
        if retrograde:
            denom = 1.0 - np.cos(i)
            hk_scale = 1.0
        else:
            denom = 1.0 + np.cos(i)
            hk_scale = 1.0
        if abs(denom) < 1e-14:
            denom = np.sign(denom) * 1e-14 if denom != 0 else 1e-14
        f = e*np.cos(raan + argp)
        g = e*np.sin(raan + argp)
        t = np.tan(i/2.0)
        h = t*np.cos(raan)
        k = t*np.sin(raan)
        return {'a': p/(1-e**2) if e < 1 else np.inf, 'p': p, 'f': f, 'g': g, 'h': h, 'k': k, 'L': L}

    def calculate_q(self, curr):
        #TODO: check implementation lol
        da2 = (curr["a"] - self.target.a)**2
        de2 = (curr["ex"] - self.target.ex)**2 + (curr["ey"] - self.target.ey)**2
        di2 = (curr["hx"] - self.target.hx)**2 + (curr["hy"] - self.target.hy)**2
        dL2 = 0.0

        if self.target.L is not None:
            dL = curr["L"] - self.target.L
            dL = self.wrap(dL)
            dL2 = dL*dL
        return self.gains.Wa*da2 + self.gains.We*de2 + self.gains.Wi*di2 + self.gainslWL*dL2
    
    def calculate_qgrad(self, mu,y):
        eps=1e-6
        qgrad = np.zeros(3)
        q0 = self.q_from_cart(mu,y)
        for i in range(3):
            dy = np.zeros_like(y)
            dy[i] = eps
            qp = self.q_from_cart(mu, y+dy)
            qm = self.q_from_cart(mu, y-dy)
            qgrad[i] = (qp - qm)/(2*eps)
        return qgrad

    def q_from_cart(self, mu,y):
        r = y[0:3]
        v = y[3:6]
        eq = self.cart2eq(mu,r,v)
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
        