from scipy.integrate import solve_ivp
import numpy as np
from .models.body import Body

Earth = Body(name="earth",mu=3.986e14)
Mars = Body(name="mars",mu=4.283e13)
Sun = Body(name="sun",mu=1.327e20)

class Propagator:
    def __init__(self, spacecraft):
        self.body = Sun
        self.spacecraft = spacecraft
        self.change_body = False
        self.t_hist = []
        self.u_hist = []
        self.q_hist = []

    

    def forward(self, y0, tspan, control):
        self.t_hist = []
        self.u_hist = []
        self.q_hist = []
        sol= solve_ivp(
            lambda t, y: self.twobody(t, y, self.spacecraft, self.body, control),
            tspan,
            y0,
            method="RK23",
            max_step=86400.0, 
        )
        return sol, np.asarray(self.t_hist), np.asarray(self.u_hist), np.asarray(self.q_hist)
    
    
    def twobody(self,t,y,spacecraft, body, control):
        r = y[0:3]
        v = y[3:6]
        m = y[6]
        T_rtn, Q = control(t,y)
        Tmag = np.linalg.norm(T_rtn)

        r_hat = r / np.linalg.norm(r)
        h_vec = np.cross(r, v)
        n_hat = h_vec / np.linalg.norm(h_vec)
       
        t_hat = np.cross(n_hat, r_hat)

        C_rtn2inertial = np.column_stack((r_hat, t_hat, n_hat))
        T = C_rtn2inertial @ T_rtn

        a = -Sun.mu*r / np.linalg.norm(r)**3 + T/m


        self.t_hist.append(t)
        self.u_hist.append(T_rtn)
        self.q_hist.append(Q)

        mdot = -Tmag/(spacecraft.Isp*spacecraft.g0)


        return np.hstack((v,a,mdot))