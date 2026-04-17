from scipy.integrate import solve_ivp
import numpy as np
from .models.body import Body
from .planet_propagator import earth_state, mars_state

Earth = Body(name="earth",mu=3.986e14)
Mars = Body(name="mars",mu=4.283e13)
Sun = Body(name="sun",mu=1.327e20)

class Propagator:
    def __init__(self, spacecraft):
        self.body = Earth
        self.spacecraft = spacecraft
        self.change_body = False

    def forward(self, y0, tspan, control):
        return solve_ivp(
            lambda t,y: self.twobody(t,y,self.spacecraft, self.body, control),
            tspan,
            y0,
            'RK45'
        )
    
    def twobody(self,t,y,spacecraft, body, control):
        r = y[0:3]
        v = y[3:6]
        m = y[6]
        T = control(t,y)
        Tmag = np.linalg.norm(T)
        print("pre propagated orbit pos ", r)

        a = -Sun.mu*r / np.linalg.norm(r)**3 + T/m

        if self.body.name == "earth":
            earth_dist, _ = earth_state(t)
            r_earth = r-earth_dist
            rmag_earth = np.linalg.norm(r_earth)
            if rmag_earth > body.earth_SOI:
                self.change_body = True
            a = -body.mu*r_earth / rmag_earth**3
        else:
            mars_dist, _ = mars_state(t)
            r_mars = r-mars_dist
            rmag_mars = np.linalg.norm(r_mars)
            if self.body.name == "sun" and rmag_mars < body.mars_SOI:
                    self.change_body = True
            elif self.body.name == "mars":
                a = -body.mu*r_mars / rmag_mars**3

        mdot = -Tmag/(spacecraft.Isp*spacecraft.g0)

        return np.hstack((v,a,mdot))