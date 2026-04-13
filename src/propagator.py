from scipy.integrate import solve_ivp
import numpy as np
from .models.body import Body

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

        if self.body.name == "earth":
            earth_dist, _ = body.sun_to_earth(t)
            r_earth = r-earth_dist
            rmag_earth = np.linalg.norm(r_earth)
            if rmag_earth > body.earth_SOI:
                self.change_body = True
            r_rel = r_earth
        else:
            mars_dist, _ = body.sun_to_mars(t)
            r_mars = r-mars_dist
            rmag_mars = np.linalg.norm(r_mars)
            if self.body.name == "sun":
                if rmag_mars < body.mars_SOI:
                    self.change_body = True
                r_rel = r
            r_rel = r_mars

        rmag_rel = np.linalg.norm(r_rel)
        a = -body.mu*r_rel / rmag_rel**3 + T/m #gravity + thrust accel
        mdot = -Tmag/(spacecraft.Isp*spacecraft.g0)

        return np.hstack((v,a,mdot))