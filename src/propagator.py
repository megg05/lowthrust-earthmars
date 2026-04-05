from scipy.integrate import solve_ivp
from .eom import twobody

class Propagator:
    def __init__(self, body, spacecraft):
        self.body = body
        self.spacecraft = spacecraft

    def forward(self, y0, tspan, control, tf):
        #TODO: better solver?
        return solve_ivp(
            lambda t,y: twobody(t,y,self.body.mu, control, self.spacecraft),
            tspan,
            y0,
            'RK45',
            tf
        )