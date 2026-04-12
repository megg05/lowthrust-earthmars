from scipy.integrate import solve_ivp
from .eom import twobody

class Propagator:
    def __init__(self, body, spacecraft):
        self.body = body
        self.spacecraft = spacecraft
    
    def change_body(self, body):
        # TODO: implement SOI switching
        self.body = body

    def forward(self, y0, tspan, control):
        #TODO: better solver?
        return solve_ivp(
            lambda t,y: twobody(t,y,self.spacecraft, self.body, control),
            tspan,
            y0,
            'RK45'
        )