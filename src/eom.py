import numpy as np

def twobody(t,y,spacecraft, body, control):
    r = y[0:3]
    v = y[3:6]
    m = y[6]
    T = control(t,y)

    Tmag = np.linalg.norm(T)
    r_rel = r-body.state(t)[0:3]
    rmag_rel = np.linalg.norm(r_rel)
    a = -body.mu*r_rel / rmag_rel**3 + T/m #gravity + thrust accel
    mdot = -Tmag/(spacecraft.Isp*spacecraft.g0)

    return np.hstack((v,a,mdot))