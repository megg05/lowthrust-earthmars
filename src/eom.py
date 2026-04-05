import numpy as np

def twobody(t,y,mu,control, spacecraft):
    r = y[0:3]
    v = y[3:6]
    m = y[6]
    T = control(t,y)

    Tmag = np.linalg.norm(T)
    rmag = np.linalg.norm(r)
    a = -mu*r / rmag**3 + T/m #gravity + thrust accel
    mdot = -Tmag/(spacecraft.Isp*spacecraft.g0)

    return np.hstack((v,a,mdot))