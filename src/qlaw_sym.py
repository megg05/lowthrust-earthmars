import sympy as sym
import numpy as np
from dataclasses import dataclass

def symbolic_qlaw():

    phase = sym.symbols("phase")
    mu = sym.symbols("mu")
    accel = sym.symbols("accel")
    a, f, g, h, k, L = sym.symbols("a f g h k L")
    oe = [a, f, g, h, k, L]
    aT, fT, gT, hT, kT, LT = sym.symbols("a_T f_T g_T h_T k_T L_T") 
    oeT = [aT, fT, gT, hT, kT, LT]
    wa, wf, wg, wh, wk, wL = sym.symbols("w_a w_f w_g w_h w_k w_L")
    W_oe = [wa, wf, wg, wh, wk, wL]
    Da, Df, Dg, Dh, Dk = sym.symbols("D_a D_f D_g D_h D_k")
    D_mee = [Da, Df, Dg, Dh, Dk]

    def calculate_Q():
        a_, f_, g_, h_, k_, L_ = oe 
        aT_, fT_, gT_, hT_, kT_, LT_ = oeT

        p = a_ * (1 - f_**2 - g_**2)
        e = sym.sqrt(f_**2 + g_**2)
        ang_mom = sym.sqrt(a_ * mu * (1 - e**2))
        rp = a_ * (1 - e)
        ta = L_ - sym.atan2(g_, f_)  # true anomaly
        r = ang_mom**2 / (mu * (1 + e * sym.cos(ta)))
        s2= 1 + h_**2 + k_**2

        e_oe = [
            a_ - aT_,
            f_ - fT_,
            g_ - gT_,
            h_ - hT_,
            k_ - kT_,
            L_ - LT_
        ]

        # element rates for given perturbing accel
        # sqrt_pmu = sym.sqrt(p / mu)
        # adot_xx = 2 * accel * a_ * sym.sqrt(a_ / mu) * sym.sqrt((1 + sym.sqrt(f_**2 + g_**2)) / (1 - sym.sqrt(f_**2 + g_**2)))
        # fdot_xx = 2 * accel * sqrt_pmu
        # gdot_xx = 2 * accel * sqrt_pmu
        # hdot_xx = 0.5 * accel * sqrt_pmu * s2/ (sym.sqrt(1 - g_**2) + f_)
        # kdot_xx = 0.5 * accel * sqrt_pmu * s2/ (sym.sqrt(1 - f_**2) + g_)
        # Ldot_xx = accel * sqrt_pmu / (1 + f_*sym.cos(L_) + g_*sym.sin(L_))
        # oedot = [adot_xx, fdot_xx, gdot_xx, hdot_xx, kdot_xx, Ldot_xx]
        # CORRECTED: Max sensitivities (accel=1 m/s² convention)
        sqrt_pmu = sym.sqrt(p / mu)
        # adot_xx = 2 * a_ * sym.sqrt(a_ / mu) * sym.sqrt((1 + sym.sqrt(f_**2 + g_**2)) / (1 - sym.sqrt(f_**2 + g_**2)))
        # fdot_xx = 2 * sqrt_pmu
        # gdot_xx = 2 * sqrt_pmu
        # hdot_xx = 0.5 * sqrt_pmu * s2 / (sym.sqrt(1 - g_**2) + f_)
        # kdot_xx = 0.5 * sqrt_pmu * s2 / (sym.sqrt(1 - f_**2) + g_)
        oedot = D_mee
        # oedot = [adot_xx, fdot_xx, gdot_xx, hdot_xx, kdot_xx]


        # S_oe = [
        #     (1 + (sym.sqrt((a_ - aT_)**2) / (3 * aT_)) ** 4) ** (1 / 2),
        #     1.0, 1.0, 1.0, 1.0, 1.0
        # ]
        S_oe = [1.0,1.0,1.0,1.0,1.0]

        q = (W_oe[0] * S_oe[0] * (e_oe[0] / oedot[0])**2 +
            W_oe[1] * S_oe[1] * (e_oe[1] / oedot[1])**2 +
            W_oe[2] * S_oe[2] * (e_oe[2] / oedot[2])**2 +
            W_oe[3] * S_oe[3] * (e_oe[3] / oedot[3])**2 +
            W_oe[4] * S_oe[4] * (e_oe[4] / oedot[4])**2
        )

        # if phase == 2:
        #     q += W_oe[5] * S_oe[5] * (e_oe[5] / oedot[5])**2


        # MEE sensitivity matrix
        cosL = sym.cos(L_)
        sinL = sym.sin(L_)
        w = 1 + f_ * cosL + g_ * sinL

        psi = [ #rtn
            [
                2 * a_**2 / ang_mom * e * sym.sin(ta),
                sqrt_pmu * sinL,
                -sqrt_pmu * cosL,
                0.0,
                0.0,
                # 0.0
            ],
            [
                2 * a_**2 / ang_mom * p / r,
                sqrt_pmu / w * ((w + 1) * cosL + f_),
                sqrt_pmu / w * ((w + 1) * sinL + g_),
                0.0,
                0.0,
                # sym.sqrt(a_ / mu) / w 
            ],
            [
                0.0,
                sqrt_pmu / w * (-g_ * (h_ * sinL - k_ * cosL)),
                sqrt_pmu / w * (f_ * (h_ * sinL - k_ * cosL)),
                sqrt_pmu / w * 0.5 * s2 * cosL,
                sqrt_pmu / w * 0.5 * s2 * sinL,
                # sym.sqrt(a_ / mu) / w * (h_ * sinL - k_ * cosL)
            ]
        ]

        dqda = sym.diff(q, a_)
        dqdf = sym.diff(q, f_)
        dqdg = sym.diff(q, g_)
        dqdh = sym.diff(q, h_)
        dqdk = sym.diff(q, k_)
        # dqdL = sym.diff(q, L_) if phase == 2 else 0
        dqdoe = [dqda, dqdf, dqdg, dqdh, dqdk]

        D1 = sum(psi[0][i] * dqdoe[i] for i in range(5))
        D2 = sum(psi[1][i] * dqdoe[i] for i in range(5))
        D3 = sum(psi[2][i] * dqdoe[i] for i in range(5))

        # RTN thrusting angles!
        Dmag = sym.sqrt(D1**2 + D2**2 + D3**2)
        D1_unit = D1 / Dmag
        D2_unit = D2 / Dmag
        D3_unit = D3 / Dmag

        # The unit vector must be the NEGATIVE of these D values
        Dmag = sym.sqrt(D1**2 + D2**2 + D3**2 + 1e-16)
        u_r = -D1 / Dmag
        u_t = -D2 / Dmag
        u_n = -D3 / Dmag

        # in/out of plane thrusting angles!
        alpha = sym.atan2(-D1_unit, -D2_unit) 
        beta = sym.atan(-D3_unit / sym.sqrt(D1_unit**2 + D2_unit**2))
        # beta = sym.atan(-D3_unit)

        # Qdot Lyapunov descent
        cosa = sym.cos(alpha)
        cosb = sym.cos(beta)
        sina = sym.sin(alpha)
        sinb = sym.sin(beta)
        dqdt = accel*(D1*cosb*cosa + D2*cosb*sina + D3*sinb)
        # u_r = cosb*cosa
        # u_t = cosb*sina
        # u_n = sinb

        # lambdify
        arg_list = [mu, accel, oe, oeT, W_oe, D_mee, phase]

        
        fun_control = sym.lambdify(arg_list, [u_r, u_t, u_n, alpha, beta, q], "numpy", cse=True)
        fun_q_dqdt = sym.lambdify(arg_list, [q, dqdt], "numpy", cse=True)
        # fun_psi = sym.lambdify(arg_list, psi, "numpy", cse=True)
        # fun_dqdoe = sym.lambdify(arg_list, dqdoe, "numpy", cse=True)
    

        return fun_control, fun_q_dqdt

    fun_control, fun_q_dqdt = calculate_Q()
    return fun_control, fun_q_dqdt