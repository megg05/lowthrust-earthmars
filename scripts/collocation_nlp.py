"""
Hermite-Simpson Direct Collocation NLP for Earth-to-Mars Low-Thrust Transfer
=============================================================================
Solver: CasADi + IPOPT
Initial guess: Q-law trajectory from best_transfer.npz
Non-dimensionalized using canonical heliocentric units.

Control: [T, ux, uy, uz] where T is thrust magnitude and u is direction.
The unit-vector constraint ||u||=1 is relaxed to ||u||<=1. When T>0, the
optimality conditions ensure ||u||=1 at the solution. During coasting (T=0),
u is irrelevant.

TUNING:
  - W1, W2: objective weights for fuel vs. time
  - N_SEGMENTS: collocation resolution
  - TOF bounds: flight time search range
  - Swap in PSYCHE_SEP parameters once BCs are compatible
"""

import os
import numpy as np
import casadi as ca

_script_dir = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# 1. PARAMETERS
# =============================================================================

MU_SUN = 1.327e20          # [m^3/s^2]
AU = 1.496e11              # [m]
G0 = 9.80665               # [m/s^2]

# Spacecraft (matched to Q-law trajectory)
M_WET = 6648.0             # [kg] wet mass
M_DRY = 1648.0             # [kg] dry mass
T_MAX = 1.120              # [N] max thrust
ISP = 1800.0               # [s] specific impulse
ETA = 0.7                  # thruster efficiency

# To use Psyche SEP:
# M_WET = 2733.0; M_DRY = 1648.0; T_MAX = 0.240; ISP = 1800.0; ETA = 0.7

N_SEG = 100                # collocation segments (reduce for faster solve)

# Objective: J = W1 * (fuel/m_wet) + W2 * (tf/tf_ref)
W1 = 1.0
W2 = 0.0

TOF_MIN_D = 400.0          # [days]
TOF_MAX_D = 1200.0         # [days]

# =============================================================================
# 2. CANONICAL UNITS
# =============================================================================

LU = AU
TU = np.sqrt(AU**3 / MU_SUN)
VU = LU / TU
MU_sc = M_WET
FU = MU_sc * LU / TU**2

mu_nd = 1.0
m_wet_nd = 1.0
m_dry_nd = M_DRY / MU_sc
T_max_nd = T_MAX / FU
v_ex_nd = (ISP * G0) / VU

print("=" * 60)
print("HERMITE-SIMPSON COLLOCATION — LOW-THRUST EARTH→MARS")
print("=" * 60)
print(f"Spacecraft: m0={M_WET} kg, T_max={T_MAX} N, Isp={ISP} s")
print(f"Units: LU={LU:.2e}m TU={TU/86400:.1f}d VU={VU:.0f}m/s FU={FU:.2e}N")
print(f"Non-dim: T_max={T_max_nd:.4e} m_dry={m_dry_nd:.4f} v_ex={v_ex_nd:.4f}")

# =============================================================================
# 3. LOAD INITIAL GUESS
# =============================================================================

qlaw = np.load(os.path.join(_script_dir, "best_transfer.npz"))
ts_q = qlaw["ts"]
sol_q = qlaw["sol"]      # (7, 3590)
u_q = qlaw["u"]         # (22740, 3) RTN thrust [N]

t0_s = ts_q[0]
tf_s = ts_q[-1]
tof_s = tf_s - t0_s
tof_nd = tof_s / TU

# Boundaries (SI)
r0 = sol_q[0:3, 0]
v0 = sol_q[3:6, 0]
rf = sol_q[0:3, -1]
vf = sol_q[3:6, -1]

print(f"\nQ-law: dep day {int(qlaw['departure'])}, TOF {int(qlaw['tof'])}d")
print(f"  mass {sol_q[6,0]:.0f}→{sol_q[6,-1]:.0f} kg, |r0|={np.linalg.norm(r0)/AU:.3f} AU")

# Non-dim boundaries
r0_nd = r0 / LU
v0_nd = v0 / VU
rf_nd = rf / LU
vf_nd = vf / VU

# Resample
N = N_SEG
nn = N + 1
t_nodes = np.linspace(t0_s, tf_s, nn)

xg_si = np.zeros((nn, 7))
for i in range(7):
    xg_si[:, i] = np.interp(t_nodes, ts_q, sol_q[i, :])

# Non-dim state guess
xg = np.zeros((nn, 7))
xg[:, 0:3] = xg_si[:, 0:3] / LU
xg[:, 3:6] = xg_si[:, 3:6] / VU
xg[:, 6] = xg_si[:, 6] / MU_sc

# Control: [T, ux, uy, uz]
nu_q = u_q.shape[0]
u_idx = np.linspace(0, nu_q - 1, nn, dtype=int)
u_rtn_s = u_q[u_idx, :]

ug = np.zeros((nn, 4))  # [T_nd, ux, uy, uz]
for k in range(nn):
    rk = xg_si[k, 0:3]
    vk = xg_si[k, 3:6]
    rh = rk / np.linalg.norm(rk)
    hv = np.cross(rk, vk)
    hn = np.linalg.norm(hv)
    nh = hv / hn if hn > 1e-10 else np.array([0, 0, 1.0])
    th = np.cross(nh, rh)
    C = np.column_stack((rh, th, nh))
    F_eci = C @ u_rtn_s[k]
    Tmag = np.linalg.norm(F_eci)
    Tmag_clip = min(Tmag, T_MAX)
    ug[k, 0] = Tmag_clip / FU
    if Tmag > 1e-10:
        ug[k, 1:4] = F_eci / Tmag
    else:
        ug[k, 1:4] = [1.0, 0.0, 0.0]

print(f"  Guess: {nn} nodes, T_nd=[{ug[:,0].min():.4e}, {ug[:,0].max():.4e}]")

# =============================================================================
# 4. DYNAMICS
# =============================================================================
# x = [r(3), v(3), m]   u = [T, ux, uy, uz]
# rdot = v
# vdot = -mu/|r|^3 * r + T/m * u_hat
# mdot = -T / v_ex

NS = 7  # states
NC = 4  # controls


def dynamics():
    x = ca.SX.sym("x", NS)
    u = ca.SX.sym("u", NC)
    r = x[0:3]; v = x[3:6]; m = x[6]
    T = u[0]; uh = u[1:4]

    rsq = ca.dot(r, r)
    rnorm = ca.sqrt(rsq)

    rdot = v
    vdot = -mu_nd / (rsq * rnorm) * r + (T / m) * uh
    mdot = -T / v_ex_nd

    return ca.Function("f", [x, u], [ca.vertcat(rdot, vdot, mdot)])


f = dynamics()

# =============================================================================
# 5. BUILD NLP
# =============================================================================

print("\nBuilding NLP...")

opti = ca.Opti()

# Decision variables
tf_v = opti.variable()
opti.set_initial(tf_v, tof_nd)
opti.subject_to(opti.bounded(TOF_MIN_D * 86400 / TU, tf_v, TOF_MAX_D * 86400 / TU))

X = opti.variable(NS, nn)
U = opti.variable(NC, nn)

# Initial guess
for k in range(nn):
    opti.set_initial(X[:, k], xg[k, :])
    opti.set_initial(U[:, k], ug[k, :])

# Variable bounds
for k in range(nn):
    # State bounds
    opti.subject_to(opti.bounded(-3.0, X[0, k], 3.0))  # rx [AU]
    opti.subject_to(opti.bounded(-3.0, X[1, k], 3.0))  # ry
    opti.subject_to(opti.bounded(-3.0, X[2, k], 3.0))  # rz
    vb = 60e3 / VU
    opti.subject_to(opti.bounded(-vb, X[3, k], vb))
    opti.subject_to(opti.bounded(-vb, X[4, k], vb))
    opti.subject_to(opti.bounded(-vb, X[5, k], vb))
    opti.subject_to(opti.bounded(m_dry_nd, X[6, k], m_wet_nd))

    # Control bounds
    opti.subject_to(opti.bounded(0, U[0, k], T_max_nd))    # T >= 0
    opti.subject_to(opti.bounded(-1, U[1, k], 1))
    opti.subject_to(opti.bounded(-1, U[2, k], 1))
    opti.subject_to(opti.bounded(-1, U[3, k], 1))

    # Unit direction constraint: ||u_hat||^2 <= 1
    opti.subject_to(U[1, k]**2 + U[2, k]**2 + U[3, k]**2 <= 1.0)

# =============================================================================
# 6. HERMITE-SIMPSON DEFECTS
# =============================================================================

h_seg = tf_v / N

for k in range(N):
    xk = X[:, k]
    xk1 = X[:, k + 1]
    uk = U[:, k]
    uk1 = U[:, k + 1]

    fk = f(xk, uk)
    fk1 = f(xk1, uk1)

    # Midpoint
    um = (uk + uk1) / 2
    xm = (xk + xk1) / 2 + (h_seg / 8) * (fk - fk1)
    fm = f(xm, um)

    # Defect = 0
    defect = xk1 - xk - (h_seg / 6) * (fk + 4 * fm + fk1)
    opti.subject_to(defect == 0)

# =============================================================================
# 7. BOUNDARY CONDITIONS
# =============================================================================

# Initial: position + velocity + mass fixed
opti.subject_to(X[0:3, 0] == r0_nd)
opti.subject_to(X[3:6, 0] == v0_nd)
opti.subject_to(X[6, 0] == m_wet_nd)

# Terminal: position + velocity fixed, mass free
opti.subject_to(X[0:3, -1] == rf_nd)
opti.subject_to(X[3:6, -1] == vf_nd)

# =============================================================================
# 8. OBJECTIVE
# =============================================================================

fuel_frac = (m_wet_nd - X[6, -1]) / m_wet_nd
time_frac = tf_v / tof_nd

J = W1 * fuel_frac + W2 * time_frac
opti.minimize(J)

# =============================================================================
# 9. SOLVER OPTIONS
# =============================================================================

p_opts = {"expand": True}
s_opts = {
    "max_iter": 10000,
    "tol": 1e-6,
    "acceptable_tol": 1e-4,
    "acceptable_iter": 15,
    "mu_strategy": "adaptive",
    "mu_init": 0.01,
    "linear_solver": "mumps",
    "nlp_scaling_method": "gradient-based",
    "print_level": 5,
    "sb": "yes",
}

opti.solver("ipopt", p_opts, s_opts)

# =============================================================================
# 10. SOLVE
# =============================================================================

print(f"\n{'='*60}")
print("SOLVING...")
print(f"{'='*60}\n")

try:
    sol = opti.solve()
    status = "Solve_Succeeded"
    tf_opt = sol.value(tf_v)
    X_opt = sol.value(X)
    U_opt = sol.value(U)
except RuntimeError as e:
    print(f"\nIPOPT did not converge: {e}")
    status = "Failed"
    tf_opt = opti.debug.value(tf_v)
    X_opt = opti.debug.value(X)
    U_opt = opti.debug.value(U)

# =============================================================================
# 11. RESULTS
# =============================================================================

tf_s_opt = tf_opt * TU
x_si = np.zeros_like(X_opt.T)
x_si[:, 0:3] = X_opt[0:3, :].T * LU
x_si[:, 3:6] = X_opt[3:6, :].T * VU
x_si[:, 6] = X_opt[6, :].T * MU_sc

u_si = np.zeros((nn, 4))
u_si[:, 0] = U_opt[0, :] * FU
u_si[:, 1:4] = U_opt[1:4, :].T

t_out = np.linspace(0, tf_s_opt, nn)
T_mag = u_si[:, 0]

print(f"\n{'='*60}")
print(f"  STATUS: {status}")
print(f"{'='*60}")
print(f"  Transfer time:   {tf_s_opt/86400:.2f} days")
print(f"  Initial mass:    {x_si[0, 6]:.2f} kg")
print(f"  Final mass:      {x_si[-1, 6]:.2f} kg")
print(f"  Propellant:      {x_si[0, 6] - x_si[-1, 6]:.2f} kg")
print(f"  Mass fraction:   {x_si[-1, 6]/x_si[0, 6]*100:.1f}%")
print(f"  Delta-v:         {ISP*G0*np.log(x_si[0,6]/max(x_si[-1,6],1)):.0f} m/s")
print(f"  |r0|={np.linalg.norm(x_si[0,0:3])/AU:.4f} AU")
print(f"  |rf|={np.linalg.norm(x_si[-1,0:3])/AU:.4f} AU")
print(f"  Thrust: [{T_mag.min():.5f}, {T_mag.max():.5f}] N")

# Save
out = os.path.join(_script_dir, "collocation_solution.npz")
np.savez(out,
    t=t_out, x=x_si, u=u_si, T_mag=T_mag, tf=tf_s_opt,
    x_nd=X_opt.T, u_nd=U_opt.T, tf_nd=tf_opt,
    status=status, LU=LU, TU=TU, MU=MU_sc, VU=VU, FU=FU,
    M_WET=M_WET, M_DRY=M_DRY, T_MAX=T_MAX, ISP=ISP, W1=W1, W2=W2)
print(f"  Saved: {out}")
