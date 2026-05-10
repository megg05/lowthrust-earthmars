"""
Sweep W1/W2 objective weights for the collocation NLP and plot results.
W1 + W2 = 1, stepped in 0.2 increments → 6 runs.
"""

import os
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

_script_dir = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# PARAMETERS (same as collocation_nlp.py)
# =============================================================================

MU_SUN = 1.327e20
AU = 1.496e11
G0 = 9.80665

M_WET = 6648.0
M_DRY = 1648.0
T_MAX = 1.120
ISP = 1800.0

N_SEG = 100
TOF_MIN_D = 400.0
TOF_MAX_D = 1200.0

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

# =============================================================================
# LOAD Q-LAW GUESS (once)
# =============================================================================

qlaw = np.load(os.path.join(_script_dir, "best_transfer.npz"))
ts_q = qlaw["ts"]
sol_q = qlaw["sol"]
u_q = qlaw["uhist"]

t0_s = ts_q[0]
tf_s = ts_q[-1]
tof_s = tf_s - t0_s
tof_nd = tof_s / TU

r0 = sol_q[0:3, 0]
v0 = sol_q[3:6, 0]
rf = sol_q[0:3, -1]
vf = sol_q[3:6, -1]

r0_nd = r0 / LU
v0_nd = v0 / VU
rf_nd = rf / LU
vf_nd = vf / VU

N = N_SEG
nn = N + 1
t_nodes = np.linspace(t0_s, tf_s, nn)

xg_si = np.zeros((nn, 7))
for i in range(7):
    xg_si[:, i] = np.interp(t_nodes, ts_q, sol_q[i, :])

xg = np.zeros((nn, 7))
xg[:, 0:3] = xg_si[:, 0:3] / LU
xg[:, 3:6] = xg_si[:, 3:6] / VU
xg[:, 6] = xg_si[:, 6] / MU_sc

nu_q = u_q.shape[0]
u_idx = np.linspace(0, nu_q - 1, nn, dtype=int)
u_rtn_s = u_q[u_idx, :]

ug = np.zeros((nn, 4))
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

# =============================================================================
# BUILD SOLVER (parameterized by W1, W2)
# =============================================================================

NS = 7
NC = 4


def build_and_solve(w1, w2):
    """Build and solve the NLP for given objective weights. Returns dict of results."""

    def dyn():
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

    f = dyn()
    opti = ca.Opti()

    tf_v = opti.variable()
    opti.set_initial(tf_v, tof_nd)
    opti.subject_to(opti.bounded(TOF_MIN_D * 86400 / TU, tf_v, TOF_MAX_D * 86400 / TU))

    X = opti.variable(NS, nn)
    U = opti.variable(NC, nn)

    for k in range(nn):
        opti.set_initial(X[:, k], xg[k, :])
        opti.set_initial(U[:, k], ug[k, :])

    vb = 60e3 / VU
    for k in range(nn):
        opti.subject_to(opti.bounded(-3.0, X[0, k], 3.0))
        opti.subject_to(opti.bounded(-3.0, X[1, k], 3.0))
        opti.subject_to(opti.bounded(-3.0, X[2, k], 3.0))
        opti.subject_to(opti.bounded(-vb, X[3, k], vb))
        opti.subject_to(opti.bounded(-vb, X[4, k], vb))
        opti.subject_to(opti.bounded(-vb, X[5, k], vb))
        opti.subject_to(opti.bounded(m_dry_nd, X[6, k], m_wet_nd))
        opti.subject_to(opti.bounded(0, U[0, k], T_max_nd))
        opti.subject_to(opti.bounded(-1, U[1, k], 1))
        opti.subject_to(opti.bounded(-1, U[2, k], 1))
        opti.subject_to(opti.bounded(-1, U[3, k], 1))
        opti.subject_to(U[1, k]**2 + U[2, k]**2 + U[3, k]**2 <= 1.0)

    h_seg = tf_v / N
    for k in range(N):
        xk = X[:, k]; xk1 = X[:, k+1]
        uk = U[:, k]; uk1 = U[:, k+1]
        fk = f(xk, uk); fk1 = f(xk1, uk1)
        um = (uk + uk1) / 2
        xm = (xk + xk1) / 2 + (h_seg / 8) * (fk - fk1)
        fm = f(xm, um)
        opti.subject_to(xk1 - xk - (h_seg / 6) * (fk + 4*fm + fk1) == 0)

    opti.subject_to(X[0:3, 0] == r0_nd)
    opti.subject_to(X[3:6, 0] == v0_nd)
    opti.subject_to(X[6, 0] == m_wet_nd)
    opti.subject_to(X[0:3, -1] == rf_nd)
    opti.subject_to(X[3:6, -1] == vf_nd)

    fuel_frac = (m_wet_nd - X[6, -1]) / m_wet_nd
    time_frac = tf_v / tof_nd
    opti.minimize(w1 * fuel_frac + w2 * time_frac)

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
        "print_level": 0,
        "sb": "yes",
    }
    opti.solver("ipopt", p_opts, s_opts)

    try:
        sol = opti.solve()
        status = "Solved"
        tf_opt = sol.value(tf_v)
        X_opt = sol.value(X)
        U_opt = sol.value(U)
    except RuntimeError:
        status = "Failed"
        tf_opt = opti.debug.value(tf_v)
        X_opt = opti.debug.value(X)
        U_opt = opti.debug.value(U)

    tf_sec = tf_opt * TU
    x_si = np.zeros((nn, 7))
    x_si[:, 0:3] = X_opt[0:3, :].T * LU
    x_si[:, 3:6] = X_opt[3:6, :].T * VU
    x_si[:, 6] = X_opt[6, :].T * MU_sc
    T_mag = U_opt[0, :] * FU
    t_out = np.linspace(0, tf_sec, nn)

    return {
        "w1": w1, "w2": w2, "status": status,
        "tf_days": tf_sec / 86400,
        "m_final": x_si[-1, 6],
        "propellant": x_si[0, 6] - x_si[-1, 6],
        "t": t_out,
        "x": x_si,
        "T_mag": T_mag,
        "x_nd": X_opt.T,
    }


# =============================================================================
# SWEEP
# =============================================================================

weight_pairs = [(1.0, 0.0), (0.8, 0.2), (0.6, 0.4),
                (0.4, 0.6), (0.2, 0.8), (0.0, 1.0)]

results = []
for w1, w2 in weight_pairs:
    print(f"Solving W1={w1:.1f}, W2={w2:.1f}...", end=" ", flush=True)
    res = build_and_solve(w1, w2)
    results.append(res)
    print(f"{res['status']} | TOF={res['tf_days']:.1f}d | m_f={res['m_final']:.1f}kg | Δm={res['propellant']:.1f}kg")

# =============================================================================
# PLOT
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# --- Row 1: XY trajectory for each case ---
for idx, res in enumerate(results):
    ax = axes[0, idx] if idx < 3 else axes[1, idx - 3]
    x_au = res["x"][:, 0:2] / AU
    ax.plot(x_au[:, 0], x_au[:, 1], "c-", linewidth=1.5, label="Trajectory")
    ax.plot(x_au[0, 0], x_au[0, 1], "bo", markersize=8, label="Earth dep")
    ax.plot(x_au[-1, 0], x_au[-1, 1], "ro", markersize=8, label="Mars arr")
    ax.plot(0, 0, "y*", markersize=15)
    ax.set_aspect("equal")
    ax.set_xlabel("X [AU]")
    ax.set_ylabel("Y [AU]")
    ax.set_title(f"W1={res['w1']:.1f}, W2={res['w2']:.1f}\n"
                 f"TOF={res['tf_days']:.0f}d, Δm={res['propellant']:.0f}kg",
                 fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2.0, 1.5)
    ax.set_ylim(-1.5, 1.5)

plt.tight_layout()
plt.savefig(os.path.join(_script_dir, "sweep_trajectories.png"), dpi=150)
print(f"\nSaved: sweep_trajectories.png")

# --- Figure 2: Thrust profiles ---
fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))

for idx, res in enumerate(results):
    ax = axes2[0, idx] if idx < 3 else axes2[1, idx - 3]
    t_days = res["t"] / 86400
    ax.plot(t_days, res["T_mag"], "m-", linewidth=1.0)
    ax.axhline(T_MAX, color="r", linestyle="--", alpha=0.5, label=f"T_max={T_MAX} N")
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Thrust [N]")
    ax.set_title(f"W1={res['w1']:.1f}, W2={res['w2']:.1f}", fontsize=10)
    ax.set_ylim(-0.05, T_MAX * 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(_script_dir, "sweep_thrust.png"), dpi=150)
print(f"Saved: sweep_thrust.png")

# --- Figure 3: Mass profiles ---
fig3, axes3 = plt.subplots(2, 3, figsize=(16, 10))

for idx, res in enumerate(results):
    ax = axes3[0, idx] if idx < 3 else axes3[1, idx - 3]
    t_days = res["t"] / 86400
    ax.plot(t_days, res["x"][:, 6], "g-", linewidth=1.5)
    ax.axhline(M_DRY, color="r", linestyle="--", alpha=0.5, label=f"m_dry={M_DRY} kg")
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Mass [kg]")
    ax.set_title(f"W1={res['w1']:.1f}, W2={res['w2']:.1f}\n"
                 f"m_f={res['m_final']:.0f} kg", fontsize=10)
    ax.set_ylim(M_DRY - 100, M_WET + 100)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(_script_dir, "sweep_mass.png"), dpi=150)
print(f"Saved: sweep_mass.png")

# --- Figure 4: Pareto front (fuel vs time) ---
fig4, ax4 = plt.subplots(1, 1, figsize=(8, 6))

tofs = [r["tf_days"] for r in results]
fuels = [r["propellant"] for r in results]
labels = [f"({r['w1']:.1f},{r['w2']:.1f})" for r in results]

ax4.plot(tofs, fuels, "ko-", markersize=8, linewidth=2)
for i, lbl in enumerate(labels):
    ax4.annotate(lbl, (tofs[i], fuels[i]), textcoords="offset points",
                 xytext=(8, 5), fontsize=9)
ax4.set_xlabel("Transfer Time [days]", fontsize=12)
ax4.set_ylabel("Propellant Consumed [kg]", fontsize=12)
ax4.set_title("Pareto Front: Fuel vs. Time\n(W1, W2) labels", fontsize=13)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(_script_dir, "sweep_pareto.png"), dpi=150)
print(f"Saved: sweep_pareto.png")

# --- Summary table ---
print(f"\n{'='*70}")
print(f"{'W1':>4} {'W2':>4} | {'Status':>8} | {'TOF [d]':>8} | {'m_f [kg]':>9} | {'Δm [kg]':>8} | {'Δv [m/s]':>8}")
print(f"{'-'*70}")
for r in results:
    dv = ISP * G0 * np.log(M_WET / max(r["m_final"], 1))
    print(f"{r['w1']:>4.1f} {r['w2']:>4.1f} | {r['status']:>8} | {r['tf_days']:>8.1f} | {r['m_final']:>9.1f} | {r['propellant']:>8.1f} | {dv:>8.0f}")
print(f"{'='*70}")

plt.show()
