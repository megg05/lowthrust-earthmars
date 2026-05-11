"""
Sweep W1/W2 objective weights for the collocation NLP and plot results.
W1 + W2 = 1, stepped in 0.2 increments → 6 runs.
"""

import os
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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

# Color scheme for 6 cases
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# --- Figure 1: All trajectories on one plot ---
fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))

for idx, res in enumerate(results):
    x_au = res["x"][:, 0:2] / AU
    label = f"({res['w1']:.1f},{res['w2']:.1f}), {res['tf_days']:.0f}d"
    ax1.plot(x_au[:, 0], x_au[:, 1], "-", color=colors[idx],
             linewidth=2.0, label=label, alpha=0.8)

# Mark start/end points (same for all)
x_au_0 = results[0]["x"][:, 0:2] / AU
ax1.plot(x_au_0[0, 0], x_au_0[0, 1], "bo", markersize=10, label="Earth dep", zorder=10)
ax1.plot(x_au_0[-1, 0], x_au_0[-1, 1], "ro", markersize=10, label="Mars arr", zorder=10)
ax1.plot(0, 0, "y*", markersize=18, label="Sun", zorder=10)

ax1.set_aspect("equal")
ax1.set_xlabel("X [AU]", fontsize=12)
ax1.set_ylabel("Y [AU]", fontsize=12)
ax1.set_title("Trajectory Comparison: All Weight Combinations", fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-2.0, 1.5)
ax1.set_ylim(-1.5, 1.5)
ax1.legend(fontsize=9, loc='upper right', title="(W1,W2), TOF")

plt.tight_layout()
plt.savefig(os.path.join(_script_dir, "sweep_trajectories.png"), dpi=150)
print(f"\nSaved: sweep_trajectories.png")

# --- Figure 2: All thrust profiles on one plot ---
fig2, ax2 = plt.subplots(1, 1, figsize=(12, 6))

for idx, res in enumerate(results):
    t_days = res["t"] / 86400
    label = f"({res['w1']:.1f},{res['w2']:.1f})"
    ax2.plot(t_days, res["T_mag"], "-", color=colors[idx],
             linewidth=1.5, label=label, alpha=0.8)

ax2.axhline(T_MAX, color="k", linestyle="--", linewidth=1.5,
            alpha=0.5, label=f"T_max={T_MAX} N")
ax2.set_xlabel("Time [days]", fontsize=12)
ax2.set_ylabel("Thrust [N]", fontsize=12)
ax2.set_title("Thrust Profile Comparison", fontsize=14)
ax2.set_ylim(-0.05, T_MAX * 1.15)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10, loc='upper right', title="(W1,W2)")

plt.tight_layout()
plt.savefig(os.path.join(_script_dir, "sweep_thrust.png"), dpi=150)
print(f"Saved: sweep_thrust.png")

# --- Figure 3: All mass profiles on one plot ---
fig3, ax3 = plt.subplots(1, 1, figsize=(12, 6))

for idx, res in enumerate(results):
    t_days = res["t"] / 86400
    label = f"({res['w1']:.1f},{res['w2']:.1f}), m_f={res['m_final']:.0f}kg"
    ax3.plot(t_days, res["x"][:, 6], "-", color=colors[idx],
             linewidth=2.0, label=label, alpha=0.8)

ax3.axhline(M_DRY, color="k", linestyle="--", linewidth=1.5,
            alpha=0.5, label=f"m_dry={M_DRY} kg")
ax3.set_xlabel("Time [days]", fontsize=12)
ax3.set_ylabel("Mass [kg]", fontsize=12)
ax3.set_title("Mass Profile Comparison", fontsize=14)
ax3.set_ylim(M_DRY - 200, M_WET + 200)
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=9, loc='upper right', title="(W1,W2), Final mass")

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

# --- Combined figure with all 4 plots ---
fig_combined = plt.figure(figsize=(18, 13))
gs = GridSpec(2, 2, figure=fig_combined, hspace=0.25, wspace=0.22,
              left=0.07, right=0.98, top=0.94, bottom=0.06)

# (a) Mass profiles
ax1 = fig_combined.add_subplot(gs[0, 0])
for idx, r in enumerate(results):
    t_days = r["t"] / 86400
    mass = r["x"][:, 6]
    mf = r["m_final"]
    ax1.plot(t_days, mass, color=colors[idx], linewidth=2.5,
            label=f"({r['w1']:.1f},{r['w2']:.1f}), $m_f$={mf:.0f}kg")
ax1.axhline(M_DRY, color='gray', linestyle='--', linewidth=2, label=f'$m_{{dry}}$={M_DRY:.0f} kg')
ax1.set_xlabel('Time [days]', fontsize=13, fontweight='bold')
ax1.set_ylabel('Mass [kg]', fontsize=13, fontweight='bold')
ax1.set_title('(a) Mass Profile Comparison', fontsize=14, fontweight='bold', pad=12)
ax1.legend(fontsize=10, loc='upper right', framealpha=0.95)
ax1.grid(True, alpha=0.3, linewidth=0.8)
ax1.tick_params(labelsize=11)

# (b) Thrust profiles
ax2 = fig_combined.add_subplot(gs[0, 1])
for idx, r in enumerate(results):
    t_days = r["t"] / 86400
    thrust = r["T_mag"]
    ax2.plot(t_days, thrust, color=colors[idx], linewidth=2.5,
            label=f"({r['w1']:.1f},{r['w2']:.1f})")
ax2.axhline(T_MAX, color='gray', linestyle='--', linewidth=2, label=f'$T_{{max}}$={T_MAX:.2f} N')
ax2.set_xlabel('Time [days]', fontsize=13, fontweight='bold')
ax2.set_ylabel('Thrust [N]', fontsize=13, fontweight='bold')
ax2.set_title('(b) Thrust Profile Comparison', fontsize=14, fontweight='bold', pad=12)
ax2.legend(fontsize=10, loc='upper right', framealpha=0.95)
ax2.grid(True, alpha=0.3, linewidth=0.8)
ax2.set_ylim([-0.05, 1.25])
ax2.tick_params(labelsize=11)

# (c) Trajectories
ax3 = fig_combined.add_subplot(gs[1, 0])
for idx, r in enumerate(results):
    x = r["x"][:, 0] / AU
    y = r["x"][:, 1] / AU
    tof = r["tf_days"]
    ax3.plot(x, y, color=colors[idx], linewidth=2.5,
            label=f"({r['w1']:.1f},{r['w2']:.1f}), {tof:.0f}d")
ax3.plot(1.0, 0.0, 'o', color='blue', markersize=12, label='Earth', zorder=10)
ax3.plot(0, 0, '*', color='gold', markersize=24, markeredgecolor='orange',
        markeredgewidth=2, label='Sun', zorder=10)
ax3.set_xlabel('X [AU]', fontsize=13, fontweight='bold')
ax3.set_ylabel('Y [AU]', fontsize=13, fontweight='bold')
ax3.set_title('(c) Trajectory Comparison', fontsize=14, fontweight='bold', pad=12)
ax3.legend(fontsize=10, loc='upper left', framealpha=0.95)
ax3.grid(True, alpha=0.3, linewidth=0.8)
ax3.set_aspect('equal')
ax3.set_xlim([-2.0, 1.8])
ax3.set_ylim([-1.5, 1.8])
ax3.tick_params(labelsize=11)

# (d) Pareto front
ax4 = fig_combined.add_subplot(gs[1, 1])
tof_vals = [r["tf_days"] for r in results]
fuel_vals = [r["propellant"] for r in results]
pareto_labels = [f"({r['w1']:.1f},{r['w2']:.1f})" for r in results]
ax4.plot(tof_vals, fuel_vals, 'o-', color='black', linewidth=3,
        markersize=12, markerfacecolor='white', markeredgewidth=2.5)
for i, label in enumerate(pareto_labels):
    ax4.annotate(label, (tof_vals[i], fuel_vals[i]),
                textcoords="offset points", xytext=(10,-10),
                fontsize=11, fontweight='bold')
ax4.set_xlabel('Transfer Time [days]', fontsize=13, fontweight='bold')
ax4.set_ylabel('Propellant Consumed [kg]', fontsize=13, fontweight='bold')
ax4.set_title('(d) Pareto Front: Fuel vs Time', fontsize=14, fontweight='bold', pad=12)
ax4.grid(True, alpha=0.3, linewidth=0.8)
ax4.tick_params(labelsize=11)

fig_combined.suptitle('Objective Weight Analysis: (W1, W2) Tradeoff Study',
                     fontsize=16, fontweight='bold')
plt.savefig(os.path.join(_script_dir, "sweep_combined.png"), dpi=200, bbox_inches='tight')
print(f"Saved: sweep_combined.png")

# --- Summary table ---
print(f"\n{'='*70}")
print(f"{'W1':>4} {'W2':>4} | {'Status':>8} | {'TOF [d]':>8} | {'m_f [kg]':>9} | {'Δm [kg]':>8} | {'Δv [m/s]':>8}")
print(f"{'-'*70}")
for r in results:
    dv = ISP * G0 * np.log(M_WET / max(r["m_final"], 1))
    print(f"{r['w1']:>4.1f} {r['w2']:>4.1f} | {r['status']:>8} | {r['tf_days']:>8.1f} | {r['m_final']:>9.1f} | {r['propellant']:>8.1f} | {dv:>8.0f}")
print(f"{'='*70}")

plt.show()
