"""Hohmann transfer visualization with plane change, parking orbits, and HCI propagation."""
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.body import Earth, Mars, Sun
from models.spacecraft import Psyche
from hohmann import hohmann_transfer, propagate_hci, propellant_mass


AU = 1.496e11


def plot_transfer_orbit(ax, result, states):
    """Top-down view of HCI propagation: Earth orbit, Mars orbit, transfer arc."""
    theta = np.linspace(0, 2 * np.pi, 300)

    r1 = Earth.a_sun
    ax.plot(r1 * np.cos(theta) / AU, r1 * np.sin(theta) / AU,
            'b-', lw=1.0, alpha=0.5, label='Earth orbit')

    r2 = Mars.a_sun
    ax.plot(r2 * np.cos(theta) / AU, r2 * np.sin(theta) / AU,
            'r-', lw=1.0, alpha=0.5, label='Mars orbit')

    ax.plot(states[:, 0] / AU, states[:, 1] / AU,
            'g-', lw=2.5, label='Transfer orbit')

    ax.plot(0, 0, 'yo', ms=14, zorder=5)
    ax.annotate('Sun', (0, 0), textcoords="offset points",
                xytext=(10, -12), fontsize=9)

    ax.plot(states[0, 0] / AU, states[0, 1] / AU, 'bo', ms=10, zorder=5)
    ax.annotate('Departure\n(LEO escape)', (states[0, 0] / AU, states[0, 1] / AU),
                textcoords="offset points", xytext=(10, 10), fontsize=8, color='blue')

    ax.plot(states[-1, 0] / AU, states[-1, 1] / AU, 'ro', ms=10, zorder=5)
    ax.annotate('Arrival\n(MSO capture)', (states[-1, 0] / AU, states[-1, 1] / AU),
                textcoords="offset points", xytext=(10, 10), fontsize=8, color='red')

    ax.set_xlabel('x [AU]')
    ax.set_ylabel('y [AU]')
    ax.set_title('Hohmann Transfer (HCI frame, top-down)')
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_3d_transfer(ax, states):
    """3D view showing out-of-plane component from plane change."""
    ax.plot(states[:, 0] / AU, states[:, 1] / AU, states[:, 2] / AU,
            'g-', lw=2, label='Transfer arc')
    ax.plot([states[0, 0] / AU], [states[0, 1] / AU], [states[0, 2] / AU],
            'bo', ms=8, label='Departure')
    ax.plot([states[-1, 0] / AU], [states[-1, 1] / AU], [states[-1, 2] / AU],
            'ro', ms=8, label='Arrival')

    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(Earth.a_sun * np.cos(theta) / AU, Earth.a_sun * np.sin(theta) / AU,
            np.zeros_like(theta), 'b-', lw=0.8, alpha=0.4)
    ax.plot(Mars.a_sun * np.cos(theta) / AU, Mars.a_sun * np.sin(theta) / AU,
            np.zeros_like(theta), 'r-', lw=0.8, alpha=0.4)

    ax.set_xlabel('x [AU]')
    ax.set_ylabel('y [AU]')
    ax.set_zlabel('z [AU]')
    ax.set_title('3D Transfer (1.85° plane change)')
    ax.legend(fontsize=8)


def plot_energy(ax, times, energies, result):
    """Energy profile along transfer + parking orbit energies."""
    ax.axhline(result.energy_transfer / 1e6, color='green', ls='-', lw=2,
               label=f'ε_transfer = {result.energy_transfer:.3e} J/kg')

    ax.axhline(result.energy_leo / 1e6, color='blue', ls='--', lw=1.5,
               label=f'ε_LEO (Earth) = {result.energy_leo:.3e} J/kg')
    ax.axhline(result.energy_mso / 1e6, color='red', ls='--', lw=1.5,
               label=f'ε_MSO (Mars) = {result.energy_mso:.3e} J/kg')

    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Specific energy [MJ/kg]')
    ax.set_title('Orbital Energy States')
    ax.legend(fontsize=7, loc='right')
    ax.grid(True, alpha=0.3)


def plot_velocity_radius(ax, times, states):
    """Velocity and radius vs time during transfer."""
    days = times / 86400.0
    r = np.linalg.norm(states[:, :3], axis=1) / AU
    v = np.linalg.norm(states[:, 3:], axis=1) / 1e3

    ax_r = ax
    ax_v = ax.twinx()

    ln1 = ax_r.plot(days, r, 'b-', lw=1.5, label='r [AU]')
    ax_r.set_xlabel('Time [days]')
    ax_r.set_ylabel('Heliocentric distance [AU]', color='blue')
    ax_r.tick_params(axis='y', labelcolor='blue')

    ln2 = ax_v.plot(days, v, 'r-', lw=1.5, label='v [km/s]')
    ax_v.set_ylabel('Speed [km/s]', color='red')
    ax_v.tick_params(axis='y', labelcolor='red')

    lines = ln1 + ln2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=8, loc='center right')
    ax.set_title('Distance & Speed vs Time (HCI)')
    ax.grid(True, alpha=0.3)


def plot_dv_budget(ax, result):
    """Delta-v breakdown bar chart."""
    labels = ['Δv₁\n(LEO escape)', 'Δv₂\n(MSO capture +\nplane change)']
    dvs = [result.dv_depart / 1e3, result.dv_capture / 1e3]
    colors = ['steelblue', 'coral']

    bars = ax.bar(labels, dvs, color=colors, width=0.5)
    for bar, dv in zip(bars, dvs):
        ax.text(bar.get_x() + bar.get_width() / 2, dv + 0.05,
                f'{dv:.3f} km/s', ha='center', va='bottom', fontsize=10)

    ax.axhline(result.dv_total / 1e3, color='gray', ls='--', lw=1,
               label=f'Total Δv = {result.dv_total / 1e3:.3f} km/s')
    ax.set_ylabel('Δv [km/s]')
    ax.set_title('Delta-V Budget (2-impulse)')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, max(dvs) * 1.3)


def plot_summary(ax, result):
    """Text summary panel."""
    ax.axis('off')
    tof_days = result.tof / 86400.0
    mp_chem = propellant_mass(result.dv_total, Psyche.m0, 320.0)
    mp_elec = propellant_mass(result.dv_total, Psyche.m0, Psyche.Isp)

    text = (
        "Mission Parameters\n"
        "─────────────────────────────\n"
        f"LEO parking orbit:   400 km alt\n"
        f"MSO parking orbit:   300 km alt\n"
        f"Plane change:        1.85°\n"
        "─────────────────────────────\n"
        f"v∞ departure:  {result.v_inf_dep / 1e3:.3f} km/s\n"
        f"v∞ arrival:    {result.v_inf_arr / 1e3:.3f} km/s\n"
        "─────────────────────────────\n"
        f"Δv₁ (escape):  {result.dv_depart / 1e3:.3f} km/s\n"
        f"Δv₂ (capture): {result.dv_capture / 1e3:.3f} km/s\n"
        f"Δv total:      {result.dv_total / 1e3:.3f} km/s\n"
        "─────────────────────────────\n"
        f"Transfer time: {tof_days:.1f} days\n"
        "─────────────────────────────\n"
        f"Psyche ({Psyche.m0:.0f} kg):\n"
        f"  Chemical:  {mp_chem:.0f} kg ({100 * mp_chem / Psyche.m0:.0f}%)\n"
        f"  SEP:       {mp_elec:.0f} kg ({100 * mp_elec / Psyche.m0:.0f}%)\n"
    )
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))


def main():
    result = hohmann_transfer(
        r1=Earth.a_sun,
        r2=Mars.a_sun,
        mu_sun=Sun.mu,
        mu_earth=Earth.mu,
        mu_mars=Mars.mu,
        R_earth=Earth.r,
        R_mars=Mars.r,
        alt_leo_km=400.0,
        alt_mso_km=300.0,
        delta_i_deg=1.85,
    )

    times, states, energies = propagate_hci(result.state_dep, Sun.mu, result.tof, n_points=500)

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle('Hohmann Transfer: Earth → Mars\n'
                 '2-impulse, 1.85° plane change, 400 km LEO → 300 km MSO, HCI frame',
                 fontsize=14, fontweight='bold', y=0.98)

    ax1 = fig.add_subplot(2, 3, 1)
    plot_transfer_orbit(ax1, result, states)

    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    plot_3d_transfer(ax2, states)

    ax3 = fig.add_subplot(2, 3, 3)
    plot_dv_budget(ax3, result)

    ax4 = fig.add_subplot(2, 3, 4)
    plot_velocity_radius(ax4, times, states)

    ax5 = fig.add_subplot(2, 3, 5)
    plot_energy(ax5, times, energies, result)

    ax6 = fig.add_subplot(2, 3, 6)
    plot_summary(ax6, result)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out_dir = ROOT / "plots"
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / "hohmann_transfer.png", dpi=150, bbox_inches='tight')
    print(f"Saved to {out_dir / 'hohmann_transfer.png'}")
    plt.show()


if __name__ == "__main__":
    main()
