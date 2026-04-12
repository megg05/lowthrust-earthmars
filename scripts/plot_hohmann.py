"""Hohmann transfer visualization: orbit diagram, velocity, propellant, phasing, and timeline."""
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Arc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.body import Earth, Mars, Sun
from models.spacecraft import Psyche
from hohmann import hohmann, propellant_mass


AU = 1.496e11  # meters


def plot_transfer_orbit(ax, result):
    """Top-down view: Earth orbit, Mars orbit, transfer ellipse."""
    theta = np.linspace(0, 2 * np.pi, 300)

    # Earth orbit (circular approx)
    r1 = Earth.a_sun
    ax.plot(r1 * np.cos(theta) / AU, r1 * np.sin(theta) / AU,
            'b-', lw=1.2, label='Earth orbit')

    # Mars orbit (circular approx)
    r2 = Mars.a_sun
    ax.plot(r2 * np.cos(theta) / AU, r2 * np.sin(theta) / AU,
            'r-', lw=1.2, label='Mars orbit')

    # Transfer ellipse (half)
    a_t = result.a_transfer
    e_t = 1 - r1 / a_t  # eccentricity of transfer orbit
    theta_t = np.linspace(0, np.pi, 200)
    r_t = a_t * (1 - e_t**2) / (1 + e_t * np.cos(theta_t))
    ax.plot(r_t * np.cos(theta_t) / AU, r_t * np.sin(theta_t) / AU,
            'g--', lw=2, label='Transfer orbit')

    # Sun
    ax.plot(0, 0, 'yo', ms=12, zorder=5)
    ax.annotate('Sun', (0, 0), textcoords="offset points",
                xytext=(10, -10), fontsize=9)

    # Departure and arrival points
    ax.plot(r1 / AU, 0, 'bo', ms=10, zorder=5)
    ax.annotate('Departure\n(Earth)', (r1 / AU, 0), textcoords="offset points",
                xytext=(10, 10), fontsize=9, color='blue')

    ax.plot(-r2 / AU, 0, 'ro', ms=10, zorder=5)
    ax.annotate('Arrival\n(Mars)', (-r2 / AU, 0), textcoords="offset points",
                xytext=(10, 10), fontsize=9, color='red')

    # Direction arrow on transfer
    mid_idx = len(theta_t) // 3
    ax.annotate('', xy=(r_t[mid_idx + 5] * np.cos(theta_t[mid_idx + 5]) / AU,
                        r_t[mid_idx + 5] * np.sin(theta_t[mid_idx + 5]) / AU),
                xytext=(r_t[mid_idx] * np.cos(theta_t[mid_idx]) / AU,
                        r_t[mid_idx] * np.sin(theta_t[mid_idx]) / AU),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))

    ax.set_xlabel('x [AU]')
    ax.set_ylabel('y [AU]')
    ax.set_title('Hohmann Transfer: Earth to Mars')
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_velocity(ax, result):
    """Velocity diagram showing circular and transfer velocities with delta-v."""
    labels = ['Departure', 'Arrival']
    v_circ = [result.v_dep_circ / 1e3, result.v_arr_circ / 1e3]
    v_trans = [result.v_dep_transfer / 1e3, result.v_arr_transfer / 1e3]
    dv = [result.dv1 / 1e3, result.dv2 / 1e3]

    x = np.arange(len(labels))
    w = 0.25

    bars1 = ax.bar(x - w, v_circ, w, label='Circular velocity', color='steelblue')
    bars2 = ax.bar(x, v_trans, w, label='Transfer velocity', color='seagreen')
    bars3 = ax.bar(x + w, dv, w, label='Δv (burn)', color='coral')

    # Value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.2,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Velocity [km/s]')
    ax.set_title('Velocity Breakdown')
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)


def plot_propellant(ax, result):
    """Bar chart: chemical vs electric propellant mass."""
    Isp_chem = 320.0
    mp_chem = propellant_mass(result.dv_total, Psyche.m0, Isp_chem)
    mp_elec = propellant_mass(result.dv_total, Psyche.m0, Psyche.Isp)

    categories = [f'Chemical\n(Isp={Isp_chem:.0f} s)',
                  f'SPT-140\n(Isp={Psyche.Isp:.0f} s)']
    masses = [mp_chem, mp_elec]
    colors = ['#d95f02', '#1b9e77']

    bars = ax.bar(categories, masses, color=colors, width=0.5)

    for bar, m in zip(bars, masses):
        pct = 100 * m / Psyche.m0
        ax.text(bar.get_x() + bar.get_width() / 2, m + 20,
                f'{m:.0f} kg\n({pct:.0f}%)', ha='center', va='bottom', fontsize=9)

    ax.axhline(Psyche.m0, color='gray', ls='--', lw=1, label=f'Wet mass ({Psyche.m0:.0f} kg)')
    ax.set_ylabel('Propellant mass [kg]')
    ax.set_title('Propellant Comparison (Tsiolkovsky)')
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, Psyche.m0 * 1.1)


def plot_phasing(ax, result):
    """Phase angle geometry at departure."""
    r1 = Earth.a_sun
    r2 = Mars.a_sun

    # Transfer time
    tof = result.tof

    # Mars angular velocity (circular)
    n_mars = np.sqrt(Sun.mu / r2**3)

    # Mars sweeps this angle during transfer
    mars_sweep = n_mars * tof

    # Phase angle: Mars must be ahead of Earth by this much at departure
    phase_angle = np.pi - mars_sweep  # target is 180 deg minus what Mars travels
    phase_angle_deg = np.degrees(phase_angle)

    theta = np.linspace(0, 2 * np.pi, 300)

    # Earth and Mars orbits
    ax.plot(r1 * np.cos(theta) / AU, r1 * np.sin(theta) / AU, 'b-', lw=1, alpha=0.5)
    ax.plot(r2 * np.cos(theta) / AU, r2 * np.sin(theta) / AU, 'r-', lw=1, alpha=0.5)

    # Earth at departure (at 0 deg)
    ax.plot(r1 / AU, 0, 'bo', ms=12, zorder=5)
    ax.annotate('Earth\n(departure)', (r1 / AU, 0),
                textcoords="offset points", xytext=(12, -15), fontsize=9, color='blue')

    # Mars at departure (phase_angle ahead)
    mars_x = r2 * np.cos(phase_angle) / AU
    mars_y = r2 * np.sin(phase_angle) / AU
    ax.plot(mars_x, mars_y, 'ro', ms=12, zorder=5)
    ax.annotate('Mars\n(departure)', (mars_x, mars_y),
                textcoords="offset points", xytext=(12, 5), fontsize=9, color='red')

    # Mars at arrival (at 180 deg)
    ax.plot(-r2 / AU, 0, 'r^', ms=10, zorder=5, alpha=0.5)
    ax.annotate('Mars\n(arrival)', (-r2 / AU, 0),
                textcoords="offset points", xytext=(10, 10), fontsize=9, color='red', alpha=0.7)

    # Phase angle arc
    arc_r = 0.4
    arc_theta = np.linspace(0, phase_angle, 50)
    ax.plot(arc_r * np.cos(arc_theta), arc_r * np.sin(arc_theta), 'k-', lw=1.5)
    mid_a = phase_angle / 2
    ax.text(arc_r * 1.3 * np.cos(mid_a), arc_r * 1.3 * np.sin(mid_a),
            f'φ = {phase_angle_deg:.1f}°', fontsize=10, ha='center', va='center')

    # Sun
    ax.plot(0, 0, 'yo', ms=10, zorder=5)

    ax.set_xlabel('x [AU]')
    ax.set_ylabel('y [AU]')
    ax.set_title('Required Phase Angle at Departure')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)


def plot_timeline(ax, result):
    """Transfer timeline with key events."""
    tof_days = result.tof / 86400

    # Synodic period for Earth-Mars
    T_earth = 365.25  # days
    T_mars = 687.0    # days
    T_syn = 1 / abs(1 / T_earth - 1 / T_mars)

    events = {
        'Departure\nburn': 0,
        'Transfer\n(coast)': tof_days / 2,
        'Arrival\nburn': tof_days,
    }

    ax.set_xlim(-20, tof_days + 40)
    ax.set_ylim(-0.5, 1.5)

    # Timeline bar
    ax.barh(0.5, tof_days, left=0, height=0.3, color='seagreen', alpha=0.6)

    # Events
    for label, t in events.items():
        color = 'coral' if 'burn' in label else 'steelblue'
        ax.plot(t, 0.5, 'o', color=color, ms=12, zorder=5)
        ax.annotate(label, (t, 0.5), textcoords="offset points",
                    xytext=(0, 25), ha='center', fontsize=9, fontweight='bold')

    # Time labels
    ax.text(0, 0.15, 'Day 0', ha='center', fontsize=8)
    ax.text(tof_days, 0.15, f'Day {tof_days:.0f}', ha='center', fontsize=8)
    ax.text(tof_days / 2, 0.15, f'{tof_days / 2:.0f} days', ha='center',
            fontsize=8, color='gray')

    # Summary text
    summary = (
        f'Transfer time: {tof_days:.1f} days ({tof_days / 30.44:.1f} months)\n'
        f'Total Δv: {result.dv_total / 1e3:.2f} km/s\n'
        f'(Δv₁ = {result.dv1 / 1e3:.2f} km/s,  Δv₂ = {result.dv2 / 1e3:.2f} km/s)\n'
        f'Launch window recurrence: every {T_syn:.0f} days ({T_syn / 30.44:.0f} months)'
    )
    ax.text(tof_days / 2, 1.2, summary, ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_xlabel('Time [days]')
    ax.set_yticks([])
    ax.set_title('Transfer Timeline')
    ax.grid(True, axis='x', alpha=0.3)


def main():
    result = hohmann(Earth.a_sun, Mars.a_sun, Sun.mu)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Hohmann Transfer: Earth → Mars', fontsize=16, fontweight='bold', y=0.98)

    plot_transfer_orbit(axes[0, 0], result)
    plot_velocity(axes[0, 1], result)
    plot_propellant(axes[0, 2], result)
    plot_phasing(axes[1, 0], result)
    plot_timeline(axes[1, 1], result)

    # Hide the unused subplot, use for text summary
    ax_summary = axes[1, 2]
    ax_summary.axis('off')
    tof_days = result.tof / 86400
    Isp_chem = 320.0
    mp_chem = propellant_mass(result.dv_total, Psyche.m0, Isp_chem)
    mp_elec = propellant_mass(result.dv_total, Psyche.m0, Psyche.Isp)
    summary_text = (
        "Mission Summary\n"
        "─────────────────────────\n"
        f"Departure orbit:  {Earth.a_sun / AU:.3f} AU\n"
        f"Arrival orbit:    {Mars.a_sun / AU:.3f} AU\n"
        f"Transfer SMA:     {result.a_transfer / AU:.3f} AU\n"
        "─────────────────────────\n"
        f"Δv₁ (departure):  {result.dv1:.1f} m/s\n"
        f"Δv₂ (arrival):    {result.dv2:.1f} m/s\n"
        f"Δv total:         {result.dv_total:.1f} m/s\n"
        "─────────────────────────\n"
        f"Transfer time:    {tof_days:.1f} days\n"
        f"                  ({tof_days / 30.44:.1f} months)\n"
        "─────────────────────────\n"
        f"Spacecraft:       Psyche ({Psyche.m0:.0f} kg)\n"
        f"Chemical fuel:    {mp_chem:.0f} kg ({100 * mp_chem / Psyche.m0:.0f}%)\n"
        f"SEP fuel:         {mp_elec:.0f} kg ({100 * mp_elec / Psyche.m0:.0f}%)\n"
        f"Savings:          {mp_chem - mp_elec:.0f} kg ({100 * (mp_chem - mp_elec) / mp_chem:.0f}%)\n"
    )
    ax_summary.text(0.1, 0.95, summary_text, transform=ax_summary.transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir = ROOT / "plots"
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / "hohmann_transfer.png", dpi=150, bbox_inches='tight')
    print(f"Saved to {out_dir / 'hohmann_transfer.png'}")
    plt.show()


if __name__ == "__main__":
    main()
