
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ==========================================
# 1. Top Panel Function (2D Plots)
# ==========================================

def _set_dual_xaxis(ax, xticks_positions, angle_labels, swap_time_ticks):
    """Bottom axis: swap time in ns. Top axis: angle (pi labels)."""

    ax.set_xticks(xticks_positions)
    ax.set_xticklabels(angle_labels)
    ax.set_xlabel('Swap angle (rad)')

    top_ax = ax.secondary_xaxis('top')
    top_ax.set_xticks(swap_time_ticks)
    # top_ax.set_xticklabels([f"{int(x)}" for x in swap_time_ticks])
    top_ax.set_xlabel('$\\tau$ (ns)')

def plot_top_panel(subfig, data_type, node_data, non_Markovian_data, interaction_time, iswap_point, quad, mhz_formatter):
    """
    Plots the iSWAP, Axes, and Robustness charts in the provided subfigure space.
    data_type: 'raw' or 'mle'
    """
    ax_name = ['iswap', 'axes', 'robustness']
    ax_iswap = subfig.add_subplot(1, len(ax_name), 1)
    ax_axes = subfig.add_subplot(1, len(ax_name), 2)
    ax_robustness = subfig.add_subplot(1, len(ax_name), 3)

    # --- 1. Calculate the time points (ns) corresponding to integer pi ---
    coupling_mhz = 44.8
    pi_time_ns = 1000 / coupling_mhz
    max_time_ns = 150
    max_pi = int(max_time_ns / pi_time_ns)

    xticks_positions = [i * pi_time_ns for i in range(max_pi + 1)]
    angle_labels = ["0"] + [(f"{i}$\\pi$" if i > 1 else "$\\pi$") for i in range(1, max_pi + 1)]
    swap_time_ticks = [0, 50, 100, 150]

    # --- 2. Prepare plotting data ---
    ds = node_data.results['ds']
    ds = ds.assign_coords(detuning=(((ds.amp-1)*iswap_point)**2 * quad)*np.sign(ds.amp-1))

    # ========== Subplot 1: iSWAP / Rabi Chevron ==========
    quadmesh = ds.sel(qubit='q0_q2').state_target.plot(
        x='time', y='amp', vmin=0.2, vmax=0.8, rasterized=True, ax=ax_iswap, add_colorbar=False
    )
    # Keep the colorbar narrow and close to panel (a) to avoid overlap with panel (c).
    cbar = subfig.colorbar(
        quadmesh,
        ax=ax_iswap,
        orientation="vertical",
        pad=0.01,
        fraction=0.038,
        ticks=[0.2, 0.4, 0.6, 0.8],
    )
    cbar.set_label("Probability $|0\\rangle$ ($p_0$)")

    ax_iswap.yaxis.set_major_formatter(mticker.FuncFormatter(mhz_formatter))
    ax_iswap.set_ylabel('Detuning (MHz)')
    ax_iswap.set_yticks([0.9374, 0.9638, 0.998, 1.0362, 1.0626])
    ax_iswap.set_xlim(0, max_time_ns)
    _set_dual_xaxis(ax_iswap, xticks_positions, angle_labels, swap_time_ticks)
    ax_iswap.text(-0.12, 1.15, "(a)", transform=ax_iswap.transAxes, ha="left", va="top")

    ax_iswap.set_title('')

    # ========== Subplot 2: Axes ==========
    # Dynamically select data based on data_type ('raw' or 'mle')
    axes_list = np.array([non_Markovian_data[data_type][key]['ellipsoid']['axes'] for key in non_Markovian_data[data_type].keys()])
    ax_axes.plot(interaction_time, axes_list[:, 0], '.', color='black', markersize=5, label='X')
    ax_axes.plot(interaction_time, axes_list[:, 1], '.', color='red', markersize=5, label='Y')
    ax_axes.plot(interaction_time, axes_list[:, 2], '.', color='blue', markersize=5, label='Z')

    ax_axes.set_ylim(0, 0.7) if data_type == 'raw' else ax_axes.set_ylim(0, 1.1)
    ax_axes.set_yticks([0.2, 0.4, 0.6]) if data_type == 'raw' else ax_axes.set_yticks([0.1, 0.4, 0.7, 1.0])
    ax_axes.set_ylabel('Axes')
    ax_axes.legend(loc='upper right', ncol=3, columnspacing=1.0)
    ax_axes.set_xlim(0, max_time_ns)
    _set_dual_xaxis(ax_axes, xticks_positions, angle_labels, swap_time_ticks)
    ax_axes.text(-0.15, 1.08, "(c)", transform=ax_axes.transAxes, ha="left", va="top")

    # ========== Subplot 3: Robustness ==========
    sim_data_type = f"sim_{data_type}"  # e.g., 'sim_raw' or 'sim_mle'

    robustness_list = np.array([non_Markovian_data[data_type][key]['quantum_information']['robustness'] for key in non_Markovian_data[data_type].keys()])
    ax_robustness.plot(interaction_time, robustness_list, '*k', markersize=5, label='Exp.')

    sim_robustness_list = np.array([non_Markovian_data[sim_data_type][key]['quantum_information']['robustness'] for key in non_Markovian_data[sim_data_type].keys()])
    ax_robustness.plot(interaction_time, sim_robustness_list, color='red', label='Sim.')

    ax_robustness.set_ylabel('Robustness')
    ax_robustness.legend(loc='upper right', ncol=2, columnspacing=1.0)
    ax_robustness.set_yticks([0.2, 0.4, 0.6]) if data_type == 'raw' else ax_robustness.set_yticks([0.1, 0.4, 0.7, 1.0])
    ax_robustness.set_xlim(0, max_time_ns)
    ax_robustness.set_ylim(-0.01, 0.7) if data_type == 'raw' else ax_robustness.set_ylim(0, 1.1)
    _set_dual_xaxis(ax_robustness, xticks_positions, angle_labels, swap_time_ticks)

    ax_robustness.text(-0.15, 1.08, "(d)", transform=ax_robustness.transAxes, ha="left", va="top")
    plt.tight_layout()

# ==========================================
# 2. Bottom Panel Function (3D Ellipsoids)
# ==========================================
def plot_bottom_panel(subfig, analyze_data):
    """
    Plots the 5 3D ellipsoids in the provided subfigure space.
    """
    ax_list = [subfig.add_subplot(1, 5, i+1, projection='3d') for i in range(5)]
    times_ns = [0, 12, 22, 34, 44]
    coupling_mhz = 44.8
    for t_ns, ax in zip(times_ns, ax_list):
        theta_in_pi = t_ns * (coupling_mhz * 1e-3)
        if np.isclose(theta_in_pi, 0, atol=1e-2):
            title_str = "0"
        elif np.isclose(theta_in_pi % 1, 0, atol=1e-2):
            title_str = f"{int(round(theta_in_pi))}$\\pi$"
        else:
            title_str = f"{theta_in_pi:.1f}$\\pi$"

        dict_key = f'interaction_time={t_ns}ns'
        analyze_data[dict_key].ellipsoid_plot(title=title_str, ax=ax)

    L = 1.1
    for ax in ax_list:
        ax.set_xlabel('X',)
        ax.set_zlabel('Z')
        ax.grid(False)
        ax.xaxis.pane.set_facecolor((1, 1, 1, 0))
        ax.yaxis.pane.set_facecolor((1, 1, 1, 0))
        ax.zaxis.pane.set_facecolor((1, 1, 1, 0))
        ax.xaxis.pane.set_edgecolor((1, 1, 1, 0))
        ax.yaxis.pane.set_edgecolor((1, 1, 1, 0))
        ax.zaxis.pane.set_edgecolor((1, 1, 1, 0))
        ax.yaxis.line.set_color((1, 1, 1, 0))

        legend = ax.get_legend()
        if legend:
            legend.remove()

        ax.set_xlim([-L, L])
        ax.set_ylim([-L, L])
        ax.set_zlim([-L, L])
        ax.tick_params(axis='x', which='both', length=0)
        ax.tick_params(axis='y', which='both', length=0)
        ax.tick_params(axis='z', which='both', length=0, color=(1, 1, 1, 0))
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([])
        ax.set_zticks([-1, 0, 1])
        ax.view_init(elev=0, azim=-90)
        ax.set_box_aspect((1, 1, 1))
        ax.set_proj_type('ortho')
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='z', labelsize=10)
        ax.text2D(-0.05, 1.1, '(b)', transform=ax.transAxes, ha="left", va="top") if ax == ax_list[0] else None


# ==========================================
# 3. Master function to combine everything
# ==========================================
def plot_combined_figure(data_type, node_data, non_Markovian_data, analyze_raw_dict, analyze_mle_dict, interaction_time, iswap_point, quad, mhz_formatter):
    """
    Main function to generate the combined figure.
    Usage:
    plot_combined_figure('raw', node_data, ...)
    or
    plot_combined_figure('mle', node_data, ...)
    """
    # Initialize the master figure
    fig_main = plt.figure(figsize=(8, 4), dpi=200)

    # Split the figure into 2 rows. You can adjust height_ratios if the 3D plots need more/less space.
    subfigs = fig_main.subfigures(2, 1, height_ratios=[1, 1.2], hspace=0.1)

    # Automatically select the correct ellipsoid dictionary based on data_type
    analyze_data = analyze_mle_dict if data_type == 'mle' else analyze_raw_dict

    # Plot the top and bottom panels into their respective subfigures
    plot_top_panel(subfigs[0], data_type, node_data, non_Markovian_data, interaction_time, iswap_point, quad, mhz_formatter)
    plot_bottom_panel(subfigs[1], analyze_data)
    plt.tight_layout()
    plt.show()
    return fig_main


