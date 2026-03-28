import numpy as np
from matplotlib import pyplot as plt
from fig_utils import _set_dual_xaxis
from quam_libs.json_saver import load_from_npz_json
non_Markovian_data = load_from_npz_json(
    npz_path="data/analyze_data/non_markovian_arrays.npz",
    json_path="data/analyze_data/non_markovian.json",
)

mle_QPT_robustness_list = [non_Markovian_data['qpt'] [key]['mle robustness'] for key in non_Markovian_data['qpt'].keys()]
mle_ellipsoid_robustness_list = [non_Markovian_data['mle'][key]['quantum_information']['robustness'] for key in non_Markovian_data['mle'].keys()]
sim_mle_ellipsoid_robustness_list = [non_Markovian_data['sim_mle'][key]['quantum_information']['robustness']  for key in non_Markovian_data['sim_mle'].keys()]

time_QPT = np.arange(0, len(mle_QPT_robustness_list), 1)
time_ellipsoid = np.arange(0, len(mle_ellipsoid_robustness_list), 1) * 2

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(time_ellipsoid, mle_ellipsoid_robustness_list, '.', color='blue', label='Ellipsoid')
ax.plot(time_ellipsoid, sim_mle_ellipsoid_robustness_list, '-', color='red', label='Sim')
ax.plot(time_QPT, mle_QPT_robustness_list, '.', color='green', label='QPT')

# --- Convert X-axis to Angle Labels ---
coupling_mhz = 44.8
pi_time_ns = 1000 / coupling_mhz

# Dynamically get the maximum time length from the datasets
max_time_ns = max(time_ellipsoid[-1], time_QPT[-1])

# Calculate how many integer pi intervals fit into the maximum time
max_pi = int(max_time_ns / pi_time_ns)

# Generate the corresponding tick positions and labels
xticks_positions = [i * pi_time_ns for i in range(max_pi + 1)]
xticks_labels = ["0"] + [(f"{i}$\\pi$" if i > 1 else "$\\pi$") for i in range(1, max_pi + 1)]

# Apply x-axis settings
angle_labels = ["0"] + [(f"{i}$\\pi$" if i > 1 else "$\\pi$") for i in range(1, max_pi + 1)]
swap_time_ticks = [0, 50, 100, 150]
ax.set_xticks(xticks_positions)
ax.set_xticklabels(xticks_labels)
ax.set_xlim(0, max_time_ns)  
ax.set_xlim(0, max_time_ns)
_set_dual_xaxis(ax, xticks_positions, angle_labels, swap_time_ticks)
# --- Other Plot Settings ---
ax.legend(loc='upper right', ncol=3, columnspacing=1.0)

# Update labels for angle
ax.set_xlabel('Swap angle (rad)')
ax.set_ylabel('Robustness')

plt.tight_layout()
plt.show()