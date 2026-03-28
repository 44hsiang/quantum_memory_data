import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/QPT_ellipsoid_GST.csv')

gate_error = df['gate_error'].values
qpt_theory_fidelities = df['qpt_theory_fidelities'].values
gate_fidelity = df['gate_fidelity'].values
ellipsoid_exp_robustness = df['ellipsoid_exp_robustness'].values
ellipsoid_exp_robustness_std = df['ellipsoid_exp_robustness_std'].values
ellipsoid_sim_robustness = df['ellipsoid_sim_robustness'].values
gst_exp_robustness = df['gst_exp_robustness'].values
gst_exp_robustness_std = df['gst_exp_robustness_std'].values


fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Plot Ellipsoid data
ax.errorbar(gate_fidelity*100, ellipsoid_exp_robustness, yerr=ellipsoid_exp_robustness_std,
                    linestyle='none', fmt='s', markersize=5, markerfacecolor='blue', 
                    markeredgecolor='blue', ecolor='blue', elinewidth=2, capsize=4, capthick=1.2,
                    label='Ellipsoid Data')

ax.hlines(ellipsoid_sim_robustness, xmin=85, xmax=100, colors="#79c7ffff", linewidth=2, label='Ellipsoid Theory')

# Plot GST data
err_gst = ax.errorbar(gate_fidelity*100, gst_exp_robustness, yerr=gst_exp_robustness_std,
                      linestyle='none', fmt='o', markersize=5, markerfacecolor='orange', 
                      markeredgecolor='orange', ecolor='orange', elinewidth=2, capsize=4, capthick=1.2)

# Customize plot
ax.set_xlim(85.5, 99.7)
ax.set_ylim(0.94, 0.995)
ax.set_xlabel('Gate Fidelity (%)', fontsize=16)
ax.set_ylabel('Robustness', fontsize=16)
ax.tick_params(axis='x', labelsize=13) 
ax.tick_params(axis='y', labelsize=13) 
ax.set_xticks([87, 90 ,93 ,96, 99])
ax.set_yticks([0.94, 0.95, 0.96, 0.97, 0.98, 0.99])
ax.grid(True, alpha=0.3)
plt.show()