import numpy as np
import matplotlib.pyplot as plt

from quam_libs.json_saver import load_from_npz_json
Markovian_data = load_from_npz_json(
    npz_path="data/analyze_data/markovian_arrays.npz",
    json_path="data/analyze_data/markovian.json",
)

xx = Markovian_data['dephasing']['dephasing_voltage_^2_phi']
gamma_phi = Markovian_data['dephasing']['dephasing_rate']
gamma_phi_std = Markovian_data['dephasing']['dephasing_rate_std']
gamma_phi_fit = Markovian_data['dephasing']['dephasing_rate_fit']

raw_dephasing_rates = np.array([Markovian_data['raw'][key]['qubit_properties']['dephasing_rate'] for key in Markovian_data['mle'].keys()])*1e-6
mle_axes_list = np.array([Markovian_data['mle'][key]['ellipsoid']['axes'] for key in Markovian_data['raw'].keys()])
fit_dephasing_rates = Markovian_data['mle_fit']['dephasing_rate']*1e-6
fit_raw_axes_x = Markovian_data['mle_fit']['x_axis_fit']
fit_raw_axes_y = Markovian_data['mle_fit']['y_axis_fit']
fit_raw_axes_z = Markovian_data['mle_fit']['z_axis_fit']*np.ones_like(fit_dephasing_rates)
mle_robustness_list = np.array([Markovian_data['mle'][key]['quantum_information']['robustness'] for key in Markovian_data['mle'].keys()])
sim_dephasing_rates = np.array([Markovian_data['sim_mle'][key]['qubit_properties']['dephasing_rate'] for key in Markovian_data['sim_mle'].keys()])*1e-6
sim_robustness_list = np.array([Markovian_data['sim_mle'][key]['quantum_information']['robustness'] for key in Markovian_data['sim_mle'].keys()])

fig = plt.figure(figsize=(12, 3))
ax_name = ['dephasing','axes', 'robustness']
ax_dephasing = fig.add_subplot(1,len(ax_name), 1)
ax_axes = fig.add_subplot(1,len(ax_name), 2)
ax_robustness = fig.add_subplot(1, len(ax_name), 3)


ax_dephasing.errorbar(xx, gamma_phi*1e-6, yerr=gamma_phi_std*1e-6, capsize=3,color='k',fmt='.', label='Exp.')
ax_dephasing.plot(xx, gamma_phi_fit, color='red', label='Sim.')
ax_dephasing.set_xlabel('$A_\\phi^2(\\frac{\\phi^2}{\\phi_0^2})$',)
ax_dephasing.set_ylabel('$\\Gamma_{\\phi}$ (MHz)',)
ax_dephasing.text(-0.25, 1, "(a)", transform=ax_dephasing.transAxes, ha="left", va="top",fontsize=14)
ax_dephasing.legend(loc='upper left')


ax_axes.plot(raw_dephasing_rates,mle_axes_list[:, 0],'.k' ,label='X')
ax_axes.plot(raw_dephasing_rates,mle_axes_list[:, 1],'.r',label='Y')
ax_axes.plot(raw_dephasing_rates,mle_axes_list[:, 2],'.b', label='Z')
ax_axes.text(-0.25, 1, "(b)", transform=ax_axes.transAxes, ha="left", va="top",fontsize=14)
ax_axes.set_ylabel('Axes')
ax_axes.set_xlabel('$\\Gamma_\\phi$(MHz)')
ax_axes.legend(loc='lower left', ncol=3, columnspacing=1.0)    


ax_robustness.plot(sim_dephasing_rates,sim_robustness_list,'red',label='Sim.')
ax_robustness.plot(raw_dephasing_rates,mle_robustness_list,color='black',marker='*',markersize=5,linestyle='none',label='Exp.')
ax_robustness.text(-0.25, 1, "(c)", transform=ax_robustness.transAxes, ha="left", va="top",fontsize=14)


ax_robustness.legend(loc='upper right')
ax_robustness.set_ylabel('Robustness',)
ax_robustness.set_xlabel('$\\Gamma_\\phi$(MHz)',)


plt.tight_layout()
plt.show()