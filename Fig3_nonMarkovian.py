import pickle
from quam_libs.json_saver import load_from_npz_json
non_Markovian_data = load_from_npz_json(
    npz_path="data/analyze_data/non_markovian_arrays.npz",
    json_path="data/analyze_data/non_markovian.json",
)

with open('data/analyze_data/non_markovian_mle_analyze_data.pkl', 'rb') as file:
    non_Markovian_analyze_mle = pickle.load(file)

with open("data/analyze_data/non_markovian_raw_analyze_data.pkl", "rb") as file:
    non_Markovian_analyze_raw = pickle.load(file)


import numpy as np
from qualibrate import QualibrationNode
from pathlib import Path
data_path = Path("data/non_Markovian_152ns").resolve()
interaction_time = np.array([int(key.split('=')[1].split('ns')[0]) for key in non_Markovian_data['raw'].keys()])

node_iswap  = QualibrationNode(name="01 10 oscillation")
node_data = node_iswap.load_from_id(6043,base_path=data_path)

ds = node_data.results['ds']
quad = node_data.machine.qubits['q2'].freq_vs_flux_01_quad_term
iswap_point =  node_data.machine.qubit_pairs['q0_q2'].gates.iSWAP_unipolar.flux_pulse_control.amplitude
flux = ds.amp*iswap_point
print(f"iSWAP point: {iswap_point}")

def mhz_formatter(y, pos):
    detuning_hz = (((y - 1) * iswap_point)**2 * quad) * np.sign(y - 1)
    detuning_mhz = -detuning_hz / 1e6

    return f"{detuning_mhz:.0f}"

from fig_utils import plot_combined_figure
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 7,         
    'axes.labelsize': 7,     
    'xtick.labelsize': 7,   
    'ytick.labelsize': 7,    
    'axes.titlesize': 7,     
    'legend.fontsize': 7,    
})

fig = plot_combined_figure(
    data_type='mle', 
    node_data=node_data, 
    non_Markovian_data=non_Markovian_data, 
    analyze_raw_dict=non_Markovian_analyze_raw, 
    analyze_mle_dict=non_Markovian_analyze_mle,
    interaction_time=interaction_time, 
    iswap_point=iswap_point, 
    quad=quad, 
    mhz_formatter=mhz_formatter
)
plt.show()