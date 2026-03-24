# %% Import 
from qualibrate import QualibrationNode
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pandas as pd
import numpy as np
import glob
import os

# %% Load data
# id_qpt, id_gst, id_ell = 1172, 1131, 1132
id_qpt, id_gst, id_ell = 1218, 1131, 1132
gst_depth = 8
from pathlib import Path
data_path = Path("data/gst_qpt_ellipsoid").resolve()
node = QualibrationNode('robustness_to_fidelity')


da_gst = node.load_from_id(id_gst, base_path=data_path).results['ds'].sel(depth=gst_depth).sel(model_type='CPTP').robustness
da_qpt = node.load_from_id(id_qpt, base_path=data_path).results['ds'].sel(model_type='mitigated').robustness
da_ell = node.load_from_id(id_ell, base_path=data_path).results['ds'].robustness
da_fid = node.load_from_id(id_qpt, base_path=data_path).results['ds'].sel(model_type='mitigated').fidelity

standard_alpha = da_fid.alpha.values
standard_fidelity = da_fid.values
# %%
# 2. 強制重洗所有 DataArray 的座標，捨棄它們原本帶有誤差的 alpha
def clean_da(da, label_name):
    # 只取數值，丟掉原本的座標，重新建立一個乾淨的
    new_da = xr.DataArray(
        da.values, 
        dims=["alpha"], 
        coords={
            "alpha": standard_alpha, 
            "fidelity": ("alpha", standard_fidelity),
            "label": label_name
        }
    )
    return new_da

# 3. 重新建立三個乾淨的 da
da_gst_clean = clean_da(da_gst, "gst")
da_qpt_clean = clean_da(da_qpt, "qpt")
da_ell_clean = clean_da(da_ell, "ell")

# %%
combined = xr.concat(
    [da_gst_clean, da_qpt_clean, da_ell_clean], 
    # 強制使用 object 類型，不要讓 pandas 使用新的 StringDtype
    dim=pd.Index(['gst', 'qpt', 'ell'], name='method', dtype=object)
).sortby("fidelity")

# %% Plot
fig, ax = plt.subplots(figsize=(8, 6))

combined.plot.line(
    x='fidelity',
    hue='method',
    marker='o',
    linewidth=1.5
)


# plt.suptitle(f'Identity', 
#              fontsize=16, fontweight='bold', y=1.02)
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# %%
from quam_libs.QPT_theory import compute_qpt_theory

alphas = np.array([round(0.01 * i, 2) for i in range(31)]) 
qpt_choi_matrices_list, qpt_theory_fidelities_list = compute_qpt_theory(alphas, q_meas=0.995,verbose=True)

# %%
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(alphas + 1, qpt_theory_fidelities_list, linestyle='-', color='r', label='Theory')
da_fid.plot.line(
    x='alpha',
    marker='o',
    linewidth=1.5
)

plt.show()
# %%
