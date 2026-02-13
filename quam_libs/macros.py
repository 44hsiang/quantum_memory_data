from quam_libs.lib.fit import fit_decay_exp, decay_exp
import numpy as np

# fitting tool
def T1_extraction(ds):

    fit_data = fit_decay_exp(ds.state, "idle_time")
    fit_data.attrs = {"long_name": "time", "units": "µs"}

    # Decay rate and its uncertainty
    decay = fit_data.sel(fit_vals="decay")
    decay.attrs = {"long_name": "decay", "units": "ns"}
    decay_res = fit_data.sel(fit_vals="decay_decay")
    decay_res.attrs = {"long_name": "decay", "units": "ns"}
    # T1 and its uncertainty
    tau = -1 / fit_data.sel(fit_vals="decay")
    tau.attrs = {"long_name": "T1", "units": "µs"}
    tau_error = -tau * (np.sqrt(decay_res) / decay)
    tau_error.attrs = {"long_name": "T1 error", "units": "µs"}

    return tau.values, tau_error.values

