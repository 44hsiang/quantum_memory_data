from quam_libs.lib.fit import fit_decay_exp, decay_exp
import numpy as np

# utility functions for data generation and analysis
def generate_uniform_sphere_angles(n_points):

    indices = np.arange(0, n_points)
    golden_angle = np.pi * (3 - np.sqrt(5))  

    z = 1 - 2 * (indices + 0.5) / n_points     
    theta = np.arccos(z)                      
    phi = (indices * golden_angle) % (2 * np.pi)  

    theta_list = theta.tolist()
    phi_list = phi.tolist()

    return theta_list, phi_list


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

# math functions for depahsing simulation
def non_Gaussian_noise(delta, a= -47701117184.95155 , x0=0.029, c=4507310.510302373+4.56e9, N=10000):
    x = np.random.uniform(x0 - delta, x0 + delta, N)
    return a * (x - x0)**2 + c

def dephasing_errorbar(T1, T2, sT1, sT2, rho=0.0, pure=False):
    T1, T2  = np.asarray(T1, float), np.asarray(T2, float)
    sT1, sT2 = np.asarray(sT1, float), np.asarray(sT2, float)
    sign = 1.0 
    val = 1.0/T2 - sign*(1.0/(2.0*T1))
    dT2 = -1.0/(T2**2)
    dT1 = -sign/(2.0*T1**2)
    var = (dT1*sT1)**2 + (dT2*sT2)**2 + 2.0*rho*(dT1*sT1)*(dT2*sT2)
    var = np.maximum(var, 0.0)
    return val, np.sqrt(var)
