import numpy as np
import matplotlib.pyplot as plt
from qutip import *
# from quam_libs.quantum_memory.marcos import *
from quam_libs.quantum_memory.legacy.NoiseAnalyze import NoiseAnalyze, Checker, EllipsoidFitParameter,QuantumMemory
from quam_libs.QI_utils import density_matrix_to_bloch_vector, bloch_vector_to_density_matrix

def theta_phi_to_alpha_beta(theta, phi):
    """
    Convert spherical coordinates (theta, phi) to alpha and beta coefficients.
    """
    alpha = np.cos(theta / 2)
    beta = np.exp(1j * phi) * np.sin(theta / 2)
    return alpha, beta

def alpah_beta_to_qutip(alpha, beta):
    """
    Convert alpha and beta coefficients to a qubit state.
    """
    norm = np.sqrt(np.abs(alpha)**2 + np.abs(beta)**2)
    return (alpha / norm) * basis(2, 0) + (beta / norm) * basis(2, 1)
def generate_uniform_sphere_angles(n_points):

    indices = np.arange(0, n_points)
    golden_angle = np.pi * (3 - np.sqrt(5))  

    z = 1 - 2 * (indices + 0.5) / n_points     
    theta = np.arccos(z)                      
    phi = (indices * golden_angle) % (2 * np.pi)  

    theta_list = theta.tolist()
    phi_list = phi.tolist()

    return theta_list, phi_list

# Default parameters (will be overridden by function arguments)
DEFAULT_T1 = 250e-9  # relaxation time
DEFAULT_T2 = 200e-9  # dephasing time
DEFAULT_ERROR_RATE = 0.16  # readout error rate
DEFAULT_G = np.pi * 10538865 * 2  # coupling strength
detuning = 0

def wait_dm(theta, phi, g=None, T1=None, T2=None, delay=None):
    """
    Simulate waiting without swap gate (free evolution with noise).
    """
    if g is None:
        g = DEFAULT_G
    if T1 is None:
        T1 = DEFAULT_T1
    if T2 is None:
        T2 = DEFAULT_T2
    if delay is None:
        delay = np.arange(0, 16e-9, 20e-9)
    # basic operators
    detuning1 = 2*np.pi*detuning
    detuning2 = 2*np.pi*detuning

    sx, sy, sz, sm, I = sigmax(), sigmay(), sigmaz(), destroy(2), qeye(2)
    H0 = detuning1 * tensor(I, sx) + detuning2 * tensor(sx, I)  # free Hamiltonian

    # T1 and T2
    gamma_relax    = 1/T1
    gamma_deph     = 1/T2 - gamma_relax/2    

    c_ops = [
        np.sqrt(gamma_relax) * tensor(I, sm),   
        np.sqrt(gamma_deph)  * tensor(I, sz)    
    ]
    # initial state
    alpha, beta = theta_phi_to_alpha_beta(theta, phi)

    env_qubit = basis(2, 0)  # environment qubit in state |0>
    target_qubit = alpah_beta_to_qutip(alpha,beta)
    psi0 = tensor(env_qubit, target_qubit)

    # solve the master equation
    result = mesolve(
        H0,                    
        psi0,                 
        delay,                
        c_ops=c_ops,                
        e_ops=[],             
        options=Options(store_states=True)  
    )

    rho_t = result.states
    return rho_t

def BR_swap_dm_theta_phi(theta, phi, g=None, T1=None, T2=None, tlist=None):
    """
    BR (Bright Rabi) swap gate simulation starting from theta, phi angles.
    """
    if g is None:
        g = DEFAULT_G
    if T1 is None:
        T1 = DEFAULT_T1
    if T2 is None:
        T2 = DEFAULT_T2
    # basic operators
    detuning1 = 2*np.pi*0.1e6      
    detuning2 = 2*np.pi*0.1e6

    sx, sy, sz, sm, I = sigmax(), sigmay(), sigmaz(), destroy(2), qeye(2)
    H = (g/2) * (tensor(sx, sx) + tensor(sy, sy))
    H0 = detuning1 * tensor(I, sx) + detuning2 * tensor(sx, I)  # free Hamiltonian

    # T1 and T2
    gamma_relax    = 1/T1
    gamma_deph     = 1/T2 - gamma_relax/2    

    c_ops = [
        np.sqrt(gamma_relax) * tensor(I, sm),   
        np.sqrt(gamma_deph)  * tensor(I, sz)    
    ]
    
    # initial state
    alpha, beta = theta_phi_to_alpha_beta(theta, phi)

    env_qubit = basis(2, 0)  # environment qubit in state |0>
    target_qubit = alpah_beta_to_qutip(alpha,beta)
    psi0 = tensor(env_qubit, target_qubit)

    # time evolution settings
    if tlist is None:
        # Default time list if not provided
        tlist = np.arange(0,110e-9,2e-9)            

    # solve the master equation
    result = mesolve(
        H+H0,                    
        psi0,                 
        tlist,                
        c_ops=c_ops,                
        e_ops=[],             
        options=Options(store_states=True)  
    )

    rho_t = result.states 
    return rho_t
    
def BR_swap_dm(inital_state, g=None, T1=None, T2=None, tlist=None):
    """
    BR (Bright Rabi) swap gate simulation from initial density matrix.
    """
    if g is None:
        g = DEFAULT_G
    if T1 is None:
        T1 = DEFAULT_T1
    if T2 is None:
        T2 = DEFAULT_T2
    # basic operators
    detuning1 = 2*np.pi*detuning     
    detuning2 = 2*np.pi*detuning

    sx, sy, sz, sm, I = sigmax(), sigmay(), sigmaz(), destroy(2), qeye(2)
    H = (g/2) * (tensor(sx, sx) + tensor(sy, sy))
    H0 = detuning1 * tensor(I, sx) + detuning2 * tensor(sx, I)  # free Hamiltonian

    # T1 and T2
    gamma_relax    = 1/T1
    gamma_deph     = 1/T2 - gamma_relax/2    

    c_ops = [
        np.sqrt(gamma_relax) * tensor(I, sm),   
        np.sqrt(gamma_deph)  * tensor(I, sz)    
    ]

    # time evolution settings
    if tlist is None:
        # Default time list if not provided
        tlist = np.arange(0,110e-9,2e-9)            

    # solve the master equation
    result = mesolve(
        H+H0,                    
        inital_state,                 
        tlist,                
        c_ops,                
        e_ops=[],             
        options=Options(store_states=True)  
    )

    rho_t = result.states 
    return rho_t

def error_gate(rho,readout_error):
    gamma = readout_error
    I = qeye(2)
    sx = sigmax()
    sy = sigmay()
    return gamma/2*(sx*rho*sx+sy*rho*sy) + (1-gamma)*I*rho*I

def run_simulation(g, t1, t2, error_rate, swap_time_list, n_points=100):
    """
    Run the nonMarkovian quantum memory simulation.
    
    Parameters:
    -----------
    g : float
        Coupling strength (rad/s)
    t1 : float
        Relaxation time (seconds)
    t2 : float
        Dephasing time (seconds)
    error_rate : float
        Readout error rate (0 to 1)
    swap_time_list : array-like
        List of swap times to simulate (seconds)
    n_points : int, optional
        Number of test points on Bloch sphere (default: 100)
    
    Returns:
    --------
    dict : Dictionary containing results indexed by swap time values:
        {
            'swap_times': array of swap times,
            'results': {
                swap_time_value_ns: {
                    'volume': float,
                    'axes': array of 3 ellipsoid semi-axes,
                    'R': array,
                    'robustness': float,
                    'data_xyz': array of Bloch vectors
                },
                ...
            }
        }
    """
    swap_time_list = np.asarray(swap_time_list)
    
    # Generate initial test points on Bloch sphere
    theta_range, phi_range = generate_uniform_sphere_angles(n_points)
    
    data_angle_list = []
    for i in range(len(swap_time_list)):
        data_angle = []
        for j in range(len(theta_range)):
            data_angle.append([theta_range[j], phi_range[j]])
        data_angle_list.append(np.array(data_angle))
    
    # Initialize simulation data structure
    simulation_data = {
        f"swap={int(swap_time_list[i]*1e9)}ns": {'data_angle': data_angle_list[i].tolist(), 'data_xyz': []}
        for i in range(len(swap_time_list))
    }
    
    # Run quantum simulations for each initial state
    print("[*] Running quantum simulations...")
    for i in range(n_points):
        if i % 20 == 0:
            print(f"    Progress: {i}/{n_points}")
        
        theta, phi = data_angle_list[0][i]
        
        # Initial evolution without swap
        delay_dm = wait_dm(theta, phi, g=g, T1=t1, T2=t2)[-1]
        
        # Evolution with swap gate at various times
        dm = BR_swap_dm(delay_dm, g=g, T1=t1, T2=t2, tlist=swap_time_list)
        
        for j in range(len(swap_time_list)):
            dm_j = dm[j].ptrace(1)
            dm_j = error_gate(dm_j, error_rate).full()
            bloch_vector = density_matrix_to_bloch_vector(dm_j)
            simulation_data[f"swap={int(swap_time_list[j]*1e9)}ns"]['data_xyz'].append(bloch_vector.tolist())
    
    # Ellipsoid fitting
    print("[*] Fitting ellipsoids...")
    ellipsoid_fit_parameters = EllipsoidFitParameter()
    
    sim_swap_data = {}
    sim_swap_analyze = {}
    
    for i, key in enumerate(simulation_data):
        data_xyz = simulation_data[key]['data_xyz']
        data_angle = simulation_data[key]['data_angle']
        data_dm = np.array([bloch_vector_to_density_matrix(data_xyz[i]) for i in range(len(data_xyz))])
        
        sim_swap_analyzer = NoiseAnalyze(data_xyz, data_angle, ellipsoid_fit_parameters)
        corrected_dm = sim_swap_analyzer.corrected_dm
        corrected_bloch = sim_swap_analyzer.corrected_bloch
        
        sim_swap_data[key] = {
            'data_xyz': data_xyz,
            'data_dm': data_dm,
            'data_angle': data_angle,
            'corrected_xyz': corrected_bloch,
            'corrected_dm': corrected_dm,
        }
        sim_swap_analyze[key] = sim_swap_analyzer
    
    # Calculate robustness and other properties
    print("[*] Calculating robustness and Choi states...")
    swap_result_dict = {}
    
    for key in sim_swap_data.keys():
        center, axes, R, volume, param = sim_swap_analyze[key].ellipsoid_fit()
        qm_analyze = QuantumMemory(axes, center, R)
        
        # Calculate robustness from the ellipsoid properties
        robustness = np.prod(axes)  # or use qm_analyze.robustness if available
        choi_state = qm_analyze.choi_state()
        
        swap_result_dict[key] = {
            'axes': axes,
            'center': center,
            'R': R,
            'param': param,
            'volume': volume,
            'robustness': robustness,
            'choi_state': choi_state,
            'data_xyz': sim_swap_data[key]['data_xyz']  # Include Bloch vectors
        }
    
    # Verify and correct Choi states
    tol = 1e-6
    swap_choi_list = [swap_result_dict[key]["choi_state"] for key in swap_result_dict.keys()]
    keys = list(swap_result_dict.keys())
    
    print("[*] Verifying Choi states...")
    for i, key in enumerate(swap_result_dict.keys()):
        checker = Checker(swap_choi_list[i])
        choi, count = checker.choi_checker(index=[1], repeat=100, tol=tol, print_reason=False)
        
        swap_result_dict[key]["choi_state"] = choi
    
    # Format results for output
    results = {}
    for key in swap_result_dict.keys():
        swap_time_ns = int(key.split('=')[1].split('ns')[0])
        result_dict = {
            'volume': float(swap_result_dict[key]['volume']),
            'center': swap_result_dict[key]['center'].tolist() if hasattr(swap_result_dict[key]['center'], 'tolist') else swap_result_dict[key]['center'],
            'axes': swap_result_dict[key]['axes'].tolist() if hasattr(swap_result_dict[key]['axes'], 'tolist') else swap_result_dict[key]['axes'],
            'R': swap_result_dict[key]['R'].tolist() if hasattr(swap_result_dict[key]['R'], 'tolist') else swap_result_dict[key]['R'],
            'robustness': float(swap_result_dict[key]['robustness']),
            'data_xyz': swap_result_dict[key]['data_xyz'],
        }
        results[swap_time_ns] = result_dict
    
    print("[*] Simulation complete!")
    
    return {
        'swap_times': swap_time_list.tolist(),
        'results': results
    }


def print_results(results, swap_time):
    """
    Print simulation results for a specific swap time.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_simulation()
    swap_time : float or int
        Swap time in nanoseconds to print results for
    """
    if swap_time in results['results']:
        result = results['results'][swap_time]
        print(f"\n{'='*60}")
        print(f"Simulation Results for swap_time = {swap_time} ns")
        print(f"{'='*60}")
        print(f"Volume:       {result['volume']:.6f}")
        print(f"Axes:         {result['axes']}")
        print(f"R:            {result['R']}")
        print(f"Robustness:   {result['robustness']:.6f}")
        print(f"Data points:  {len(result['data_xyz'])} Bloch vectors")
        print(f"{'='*60}\n")
    else:
        print(f"Error: swap_time {swap_time} ns not found in results")
        print(f"Available swap times: {list(results['results'].keys())}")


if __name__ == "__main__":
    # Example usage
    print("[*] Starting nonMarkovian Quantum Memory Simulation")
    print("[*] ================================================")
    
    # Input parameters
    g = np.pi * 11363636 * 2  # coupling strength
    t1 = 250e-9  # relaxation time
    t2 = 200e-9  # dephasing time
    error_rate = 0.16  # readout error rate
    
    # Define swap times to simulate
    swap_time_list = np.arange(0, 153e-9, 2e-9)
    
    # Run simulation
    results = run_simulation(g, t1, t2, error_rate, swap_time_list, n_points=100)
    
    # Print results for specific swap times
    print_results(results, 0)      # First swap time
    print_results(results, 76)     # Middle swap time
    print_results(results, 152)    # Last swap time
