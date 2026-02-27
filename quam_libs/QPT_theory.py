"""
QPT Theory and MLE Tomography Module

This module provides functions to simulate quantum process tomography (QPT) data
with gate errors and reconstruct the Choi matrix using Maximum Likelihood Estimation (MLE).
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, List


# ==========================================
# 1. Fundamental Physical Quantities and Ideal Configuration
# ==========================================

# Ideal measurement projection operators
E_0 = np.array([[1, 0], [0, 0]], dtype=complex)
E_1 = np.array([[0, 0], [0, 1]], dtype=complex)
E_plus = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)
E_minus = 0.5 * np.array([[1, -1], [-1, 1]], dtype=complex)
E_R = 0.5 * np.array([[1, -1j], [1j, 1]], dtype=complex)
E_L = 0.5 * np.array([[1, 1j], [-1j, 1]], dtype=complex)
IDEAL_MEAS = [E_0, E_1, E_plus, E_minus, E_R, E_L]

# Ideal input state transposes for QPT theory formula
rho_0_T = np.array([[1, 0], [0, 0]], dtype=complex)
rho_1_T = np.array([[0, 0], [0, 1]], dtype=complex)
rho_plus_T = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)
rho_R_T = 0.5 * np.array([[1, 1j], [-1j, 1]], dtype=complex)  # Transpose of |R> equals |L>
IDEAL_RHOS_T = [rho_0_T, rho_1_T, rho_plus_T, rho_R_T]

# Perfect Identity Choi Matrix (unnormalized)
J_IDEAL = np.array([[1, 0, 0, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1, 0, 0, 1]], dtype=complex)


# ==========================================
# 2. Experimental Error Data Generator
# ==========================================

def simulate_error_identity_data(alpha: float) -> np.ndarray:
    """
    Simulate quantum state preparation data with rotation error.
    
    Parameters
    ----------
    alpha : float
        Over-rotation error parameter. Gate rotation angle becomes (1+alpha)*θ.
        
    Returns
    -------
    np.ndarray
        Shape (4, 6) array of measurement probabilities for 4 input states 
        and 6 measurement bases.
    """
    def Ry(theta):
        return np.array([[np.cos(theta/2), -np.sin(theta/2)], 
                         [np.sin(theta/2), np.cos(theta/2)]], dtype=complex)
    
    def Rx(theta):
        return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)], 
                         [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=complex)
    
    psi_0 = np.array([1, 0], dtype=complex)
    
    # Add (1+alpha) rotation error
    rhos_prep = [
        np.outer(psi_0, psi_0.conj()),                                                   # |0> perfect
        np.outer(Ry(np.pi * (1+alpha)) @ psi_0, (Ry(np.pi * (1+alpha)) @ psi_0).conj()), # |1> with error
        np.outer(Ry((np.pi/2) * (1+alpha)) @ psi_0, (Ry((np.pi/2) * (1+alpha)) @ psi_0).conj()), # |+> with error
        np.outer(Rx((-np.pi/2) * (1+alpha)) @ psi_0, (Rx((-np.pi/2) * (1+alpha)) @ psi_0).conj())# |R> with error (y basis)
    ]
    
    p_error = np.zeros((4, 6))
    for k, rho_out in enumerate(rhos_prep):
        for m, E in enumerate(IDEAL_MEAS):
            p_error[k, m] = np.real(np.trace(E @ rho_out))
    
    return p_error


# ==========================================
# 3. MLE Optimization Algorithm
# ==========================================

def t_params_to_choi(t: np.ndarray) -> np.ndarray:
    """
    Convert 16 real parameters to Hermitian Choi matrix via T.
    
    Parameters
    ----------
    t : np.ndarray
        16 real parameters representing the lower triangular part of matrix T.
        
    Returns
    -------
    np.ndarray
        4x4 Choi matrix J = T†T (guaranteed positive semidefinite).
    """
    T = np.zeros((4, 4), dtype=complex)
    T[0, 0] = t[0]
    T[1, 0] = t[1] + 1j*t[2]
    T[1, 1] = t[3]
    T[2, 0] = t[4] + 1j*t[5]
    T[2, 1] = t[6] + 1j*t[7]
    T[2, 2] = t[8]
    T[3, 0] = t[9] + 1j*t[10]
    T[3, 1] = t[11] + 1j*t[12]
    T[3, 2] = t[13] + 1j*t[14]
    T[3, 3] = t[15]
    
    return T.conj().T @ T


def tp_constraints(t: np.ndarray) -> np.ndarray:
    """
    Trace-Preserving constraint: Tr_out(J) = I.
    
    Parameters
    ----------
    t : np.ndarray
        16 real parameters.
        
    Returns
    -------
    np.ndarray
        Constraint values (should be close to 0).
    """
    J = t_params_to_choi(t)
    return np.array([
        np.real(J[0, 0] + J[1, 1]) - 1.0,
        np.real(J[2, 2] + J[3, 3]) - 1.0,
        np.real(J[0, 2] + J[1, 3]),
        np.imag(J[0, 2] + J[1, 3])
    ])


def perform_mle_tomography(p_data: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Perform MLE tomography to reconstruct the Choi matrix.
    
    Parameters
    ----------
    p_data : np.ndarray
        Shape (4, 6) array of measurement probabilities.
    seed : int, optional
        Random seed for reproducibility (default: 42).
        
    Returns
    -------
    np.ndarray
        Reconstructed 4x4 Choi matrix.
    """
    def objective_function(t):
        J_guess = t_params_to_choi(t)
        p_model = np.zeros((4, 6))
        for k in range(4):
            for m in range(6):
                M = np.kron(IDEAL_RHOS_T[k], IDEAL_MEAS[m])
                p_model[k, m] = np.real(np.trace(M @ J_guess))
        return np.sum((p_data - p_model)**2)
    
    # Provide initial guess (slightly mixed state to avoid gradient getting stuck at singular points)
    np.random.seed(seed)
    t0 = np.random.rand(16) * 0.1
    t0[0] = 1.0
    t0[15] = 1.0
    
    cons = {'type': 'eq', 'fun': tp_constraints}
    # Use SLSQP for constrained optimization
    res = minimize(objective_function, t0, method='SLSQP', 
                   constraints=cons, options={'maxiter': 500, 'ftol': 1e-8})
    
    return t_params_to_choi(res.x)


def calculate_process_fidelity(J: np.ndarray) -> float:
    """
    Calculate process fidelity between reconstructed and ideal Choi matrix.
    
    Parameters
    ----------
    J : np.ndarray
        4x4 Choi matrix.
        
    Returns
    -------
    float
        Process fidelity (between 0 and 1).
    """
    return np.real(np.trace(J_IDEAL @ J)) / 4.0


# ==========================================
# 4. Main Function: Compute Choi States and Fidelity for Given Gate Error Range
# ==========================================

def compute_qpt_theory(
    gate_errors: np.ndarray,
    verbose: bool = True
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Compute Choi matrices and fidelities for a range of gate errors.
    
    Parameters
    ----------
    gate_errors : np.ndarray or list
        Array of gate error values (alpha parameter).
        Example: np.linspace(-0.2, 0.2, 21) for ±20% error range.
    verbose : bool, optional
        If True, print progress information (default: True).
        
    Returns
    -------
    Tuple[List[np.ndarray], List[float]]
        - choi_matrices: List of 4x4 Choi matrices for each gate error.
        - fidelities: List of process fidelities for each gate error.
        
    Examples
    --------
    >>> alphas = np.linspace(-0.2, 0.2, 21)
    >>> choi_matrices, fidelities = compute_qpt_theory(alphas)
    >>> print(f"Fidelity at alpha=0: {fidelities[10]:.4f}")
    """
    gate_errors = np.asarray(gate_errors)
    choi_matrices = []
    fidelities = []
    
    if verbose:
        print(f"Starting QPT theory computation ({len(gate_errors)} points)...")
    
    for i, alpha in enumerate(gate_errors):
        p_err = simulate_error_identity_data(alpha)
        J_mle = perform_mle_tomography(p_err)
        choi_matrices.append(J_mle)
        
        fid = calculate_process_fidelity(J_mle)
        fidelities.append(fid)
        
        if verbose:
            print(f"[{i+1:3d}/{len(gate_errors)}] Alpha = {alpha:+.4f} | "
                  f"Process Fidelity = {fid:.6f}")
    
    return choi_matrices, fidelities


if __name__ == "__main__":
    # Example: Compute gate error from -20% to +20%
    alphas = np.linspace(-0.2, 0.2, 21)
    choi_matrices, fidelities = compute_qpt_theory(alphas)
    
    print("\n" + "="*60)
    print("Computation Results:")
    print("="*60)
    print(f"Alpha range: [{alphas[0]:.2f}, {alphas[-1]:.2f}]")
    print(f"Fidelity range: [{min(fidelities):.6f}, {max(fidelities):.6f}]")
