import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ==========================================
# 1. 基礎物理量與理想設定
# ==========================================
E_0 = np.array([[1, 0], [0, 0]], dtype=complex)
E_1 = np.array([[0, 0], [0, 1]], dtype=complex)
E_plus = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)
E_minus = 0.5 * np.array([[1, -1], [-1, 1]], dtype=complex)
E_R = 0.5 * np.array([[1, -1j], [1j, 1]], dtype=complex)
E_L = 0.5 * np.array([[1, 1j], [-1j, 1]], dtype=complex)
ideal_meas = [E_0, E_1, E_plus, E_minus, E_R, E_L]

rho_0_T = np.array([[1, 0], [0, 0]], dtype=complex)
rho_1_T = np.array([[0, 0], [0, 1]], dtype=complex)
rho_plus_T = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)
rho_R_T = 0.5 * np.array([[1, 1j], [-1j, 1]], dtype=complex)
ideal_rhos_T = [rho_0_T, rho_1_T, rho_plus_T, rho_R_T]

J_ideal = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]], dtype=complex)

def Rz(theta): 
    return np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]], dtype=complex)
def Ry(theta): 
    return np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]], dtype=complex)
def Rx(theta): 
    return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)], [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=complex)

# ==========================================
# 2. Gate Error 圓錐誤差數據產生器
# ==========================================
def generate_cone_mixed_state(U_ideal, ideal_angle, gate_error, num_samples=300):
    """
    產生環繞著目標態的立體角圓錐混合態。
    deviation_angle = gate_error * ideal_angle
    """
    psi_0 = np.array([1, 0], dtype=complex)
    deviation_angle = gate_error * ideal_angle
    rho_mixed = np.zeros((2, 2), dtype=complex)
    
    for _ in range(num_samples):
        phi = np.random.uniform(0, 2 * np.pi)
        # 1. 在北極產生偏差 (Rx) 並旋轉形成圓錐 (Rz)
        # 2. 用 U_ideal 將圓錐轉到目標態位置
        psi_err = U_ideal @ Rz(phi) @ Rx(deviation_angle) @ psi_0
        rho_mixed += np.outer(psi_err, psi_err.conj())
        
    return rho_mixed / num_samples

def simulate_gate_error_data(gate_error, num_samples=300):
    psi_0 = np.array([1, 0], dtype=complex)
    
    # 假設 |0> 態的製備 (ground state cooling) 是完美的
    rho_0 = np.outer(psi_0, psi_0.conj())
    
    # 計算 |1>, |+>, |R> 的誤差圓錐混合態
    rho_1 = generate_cone_mixed_state(Ry(np.pi), np.pi, gate_error, num_samples)
    rho_plus = generate_cone_mixed_state(Ry(np.pi/2), np.pi/2, gate_error, num_samples)
    rho_R = generate_cone_mixed_state(Rx(-np.pi/2), np.pi/2, gate_error, num_samples)
    
    rhos_prep = [rho_0, rho_1, rho_plus, rho_R]
    
    p_error = np.zeros((4, 6))
    for k, rho_out in enumerate(rhos_prep):
        for m, E in enumerate(ideal_meas):
            p_error[k, m] = np.real(np.trace(E @ rho_out))
    return p_error

# ==========================================
# 3. MLE 最佳化演算法
# ==========================================
def t_params_to_choi(t):
    T = np.zeros((4, 4), dtype=complex)
    T[0,0] = t[0]
    T[1,0] = t[1] + 1j*t[2];  T[1,1] = t[3]
    T[2,0] = t[4] + 1j*t[5];  T[2,1] = t[6] + 1j*t[7];  T[2,2] = t[8]
    T[3,0] = t[9] + 1j*t[10]; T[3,1] = t[11]+ 1j*t[12]; T[3,2] = t[13]+ 1j*t[14]; T[3,3] = t[15]
    return T.conj().T @ T

def tp_constraints(t):
    J = t_params_to_choi(t)
    return np.array([
        np.real(J[0,0] + J[1,1]) - 1.0, np.real(J[2,2] + J[3,3]) - 1.0,
        np.real(J[0,2] + J[1,3]), np.imag(J[0,2] + J[1,3])
    ])

def perform_mle_tomography(p_data):
    def objective_function(t):
        J_guess = t_params_to_choi(t)
        p_model = np.zeros((4, 6))
        for k in range(4):
            for m in range(6):
                M = np.kron(ideal_rhos_T[k], ideal_meas[m])
                p_model[k, m] = np.real(np.trace(M @ J_guess))
        return np.sum((p_data - p_model)**2)
    
    np.random.seed(42)
    t0 = np.random.rand(16) * 0.1
    t0[0] = 1.0; t0[15] = 1.0 
    cons = {'type': 'eq', 'fun': tp_constraints}
    res = minimize(objective_function, t0, method='SLSQP', constraints=cons, options={'maxiter': 500, 'ftol': 1e-8})
    return t_params_to_choi(res.x)




# ==========================================
# 4. 主程式與繪圖 (以 Gate Error 為橫軸)
# ==========================================
if __name__ == "__main__":

    gate_errors = np.linspace(0, 0.50, 51) 
    fidelities = []

    print("開始執行隨機相位 Gate Error 模擬與 MLE (共 51 個點)...")
    for e in gate_errors:
        p_err = simulate_gate_error_data(e, num_samples=400) 
        J_mle = perform_mle_tomography(p_err)
        
        fid = np.real(np.trace(J_ideal @ J_mle)) / 4.0
        fidelities.append(fid)
        print(f"Gate Error = {e*100:5.1f}% | Process Fidelity = {fid:.4f}")

    # 繪圖
    plt.figure(figsize=(8, 6))
    plt.plot(gate_errors * 100, fidelities, marker='D', linestyle='-', color='purple', label='Phase-Averaged Error (Cone)')
    plt.axhline(1, color='gray', linestyle='--', alpha=0.6)
    plt.title('Process Fidelity vs. Gate Error ', fontsize=14)
    plt.xlabel('Gate Error $\\alpha$ (%)', fontsize=12)
    plt.ylabel('Process Fidelity $\\mathcal{F}$', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
