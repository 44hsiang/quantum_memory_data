import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Tuple, List


# ==========================================
# 1. 理想設定
# ==========================================

# 理想量測投影算符 (POVM elements)
E_0 = np.array([[1, 0], [0, 0]], dtype=complex)
E_1 = np.array([[0, 0], [0, 1]], dtype=complex)
E_plus = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)
E_minus = 0.5 * np.array([[1, -1], [-1, 1]], dtype=complex)
E_R = 0.5 * np.array([[1, -1j], [1j, 1]], dtype=complex)
E_L = 0.5 * np.array([[1, 1j], [-1j, 1]], dtype=complex)
IDEAL_MEAS = [E_0, E_1, E_plus, E_minus, E_R, E_L]

# QPT公式中使用的輸入態轉置
rho_0_T = np.array([[1, 0], [0, 0]], dtype=complex)
rho_1_T = np.array([[0, 0], [0, 1]], dtype=complex)
rho_plus_T = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)
rho_R_T = 0.5 * np.array([[1, 1j], [-1j, 1]], dtype=complex)
IDEAL_RHOS_T = [rho_0_T, rho_1_T, rho_plus_T, rho_R_T]

# 理想 identity channel 的 Choi matrix（未正規化）
J_IDEAL = np.array([[1, 0, 0, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1, 0, 0, 1]], dtype=complex)


# ==========================================
# 2. Noisy measurement 模型
# ==========================================

def make_noisy_measurements(q: float) -> List[np.ndarray]:
    """
    建立含有 readout error 的量測算符 (POVM)

    假設量測誤差為 depolarizing white noise：
        E_noisy = q * E + (1-q) * I/2

    物理意義：
        q = measurement quality
        q = 1 → 完美量測
        q < 1 → 量測結果受到white noise污染

    此模型可確保：
        1. 每個 POVM element 仍為正定
        2. POVM completeness：sum(E_noisy) = I
    """
    I2 = np.eye(2, dtype=complex)
    noisy_meas = []

    for E in IDEAL_MEAS:
        E_noisy = q * E + (1 - q) * I2 / 2
        noisy_meas.append(E_noisy)

    return noisy_meas


# ==========================================
# 3. 模擬實驗資料
# ==========================================

def simulate_error_identity_data(alpha: float, q_meas: float = 1.0) -> np.ndarray:
    """
    模擬 QPT 實驗量測機率資料

    包含兩種誤差來源：

    1 state preparation over-rotation error
        每個準備態的旋轉角度變為 (1+alpha)*theta

    2 measurement white noise
        使用 noisy POVM 進行量測

    回傳：
        p_data (shape = 4 × 6)
        四個輸入態 × 六個量測結果的機率
    """

    def Ry(theta):
        return np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                         [np.sin(theta / 2),  np.cos(theta / 2)]], dtype=complex)

    def Rx(theta):
        return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                         [-1j * np.sin(theta / 2), np.cos(theta / 2)]], dtype=complex)

    psi_0 = np.array([1, 0], dtype=complex)

    # 含 over-rotation 的輸入態
    rhos_prep = [
        np.outer(psi_0, psi_0.conj()),
        np.outer(Ry(np.pi * (1 + alpha)) @ psi_0,
                 (Ry(np.pi * (1 + alpha)) @ psi_0).conj()),
        np.outer(Ry((np.pi / 2) * (1 + alpha)) @ psi_0,
                 (Ry((np.pi / 2) * (1 + alpha)) @ psi_0).conj()),
        np.outer(Rx((-np.pi / 2) * (1 + alpha)) @ psi_0,
                 (Rx((-np.pi / 2) * (1 + alpha)) @ psi_0).conj())
    ]

    meas_ops = make_noisy_measurements(q_meas)

    p_error = np.zeros((4, 6))
    for k, rho_out in enumerate(rhos_prep):
        for m, E in enumerate(meas_ops):
            p_error[k, m] = np.real(np.trace(E @ rho_out))

    return p_error


# ==========================================
# 4. MLE tomography
# ==========================================

def t_params_to_choi(t: np.ndarray) -> np.ndarray:
    """
    使用 T†T 參數化方式建立 Choi matrix

     自動保證 Choi matrix 為 positive semidefinite
    """
    T = np.zeros((4, 4), dtype=complex)
    T[0, 0] = t[0]
    T[1, 0] = t[1] + 1j * t[2]
    T[1, 1] = t[3]
    T[2, 0] = t[4] + 1j * t[5]
    T[2, 1] = t[6] + 1j * t[7]
    T[2, 2] = t[8]
    T[3, 0] = t[9] + 1j * t[10]
    T[3, 1] = t[11] + 1j * t[12]
    T[3, 2] = t[13] + 1j * t[14]
    T[3, 3] = t[15]
    return T.conj().T @ T


def tp_constraints(t: np.ndarray) -> np.ndarray:
    """
    Trace-preserving 條件：
        Tr_out(J) = I
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
    使用 Maximum Likelihood Estimation 重建 Choi matrix

    重建模型假設：
        使用理想量測模型進行 tomography
    """
    def objective_function(t):
        J_guess = t_params_to_choi(t)
        p_model = np.zeros((4, 6))
        for k in range(4):
            for m in range(6):
                M = np.kron(IDEAL_RHOS_T[k], IDEAL_MEAS[m])
                p_model[k, m] = np.real(np.trace(M @ J_guess))
        return np.sum((p_data - p_model) ** 2)

    np.random.seed(seed)
    t0 = np.random.rand(16) * 0.1
    t0[0] = 1.0
    t0[15] = 1.0

    cons = {'type': 'eq', 'fun': tp_constraints}

    res = minimize(
        objective_function,
        t0,
        method='SLSQP',
        constraints=cons,
        options={'maxiter': 500, 'ftol': 1e-8}
    )

    return t_params_to_choi(res.x)


def calculate_process_fidelity(J: np.ndarray) -> float:
    """
    計算 process fidelity：
        F = Tr(J_ideal J) / 4
    """
    return np.real(np.trace(J_IDEAL @ J)) / 4.0


# ==========================================
# 5. 主計算流程
# ==========================================

def compute_qpt_theory(
    gate_errors: np.ndarray,
    q_meas: float = 1.0,
    verbose: bool = True
) -> Tuple[List[np.ndarray], List[float]]:

    gate_errors = np.asarray(gate_errors)
    choi_matrices = []
    fidelities = []

    if verbose:
        print(f"Starting QPT theory computation ({len(gate_errors)} points)...")
        print(f"Measurement quality q = {q_meas:.4f}")

    for i, alpha in enumerate(gate_errors):

        p_err = simulate_error_identity_data(alpha, q_meas=q_meas)
        J_mle = perform_mle_tomography(p_err)

        choi_matrices.append(J_mle)

        fid = calculate_process_fidelity(J_mle)
        fidelities.append(fid)

        if verbose:
            print(f"[{i+1:3d}/{len(gate_errors)}] "
                  f"Alpha = {alpha:+.4f} | "
                  f"Process Fidelity = {fid:.6f}")

    return choi_matrices, fidelities


# ==========================================
# 6. 畫點圖
# ==========================================

def plot_qpt_points_vs_alpha(alphas, fidelities, q_meas):

    plt.figure(figsize=(8, 6))
    plt.plot(alphas, fidelities, 'go-', linewidth=2, markersize=6,
             label=f'QPT Theory Points (q={q_meas:.3f})')

    plt.xlabel('Alpha', fontsize=14)
    plt.ylabel('Process Fidelity', fontsize=14)
    plt.title('QPT Theory Points with Measurement Error', fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


# ==========================================
# 7. 執行
# ==========================================

if __name__ == "__main__":

    alphas = np.linspace(-0.2, 0.2, 21)
    q_meas = 0.995

    choi_matrices, fidelities = compute_qpt_theory(
        alphas,
        q_meas=q_meas,
        verbose=True
    )

    plot_qpt_points_vs_alpha(alphas, fidelities, q_meas)