from scipy.linalg import eigvalsh
import numpy as np
import cvxpy as cp


identity = np.array([[1, 0], [0, 1]])
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])


# Bloch vector to density matrix and vice versa
def bloch_vector_to_density_matrix(bloch_vector):
    return 0.5 * (identity + bloch_vector[0] * sigma_x + bloch_vector[1] * sigma_y + bloch_vector[2] * sigma_z)

def density_matrix_to_bloch_vector(rho):
    r_x = np.real(np.trace(rho @ sigma_x))
    r_y = np.real(np.trace(rho @ sigma_y))
    r_z = np.real(np.trace(rho @ sigma_z))
    return np.array([r_x, r_y, r_z])

#quantum memory 
from qutip import Qobj
from picos import Problem, value
from picos.expressions.variables import HermitianVariable
from picos.expressions.algebra import trace, partial_transpose
def entanglementRobustness(state, solver='mosek', **extra_options) :
    if isinstance(state, Qobj):
        state = (state).full()
    
    SP = Problem()

    # add variable
    gamma = HermitianVariable("gamma", (4, 4))
    rho = HermitianVariable("rho", (4, 4))

    # add constraints
    SP.add_constraint(
        (state + gamma) - rho == 0
    )
    SP.add_constraint(
        partial_transpose(rho, 0) >> 0
    )
    SP.add_constraint(
        gamma >> 0
    )
    SP.add_constraint(
        trace(rho) - 1 >> 0
    )
    
    # find the solution
    SP.set_objective(
        'min',
        trace(rho) - 1
    )

    # solve the problem
    SP.solve(solver=solver, **extra_options)

    # return results
    return max(SP.value, 0)


class CorrectBloch:
    """
    Correct the 2*2 density matrix to ensure it is Hermitian, positive semi-definite, and has a trace of 1.
    """
    def __init__(self, uncorrect_bloch_vector):
        "uncorrect_bloch_vector: List"
        self.bloch_vector = np.array(uncorrect_bloch_vector).reshape(-1,3)
        self.dm = self.density_matrix()

    def density_matrix(self):
        experimental_dm = []
        for i in range(len(self.bloch_vector)):
            experimental_dm.append(bloch_vector_to_density_matrix(self.bloch_vector[i]))
        experimental_dm = np.array(experimental_dm).reshape(-1, 2, 2)
        return np.array(experimental_dm)

    def project_to_cptp(self,dims=(2, 2)):
        corrected_dm = []
        for dm_i in self.dm:
            X = cp.Variable(dims, hermitian=True)
            obj = cp.Minimize(cp.norm(X - dm_i, "fro"))
            constraints = [
                X >> 0,
                cp.trace(X) == 1
            ]
            prob = cp.Problem(obj, constraints)
            prob.solve(solver=cp.SCS)   
            if prob.status not in ("optimal", "optimal_inaccurate"):
                raise RuntimeError(f"Projection fail: {prob.status}")
            corrected_dm.append(X.value)
        corrected_dm = np.array(corrected_dm).reshape(-1, dims[0], dims[1])
        corrected_bloch = np.array([density_matrix_to_bloch_vector(dm) for dm in corrected_dm])
        return corrected_dm, corrected_bloch

class CorrectChoi:
    """Correct and validate a 4x4 Choi state.

    Usage
    -----
    cc = CorrectChoi(choi)
    corrected = cc.correct(max_iterations=100, tol=1e-8)

    After correction:
    - cc.corrected_choi: corrected Choi matrix
    - cc.n_corrections: number of projection steps used
    - cc.is_valid_state: whether final state is valid CP/TP state
    """
    def __init__(self, choi_state):
        self.choi_state = np.array(choi_state, dtype=complex).reshape(4, 4)
        self.corrected_choi = self.choi_state.copy()
        self.n_corrections = 0
        self.is_valid_state = False

    @staticmethod
    def _partial_trace_out(choi, d=2):
        # For a dxd channel Choi matrix of shape (d^2, d^2)
        choi4 = np.asarray(choi).reshape(d, d, d, d)  # [i,k,j,l]
        rho_in = np.zeros((d, d), dtype=complex)
        for i in range(d):
            rho_in += choi4[i, :, i, :]
        return rho_in

    @staticmethod
    def _trace_norm(rho, sigma):
        evals = eigvalsh(rho - sigma)
        return 0.5 * np.sum(np.abs(evals))

    def check_validity(self, choi=None, d=2, tol=1e-8, print_reason=False):
        mat = self.corrected_choi if choi is None else np.asarray(choi).reshape(d * d, d * d)

        is_hermitian = np.allclose(mat, mat.conj().T, atol=tol)
        eigenvalues = eigvalsh(mat)
        is_psd = np.all(eigenvalues >= -tol)
        trace_is_one = abs(np.trace(mat) - 1) < tol

        target = np.eye(d) / d
        pt = self._partial_trace_out(mat, d=d)
        pt_trace_is_one = self._trace_norm(pt, target) < tol

        is_valid = bool(is_hermitian and is_psd and trace_is_one and pt_trace_is_one)

        if (not is_valid) and print_reason:
            print("Choi state is invalid:")
            if not is_hermitian:
                print("❌ Choi state is not Hermitian.")
            if not is_psd:
                print("❌ Choi state is not positive semi-definite. Eigenvalues:", eigenvalues)
            if not trace_is_one:
                print(f"❌ Trace of the Choi state is not 1. Got trace = {np.trace(mat)}")
            if not pt_trace_is_one:
                print(f"❌ Partial trace out is not I/d. distance = {self._trace_norm(pt, target)}")

        return is_valid

    def project_to_cp_tp(self, choi=None, d=2, solver=cp.SCS, use_soft_constraint=True):
        """Project to CP/TP state with optional soft constraint on partial trace.
        
        Args:
            choi: Choi matrix to project
            d: qubit dimension (default 2)
            solver: CVXPY solver (tries fallback: mosek → cvxopt → scs)
            use_soft_constraint: If True, use soft constraint for partial trace (recommended)
                If False, use hard constraint (may fail to converge)
        """
        ref_choi = self.corrected_choi if choi is None else np.asarray(choi).reshape(d * d, d * d)
        X = cp.Variable((d * d, d * d), hermitian=True)

        def ptrace_out_cp(X, d_in=2, d_out=2):
            X4 = cp.reshape(X, (d_out, d_in, d_out, d_in), order='C')   # [i,k,j,l]
            rho_in = 0
            for i in range(d_out):
                rho_in += X4[i, :, i, :]
            return rho_in

        pt_target = np.eye(d) / d
        
        if use_soft_constraint:
            # Use soft constraint: minimize ||ptrace_out(X) - I/d||_F^2
            # This allows the optimization to balance all constraints
            pt_deviation = ptrace_out_cp(X, d_in=d, d_out=d) - pt_target
            objective = cp.Minimize(
                cp.norm(X - ref_choi, 'fro') + 0.1 * cp.norm(pt_deviation, 'fro')
            )
            constraints = [
                X >> 0,
                cp.trace(X) == 1,
            ]
        else:
            # Hard constraint (original approach)
            objective = cp.Minimize(cp.norm(X - ref_choi, 'fro'))
            constraints = [
                X >> 0,
                cp.trace(X) == 1,
                ptrace_out_cp(X, d_in=d, d_out=d) == pt_target,
            ]

        prob = cp.Problem(objective, constraints)
        
        # Solver fallback strategy
        solvers_to_try = [solver]
        if solver != cp.SCS:
            solvers_to_try.append(cp.SCS)
        
        result = None
        for solver_option in solvers_to_try:
            try:
                prob.solve(solver=solver_option, verbose=False)
                if prob.status in ("optimal", "optimal_inaccurate"):
                    result = np.asarray(X.value)
                    break
            except:
                continue
        
        if result is None:
            raise RuntimeError(f"Projection failed with all solvers. Status: {prob.status}")

        return result

    def correct(self, max_iterations=100, d=2, tol=1e-4, print_reason=True, solver=cp.SCS):
        """Iteratively correct Choi state to valid CP state, relaxing constraints as needed.
        
        Strategy:
        1. Try strict CP/TP constraints (both PSD and partial trace == I/d)
        2. If fails, relax to CP constraint only (PSD + trace = 1)
        3. Return the best result (either valid or best attempt)
        
        Args:
            max_iterations: Maximum number of projection iterations
            d: Qubit dimension (default 2)
            tol: Tolerance for validity check
            print_reason: Print detailed validation messages
            solver: CVXPY solver to use
        
        Returns:
            Corrected Choi matrix
        """
        choi = self.corrected_choi.copy()
        count = 0
        
        # First, try to satisfy all constraints (CP + TP)
        for _ in range(max_iterations):
            if self.check_validity(choi=choi, d=d, tol=tol, print_reason=False):
                self.corrected_choi = choi
                self.n_corrections = count
                self.is_valid_state = True
                if print_reason:
                    print(f"✓ Choi state is valid (CP+TP) after {count} correction(s).")
                return self.corrected_choi

            try:
                choi = self.project_to_cp_tp(choi=choi, d=d, solver=solver, use_soft_constraint=True)
            except Exception as e:
                if print_reason:
                    print(f"  Note: CP+TP projection step {count} encountered issue: {type(e).__name__}")
                break
                
            count += 1

        # If strict CP+TP fails, try CP constraint only (weaker but more achievable)
        if not self.is_valid_state:
            if print_reason:
                print(f"Could not achieve CP+TP after {count} iterations. Trying CP-only correction...")
            
            choi_cp_only = self.corrected_choi.copy()
            count_cp = 0
            
            for _ in range(max_iterations // 2):
                # Check if it's at least a valid CP state (PSD + trace = 1)
                evals = eigvalsh(choi_cp_only)
                is_psd = np.all(evals >= -tol)
                is_trace_one = abs(np.trace(choi_cp_only) - 1) < tol
                
                if is_psd and is_trace_one:
                    self.corrected_choi = choi_cp_only
                    self.n_corrections = count + count_cp
                    self.is_valid_state = False  # Mark as partially valid
                    if print_reason:
                        print(f"✓ Achieved valid CP state after {count + count_cp} corrections (TP not satisfied)")
                    return self.corrected_choi
                
                try:
                    # Simple projection to PSD + normalized
                    choi_cp_only = self._project_to_cp_only(choi_cp_only, d=d)
                except:
                    break
                    
                count_cp += 1

        # Final result (whether valid or best attempt)
        self.corrected_choi = choi
        self.n_corrections = count
        self.is_valid_state = self.check_validity(choi=choi, d=d, tol=tol, print_reason=False)

        if print_reason:
            evals = eigvalsh(choi)
            trace_val = np.trace(choi)
            pt = self._partial_trace_out(choi, d=d)
            pt_error = self._trace_norm(pt, np.eye(d) / d)
            
            if self.is_valid_state:
                print(f"✓ Choi state is fully valid after {count} iterations.")
            else:
                print(f"⚠ Choi state is partially corrected after {count} iterations:")
                print(f"  - PSD: {np.all(evals >= -1e-8)}")
                print(f"  - Trace=1: {abs(trace_val - 1) < 1e-8}")
                print(f"  - TP (partial trace error): {pt_error:.3e}")
                print(f"  Consider this as 'best effort' for non-ideal channels")

        return self.corrected_choi
    
    def _project_to_cp_only(self, choi, d=2):
        """Simple projection to CP state (PSD + normalized) without TP constraint."""
        X = cp.Variable((d * d, d * d), hermitian=True)
        
        objective = cp.Minimize(cp.norm(X - choi, 'fro'))
        constraints = [
            X >> 0,
            cp.trace(X) == 1,
        ]
        
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, verbose=False)
        
        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"CP projection failed: {prob.status}")
        
        return np.asarray(X.value)

    @property
    def result(self):
        return {
            "corrected_choi": self.corrected_choi,
            "n_corrections": self.n_corrections,
            "is_valid_state": self.is_valid_state,
        }
