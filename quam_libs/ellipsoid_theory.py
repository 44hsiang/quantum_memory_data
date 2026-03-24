"""Utilities to simulate Markovian ellipsoid data and compute memory robustness.

This module mirrors the "raw/simulation" flow used in `Markovian_data.ipynb`,
but wraps it into reusable functions.
"""

from __future__ import annotations

import io
import sys
from itertools import permutations, product
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np

# Allow running this file directly as a script:
# `python quam_libs/ellipsoid_theory.py`
if __package__ in (None, ""):
    workspace_root = Path(__file__).resolve().parents[1]
    workspace_root_str = str(workspace_root)
    if workspace_root_str not in sys.path:
        sys.path.insert(0, workspace_root_str)

from quam_libs.QI_utils import BR_density_matrix, density_matrix_to_bloch_vector
from quam_libs.macros import generate_uniform_sphere_angles, non_Gaussian_noise
from quam_libs.quantum_channel_utils import error_gate
from quam_libs.quantum_memory.entanglement_robustness import entanglementRobustness
from quam_libs.quantum_memory.legacy.NoiseAnalyze import (
    Checker,
    EllipsoidFitParameter,
    NoiseAnalyze,
    QuantumMemory,
)


def _simulate_single_dataset(
    t1: float,
    t2: float,
    error: float,
    n_points: int,
    t_delay: float,
    flux_noise: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate simulated Bloch vectors/angles/detuning for one T2 value."""
    theta_range, phi_range = generate_uniform_sphere_angles(n_points)
    data_angle = np.column_stack([theta_range, phi_range])

    # Match notebook behavior: sample non-Gaussian detuning once and accumulate.
    # detuning = non_Gaussian_noise(flux_noise, N=n_points) - non_Gaussian_noise(0, N=n_points)
    detuning = 0 
    bloch_vectors = []
    detuning_sum = 0.0
    for i in range(n_points):
        # detuning_sum += detuning[i]
        br_state = BR_density_matrix(
            theta_range[i],
            phi_range[i],
            t1,
            t2,
            t_delay,
            detuning=detuning,
        )
        error_br_state = error_gate(br_state, error)
        bloch_vectors.append(density_matrix_to_bloch_vector(error_br_state))

    data_xyz = np.array(bloch_vectors)
    return data_xyz, data_angle, detuning


def _qm_value(axes: np.ndarray, center: np.ndarray, r_matrix: np.ndarray) -> float:
    """Compute robustness for a given ellipsoid parameterization."""
    choi_state = QuantumMemory(axes, center, r_matrix).choi_state()
    checker = Checker(choi_state)
    with redirect_stdout(io.StringIO()):
        choi_fixed, _ = checker.choi_checker(index=[1], repeat=100, print_reason=False)
    return float(QuantumMemory.memory_robustness(choi_fixed))


def _improve_low_robustness_orientation(
    axes: np.ndarray,
    center: np.ndarray,
    r_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Exhaustively search axis/R re-labelings and keep the max-robustness Choi.

    We test all 3! axis permutations and all 2^3 column sign flips on the
    rotation matrix, then choose the combination with maximal robustness.
    """
    axes = np.array(axes, copy=True)
    center = np.array(center, copy=True)
    r_matrix = np.array(r_matrix, copy=True)

    best_axes = np.array(axes, copy=True)
    best_r = np.array(r_matrix, copy=True)
    best_robustness = _qm_value(best_axes, center, best_r)

    for perm in permutations(range(3)):
        perm_axes = axes[list(perm)]
        perm_r = r_matrix[:, list(perm)]

        for signs in product((1.0, -1.0), repeat=3):
            sign_arr = np.array(signs, dtype=float)
            candidate_r = perm_r * sign_arr
            candidate_robustness = _qm_value(perm_axes, center, candidate_r)

            if candidate_robustness > best_robustness:
                best_axes = np.array(perm_axes, copy=True)
                best_r = np.array(candidate_r, copy=True)
                best_robustness = candidate_robustness

    best_choi = QuantumMemory(best_axes, center, best_r).choi_state()
    return best_axes, best_r, float(best_robustness), best_choi


def simulate_markovian_robustness(
    t1: float,
    t2: float,
    error: float,
    n_points: int,
    *,
    t_delay: float = 16e-9,
    flux_noise: float = 0.002,
    random_seed: int | None = None,
    robustness_tol: float = 1e-6,
    checker_repeat: int = 100,
    checker_tol: float = 1e-6,
    return_details: bool = False,
) -> float | dict[str, Any]:
    """Run Markovian-style simulation and return memory robustness.

    Args:
        t1: Qubit T1 in seconds.
        t2: Qubit T2 in seconds.
        error: Gate error parameter passed into `error_gate`.
        n_points: Number of Bloch sphere sampling points.
        t_delay: Delay time in seconds. Defaults to 16 ns.
        flux_noise: Flux noise amplitude used by `non_Gaussian_noise`.
        random_seed: Optional NumPy random seed for reproducibility.
        robustness_tol: Threshold to trigger orientation correction.
        checker_repeat: Number of Choi correction iterations.
        checker_tol: Tolerance used in Choi checker.
        return_details: If True, return a dict with intermediate results.

    Returns:
        Robustness value (float) by default, or a detail dict if
        `return_details=True`.
    """
    if n_points <= 0:
        raise ValueError("n_points must be a positive integer")
    if t1 <= 0 or t2 <= 0:
        raise ValueError("t1 and t2 must be positive")

    if random_seed is not None:
        np.random.seed(random_seed)

    data_xyz, data_angle, detuning = _simulate_single_dataset(
        t1=t1,
        t2=t2,
        error=error,
        n_points=n_points,
        t_delay=t_delay,
        flux_noise=flux_noise,
    )

    fit_params = EllipsoidFitParameter(filter_method="convex", convex=True)
    analyzer = NoiseAnalyze(
        data_xyz,
        data_angle,
        ellipsoid_fit_parameters=fit_params,
    )

    center, axes, r_matrix, volume, fit_param = analyzer.ellipsoid_fit()
    qm = QuantumMemory(axes, center, r_matrix)
    choi = qm.choi_state()
    robustness = float(qm.memory_robustness)

    if robustness < robustness_tol:
        axes, r_matrix, robustness, choi = _improve_low_robustness_orientation(
            np.array(axes),
            np.array(center),
            np.array(r_matrix),
        )

    checker = Checker(choi)
    with redirect_stdout(io.StringIO()):
        choi_fixed, _ = checker.choi_checker(
            index=[1],
            repeat=checker_repeat,
            tol=checker_tol,
            print_reason=False,
        )
    robustness = float(entanglementRobustness(choi_fixed))

    if not return_details:
        return robustness

    return {
        "robustness": robustness,
        "qubit_properties": {
            "t1": t1,
            "t2": t2,
            "dephasing_rate": 1 / t2 - 1 / (2 * t1),
        },
        "ellipsoid": {
            "axes": np.array(axes),
            "center": np.array(center),
            "rotation_matrix": np.array(r_matrix),
            "volume": volume,
            "fit_param": fit_param,
        },
        "quantum_information": {
            "choi": choi_fixed,
            "robustness": robustness,
        },
        "data": {
            "original_xyz": data_xyz,
            "angle": data_angle,
            "corrected_xyz": np.array(analyzer.corrected_bloch),
            "corrected_dm": np.array(analyzer.corrected_dm),
            "detuning": np.array(detuning),
        },
    }


if __name__ == "__main__":
    result = simulate_markovian_robustness(
        t1=16e-6,
        t2=10e-6,
        t_delay=16e-9,
        error=0,
        n_points=200,
        return_details=False,
    )
    print(f"robustness = {result:.6g}")
    