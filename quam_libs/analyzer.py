from dataclasses import dataclass

from .QI_utils import CorrectBloch, CorrectChoi, entanglementRobustness
from .ellipsoid_utils.ellipsoid_utils import EllipsoidTool
from .quantum_channel_utils import compute_choi_state, compute_memory_robustness
import numpy as np

@dataclass
class EllipsoidFitParameters:
    filter_method: str = "ransac"
    ransac_threshold: float = 0.03
    ransac_iterations: int = 1500
    random_state: int | None = 42
    correct_rotation_orientation: bool = False
    find_best_order:bool = False

class QuantumMemoryAnalyze:
    
    def __init__(self, measurement_data, ideal_data, ellipsoid_fit_parameters=None):
        self.measurement_data = measurement_data
        self.ideal_data = ideal_data
        self.ellipsoid_fit_parameters = self._normalize_ellipsoid_fit_parameters(
            ellipsoid_fit_parameters
        )
        self.corrected_dm, self.corrected_bloch = self.valid_data()
        self.ellipsoid_result = self.ellipsoid_fit_results()
        center, axes, R = self.ellipsoid_result['center'], self.ellipsoid_result['axes'], self.ellipsoid_result['rotation_matrix']
        self.choi = self.choi_state(center, axes, R)
        self.memory_robustness = self.memory_robustness(self.choi)
    def valid_data(self,method="manual"):
        """
        project measurement data to density matrix by using MLE,
        and make sure density matrix is
        1. Hermitian => U = U^dagger
        2. Positive semi-definite => all eigenvalues are non-negative
        3. Trace = 1 => sum of diagonal elements is 1
        Returns:
            valid density matrix
            valid Bloch vector
        """ 
        corrector = CorrectBloch(self.measurement_data)
        corrected_dm, corrected_bloch = corrector.project_to_cptp()
        return corrected_dm, corrected_bloch


# ellipsoid fitting and plotting
    @staticmethod
    def _normalize_ellipsoid_fit_parameters(ellipsoid_fit_parameters):
        if ellipsoid_fit_parameters is None:
            return EllipsoidFitParameters()

        if isinstance(ellipsoid_fit_parameters, EllipsoidFitParameters):
            return ellipsoid_fit_parameters

        if isinstance(ellipsoid_fit_parameters, dict):
            return EllipsoidFitParameters(**ellipsoid_fit_parameters)

        raise TypeError(
            "ellipsoid_fit_parameters must be None, EllipsoidFitParameters, or dict."
        )
    
    def _build_ellipsoid_tool(self):
        p = self.ellipsoid_fit_parameters
        return EllipsoidTool(
            self.corrected_bloch,
            filter_method=p.filter_method,
            ransac_threshold=p.ransac_threshold,
            ransac_iterations=p.ransac_iterations,
            random_state=p.random_state,
            correct_rotation_orientation=p.correct_rotation_orientation,
        )

    def ellipsoid_fit_results(self):
        """
        Fit the ellipsoid from input data(Bloch vector)
        """
        return self._build_ellipsoid_tool().results

    def ellipsoid_plot(self, ax=None,title=None):
        """
        Plot the ellipsoid
        """   
        return self._build_ellipsoid_tool().plot(ax=ax, title=title)

    # quantum channel analysis
    @staticmethod
    def choi_state(center, axes, R):
        """Compute the Choi state from ellipsoid parameters.
        
        Args:
            center: Center of the ellipsoid (Bloch vector)
            axes: Semi-axes of the ellipsoid
            R: Rotation matrix of the ellipsoid
            
        Returns:
            Corrected Choi matrix
        """
        return compute_choi_state(center, axes, R)
    
    @staticmethod
    def memory_robustness(choi):
        """Return the entanglement robustness of `choi`.
        Usage: `QuantumMemory.memory_robustness(choi)`.
        """
        return compute_memory_robustness(choi)

