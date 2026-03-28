# quantum_memory_data

Data availability repository for the publication in PRX Quantum. This project contains experimental data, analysis code, and generated figures for quantum memory research including Markovian dynamics, non-Markovian dynamics, and iSWAP gate characterization.

## Project Structure

### 📊 `data/` - Raw Experimental Data
The central repository for all raw experimental datasets. Organized by experiment type and date:

- **`Markovian/`** - Markovian quantum memory dynamics data, organized by measurement dates
- **`non_Markovian/`** - Non-Markovian memory effects data with 150ns timescale
- **`non_Markovian_152ns/`** - Extended non-Markovian measurements at 152ns interval
- **`iSWAP_t1t2/`** - T1 and T2 characterization data for iSWAP gate performance
- **`QPT_gate_error_vs_fidelity/`** - Quantum process tomography data correlating gate error and fidelity metrics
- **`QPT_vs_ellipsoid/`** - Quantum state/channel characterization using ellipsoid reconstruction methods

Each subdirectory contains timestamped folders with raw experimental measurements and intermediate analysis results.

### 📁 `quam_libs/` - Data Analysis Library
Python library containing all analysis tools and utilities for processing experimental data:

- **`analyzer.py`** - Main data analysis routines
- **`quantum_memory/`** - Specialized modules for quantum memory analysis
  - `swap_analyze.py` - iSWAP gate analysis
  - `noise_analyze.py` - Noise characterization
  - `iswap_simulation.py` - Theoretical simulations for iSWAP
- **`ellipsoid_utils/`** - Ellipsoid fitting and state reconstruction methods
- **`lib/`** - Utility functions for fitting, plotting, data handling
- **`experiments/`** - Experimental execution and parameter definitions
- **`components/`** - Hardware component definitions and characterization
- **`QPT_theory.py`** - Quantum process tomography theoretical framework
- **`quantum_channel_utils.py`** - Quantum channel analysis utilities
- Additional utilities: QI functions, simulation tools, noise modeling

### 📈 `paper_figure/` - Publication Figures
Contains all figures generated for publication:

- Figure files (PDF format): FIg1.pdf, FIg2.pdf, Fig3_raw.pdf, Fig3_mle.pdf
- Appendix figures and detailed analysis plots (AppendixC_raw.pdf, AppendixC_mle.pdf)
- iSWAP characterization plots (T1 exponential decay, Rabi chevron patterns)

These are the final polished figures ready for inclusion in the manuscript.

### 📓 Analysis Notebooks (Root Level)
Jupyter notebooks containing step-by-step data analysis workflows:

- **`Markovian_data.ipynb`** - Analysis of Markovian quantum memory dynamics
- **`non_Markovian_data.ipynb`** - Analysis of non-Markovian effects in quantum memory
- **`non_Markovian150ns_data.ipynb`** - Non-Markovian analysis with 150ns time resolution
- **`iSWAP_t1t2.ipynb`** - T1/T2 relaxation analysis for iSWAP gate operations
- **`QPT_gate_error_vs_fidelity.ipynb`** - Quantum process tomography and gate fidelity correlation analysis

Each notebook documents the experimental data processing, visualization, and interpretation steps.

### 🛠️ Additional Resources

- **`pyproject.toml`** - Project dependencies and configuration (Python ≥3.12)
  - Key dependencies: QuAM, PennyLane, QuTiP, CVXPY, Qiskit Experiments
  
- **`dev/`** - Development scripts and supplementary analyses
  
- **`quam_libs/pyproject.toml`** - Workspace member configuration for the analysis library

## Dependencies

The project uses modern quantum computing Python stack:
- **QuAM & QuAM-libs**: Quantum machine definition and utilities
- **PennyLane**: Quantum machine learning and simulation
- **QuTiP**: Quantum information theory computing
- **Qiskit Experiments**: Quantum circuit experiments framework
- **CVXPY**: Convex optimization for ellipsoid fitting
- **xarray & netCDF4**: Large dataset handling and storage

## Workflow

1. **Data Collection** → Raw data stored in `data/` subdirectories
2. **Analysis** → Python scripts in `quam_libs/` process and analyze datasets
3. **Visualization** → Jupyter notebooks in root apply analysis and generate visualizations
4. **Publication** → Final figures exported to `paper_figure/` for manuscript

## License & Citation

For data availability statement in PRX Quantum publication.
