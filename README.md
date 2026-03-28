# quantum_memory_data

Data availability repository for the publication in PRX Quantum. This project contains experimental data, analysis code, and automated figure generation.

## Project Structure

### 📊 `data/` - Raw Experimental Data
The central repository for all raw experimental datasets. Organized by experiment type and date:

- **`Markovian/`** - Markovian quantum memory dynamics data, organized by measurement dates
- **`non_Markovian/`** - Non-Markovian memory effects data with 150ns timescale
- **`non_Markovian_152ns/`** - Extended non-Markovian measurements at 152ns interval
- **`iSWAP_t1t2/`** - T1 and T2 characterization data for iSWAP gate performance
- **`QPT_gate_error_vs_fidelity/`** - Quantum process tomography data correlating gate error and fidelity metrics
- **`QPT_vs_ellipsoid/`** - Quantum state/channel characterization using ellipsoid reconstruction methods
- **`gst_qpt_ellipsoid/`** - GST, QPT, and ellipsoid characterization data
- **`GST_gate_error/`** - Gate set tomography error analysis

Each subdirectory contains timestamped folders with raw experimental measurements and intermediate analysis results.

### 🔬 `data_processing/` - Data Analysis Notebooks
Jupyter notebooks containing step-by-step data processing and analysis workflows:

- **`Markovian_data.ipynb`** - Analysis of Markovian quantum memory dynamics
- **`non_Markovian150ns_data.ipynb`** - Non-Markovian analysis with 150ns time resolution
- **`Raw_nonMarkovian_Markobian_data.ipynb`** - Comparative analysis of raw Markovian and non-Markovian data
- **`QPT_gate_error_vs_fidelity.ipynb`** - Quantum process tomography and gate fidelity correlation analysis
- **`resample_ellipsoid_QPT.ipynb`** - Ellipsoid resampling and QPT analysis

Each notebook documents the experimental data processing, visualization, and interpretation steps with detailed comments.

### 📈 `paper_figure/` - Publication Figures
Contains all figures generated for publication (PDF format):

- **`Fig2.pdf`** - Ellipsoid and QPT characterization for GST data
- **`Fig3.pdf`** - Non-Markovian dynamics and robustness analysis
- **`Fig4.pdf`** - Markovian dynamics and dephasing characterization
- **`Fig6.pdf`** - Ellipsoid and GST analysis
- **`Fig7.pdf`** - Non-Markovian ellipsoid and QPT comparison
- **`Fig8.pdf`** - Raw data visualization and analysis

These are the final polished figures ready for inclusion in the manuscript.

### 🎨 Figure Generation Scripts (Root Level)
Python scripts to automatically generate publication-quality figures:

- **`Fig2_ellipsoid_QPT_GST.py`** - Plot ellipsoid, QPT, and GST comparsion figures
- **`Fig3_nonMarkovian.py`** - Plot non-Markovian dynamics figures
- **`Fig4_Markovian.py`** - Plot Markovian dynamics and dephasing figures
- **`Fig6_ellipsoid_GST.py`** - Plot ellipsoid and GST comparsion figures
- **`Fig7_nonMarkovian_ellipsoid_QPT.py`** - Plot non-Markovian ellipsoid and QPT comparison figures
- **`Fig8_raw_data.py`** - Plot raw data figures
- **`fig_utils.py`** - Shared utilities for figure generation  mostly use for non-Markovian data analysis

Each script can be run independently to regenerate its corresponding publication figure.

### 📁 `quam_libs/` - Data Analysis Library
Python library containing all analysis tools and utilities for processing experimental data:

- **`analyzer.py`** - Main data analysis routines
- **`quantum_memory/`** - Specialized modules for quantum memory analysis
- **`ellipsoid_utils/`** - Ellipsoid fitting and state reconstruction methods
- **`lib/`** - Utility functions for fitting, plotting, data handling
- **`experiments/`** - Experimental execution and parameter definitions
- **`components/`** - Hardware component definitions and characterization
- **`QPT_theory.py`** - Quantum process tomography theoretical framework
- **`quantum_channel_utils.py`** - Quantum channel analysis utilities
- Additional utilities: QI functions, simulation tools, noise modeling

### 🛠️ Additional Resources

- **`pyproject.toml`** - Project dependencies and configuration (Python ≥3.12)
  - Key dependencies: QuAM,, QuTiP, CVXPY
  - Uses `uv` for dependency management
  

- **`_not_in_used/`** - Deprecated analysis scripts and notebooks (kept for reference)

## Workflow

1. **Data Collection** → Raw data stored in `data/` subdirectories by date
2. **Processing** → Jupyter notebooks in `data_processing/` process and analyze datasets
3. **Figure Generation** → Python scripts (Fig*.py) in root directory automatically generate publication-quality figures
4. **Publication** → Final figures exported to `paper_figure/` PDF files for manuscript

## Running Figure Generation

To regenerate all publication figures:
```bash
python Fig2_ellipsoid_QPT_GST.py
python Fig3_nonMarkovian.py
python Fig4_Markovian.py
python Fig6_ellipsoid_GST.py
python Fig7_nonMarkovian_ellipsoid_QPT.py
python Fig8_raw_data.py
```

## License & Citation

For data availability statement in PRX Quantum publication.
