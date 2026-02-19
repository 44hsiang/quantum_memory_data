from quam_libs.components import QuAM
from quam_libs.quantum_memory.legacy.marcos import *
from quam_libs.analyzer import QuantumMemoryAnalyze, EllipsoidFitParameters
from quam_libs.QI_utils import *

def xyz_data_to_gate_fidelity(
    ds, 
    confusion_matrix,
    qubit,
    n_runs,
    apply_mitigation=False,
    operation_name='id',
    verbose=False
):
    """
    Convert XYZ measurement data to gate fidelity metrics.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing measurement results with dimensions: qubit, initial_state, axis
    confusion_matrix : np.ndarray
        Readout confusion matrix (transposed)
    qubit : object
        Qubit object with name and resonator attributes
    n_runs : int
        Number of measurement runs
    apply_mitigation : bool
        Whether to apply readout error mitigation (default: False)
    operation_name : str
        Name of the quantum operation being characterized (default: 'id')
    
    Returns:
    --------
    results : dict
        Dictionary containing raw and optionally mitigated results with keys:
        - 'raw': raw data analysis results
        - 'mitigated': (optional) mitigated data analysis results
        - 'quantum_information': PTM, superoperator, and Choi matrix data
    """

    results = {}
    data = {}
    mitigate_data = {}
    desired_state_name = ['0', '1', '+', 'i+']
    desired_state = [[0, 0], [np.pi, 0], [np.pi/2, 0], [np.pi/2, np.pi/2]]
    
    if verbose:
        print(f"ideal Bloch vector: {np.rad2deg(desired_state[0][0]):.3} and {np.rad2deg(desired_state[0][1]):.3} in degree")
    
    # ========== RAW DATA PROCESSING ==========
    qn = qubit.name
    data[qn] = {}
    
    # Create raw results for each prepared initial state
    for idx, initial_state in enumerate(ds.initial_state.values):
        theta, phi = desired_state[idx]
        ds_ = ds.sel(initial_state=initial_state)
        
        # Raw counts (ensure both outcomes exist)
        x_count = np.bincount(ds_.sel(qubit=qn, axis='x').state.values, minlength=2)
        y_count = np.bincount(ds_.sel(qubit=qn, axis='y').state.values, minlength=2)
        z_count = np.bincount(ds_.sel(qubit=qn, axis='z').state.values, minlength=2)
        
        # Raw Bloch vector (from counts)
        bloch = [
            (x_count[0] - x_count[1]) / n_runs,
            (y_count[0] - y_count[1]) / n_runs,
            (z_count[0] - z_count[1]) / n_runs,
        ]
        res = QuantumStateAnalysis(bloch, [theta, phi])
        
        data[qn][initial_state] = {
            'Bloch vector': bloch,
            'density matrix': res.rho,
            'fidelity': res.fidelity,
            'trace_distance': res.trace_distance,
        }
        
        # Brief per-state summary
        if verbose:
            print(f"\n{qn} - {initial_state} (raw)")
            print(f"Bloch vector: {bloch}")
            print(
                f"theta, phi: {np.rad2deg(res.theta):.3}, {np.rad2deg(res.phi):.3}"
            )
            print(f"fidelity, trace distance: {res.fidelity:.3}, {res.trace_distance:.3}")
    
    # Pack per-qubit results for convenience
    results[qn] = {
        name: {'raw': data[qn][name]}
        for name in desired_state_name
    }
    
    # Build superoperators (raw) for this qubit
    inputs = [rho_0, rho_1, rho_plus, rho_plus_i]
    outputs_raw = [
        data[qn]['0']['density matrix'],
        data[qn]['1']['density matrix'],
        data[qn]['+']['density matrix'],
        data[qn]['i+']['density matrix'],
    ]
    
    ptm = build_pauli_transfer_matrix(inputs, outputs_raw)
    superop = ptm_to_superop(ptm)
    choi = superop_to_choi(superop, 2, 2) / 2
    checker = CorrectChoi(choi)
    choi_cptp, count = checker.choi_checker(index=[1], repeat=100, tol=1e-6,print_reason=False)
        
    # Display result summaries
    np.set_printoptions(precision=3, suppress=True)
    if verbose:
        print(f"\n{qn} - operation name: {operation_name}")
        print("Process fidelity with respect to the target process (RAW):")
        print(f"raw: {process_fidelity(ptm, target=operation_name):.3}")
    
    # Save to results
    results[qn]['quantum_information'] = {
        'raw': {
            'ptm': ptm,
            'superoperator': superop,
            'choi': choi_cptp,
            'Quantum Memory Robustness': entanglementRobustness(choi_cptp),
            'fidelity': process_fidelity(ptm, target=operation_name),
        }
    }
    
    # ========== MITIGATED DATA PROCESSING ==========
    if apply_mitigation:
        mitigate_data[qn] = {}
        # import warnings
        # # Build the readout mitigator once per qubit
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore", category=DeprecationWarning)
        #     from qiskit.result import CorrelatedReadoutMitigator
        #     mitigator = CorrelatedReadoutMitigator(
        #         assignment_matrix=np.array(qubit.resonator.confusion_matrix).T,
        #         qubits=[0],
        #     )
        
        # Create mitigated results for each prepared initial state
        for idx, initial_state in enumerate(ds.initial_state.values):
            theta, phi = desired_state[idx]
            ds_ = ds.sel(initial_state=initial_state)
            
            # Raw counts
            x_count = np.bincount(ds_.sel(qubit=qn, axis='x').state.values, minlength=2)
            y_count = np.bincount(ds_.sel(qubit=qn, axis='y').state.values, minlength=2)
            z_count = np.bincount(ds_.sel(qubit=qn, axis='z').state.values, minlength=2)
            
            # Mitigate the readout for each measurement axis
            new_px_0 = np.array([MLE([x_count[0]/n_runs, x_count[1]/n_runs], confusion_matrix)[0]])
            new_py_0 = np.array([MLE([y_count[0]/n_runs, y_count[1]/n_runs], confusion_matrix)[0]])
            new_pz_0 = np.array([MLE([z_count[0]/n_runs, z_count[1]/n_runs], confusion_matrix)[0]])
            
            # Mitigated Bloch vector (already probabilities)
            m_bloch = np.array([2*new_px_0-1, 2*new_py_0-1, 2*new_pz_0-1], dtype=float).ravel()
            if np.linalg.norm(m_bloch) > 1:
                m_bloch = m_bloch / np.linalg.norm(m_bloch)
            m_res = QuantumStateAnalysis(m_bloch, [theta, phi])
            
            mitigate_data[qn][initial_state] = {
                'Bloch vector': m_bloch,
                'density matrix': m_res.rho,
                'fidelity': m_res.fidelity,
                'trace_distance': m_res.trace_distance,
            }
            
            # Brief per-state summary
            if verbose:
                print(f"\n{qn} - {initial_state} (mitigated)")
                print(f"Bloch vector: {m_bloch}")
                print(f"theta, phi: {np.rad2deg(m_res.theta):.3}, {np.rad2deg(m_res.phi):.3}")
                print(f"fidelity, trace distance: {m_res.fidelity:.3}, {m_res.trace_distance:.3}")
        
        # Add mitigated results to results dictionary
        for name in desired_state_name:
            results[qn][name]['mitigated'] = mitigate_data[qn][name]
        
        # Build superoperators (mitigated) for this qubit
        inputs = [rho_0, rho_1, rho_plus, rho_plus_i]
        outputs_mit = [
            mitigate_data[qn]['0']['density matrix'],
            mitigate_data[qn]['1']['density matrix'],
            mitigate_data[qn]['+']['density matrix'],
            mitigate_data[qn]['i+']['density matrix'],
        ]
        
        m_ptm = build_pauli_transfer_matrix(inputs, outputs_mit)
        m_superop = ptm_to_superop(m_ptm)
        m_choi = superop_to_choi(m_superop, 2, 2) / 2
        checker = CorrectChoi(choi)
        m_choi_cptp, count = checker.choi_checker(index=[1], repeat=100, tol=1e-6,print_reason=False)
        if verbose:
            print(f"\n{qn} - operation name: {operation_name}")
            print("Process fidelity with respect to the target process (MITIGATED):")
            print(f"mitigated: {process_fidelity(m_ptm, target=operation_name):.3}")
        
        # Add mitigated quantum information to results
        results[qn]['quantum_information']['mitigated'] = {
            'ptm': m_ptm,
            'superoperator': m_superop,
            'choi': m_choi_cptp,
            'Quantum Memory Robustness': entanglementRobustness(m_choi_cptp),
            'fidelity': process_fidelity(m_ptm, target=operation_name),
        }
    
    return results

import numpy as np
import xarray as xr

def random_sampling(ds, n_samples=100, sample_size=10000):
    """
    Perform random sampling with replacement on xarray data, maintain xarray format output
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Input data with shape (1, 10000, 4, 3)
    n_samples : int
        Number of sampling times, default 100
    sample_size : int
        Number of data points per sample, default 5000
    
    Returns:
    --------
    all_samples : xarray.Dataset
        All sampled data with new dimension 'sample', shape (sample: n_samples, ..., N: sample_size, ...)
    """
    sampled_data_list = []
    
    for i in range(n_samples):
        # Randomly select sample_size indices from 10000 (with replacement)
        indices = np.random.choice(10000, size=sample_size, replace=True)
        # Sample data according to these indices
        sampled = ds.isel(N=indices)
        sampled_data_list.append(sampled)
    
    # Use xarray.concat to combine all samples along new dimension, maintain xarray format
    # Use join='override' to handle duplicate index problem
    all_samples = xr.concat(sampled_data_list, dim='sample', join='override')
    return all_samples


def random_sampling_without_replacement(ds, n_samples=100, sample_size=100):
    """
    从n_points个点中不重复地随机抽样，进行多次采样。
    
    每次采样从数据集的所有n_points个点中，不重复地随机抽取sample_size个点，
    重复n_samples次，生成多个子数据集。
    
    Parameters:
    -----------
    ds : xarray.Dataset
        原始数据集，包含多个三维点（Bloch向量）
        维度通常为: (qubit, n_points, axis, ...) 或 (qubit, N, axis, ...)
    
    n_samples : int
        采样次数（生成多少个子数据集），默认100次
    
    sample_size : int
        每次采样选择的点数（必须 <= n_points），默认100
        这是用来fitting椭圆的点数
    
    Returns:
    --------
    sampled_datasets : list of xarray.Dataset
        包含n_samples个数据集的列表，每个数据集包含sample_size个点
    
    Example:
    --------
    >>> ds_list = random_sampling_without_replacement(ds, n_samples=100, sample_size=100)
    >>> # 现在有100个数据集，每个包含100个点
    >>> robustness_list = []
    >>> for ds_sample in ds_list:
    ...     rob, _ = compute_robustness(ds_sample, confusion_matrix)
    ...     robustness_list.append(rob)
    """
    # 识别数据点维度的名称 (可能是 'N' 或 'n_points')
    dim_name = None
    if 'n_points' in ds.dims:
        dim_name = 'n_points'
        n_points_total = len(ds.n_points.values)
    elif 'N' in ds.dims:
        dim_name = 'N'
        n_points_total = len(ds.N.values)
    else:
        # 自动寻找数据维度
        for dim in ds.dims:
            if dim not in ['qubit', 'axis', 'initial_state', 'theta', 'phi', 'sample']:
                dim_name = dim
                n_points_total = len(ds[dim].values)
                break
    
    if dim_name is None:
        raise ValueError(f"无法找到数据点维度。可用维度: {list(ds.dims)}")
    
    # 验证sample_size不超过总点数
    if sample_size > n_points_total:
        raise ValueError(f"sample_size ({sample_size}) 不能超过总点数 ({n_points_total})")
    
    sampled_datasets = []
    
    for i in range(n_samples):
        # 从所有n_points中不重复地随机选择sample_size个点
        sampled_indices = np.random.choice(n_points_total, size=sample_size, replace=False)
        sampled_indices = np.sort(sampled_indices)
        
        # 根据索引提取子数据集
        sampled_ds = ds.isel({dim_name: sampled_indices})
        sampled_datasets.append(sampled_ds)
    
    return sampled_datasets
    all_samples = xr.concat(sampled_data_list, dim='sample', join='override')
    return all_samples


def compute_robustness(ds, confusion_matrix, apply_mitigation=False, verbose=False):

    
    # 提取Bloch向量和角度数据 - 完全按照100f的方式
    data_xyz = np.array([
        [ds.Bloch_vector_x.values[0][i], 
         ds.Bloch_vector_y.values[0][i], 
         ds.Bloch_vector_z.values[0][i]] 
        for i in range(len(ds.n_points.values))
    ])
    
    data_angle = np.array([
        [ds.theta.values[i], ds.phi.values[i]] 
        for i in range(len(ds.n_points.values))
    ])
    
    if verbose:
        print(f"Extracted data_xyz shape: {data_xyz.shape}")
        print(f"Extracted data_angle shape: {data_angle.shape}")
    
    # 如果需要应用MLE缓解
    if apply_mitigation:
        # 计算概率 - 完全按照100f的方式
        x = data_xyz[:, 0]
        y = data_xyz[:, 1]
        z = data_xyz[:, 2]
        px, py, pz = (1 - x) / 2, (1 - y) / 2, (1 - z) / 2
        
        # 应用MLE - 完全按照100f的方式
        new_px_0 = np.array([MLE([1 - px[j], px[j]], confusion_matrix)[0] for j in range(len(px))])
        new_py_0 = np.array([MLE([1 - py[j], py[j]], confusion_matrix)[0] for j in range(len(py))])
        new_pz_0 = np.array([MLE([1 - pz[j], pz[j]], confusion_matrix)[0] for j in range(len(pz))])
        
        # 转换回Bloch向量 - 完全按照100f的方式
        data_xyz = np.array([2 * new_px_0 - 1, 2 * new_py_0 - 1, 2 * new_pz_0 - 1]).T
        
        if verbose:
            print(f"After MLE correction, data_xyz shape: {data_xyz.shape}")
    
    # 进行噪声分析和椭圆拟合 - 完全按照100f的方式
    noise_analyzer = QuantumMemoryAnalyze(data_xyz, data_angle)
    corrected_bloch = noise_analyzer.corrected_bloch
    ellipsoid_results = noise_analyzer.ellipsoid_fit()

    center, axes, R, volume, param = ellipsoid_results['center'], ellipsoid_results['axes'], ellipsoid_results['rotation_matrix'], ellipsoid_results['volume'], ellipsoid_results['fit_param']
    axes, R = find_best_fit(center, axes, R, param)
    
    if verbose:
        print(f"Ellipsoid fitting complete")
        print(f"  Center: {center}")
        print(f"  Axes: {axes}")
        print(f"  Volume: {volume}")

    # 计算robustness
    robustness = noise_analyzer.memory_robustness
    
    results_dict = {
        'axes': axes,
        'center': center,
        'R': R,
        'volume': volume,
        'param': param,
        'choi': noise_analyzer.choi,
        'robustness': robustness,
        'data_xyz': data_xyz,
        'corrected_bloch': corrected_bloch,
        'is_valid': True,  # 默认为有效（因为通过了choi_checker）
    }
    
    return robustness, results_dict


def filter_robustness_outliers(robustness_array, method='iqr', threshold=1.5, verbose=False):
    """
    使用科学方法移除robustness异常值
    
    Parameters:
    -----------
    robustness_array : np.ndarray
        robustness值数组
    
    method : str
        'iqr': 四分位数方法（推荐）
        'zscore': Z-score方法（threshold通常为3）
        'mad': 中位数绝对偏差方法
    
    threshold : float
        阈值
        - IQR方法：通常1.5（标准），3.0（极端）
        - Z-score方法：通常2.5（更严格）或3（标准）
    
    verbose : bool
        是否打印调试信息
    
    Returns:
    --------
    filtered_array : np.ndarray
        过滤后的数组
    
    mask : np.ndarray
        布尔掩码，True表示保留的数据
    
    stats : dict
        包含过滤统计信息的字典
    """
    original_count = len(robustness_array)
    
    if method == 'iqr':
        Q1 = np.percentile(robustness_array, 25)
        Q3 = np.percentile(robustness_array, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (robustness_array >= lower_bound) & (robustness_array <= upper_bound)
        
        stats = {
            'method': 'iqr',
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'threshold': threshold,
        }
    
    elif method == 'zscore':
        mean = np.mean(robustness_array)
        std = np.std(robustness_array)
        z_scores = np.abs((robustness_array - mean) / std)
        mask = z_scores < threshold
        
        stats = {
            'method': 'zscore',
            'mean': mean,
            'std': std,
            'threshold': threshold,
        }
    
    elif method == 'mad':  # Median Absolute Deviation
        median = np.median(robustness_array)
        mad = np.median(np.abs(robustness_array - median))
        modified_z_scores = 0.6745 * (robustness_array - median) / mad
        mask = np.abs(modified_z_scores) < threshold
        
        stats = {
            'method': 'mad',
            'median': median,
            'mad': mad,
            'threshold': threshold,
        }
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    filtered_array = robustness_array[mask]
    removed_count = original_count - len(filtered_array)
    
    stats['original_count'] = original_count
    stats['filtered_count'] = len(filtered_array)
    stats['removed_count'] = removed_count
    stats['removed_percentage'] = 100 * removed_count / original_count
    
    if verbose:
        print(f"Outlier Detection Results ({method}):")
        print(f"  Original count: {original_count}")
        print(f"  Filtered count: {len(filtered_array)}")
        print(f"  Removed: {removed_count} ({stats['removed_percentage']:.1f}%)")
        if method == 'iqr':
            print(f"  Bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")
        print(f"  Filtered data - Mean: {np.mean(filtered_array):.6f}, Std: {np.std(filtered_array):.6f}")
    
    return filtered_array, mask, stats

