"""
參數擬合工具

提供各種參數估計和不確定性分析方法。
"""

import numpy as np
from scipy.optimize import minimize
from scipy import stats

# 導入 lmfit 工具
from ..utils.lmfit_tools import lmfit_curve_fit, curve_fit_compatible

def parameter_fitting_analysis(fit_func, x_data, y_data, initial_guess, 
                             param_names=None, error_data=None):
    """
    執行參數擬合分析
    
    Parameters:
    -----------
    fit_func : callable
        擬合函數
    x_data : array-like
        自變量數據
    y_data : array-like
        因變量數據
    initial_guess : array-like
        初始參數猜測
    param_names : list, optional
        參數名稱列表
    error_data : array-like, optional
        測量誤差
    
    Returns:
    --------
    dict : 擬合結果
    """
    
    if param_names is None:
        param_names = [f"param_{i}" for i in range(len(initial_guess))]
    
    try:
        # 準備 lmfit 參數
        initial_params = {name: val for name, val in zip(param_names, initial_guess)}
        
        # 準備權重
        weights = None
        if error_data is not None:
            weights = 1.0 / np.asarray(error_data)
        
        # 執行擬合（使用 L-BFGS-B 演算法）
        best_params, param_errors, y_fit, fit_report = lmfit_curve_fit(
            fit_func, x_data, y_data, initial_params, weights, method='lbfgsb'
        )
        
        # 為兼容性創建 popt 和 pcov
        popt = np.array([best_params[name] for name in param_names])
        param_errors_array = np.array([param_errors[name] for name in param_names])
        pcov = np.diag(param_errors_array**2)
        
        # 計算殘差
        residuals = y_data - y_fit
        
        # 計算統計指標
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        n = len(y_data)
        p = len(popt)
        adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
        
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        
        # 參數不確定性
        param_errors = np.sqrt(np.diag(pcov)) if pcov is not None else np.zeros(len(popt))
        
        # 置信區間 (95%)
        alpha = 0.05
        dof = n - p
        t_val = stats.t.ppf(1 - alpha/2, dof)
        confidence_intervals = [(popt[i] - t_val * param_errors[i], 
                               popt[i] + t_val * param_errors[i]) 
                              for i in range(len(popt))]
        
        return {
            'success': True,
            'parameters': dict(zip(param_names, popt)),
            'parameter_errors': dict(zip(param_names, param_errors)),
            'confidence_intervals': dict(zip(param_names, confidence_intervals)),
            'covariance_matrix': pcov,
            'fitted_values': y_fit,
            'residuals': residuals,
            'r_squared': r_squared,
            'adjusted_r_squared': adjusted_r_squared,
            'rmse': rmse,
            'mae': mae,
            'aic': n * np.log(ss_res/n) + 2*p,
            'bic': n * np.log(ss_res/n) + p*np.log(n),
            'degrees_of_freedom': dof
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def bootstrap_uncertainty(fit_func, x_data, y_data, initial_guess, 
                        n_bootstrap=1000, confidence_level=0.95):
    """
    使用 Bootstrap 方法估計參數不確定性
    
    Parameters:
    -----------
    fit_func : callable
        擬合函數
    x_data : array-like
        自變量數據
    y_data : array-like
        因變量數據
    initial_guess : array-like
        初始參數猜測
    n_bootstrap : int
        Bootstrap 樣本數
    confidence_level : float
        置信水平
    
    Returns:
    --------
    dict : Bootstrap 不確定性分析結果
    """
    
    n_data = len(y_data)
    n_params = len(initial_guess)
    bootstrap_params = []
    
    print(f"執行 Bootstrap 不確定性分析 ({n_bootstrap} 次採樣)")
    
    for i in range(n_bootstrap):
        # 重採樣
        indices = np.random.choice(n_data, size=n_data, replace=True)
        x_boot = x_data[indices]
        y_boot = y_data[indices]
        
        try:
            # 準備 lmfit 參數（使用預設參數名稱）
            param_names = [f"param_{j}" for j in range(len(initial_guess))]
            initial_params = {name: val for name, val in zip(param_names, initial_guess)}
            
            # 擬合（使用 L-BFGS-B 演算法）
            best_params, _, _, _ = lmfit_curve_fit(
                fit_func, x_boot, y_boot, initial_params, method='lbfgsb'
            )
            
            # 轉換為陣列格式
            popt = np.array([best_params[name] for name in param_names])
            bootstrap_params.append(popt)
        except:
            continue
        
        if i % 100 == 0:
            print(f"  完成 {i}/{n_bootstrap}")
    
    if len(bootstrap_params) == 0:
        return {'success': False, 'error': '所有 Bootstrap 樣本擬合失敗'}
    
    bootstrap_params = np.array(bootstrap_params)
    
    # 計算統計量
    alpha = 1 - confidence_level
    percentiles = [100 * alpha/2, 50, 100 * (1 - alpha/2)]
    
    results = {}
    for i in range(n_params):
        param_values = bootstrap_params[:, i]
        lower, median, upper = np.percentile(param_values, percentiles)
        
        results[f'param_{i}'] = {
            'mean': np.mean(param_values),
            'std': np.std(param_values),
            'median': median,
            'confidence_interval': (lower, upper),
            'all_values': param_values
        }
    
    return {
        'success': True,
        'n_successful_fits': len(bootstrap_params),
        'parameters': results,
        'confidence_level': confidence_level
    }

def sensitivity_analysis(fit_func, x_data, y_data, best_params, param_names=None, 
                        perturbation_fraction=0.1):
    """
    參數敏感性分析
    
    Parameters:
    -----------
    fit_func : callable
        擬合函數
    x_data : array-like
        自變量數據
    y_data : array-like
        因變量數據
    best_params : array-like
        最佳參數值
    param_names : list, optional
        參數名稱
    perturbation_fraction : float
        擾動比例
    
    Returns:
    --------
    dict : 敏感性分析結果
    """
    
    if param_names is None:
        param_names = [f"param_{i}" for i in range(len(best_params))]
    
    # 基準擬合
    y_base = fit_func(x_data, *best_params)
    base_rmse = np.sqrt(np.mean((y_data - y_base)**2))
    
    sensitivity_results = {}
    
    for i, (param_name, param_value) in enumerate(zip(param_names, best_params)):
        # 正向擾動
        perturbed_params_pos = best_params.copy()
        perturbed_params_pos[i] *= (1 + perturbation_fraction)
        y_pos = fit_func(x_data, *perturbed_params_pos)
        rmse_pos = np.sqrt(np.mean((y_data - y_pos)**2))
        
        # 負向擾動
        perturbed_params_neg = best_params.copy()
        perturbed_params_neg[i] *= (1 - perturbation_fraction)
        y_neg = fit_func(x_data, *perturbed_params_neg)
        rmse_neg = np.sqrt(np.mean((y_data - y_neg)**2))
        
        # 計算敏感性
        sensitivity_pos = (rmse_pos - base_rmse) / (perturbation_fraction * abs(param_value))
        sensitivity_neg = (rmse_neg - base_rmse) / (perturbation_fraction * abs(param_value))
        avg_sensitivity = (abs(sensitivity_pos) + abs(sensitivity_neg)) / 2
        
        sensitivity_results[param_name] = {
            'sensitivity_positive': sensitivity_pos,
            'sensitivity_negative': sensitivity_neg,
            'average_sensitivity': avg_sensitivity,
            'rmse_positive': rmse_pos,
            'rmse_negative': rmse_neg
        }
    
    return {
        'base_rmse': base_rmse,
        'perturbation_fraction': perturbation_fraction,
        'sensitivities': sensitivity_results
    }