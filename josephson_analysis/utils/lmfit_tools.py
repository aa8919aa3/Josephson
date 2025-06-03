"""
LMfit 工具模組

提供基於 lmfit 的參數擬合工具，使用 L-BFGS-B 演算法
"""

import numpy as np
import lmfit
from typing import Callable, Dict, List, Optional, Tuple, Union


def create_lmfit_parameters(param_dict: Dict[str, Union[float, Tuple[float, float, float]]]) -> lmfit.Parameters:
    """
    創建 lmfit Parameters 對象
    
    Parameters:
    -----------
    param_dict : dict
        參數字典，格式為：
        - 'param_name': value  (固定值)
        - 'param_name': (value, min, max)  (帶邊界的參數)
        
    Returns:
    --------
    lmfit.Parameters : lmfit 參數對象
    """
    params = lmfit.Parameters()
    
    for name, value in param_dict.items():
        if isinstance(value, (tuple, list)) and len(value) == 3:
            # 帶邊界的參數
            val, vmin, vmax = value
            params.add(name, value=val, min=vmin, max=vmax)
        else:
            # 固定值參數
            params.add(name, value=value)
    
    return params


def lmfit_curve_fit(fit_func: Callable, x_data: np.ndarray, y_data: np.ndarray, 
                    initial_params: Dict[str, Union[float, Tuple[float, float, float]]],
                    weights: Optional[np.ndarray] = None,
                    method: str = 'lbfgsb') -> Tuple[Dict[str, float], Dict[str, float], np.ndarray, Dict]:
    """
    使用 lmfit 進行參數擬合，替代 scipy.optimize.curve_fit
    
    Parameters:
    -----------
    fit_func : callable
        擬合函數
    x_data : array-like
        自變量數據
    y_data : array-like
        因變量數據
    initial_params : dict
        初始參數字典
    weights : array-like, optional
        權重陣列（如果提供，會作為 sigma 的倒數）
    method : str
        擬合方法，默認使用 'lbfgsb' (L-BFGS-B)
        
    Returns:
    --------
    tuple : (best_params, param_errors, fitted_values, fit_report)
        - best_params: 最佳擬合參數字典
        - param_errors: 參數誤差字典
        - fitted_values: 擬合值陣列
        - fit_report: 詳細擬合報告
    """
    
    def residual_func(params, x, y, weights=None):
        """殘差函數"""
        # 提取參數值
        param_values = {name: params[name].value for name in params.keys()}
        
        # 計算模型值
        model_y = fit_func(x, **param_values)
        
        # 計算殘差
        residuals = y - model_y
        
        # 應用權重
        if weights is not None:
            residuals *= weights
            
        return residuals
    
    # 創建 lmfit Parameters
    params = create_lmfit_parameters(initial_params)
    
    # 準備權重
    if weights is not None:
        weights = np.asarray(weights)
    
    # 執行擬合
    minimizer = lmfit.Minimizer(residual_func, params, 
                               fcn_args=(x_data, y_data, weights))
    
    result = minimizer.minimize(method=method)
    
    # 提取結果
    best_params = {name: result.params[name].value for name in result.params.keys()}
    param_errors = {name: result.params[name].stderr or 0.0 for name in result.params.keys()}
    
    # 計算擬合值
    fitted_values = fit_func(x_data, **best_params)
    
    # 創建詳細報告
    fit_report = {
        'success': result.success,
        'chi_squared': result.chisqr,
        'reduced_chi_squared': result.redchi,
        'aic': result.aic,
        'bic': result.bic,
        'nfev': result.nfev,
        'message': result.message,
        'method': result.method,
        'params': result.params
    }
    
    return best_params, param_errors, fitted_values, fit_report


def estimate_parameter_uncertainties(fit_func: Callable, x_data: np.ndarray, y_data: np.ndarray,
                                   best_params: Dict[str, float], 
                                   weights: Optional[np.ndarray] = None,
                                   confidence_level: float = 0.95) -> Dict[str, Tuple[float, float]]:
    """
    估計參數的置信區間
    
    Parameters:
    -----------
    fit_func : callable
        擬合函數
    x_data : array-like
        自變量數據
    y_data : array-like
        因變量數據
    best_params : dict
        最佳擬合參數
    weights : array-like, optional
        權重陣列
    confidence_level : float
        置信水平 (0-1)
        
    Returns:
    --------
    dict : 參數置信區間字典
    """
    
    def residual_func(params, x, y, weights=None):
        param_values = {name: params[name].value for name in params.keys()}
        model_y = fit_func(x, **param_values)
        residuals = y - model_y
        if weights is not None:
            residuals *= weights
        return residuals
    
    # 重新創建參數
    params = lmfit.Parameters()
    for name, value in best_params.items():
        params.add(name, value=value)
    
    # 執行擬合
    minimizer = lmfit.Minimizer(residual_func, params, 
                               fcn_args=(x_data, y_data, weights))
    result = minimizer.minimize(method='lbfgsb')
    
    # 計算置信區間
    try:
        ci = lmfit.conf_interval(minimizer, result, sigmas=[confidence_level])
        confidence_intervals = {}
        
        for param_name in best_params.keys():
            if param_name in ci:
                intervals = ci[param_name]
                # 提取置信區間
                lower = intervals[0][1]  # 下界
                upper = intervals[-1][1]  # 上界
                confidence_intervals[param_name] = (lower, upper)
            else:
                confidence_intervals[param_name] = (best_params[param_name], best_params[param_name])
                
        return confidence_intervals
        
    except Exception as e:
        print(f"警告：無法計算置信區間 - {e}")
        return {name: (value, value) for name, value in best_params.items()}


def convert_scipy_to_lmfit_params(param_names: List[str], initial_guess: List[float],
                                 bounds: Optional[List[Tuple[float, float]]] = None) -> Dict[str, Union[float, Tuple[float, float, float]]]:
    """
    將 scipy.optimize.curve_fit 格式的參數轉換為 lmfit 格式
    
    Parameters:
    -----------
    param_names : list
        參數名稱列表
    initial_guess : list
        初始猜測值列表
    bounds : list of tuples, optional
        參數邊界列表 [(min, max), ...]
        
    Returns:
    --------
    dict : lmfit 參數字典
    """
    param_dict = {}
    
    for i, name in enumerate(param_names):
        if bounds is not None and i < len(bounds):
            min_val, max_val = bounds[i]
            param_dict[name] = (initial_guess[i], min_val, max_val)
        else:
            param_dict[name] = initial_guess[i]
    
    return param_dict


# 兼容性函數：模擬 scipy.optimize.curve_fit 的接口
def curve_fit_compatible(fit_func: Callable, x_data: np.ndarray, y_data: np.ndarray,
                        p0: Optional[List[float]] = None,
                        sigma: Optional[np.ndarray] = None,
                        bounds: Optional[Tuple[List[float], List[float]]] = None,
                        maxfev: int = 1000,
                        **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    與 scipy.optimize.curve_fit 兼容的接口
    
    Returns:
    --------
    tuple : (popt, pcov)
        - popt: 最佳擬合參數陣列
        - pcov: 參數協方差矩陣（簡化版）
    """
    
    # 獲取函數參數名稱
    import inspect
    sig = inspect.signature(fit_func)
    param_names = list(sig.parameters.keys())[1:]  # 排除第一個參數（通常是 x）
    
    # 設置初始猜測
    if p0 is None:
        p0 = [1.0] * len(param_names)
    
    # 準備參數字典
    if bounds is not None:
        lower_bounds, upper_bounds = bounds
        param_bounds = list(zip(lower_bounds, upper_bounds))
        initial_params = convert_scipy_to_lmfit_params(param_names, p0, param_bounds)
    else:
        initial_params = convert_scipy_to_lmfit_params(param_names, p0)
    
    # 準備權重
    weights = None
    if sigma is not None:
        weights = 1.0 / np.asarray(sigma)
    
    # 執行擬合
    best_params, param_errors, fitted_values, fit_report = lmfit_curve_fit(
        fit_func, x_data, y_data, initial_params, weights, method='lbfgsb'
    )
    
    # 轉換為 numpy 陣列格式
    popt = np.array([best_params[name] for name in param_names])
    
    # 創建簡化的協方差矩陣（對角矩陣）
    param_errors_array = np.array([param_errors[name] for name in param_names])
    pcov = np.diag(param_errors_array**2)
    
    return popt, pcov
